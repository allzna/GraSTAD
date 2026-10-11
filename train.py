import copy
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
from model import build_graph, build_temporal_edges
from utils.timing import record_time


def train_model(
    model, train_loader, train_patches, device, num_iter=100, lr=0.0001, timings=None
):
    """Train with graph-neighbor positives, non-neighbor negatives and a margin triplet loss."""
    radius = 2
    K_pos = 1
    temperature = 1.0
    initial_lr = lr
    final_lr = lr / 10

    def cosine_annealed_lr(iteration):
        t = min(iteration, num_iter)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * t / num_iter))
        return final_lr + (initial_lr - final_lr) * cosine_factor

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=0.0001
    )
    iteration_count = 0
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    print("    [Training] Graph-guided triplet learning")
    pbar = tqdm(total=num_iter, desc="    >> Training", ncols=80)
    _offsets = torch.tensor(
        [*range(-radius, 0), *range(1, radius + 1)], dtype=torch.long
    )
    while iteration_count < num_iter:
        for batch_data, batch_indexes in train_loader:
            if iteration_count >= num_iter:
                break
            iteration_count += 1
            lr = cosine_annealed_lr(iteration_count)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            batch_data = batch_data.to(device, non_blocking=True)
            batch_indexes = batch_indexes.squeeze(1)
            M = batch_data.shape[0]
            mu = 10
            total_len = len(train_patches)
            # Include one temporal neighbor per anchor before constructing the graph.
            _cand = batch_indexes.view(-1, 1) + _offsets.view(1, -1)
            _valid = (_cand >= 0) & (_cand < total_len)
            all_pos_idx = []
            for k in range(K_pos):
                if _valid.any():
                    _noise = torch.rand_like(_cand.float())
                    _score = torch.where(
                        _valid, _noise, -torch.ones_like(_noise) * 1000000000.0
                    )
                    _choice = _score.argmax(dim=1)
                    _pos_idx = _cand.gather(1, _choice.unsqueeze(1)).squeeze(1)
                    _none_valid = _valid.sum(dim=1) == 0
                    if _none_valid.any():
                        _pos_idx[_none_valid] = batch_indexes[_none_valid]
                    _valid[torch.arange(M, device=device), _choice] = False
                else:
                    _pos_idx = batch_indexes.clone()
                all_pos_idx.append(_pos_idx)
            all_pos_idx_t = torch.stack(all_pos_idx, dim=1).reshape(-1)
            raw_all_indices = torch.cat([batch_indexes, all_pos_idx_t], dim=0)
            # Encode repeated patch indices once, then recover each anchor's location.
            unique_indices, inverse_map = torch.unique(
                raw_all_indices, return_inverse=True
            )
            N_unique = unique_indices.shape[0]
            unique_patches = torch.stack(
                [train_patches[i] for i in unique_indices.tolist()], dim=0
            ).to(device, non_blocking=True)
            anchor_local = inverse_map[:M]
            all_cnn = model.cnn_embedding(unique_patches)
            C = all_cnn.shape[1]
            with record_time(timings, "Graph_Time_s", device):
                temporal_ei = build_temporal_edges(unique_indices)
            triplet_loss = torch.tensor(0.0, device=device)
            for c in range(C):
                all_cnn_c = all_cnn[:, c, :].contiguous()
                raw_c = unique_patches[:, c : c + 1, :].contiguous()
                with record_time(timings, "Graph_Time_s", device):
                    edge_index_c = build_graph(
                        raw_c, unique_indices, temporal_ei=temporal_ei
                    )
                all_feat_c = all_cnn_c
                h_anchors_feat = all_feat_c[anchor_local]
                src = edge_index_c[0]
                dst = edge_index_c[1]
                adj_mat = torch.zeros(N_unique, N_unique, device=device)
                adj_mat[src, dst] = 1.0
                adj_mat[dst, src] = 1.0
                adj_mat.fill_diagonal_(0.0)
                # Draw positives from graph neighbors and negatives from non-neighbors.
                pos_scores = torch.rand(N_unique, N_unique, device=device)
                pos_scores = pos_scores * adj_mat
                no_neighbour = pos_scores.sum(dim=1) == 0
                pos_scores[no_neighbour, 0] = 1.0
                pos_idx = pos_scores.argmax(dim=1)
                not_nb = 1.0 - adj_mat
                not_nb[
                    torch.arange(N_unique, device=device),
                    torch.arange(N_unique, device=device),
                ] = 0.0
                neg_scores = torch.rand(N_unique, N_unique, device=device)
                neg_scores = neg_scores * not_nb
                no_candidate = neg_scores.sum(dim=1) == 0
                neg_idx = neg_scores.argmax(dim=1)
                neg_idx[no_candidate] = -1
                anchor_u = anchor_local
                h_pos_list = all_feat_c[pos_idx[anchor_u]]
                z_anchor = F.normalize(model.projection(h_anchors_feat), dim=1)
                z_pos = F.normalize(model.projection(h_pos_list), dim=1)
                pos_dists = 1 - (z_anchor * z_pos).sum(dim=1) / temperature
                neg_u = neg_idx[anchor_u]
                fallback_mask = neg_u < 0
                neg_u_safe = neg_u.clamp(min=0)
                h_neg_list = all_feat_c[neg_u_safe]
                z_neg = F.normalize(model.projection(h_neg_list), dim=1)
                neg_sim = (z_anchor * z_neg).sum(dim=1) / temperature
                hard_neg_dists = 1 - neg_sim
                # Fall back to batch hard negatives when no non-neighbor is available.
                if fallback_mask.any():
                    _sim_aa = z_anchor @ z_anchor.T / temperature
                    _sim_aa.diagonal().fill_(-float("inf"))
                    node_is_a = torch.zeros(N_unique, dtype=torch.bool, device=device)
                    node_is_a[anchor_local] = True
                    both_a = node_is_a[src] & node_is_a[dst]
                    ap_adj = torch.zeros(M, M, dtype=torch.bool, device=device)
                    if both_a.any():
                        n2a = torch.full(
                            (N_unique,), -1, dtype=torch.long, device=device
                        )
                        n2a[anchor_local] = torch.arange(M, device=device)
                        es = n2a[src[both_a]]
                        ed = n2a[dst[both_a]]
                        ok = (es >= 0) & (ed >= 0)
                        ap_adj[es[ok], ed[ok]] = True
                    _sim_aa[ap_adj] = -float("inf")
                    fb_sim, _ = torch.max(_sim_aa, dim=1)
                    fb_idx = fallback_mask.nonzero(as_tuple=True)[0]
                    hard_neg_dists[fb_idx] = 1 - fb_sim[fb_idx]
                channel_loss = F.relu(pos_dists - hard_neg_dists + 0.1).mean() / mu
                triplet_loss = triplet_loss + channel_loss
            triplet_loss = triplet_loss / C
            final_loss = triplet_loss
            optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            optimizer.step()
            pbar.update(1)
            if final_loss.item() < best_loss:
                best_loss = final_loss.item()
                best_model_wts = copy.deepcopy(model.state_dict())
    pbar.close()
    model.load_state_dict(best_model_wts)
