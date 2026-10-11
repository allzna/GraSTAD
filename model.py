import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import RevIN1d


def build_temporal_edges(batch_indices):
    """Build temporal edges with a fixed radius of three patch positions.

    Args:
        batch_indices: (N,)  absolute patch indices in the full sequence
    Returns:
        edge_index: (2, E_t)  undirected (includes both directions)
    """
    N = len(batch_indices)
    device = batch_indices.device
    idx_list = batch_indices.tolist()
    idx_to_nodes = {}
    for pos, gid in enumerate(idx_list):
        idx_to_nodes.setdefault(gid, []).append(pos)
    t_edges = set()
    for i in range(N):
        gi = idx_list[i]
        for offset in range(1, 4):
            for direction in (-1, 1):
                target = gi + offset * direction
                for j in idx_to_nodes.get(target, ()):
                    if i != j:
                        edge = (i, j) if i < j else (j, i)
                        t_edges.add(edge)
    all_edges = list(t_edges)
    if not all_edges:
        all_edges = [(i, i) for i in range(N)]
    edge_index = (
        torch.tensor(all_edges, dtype=torch.long, device=device).t().contiguous()
    )
    rev = edge_index.flip(0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    return edge_index


def build_graph(raw_patches, batch_indices, temporal_ei=None):
    """Build undirected temporal and mean-centered cosine-similarity edges.

    Temporal radius is 3; each patch selects up to 5 similarity neighbors."""
    N = raw_patches.shape[0]
    device = raw_patches.device
    if temporal_ei is None:
        temporal_ei = build_temporal_edges(batch_indices)
    temporal_ei = temporal_ei.to(device)
    if raw_patches.ndim == 2:
        raw_patches = raw_patches.unsqueeze(1)
    flat_raw = raw_patches.reshape(N, -1).float()
    # Center each patch so similarity is invariant to level shifts.
    flat_raw = flat_raw - flat_raw.mean(dim=1, keepdim=True)
    fn = F.normalize(flat_raw, p=2, dim=1)
    fused = fn @ fn.T
    fused.diagonal().fill_(-1.0)
    k = min(5, N - 1)
    _, topk = torch.topk(fused, k=k, dim=1)
    src = torch.arange(N, device=device).unsqueeze(1).expand(N, k).reshape(-1)
    dst = topk.reshape(-1)
    swap = src > dst
    src2 = torch.where(swap, dst, src)
    dst2 = torch.where(swap, src, dst)
    edges = torch.stack([src2, dst2], dim=0)
    keep = edges[0] != edges[1]
    edges = edges[:, keep]
    edges = torch.unique(edges, dim=1)
    if edges.size(1) == 0:
        idx = torch.arange(N, device=device)
        edges = torch.stack([idx, idx], dim=0)
    rev = edges.flip(0)
    sim_ei = torch.cat([edges, rev], dim=1)
    edge_index = torch.cat([temporal_ei, sim_ei], dim=1)
    return edge_index


class PatchEncoder(nn.Module):

    def __init__(self, in_channels=1, use_revin=True):
        projection_dim = 256
        per_channel_dim = 32
        intermediate_dim = 64
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        C = in_channels
        C_eff = C
        if C == 1:
            per_channel_dim = 64
        else:
            budget = 4096
            per_channel_dim = min(per_channel_dim, max(1, budget // C_eff))
            intermediate_dim = min(intermediate_dim, max(1, budget // C_eff))
        self.per_channel_dim = per_channel_dim
        self.revin = None
        if use_revin:
            self.revin = RevIN1d()
        if C == 1:
            raw_layers = [128, 256, 128, 64]
            raw_kss = [7, 5, 3, 3]
            self.grouped_convs = nn.Sequential()
            for i in range(4):
                in_ch = raw_layers[i - 1] if i > 0 else 1
                out_ch = raw_layers[i]
                self.grouped_convs.add_module(
                    f"conv_raw_{i}",
                    nn.Sequential(
                        nn.Conv1d(
                            in_ch,
                            out_ch,
                            kernel_size=raw_kss[i],
                            stride=1,
                            padding=raw_kss[i] // 2,
                            bias=False,
                        ),
                        nn.BatchNorm1d(out_ch),
                        nn.ReLU(inplace=True),
                    ),
                )
        else:
            self.grouped_convs = nn.Sequential(
                nn.Conv1d(
                    C_eff,
                    C_eff * intermediate_dim,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    groups=C_eff,
                    bias=False,
                ),
                nn.BatchNorm1d(C_eff * intermediate_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    C_eff * intermediate_dim,
                    C_eff * intermediate_dim,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    groups=C_eff,
                    bias=False,
                ),
                nn.BatchNorm1d(C_eff * intermediate_dim),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    C_eff * intermediate_dim,
                    C_eff * per_channel_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=C_eff,
                    bias=False,
                ),
                nn.BatchNorm1d(C_eff * per_channel_dim),
                nn.ReLU(inplace=True),
            )
        self.fc_embedding = nn.AdaptiveAvgPool1d(output_size=1)
        self.projection_head = nn.Sequential(
            nn.Linear(per_channel_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def _cnn_forward(self, x):
        """Grouped conv → per-channel CNN embedding.

        Args:
            x: (N, C, L)  N=#patches, C=#channels, L=patch_len
        Returns:
            h: (N, C, D)  D=per_channel_dim, channel-independent embeddings
        """
        N, C_orig, L = x.shape
        if self.revin is not None:
            x = self.revin.norm(x)
        x = self.grouped_convs(x)
        h = self.fc_embedding(x).flatten(start_dim=1)
        h = h.view(N, C_orig, self.per_channel_dim)
        return h

    def cnn_embedding(self, x):
        """Return per-channel raw CNN embedding: (N, C, D)."""
        return self._cnn_forward(x)

    def projection(self, h):
        """Project single-channel embedding to metric space.
        h: (N, D) → (N, projection_dim).
        """
        return self.projection_head(h)
