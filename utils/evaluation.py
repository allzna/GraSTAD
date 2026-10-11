import torch
import torch.nn.functional as F
import numpy as np
from time import perf_counter
from utils.timing import record_time


def _subsample_bank(bank, timings=None):
    """Select representatives using the fixed 10% coreset and 500-point floor."""
    n_samples = bank.shape[0]
    k = int(round(0.1 * n_samples))
    min_eff = min(500, max(1, n_samples - 1))
    k = max(min_eff, min(k, n_samples - 1))
    if k >= n_samples:
        return bank
    from sklearn.cluster import MiniBatchKMeans

    with record_time(timings, "Cluster_Time_s", bank.device):
        flat = F.normalize(bank, p=2, dim=1)
        mbk = MiniBatchKMeans(
            n_clusters=k,
            init="k-means++",
            random_state=42,
            batch_size=max(8192, k),
            max_iter=50,
            n_init=1,
            reassignment_ratio=0.01,
        )
        mbk.fit(flat.numpy())
        centers = torch.tensor(mbk.cluster_centers_, dtype=flat.dtype)
        distances = torch.cdist(flat, centers, p=2)
        core_indices = torch.argmin(distances, dim=0)
        return bank[core_indices]


def _build_amp_stats(all_mu, all_sigma):
    """Fit a regularized Gaussian to (patch mean, log standard deviation) per channel."""
    all_mu = torch.nan_to_num(all_mu, nan=0.0, posinf=0.0, neginf=0.0)
    all_sigma = torch.nan_to_num(all_sigma, nan=1e-05, posinf=1.0, neginf=1e-05)
    C = all_mu.shape[1]
    log_sigma_all = torch.log(all_sigma.clamp(min=1e-05))
    log_sigma_all = torch.nan_to_num(log_sigma_all, nan=0.0, posinf=0.0, neginf=0.0)
    mahalanobis_stats = []
    eye2 = torch.eye(2, dtype=torch.float64)
    for c in range(C):
        ac = torch.stack([all_mu[:, c], log_sigma_all[:, c]], dim=1).to(torch.float64)
        mean_vec = ac.mean(dim=0)
        centered = ac - mean_vec
        n_eff = max(1, ac.shape[0] - 1)
        cov = centered.T @ centered / n_eff + 0.0001 * eye2
        cov = torch.nan_to_num(cov, nan=0.0001, posinf=1.0, neginf=-1.0)
        try:
            inv_cov = torch.linalg.pinv(cov)
        except RuntimeError:
            inv_cov = eye2.clone()
        inv_cov = torch.nan_to_num(inv_cov, nan=0.0, posinf=0.0, neginf=0.0)
        mahalanobis_stats.append({"mean": mean_vec.float(), "inv_cov": inv_cov.float()})
    return {"mahalanobis": mahalanobis_stats}


def _mahalanobis_per_channel(mu_raw, sigma_raw, amp_stats, device):
    """(μ, logσ) → Mahalanobis distance per channel → (C, B)."""
    mu = torch.nan_to_num(mu_raw, nan=0.0, posinf=0.0, neginf=0.0)
    log_s = torch.log(sigma_raw.clamp(min=1e-05))
    log_s = torch.nan_to_num(log_s, nan=0.0, posinf=0.0, neginf=0.0)
    C_amp = mu.shape[1]
    m_stats = amp_stats["mahalanobis"]
    dists = []
    for c in range(C_amp):
        amp_vec = torch.stack([mu[:, c], log_s[:, c]], dim=1)
        mean_c = m_stats[c]["mean"].to(device)
        inv_cov_c = m_stats[c]["inv_cov"].to(device)
        diff = amp_vec - mean_c.unsqueeze(0)
        quad = (diff @ inv_cov_c * diff).sum(dim=1)
        md = torch.sqrt(quad.clamp(min=0))
        md = torch.nan_to_num(md, nan=0.0, posinf=0.0, neginf=0.0)
        dists.append(md)
    return torch.stack(dists, dim=0)


@torch.no_grad()
def create_per_channel_memory_bank(model, data_loader, device, timings=None):
    """Create independent shape memory banks and per-channel Mahalanobis statistics."""
    model.eval()
    has_revin = hasattr(model, "revin") and model.revin is not None
    print("    [Memory] Extracting training representations...", flush=True)
    all_h = []
    all_mu = []
    all_sigma = []
    for data, batch_indices in data_loader:
        data = data.to(device)
        h = model.cnn_embedding(data)
        all_h.append(h.detach().cpu().float())
        if has_revin:
            all_mu.append(model.revin._mu.squeeze(-1).detach().cpu().float())
            all_sigma.append(model.revin._sigma.squeeze(-1).detach().cpu().float())
    all_h = torch.cat(all_h, dim=0)
    N, C, D = all_h.shape
    print(f"    [Memory] {N} patches, {C} channels, {D} features/channel", flush=True)
    per_channel_banks = []
    for c in range(C):
        started = perf_counter()
        print(
            f"    [Memory] Channel {c + 1}/{C}: clustering and selecting representatives...",
            flush=True,
        )
        per_channel_banks.append(_subsample_bank(all_h[:, c, :], timings=timings))
        print(
            f"    [Memory] Channel {c + 1}/{C}: {len(per_channel_banks[-1])} vectors, {perf_counter() - started:.2f}s",
            flush=True,
        )
    amp_stats = None
    if has_revin and all_mu:
        all_mu = torch.cat(all_mu, dim=0)
        all_sigma = torch.cat(all_sigma, dim=0)
        print("    [Statistics] Fitting Mahalanobis references...", flush=True)
        amp_stats = _build_amp_stats(all_mu, all_sigma)
    return (per_channel_banks, amp_stats)


@torch.inference_mode()
def calculate_per_channel_anomaly_scores(
    model,
    data_loader,
    per_channel_banks,
    device,
    train_loader=None,
    amp_stats=None,
    scoring_stats=None,
    return_scoring_stats=False,
):
    """Average per-channel shape and amplitude scores.

    With RevIN, calibrate both scores on training data and fuse as
    z_shape + 0.3 * z_amplitude. Without RevIN, use raw shape scores.
    Set return_scoring_stats to fit calibration without scoring data_loader."""
    model.eval()
    has_amp = (
        amp_stats is not None and hasattr(model, "revin") and (model.revin is not None)
    )
    norm_banks = [
        F.normalize(b.to(device, dtype=torch.float32), dim=1, eps=1e-12)
        for b in per_channel_banks
    ]
    C = len(norm_banks)

    def _score_batch(h_per_ch):
        """h_per_ch: (B, C, D) → per-channel cosine distances → (C, B)."""
        per_channel_scores = []
        for c in range(C):
            feats = h_per_ch[:, c, :]
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            feats = F.normalize(feats, dim=1, eps=1e-12)
            feats = torch.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            sims = feats @ norm_banks[c].T
            sims = torch.nan_to_num(sims, nan=-1.0, posinf=1.0, neginf=-1.0)
            k = min(3, sims.shape[1])
            topk_sim, _ = torch.topk(sims, k=k, dim=1, largest=True)
            dists = 1.0 - topk_sim
            scores_c = dists.mean(dim=1)
            scores_c = torch.nan_to_num(scores_c, nan=1.0, posinf=1.0, neginf=0.0)
            per_channel_scores.append(scores_c)
        return torch.stack(per_channel_scores, dim=0)

    if scoring_stats is not None and train_loader is not None:
        raise ValueError("Pass either scoring_stats or train_loader, not both")
    train_scores_per_ch = (
        None
        if not has_amp
        else [[] for _ in range(C)] if train_loader is not None else None
    )
    train_amp_per_ch = (
        None
        if not has_amp
        else [[] for _ in range(C)] if train_loader is not None else None
    )
    if has_amp and train_loader is not None:
        for data, batch_indices in train_loader:
            data = data.to(device, non_blocking=True, dtype=torch.float32)
            h = model.cnn_embedding(data)
            stacked = _score_batch(h)
            for c in range(C):
                train_scores_per_ch[c].append(stacked[c])
            if has_amp:
                mu_raw = model.revin._mu.squeeze(-1)
                sigma_raw = model.revin._sigma.squeeze(-1)
                amp_dists = _mahalanobis_per_channel(
                    mu_raw, sigma_raw, amp_stats, device
                )
                for c in range(C):
                    train_amp_per_ch[c].append(amp_dists[c])
    ch_mu_shape = torch.zeros(C, device=device)
    ch_sigma_shape = torch.ones(C, device=device)
    ch_mu_amp = torch.zeros(C, device=device)
    ch_sigma_amp = torch.ones(C, device=device)
    if train_scores_per_ch is not None:
        for c in range(C):
            all_c = torch.cat(train_scores_per_ch[c], dim=0)
            ch_mu_shape[c] = all_c.mean()
            ch_sigma_shape[c] = all_c.std().clamp(min=0.01)
    if has_amp and train_amp_per_ch is not None:
        for c in range(C):
            if train_amp_per_ch[c]:
                all_amp_c = torch.cat(train_amp_per_ch[c], dim=0)
                ch_mu_amp[c] = all_amp_c.mean()
                ch_sigma_amp[c] = all_amp_c.std().clamp(min=0.01)
    if scoring_stats is None:
        scoring_stats = {
            "ch_mu_shape": ch_mu_shape,
            "ch_sigma_shape": ch_sigma_shape,
            "ch_mu_amp": ch_mu_amp,
            "ch_sigma_amp": ch_sigma_amp,
        }
    else:
        ch_mu_shape = scoring_stats["ch_mu_shape"].to(device)
        ch_sigma_shape = scoring_stats["ch_sigma_shape"].to(device)
        ch_mu_amp = scoring_stats["ch_mu_amp"].to(device)
        ch_sigma_amp = scoring_stats["ch_sigma_amp"].to(device)
    if return_scoring_stats:
        return scoring_stats
    all_scores = []
    sum_abs_shape = 0.0
    sum_abs_amp = 0.0
    for data, batch_indices in data_loader:
        data = data.to(device, non_blocking=True, dtype=torch.float32)
        h_per_ch = model.cnn_embedding(data)
        stacked = _score_batch(h_per_ch)
        z_shape = (stacked - ch_mu_shape.unsqueeze(1)) / (
            ch_sigma_shape.unsqueeze(1) + 1e-08
        )
        z_fused = z_shape
        if has_amp:
            mu_raw = model.revin._mu.squeeze(-1)
            sigma_raw = model.revin._sigma.squeeze(-1)
            amp_dists = _mahalanobis_per_channel(mu_raw, sigma_raw, amp_stats, device)
            z_amp = (amp_dists - ch_mu_amp.unsqueeze(1)) / (
                ch_sigma_amp.unsqueeze(1) + 1e-08
            )
            sum_abs_shape += z_shape.abs().sum().item()
            sum_abs_amp += (0.3 * z_amp).abs().sum().item()
            z_fused = z_shape + 0.3 * z_amp
        per_ch_final = z_fused if has_amp else stacked
        batch_scores = per_ch_final.mean(dim=0)
        batch_scores = torch.nan_to_num(batch_scores, nan=0.0, posinf=0.0, neginf=0.0)
        all_scores.extend(batch_scores.cpu().tolist())
    if has_amp:
        total = sum_abs_shape + sum_abs_amp
        shape_pct = sum_abs_shape / total * 100 if total > 0 else 0
        amp_pct = sum_abs_amp / total * 100 if total > 0 else 0
        print(
            f"    [info] score contribution: shape={shape_pct:.1f}%  amplitude(Mahalanobis)={amp_pct:.1f}% (weight=0.30)"
        )
    return all_scores


def distribute_patch_scores_to_points(patch_scores, patch_size, num_points):
    patch_scores = np.nan_to_num(
        np.asarray(patch_scores, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    kernel = np.ones(patch_size, dtype=np.float32)
    sums = np.convolve(patch_scores, kernel, mode="full")[:num_points]
    counts = np.convolve(np.ones_like(patch_scores), kernel, mode="full")[:num_points]
    if len(sums) < num_points:
        sums = np.pad(sums, (0, num_points - len(sums)), constant_values=0.0)
        counts = np.pad(counts, (0, num_points - len(counts)), constant_values=0.0)
    point_scores = np.divide(
        sums, counts, out=np.zeros(num_points, dtype=np.float32), where=counts != 0
    )
    return np.nan_to_num(point_scores, nan=0.0, posinf=0.0, neginf=0.0)
