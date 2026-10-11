# RevIN source: https://github.com/decisionintelligence/CATCH/blob/master/ts_benchmark/baselines/catch/layers/RevIN.py
import torch
import torch.nn as nn


class RevIN1d(nn.Module):
    """Normalize patches and retain their per-channel amplitude statistics."""

    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self.min_sigma = 1e-5
        self._mu = None
        self._sigma = None

    @torch.no_grad()
    def _stats(self, x):
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        sigma = (var + self.eps).sqrt().clamp_min(self.min_sigma)
        return (mu, sigma)

    def norm(self, x):
        self._mu, self._sigma = self._stats(x)
        x_hat = (x - self._mu) / self._sigma
        return x_hat
