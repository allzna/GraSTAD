"""Accumulate CPU/CUDA component wall times without changing function returns."""
from contextlib import contextmanager
from time import perf_counter

import torch


@contextmanager
def record_time(timings, key, device=None):
    """Add seconds to an optional per-series dict; synchronize CUDA boundaries."""
    if timings is None:
        yield
        return

    cuda_device = None
    if device is not None and torch.device(device).type == "cuda":
        cuda_device = torch.device(device)
        torch.cuda.synchronize(cuda_device)
    started = perf_counter()
    try:
        yield
    finally:
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        timings[key] = timings.get(key, 0.0) + perf_counter() - started
