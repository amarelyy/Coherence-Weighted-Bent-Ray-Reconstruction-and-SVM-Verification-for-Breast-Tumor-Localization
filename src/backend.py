"""
src/backend.py
GPU/CPU backend abstraction. Auto-detects CuPy availability and provides
a unified array API. All downstream modules import `xp` from here instead
of importing numpy/cupy directly.

Usage:
    from src.backend import xp, HAS_GPU, to_gpu, to_cpu, get_array_module
"""

import numpy as np

_HAS_GPU = False
_xp = np  # default: numpy

try:
    import cupy as cp
    # Verify a CUDA device is actually available
    if cp.cuda.runtime.getDeviceCount() > 0:
        _HAS_GPU = True
        _xp = cp
        # Use managed memory for easier CPU<->GPU interop
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
except (ImportError, Exception):
    pass

xp = _xp
HAS_GPU = _HAS_GPU


def to_gpu(arr):
    """Move a numpy/cupy array to GPU. No-op if already on GPU or no GPU."""
    if not _HAS_GPU:
        return np.asarray(arr)
    return cp.asarray(arr)


def to_cpu(arr):
    """Move a cupy array back to CPU numpy. No-op if already numpy."""
    if _HAS_GPU:
        try:
            return cp.asnumpy(arr)
        except TypeError:
            return np.asarray(arr)
    return np.asarray(arr)


def get_array_module(arr):
    """Return cupy if arr is a cupy array, else numpy."""
    if _HAS_GPU:
        return cp.get_array_module(arr)
    return np


def clear_gpu_cache():
    """Free GPU memory pool. Call between large scans if needed."""
    if _HAS_GPU:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def gpu_info():
    """Print GPU device info. Useful for debugging."""
    if not _HAS_GPU:
        return "No GPU detected — running on CPU (NumPy)"
    dev = cp.cuda.runtime.getDeviceProperties(0)
    mem_free, mem_total = cp.cuda.runtime.memGetInfo(0)
    return (
        f"GPU: {dev['name'].decode()}\n"
        f"  Memory: {mem_free/1e9:.1f} GB free / {mem_total/1e9:.1f} GB total\n"
        f"  Compute capability: {dev['major']}.{dev['minor']}"
    )