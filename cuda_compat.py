"""Runtime compatibility helpers for mixed CUDA toolchains.

PyTorch wheels ship CUDA 13 libraries while CTranslate2 (faster-whisper) is
built against CUDA 12.  When the ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12``
wheels are installed, pre-loading their shared objects into the process lets
CTranslate2 resolve ``libcublas.so.12`` without touching LD_LIBRARY_PATH.
"""
import ctypes
import glob
import logging
import os
import site
from typing import Callable, List, Sequence

logger = logging.getLogger(__name__)

_PATTERNS = (
    'nvidia/cublas/lib/libcublas.so.12',
    'nvidia/cublas/lib/libcublasLt.so.12',
    'nvidia/cudnn/lib/libcudnn.so.9',
    'nvidia/cudnn/lib/libcudnn_ops.so.9',
    'nvidia/cudnn/lib/libcudnn_cnn.so.9',
    'nvidia/cudnn/lib/libcudnn_adv.so.9',
)
_loaded: List[str] = []


def preload_cuda12_libs() -> List[str]:
    """Load CUDA 12 cuBLAS/cuDNN wheels into the process (idempotent, best effort)."""
    if _loaded:
        return _loaded
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        for pattern in _PATTERNS:
            for path in glob.glob(os.path.join(root, pattern)):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    _loaded.append(path)
                except OSError as e:
                    logger.debug('could not preload %s: %s', path, e)
    if _loaded:
        logger.info('Pre-loaded %d CUDA 12 libraries for CTranslate2', len(_loaded))
    return _loaded


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def is_cuda_oom(exc: BaseException) -> bool:
    """True for PyTorch / CUDA out-of-memory errors (torch.cuda.OutOfMemoryError is a RuntimeError)."""
    return isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower()


def _empty_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _join(a, b):
    if a is None or b is None:
        return None
    if isinstance(a, tuple):
        return tuple(_join(x, y) for x, y in zip(a, b))
    if isinstance(a, list):
        return a + b
    import numpy as np
    return np.concatenate([a, b])


def halve_on_oom(fn: Callable, items: Sequence):
    """``fn(items)``; on a CUDA out-of-memory error, free the cache and retry on two halves.

    A shared GPU (or a small one) used to make a whole analysis fall back to the
    weak heuristics when one batch did not fit.  Halving keeps the models on the
    GPU; the per-item results are the same.  Halves are joined element-wise
    (arrays concatenated, lists added, tuples joined per field).  A single item
    that still does not fit re-raises, so callers fall back as before.
    """
    try:
        return fn(items)
    except RuntimeError as e:
        if not is_cuda_oom(e) or len(items) <= 1:
            raise
        _empty_cache()
        mid = len(items) // 2
        logger.warning('CUDA out of memory on a batch of %d; retrying as %d + %d',
                       len(items), mid, len(items) - mid)
        first = halve_on_oom(fn, items[:mid])
        return _join(first, halve_on_oom(fn, items[mid:]))
