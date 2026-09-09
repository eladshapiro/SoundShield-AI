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
from typing import List

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
