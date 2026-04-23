"""Lazy import shim for the ``cudnn`` (nvidia-cudnn-frontend) module.

Rationale
---------

The ``cudnn`` wheel loads ``cudnn64_*.dll`` at import time via ctypes. On
Windows that DLL lives under ``<torch>/lib`` and is not on the default DLL
search path, so an eager ``import cudnn`` at ``import flashinfer`` time
either fails outright or swallows a useful diagnostic behind a bare
``except``.

On consumer-grade Windows (SM120 RTX 5060/70/80/90) the cudnn attention
backend is not needed — FA2/CUTLASS paths cover the common LLM workloads.
There is no reason to pay the DLL-search cost or risk a silent
import-time failure just because some users *might* ask for the cudnn
backend later.

This module defers the import (and the Windows-only DLL-directory fix)
to the first call of :func:`get_cudnn`. On failure it emits a single
``RuntimeWarning`` so that callers can see *why* the cudnn path is
unreachable, then returns ``None`` on every subsequent call.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings
from typing import Optional

_lock = threading.Lock()
_cudnn = None  # type: Optional[object]
_import_failed = False
_warned = False


def _add_windows_dll_dir() -> None:
    """Make ``cudnn64_*.dll`` (bundled under ``<torch>/lib``) discoverable.

    Best-effort; silently falls through on anything unexpected so we don't
    block import of :mod:`flashinfer.cudnn` when the user doesn't actually
    use cuDNN.
    """
    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):
        return
    try:
        import torch  # noqa: WPS433 — deliberately local
    except Exception:
        return
    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    if not os.path.isdir(torch_lib):
        return
    try:
        os.add_dll_directory(torch_lib)
    except (OSError, FileNotFoundError):
        pass


def get_cudnn():
    """Return the ``cudnn`` module, or ``None`` if unavailable.

    Thread-safe; performs the import at most once. The first failure emits
    a single ``RuntimeWarning`` so consumer users can see the diagnostic
    without breaking ``import flashinfer``.
    """
    global _cudnn, _import_failed, _warned
    if _cudnn is not None:
        return _cudnn
    if _import_failed:
        return None
    with _lock:
        if _cudnn is not None:
            return _cudnn
        if _import_failed:
            return None
        _add_windows_dll_dir()
        try:
            import cudnn as _c  # noqa: WPS433

            _cudnn = _c
            return _c
        except Exception as e:
            _import_failed = True
            if not _warned:
                _warned = True
                warnings.warn(
                    "flashinfer.cudnn backend is unavailable "
                    f"({type(e).__name__}: {e}). "
                    "FA2 / CUTLASS attention paths cover SM120 consumer "
                    "GPUs without cuDNN — install "
                    "nvidia-cudnn-frontend only if you specifically need "
                    "the cuDNN-backed kernels.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            return None


def is_available() -> bool:
    """Return whether ``cudnn`` can be imported in this process.

    Triggers the lazy import on first call.
    """
    return get_cudnn() is not None
