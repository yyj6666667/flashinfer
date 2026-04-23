"""SM12.x normalize-arch fallback behavior.

These tests don't need a GPU — they directly exercise
``CompilationContext._normalize_cuda_arch`` with synthetic inputs. They're
a regression guard on the sglang scheduler-subprocess crash fixed in
commit ``bb24abc3`` where ``torch.version.cuda == "12.8"`` was causing
every SM120 target to raise ``RuntimeError("SM 12.x requires CUDA >= 12.9")``,
silently emptying ``TARGET_CUDA_ARCHS`` and producing a misleading
"requires GPUs with sm75 or higher" downstream.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from flashinfer.compilation_context import CompilationContext


def _mock_cuda_version(version_str: str):
    """Make ``is_cuda_version_at_least`` see ``version_str`` as the active CUDA."""
    from packaging.version import Version

    def fake(target: str) -> bool:
        return Version(version_str) >= Version(target)

    return patch(
        "flashinfer.jit.cpp_ext.is_cuda_version_at_least",
        side_effect=fake,
    )


def test_sm120_on_cuda_12_9_emits_120f():
    with _mock_cuda_version("12.9"):
        major, minor = CompilationContext._normalize_cuda_arch(12, 0)
    assert (major, minor) == (12, "0f"), (
        "On CUDA 12.9, SM120 should normalize to the 'f' feature-set variant; "
        f"got ({major}, {minor!r})"
    )


def test_sm120_on_cuda_12_8_falls_back_to_120a():
    """Regression guard for the sglang scheduler subprocess crash."""
    with _mock_cuda_version("12.8"):
        major, minor = CompilationContext._normalize_cuda_arch(12, 0)
    assert (major, minor) == (12, "0a"), (
        "On CUDA 12.8, SM120 MUST fall back to compute_120a (not raise). "
        f"got ({major}, {minor!r})"
    )


def test_sm121_on_cuda_12_8_still_errors():
    """sm_121 has no sm_121a base variant; CUDA 12.8 must error clearly."""
    with _mock_cuda_version("12.8"):
        with pytest.raises(RuntimeError, match="CUDA 12.8 supports sm_120a only"):
            CompilationContext._normalize_cuda_arch(12, 1)


def test_sm121_on_cuda_12_9_emits_121f():
    with _mock_cuda_version("12.9"):
        major, minor = CompilationContext._normalize_cuda_arch(12, 1)
    assert (major, minor) == (12, "1f")


def test_init_does_not_swallow_normalize_errors(monkeypatch):
    """If _normalize_cuda_arch raises a genuine RuntimeError, CompilationContext
    __init__ must propagate it rather than silently produce an empty
    TARGET_CUDA_ARCHS that downstream gates misreport as
    'requires GPUs with sm75 or higher'.
    """
    import torch

    monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda i: (12, 1))
    with _mock_cuda_version("12.8"):
        with pytest.raises(RuntimeError, match="CUDA 12.8 supports sm_120a only"):
            CompilationContext()
