"""End-to-end JIT smoke test: compile rmsnorm, run on CUDA, validate output.

Skipped on machines without CUDA. Serves as the minimum "did the full
Windows JIT toolchain (cpp_ext → ninja → nvcc → cl.exe → tvm-ffi load)
stay intact" check for CI and post-install smoke on consumer machines.
"""
from __future__ import annotations

import os
import sys

import pytest

pytest.importorskip("torch")
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="JIT smoke requires an NVIDIA GPU on PATH.",
)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


def test_rmsnorm_jit_compile_and_run(device):
    """Compile rmsnorm via JIT and compare against a torch-native reference.

    Why rmsnorm: it's one of the simplest JitSpecs in flashinfer (single .cu
    + jinja template), so any breakage in the JIT toolchain is expressed
    with minimal noise. On Windows it also exercises every one of our port
    patches:

    - ``flashinfer/jit/cpp_ext.py`` (MSVC cflags, ninja escaping, /utf-8)
    - ``flashinfer/jit/core.py`` (SHARED_LIB_EXT, DLL search on load)
    - ``include/flashinfer/_compat.h`` (FI_RESTRICT on any restrict-ed pointers)

    The test fails fast on any reintroduced regression.
    """
    import flashinfer

    torch.manual_seed(0)
    batch = 8
    hidden = 1024
    eps = 1e-6

    x = torch.randn(batch, hidden, dtype=torch.float16, device=device)
    w = torch.randn(hidden, dtype=torch.float16, device=device)

    # JIT compile + run
    out = flashinfer.rmsnorm(x, w, eps=eps)

    # torch-native reference — matches benchmarks/bench_rmsnorm_vs_torch.py
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    ref = (x.float() / rms * w.float()).to(x.dtype)

    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_compilation_context_detects_device():
    """CompilationContext must see the live device — not silently empty
    and produce "requires GPUs with sm75 or higher" via `check_cuda_arch`.
    """
    from flashinfer.compilation_context import CompilationContext

    ctx = CompilationContext()
    assert ctx.TARGET_CUDA_ARCHS, (
        "CompilationContext produced an empty TARGET_CUDA_ARCHS on a box "
        "that has an NVIDIA GPU; likely the `_normalize_cuda_arch` fallback "
        "regressed."
    )
    major, minor = torch.cuda.get_device_capability(0)
    assert any(
        arch_major == major for (arch_major, _) in ctx.TARGET_CUDA_ARCHS
    ), f"TARGET_CUDA_ARCHS {ctx.TARGET_CUDA_ARCHS} doesn't match device major={major}"


def test_check_cuda_arch_passes():
    """The gate that sglang's scheduler subprocess blew up on — no-op
    happy path here, but the test exists to catch any future change that
    re-introduces the empty-set path."""
    from flashinfer.jit.core import check_cuda_arch

    check_cuda_arch()  # must not raise
