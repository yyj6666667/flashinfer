"""Quick smoke test for the Windows FlashInfer port.

Run it after ``scripts\\windows\\build.bat`` finishes:

    python scripts\\windows\\smoke_test.py

It exits with 0 on success, 1 on import failure, 2 on kernel failure.

The test intentionally exercises a small JIT path (rmsnorm) so that a real
MSVC+nvcc compilation happens end-to-end; this is what we actually want to
validate on a freshly-ported toolchain.
"""
from __future__ import annotations

import sys
import time
import traceback


def step(title: str) -> None:
    print(f"\n==> {title}", flush=True)


def main() -> int:
    step("Importing flashinfer")
    t0 = time.time()
    try:
        import flashinfer  # noqa: F401
    except Exception:
        traceback.print_exc()
        print("[FAIL] flashinfer import")
        return 1
    print(f"[OK] flashinfer import ({time.time() - t0:.2f}s)")
    print(f"     module: {flashinfer.__file__}")

    step("Importing torch and querying CUDA")
    try:
        import torch
    except Exception:
        traceback.print_exc()
        print("[FAIL] torch import")
        return 1
    print(f"[OK] torch={torch.__version__}  cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[FAIL] torch.cuda.is_available() == False")
        print("       On Windows this usually means either the NVIDIA driver")
        print("       is older than the CUDA runtime, or the wrong torch wheel")
        print("       was installed. Verify with `nvidia-smi`.")
        return 1
    dev = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"     device[0]: {dev} (sm_{cap[0]}{cap[1]})")

    step("JIT-compiling and running flashinfer.rmsnorm")
    try:
        # rmsnorm is the simplest registered JIT op — no Jinja templates,
        # a single .cu source — which makes it ideal for a first build smoke.
        x = torch.randn(4, 128, device="cuda", dtype=torch.float16)
        w = torch.randn(128, device="cuda", dtype=torch.float16)
        t0 = time.time()
        y = flashinfer.rmsnorm(x, w)
        torch.cuda.synchronize()
        dt = time.time() - t0
    except Exception:
        traceback.print_exc()
        print("[FAIL] rmsnorm invocation")
        return 2
    print(f"[OK] rmsnorm produced {tuple(y.shape)} tensor in {dt:.2f}s")
    print("     (first call includes JIT compile; subsequent calls are fast)")

    step("Second invocation (cache hit)")
    t0 = time.time()
    y2 = flashinfer.rmsnorm(x, w)
    torch.cuda.synchronize()
    dt2 = time.time() - t0
    print(f"[OK] second call: {dt2 * 1000:.2f} ms")
    if not torch.allclose(y, y2, atol=1e-3, rtol=1e-3):
        print("[FAIL] results differ across invocations")
        return 2

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
