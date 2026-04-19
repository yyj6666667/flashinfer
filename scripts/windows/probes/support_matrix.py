"""Print the GEMM / MoE / attention backend support matrix for this GPU.

Run via:
    scripts\\windows\\debug_run.ps1 python scripts\\windows\\probes\\support_matrix.py

Answers the question: "which FlashInfer APIs does the current GPU support,
and via which backends?" — drives the Windows port verification plan.
"""
from __future__ import annotations

import torch
from flashinfer.utils import get_compute_capability


def probe(mod, mod_name: str, cc: int, backends: list[str]) -> None:
    print(f"\n=== {mod_name} ===")
    try:
        ok_cc = mod.is_compute_capability_supported(cc)
        print(f"  is_compute_capability_supported({cc}) = {ok_cc}")
    except Exception as e:
        print(f"  is_compute_capability_supported: ERROR {e!r}")
        return
    for b in backends:
        try:
            ok = mod.is_backend_supported(b, cc)
            print(f"    backend {b:<10s} -> {ok}")
        except Exception as e:
            print(f"    backend {b:<10s} -> ERROR {e!r}")


def main() -> int:
    cc_tuple = get_compute_capability(torch.device("cuda"))
    cc = cc_tuple[0] * 10 + cc_tuple[1]
    dev = torch.cuda.get_device_name(0)
    print(f"GPU:  {dev}  (sm_{cc})")
    print(f"torch={torch.__version__}  cuda={torch.version.cuda}")

    # GEMM family — import each independently so one missing API doesn't
    # abort the whole probe.
    targets = [
        ("flashinfer", "mm_bf16", ["cudnn", "cutlass", "tgv"]),
        ("flashinfer", "mm_fp8", ["cudnn", "cutlass", "cublas"]),
        ("flashinfer", "bmm_fp8", ["cudnn", "cutlass", "cublas"]),
        ("flashinfer", "mm_fp4", ["cudnn", "cutlass", "trtllm"]),
        ("flashinfer.gemm", "group_gemm_fp8_nt_groupwise", ["cudnn", "cutlass"]),
    ]
    import importlib
    for pkg, attr, backends in targets:
        try:
            mod = importlib.import_module(pkg)
            api = getattr(mod, attr)
        except (ImportError, AttributeError) as e:
            print(f"\n=== {attr} ===  NOT AVAILABLE: {e!r}")
            continue
        probe(api, attr, cc, backends)

    print("\n(Use this to pick GEMM tests that actually run on this GPU.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
