#!/usr/bin/env python3
"""Benchmark flashinfer.mm_fp8 vs torch bf16 matmul.

On SM120 mm_fp8 routes through the CUTLASS fp8 groupwise fallback (TRTLLM
low-latency cubin is SM100-only)."""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer import mm_fp8
from flashinfer.testing.utils import bench_gpu_time


def to_float8(x, dtype):
    finfo = torch.finfo(dtype)
    scale = finfo.max / x.abs().amax().clamp(min=1e-6)
    return (x * scale).to(dtype), 1.0 / scale.float()


@torch.inference_mode()
def torch_bf16(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.t()


@torch.inference_mode()
def bench_one(m: int, n: int, k: int, num_iters: int):
    device = torch.device("cuda")
    a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, k, device=device, dtype=torch.bfloat16)
    a_fp8, a_inv_s = to_float8(a, torch.float8_e4m3fn)
    b_fp8, b_inv_s = to_float8(b, torch.float8_e4m3fn)
    prepared_b = flashinfer.prepare_low_latency_gemm_weights(b_fp8, {})
    alpha = (a_inv_s * b_inv_s).to(device)

    _ = mm_fp8(a_fp8, prepared_b, alpha, out_dtype=torch.bfloat16)
    _ = torch_bf16(a, b)

    fi_ms = np.median(
        bench_gpu_time(
            lambda: mm_fp8(a_fp8, prepared_b, alpha, out_dtype=torch.bfloat16),
            repeat_iters=num_iters,
        )
    )
    torch_ms = np.median(
        bench_gpu_time(lambda: torch_bf16(a, b), repeat_iters=num_iters)
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--n", nargs="+", type=int, default=[1024, 4096])
    p.add_argument("--k", nargs="+", type=int, default=[1024, 4096])
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name()} | mm_fp8 (SM120 -> cutlass fallback)")
    header = (
        f"{'m':>6} {'n':>6} {'k':>6} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for m in args.m:
        for n in args.n:
            for k in args.k:
                if m % 4 or n % 128 or k % 128:
                    continue
                t_ms, fi_ms = bench_one(m, n, k, args.num_iters)
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{m:>6} {n:>6} {k:>6} "
                    f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
                )

    if speedups:
        print("=" * len(header))
        print(
            f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
            f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
        )


if __name__ == "__main__":
    main()
