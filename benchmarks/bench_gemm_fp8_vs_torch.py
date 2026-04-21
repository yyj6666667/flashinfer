#!/usr/bin/env python3
"""Benchmark flashinfer.gemm.gemm_fp8_nt_groupwise(cutlass) vs torch bf16 matmul."""

import argparse
from typing import List

import numpy as np
import torch

from flashinfer.gemm import gemm_fp8_nt_groupwise
from flashinfer.testing.utils import bench_gpu_time, quantize_fp8


@torch.inference_mode()
def torch_bf16_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b.t()


@torch.inference_mode()
def bench_one(m: int, n: int, k: int, mode: str, num_iters: int):
    device = torch.device("cuda")
    tile = 128

    a_bf = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    b_bf = torch.randn(n, k, dtype=torch.bfloat16, device=device) / (k ** 0.5)

    if mode == "K":
        a_scale_shape = (m, k // tile)
        b_scale_shape = (n // tile, k // tile)
    else:
        a_scale_shape = (k // tile, m)
        b_scale_shape = (k // tile, n // tile)
    a_fp8, a_scale = quantize_fp8(a_bf.float(), a_scale_shape, (1, tile), mode)
    b_fp8, b_scale = quantize_fp8(b_bf.float(), b_scale_shape, (tile, tile), mode)

    _ = gemm_fp8_nt_groupwise(a_fp8, b_fp8, a_scale, b_scale, mode,
                              out_dtype=torch.bfloat16, backend="cutlass")
    _ = torch_bf16_matmul(a_bf, b_bf)

    fi_ms = np.median(
        bench_gpu_time(
            lambda: gemm_fp8_nt_groupwise(
                a_fp8, b_fp8, a_scale, b_scale, mode,
                out_dtype=torch.bfloat16, backend="cutlass",
            ),
            repeat_iters=num_iters,
        )
    )
    torch_ms = np.median(
        bench_gpu_time(lambda: torch_bf16_matmul(a_bf, b_bf), repeat_iters=num_iters)
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--n", nargs="+", type=int, default=[1024, 4096])
    p.add_argument("--k", nargs="+", type=int, default=[1024, 4096])
    p.add_argument("--mode", choices=["MN", "K"], default="MN")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name()} | cutlass backend | mode={args.mode}")
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
                t_ms, fi_ms = bench_one(m, n, k, args.mode, args.num_iters)
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
