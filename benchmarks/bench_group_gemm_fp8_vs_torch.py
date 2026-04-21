#!/usr/bin/env python3
"""Benchmark flashinfer.gemm.group_gemm_fp8_nt_groupwise vs torch bf16 einsum.

On SM120/121 the library explicitly rejects num_groups > 1 with a
RuntimeError ('has correctness issues for num_groups > 1 on SM120/121')
so the bench here only covers the 1-group / single-GEMM case."""

import argparse
from typing import List

import numpy as np
import torch

from flashinfer.gemm import group_gemm_fp8_nt_groupwise
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_bf16_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: (m, k) bf16, b: (n, k) bf16 -> (m, n)
    return a @ b.t()


@torch.inference_mode()
def bench_one(m: int, n: int, k: int, num_iters: int):
    device = torch.device("cuda")
    block_size = 128
    m_indptr = torch.tensor([0, m], dtype=torch.int32, device=device)

    a_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_bf16 = torch.randn(n, k, device=device, dtype=torch.bfloat16)

    a = a_bf16.to(torch.float8_e4m3fn)
    b = b_bf16.unsqueeze(0).to(torch.float8_e4m3fn)  # (1, n, k)
    a_scale = torch.ones(k // block_size, m, dtype=torch.float32, device=device)
    b_scale = torch.ones(1, k // block_size, n // block_size, dtype=torch.float32, device=device)

    _ = group_gemm_fp8_nt_groupwise(
        a, b, a_scale, b_scale, m_indptr,
        scale_granularity_mnk=(1, 128, 128),
        scale_major_mode="MN",
        out_dtype=torch.bfloat16,
    )
    _ = torch_bf16_matmul(a_bf16, b_bf16)

    fi_ms = np.median(
        bench_gpu_time(
            lambda: group_gemm_fp8_nt_groupwise(
                a, b, a_scale, b_scale, m_indptr,
                scale_granularity_mnk=(1, 128, 128),
                scale_major_mode="MN",
                out_dtype=torch.bfloat16,
            ),
            repeat_iters=num_iters,
        )
    )
    torch_ms = np.median(
        bench_gpu_time(lambda: torch_bf16_matmul(a_bf16, b_bf16), repeat_iters=num_iters)
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--n", nargs="+", type=int, default=[1024, 4096])
    p.add_argument("--k", nargs="+", type=int, default=[1024, 4096])
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name()} | num_groups=1 (SM120 guard)")
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
                if m % 4 != 0 or n % 128 != 0 or k % 128 != 0:
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
