#!/usr/bin/env python3
"""Benchmark flashinfer.bmm_fp8 vs a bfloat16 torch.bmm baseline.

FP8 vs BF16 is not an apples-to-apples accuracy comparison, but it is the
common "how fast is the FP8 path relative to what most people would run
otherwise" question we care about here. The torch reference uses bfloat16
since fp8 matmul is not supported by torch.bmm on most stacks.
"""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_bmm_bf16(a_bf16: torch.Tensor, b_bf16: torch.Tensor) -> torch.Tensor:
    return torch.bmm(a_bf16, b_bf16)


@torch.inference_mode()
def bench_one(batch: int, m: int, k: int, n: int, num_iters: int):
    device = torch.device("cuda")
    a_bf16 = torch.randn(batch, m, k, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(batch, k, n, dtype=torch.bfloat16, device=device)
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)
    a_scale = torch.ones(batch, device=device)
    b_scale = torch.ones(batch, device=device)

    _ = flashinfer.bmm_fp8(a_fp8, b_fp8, a_scale, b_scale, dtype=torch.bfloat16)
    _ = torch_bmm_bf16(a_bf16, b_bf16)

    fi_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.bmm_fp8(a_fp8, b_fp8, a_scale, b_scale, dtype=torch.bfloat16),
            repeat_iters=num_iters,
        )
    )
    torch_ms = np.median(
        bench_gpu_time(lambda: torch_bmm_bf16(a_bf16, b_bf16), repeat_iters=num_iters)
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--shapes",
        nargs="+",
        default=["1,128,512,256", "1,512,4096,4096", "1,2048,4096,4096"],
        help="comma-separated B,M,K,N tuples",
    )
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name()}")
    header = (
        f"{'B':>3} {'M':>6} {'K':>6} {'N':>6} "
        f"{'torch bf16 (us)':>17} {'fi fp8 (us)':>12} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for s in args.shapes:
        b, m, k, n = (int(x) for x in s.split(","))
        t_ms, fi_ms = bench_one(b, m, k, n, args.num_iters)
        sp = t_ms / fi_ms
        speedups.append(sp)
        print(
            f"{b:>3} {m:>6} {k:>6} {n:>6} "
            f"{t_ms * 1e3:>17.2f} {fi_ms * 1e3:>12.2f} {sp:>8.2f}x"
        )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
