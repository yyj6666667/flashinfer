#!/usr/bin/env python3
"""Benchmark flashinfer.fused_add_rmsnorm vs torch eager (add + rmsnorm)."""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_fused_add_rmsnorm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
):
    residual_out = residual + input
    x = residual_out.to(torch.float32)
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    out = (x * rms).to(input.dtype) * weight
    return out, residual_out


@torch.inference_mode()
def bench_one(batch: int, hidden: int, dtype, num_iters: int):
    device = torch.device("cuda")
    w = torch.randn(hidden, dtype=dtype, device=device)

    def make():
        x = torch.randn(batch, hidden, dtype=dtype, device=device)
        r = torch.randn(batch, hidden, dtype=dtype, device=device)
        return x, r

    x0, r0 = make()
    _ = flashinfer.fused_add_rmsnorm(x0.clone(), r0.clone(), w)
    _ = torch_fused_add_rmsnorm(x0.clone(), r0.clone(), w)

    def run_fi():
        xi, ri = make()
        flashinfer.fused_add_rmsnorm(xi, ri, w)

    def run_torch():
        xi, ri = make()
        torch_fused_add_rmsnorm(xi, ri, w)

    fi_ms = np.median(bench_gpu_time(run_fi, repeat_iters=num_iters))
    torch_ms = np.median(bench_gpu_time(run_torch, repeat_iters=num_iters))
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 99, 989])
    p.add_argument("--hiddens", nargs="+", type=int, default=[1024, 4096, 8192, 16384])
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'hidden':>7} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for h in args.hiddens:
            t_ms, fi_ms = bench_one(b, h, dtype, args.num_iters)
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{b:>6} {h:>7} "
                f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
