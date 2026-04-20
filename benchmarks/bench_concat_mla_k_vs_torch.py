#!/usr/bin/env python3
"""Benchmark flashinfer.concat_ops.concat_mla_k vs torch cat+broadcast.

The kernel is tuned for num_heads=128, nope_dim=128, rope_dim=64 per its docs."""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer.concat_ops as concat_ops
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_concat_mla_k(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor):
    # k[:, :, :D_nope] = k_nope, k[:, :, D_nope:] = k_rope broadcast across heads
    D_nope = k_nope.shape[-1]
    k[..., :D_nope] = k_nope
    k[..., D_nope:] = k_rope  # broadcasts [N, 1, D_rope] -> [N, H, D_rope]


@torch.inference_mode()
def bench_one(num_tokens: int, heads: int, nope: int, rope: int, dtype, num_iters: int):
    device = torch.device("cuda")
    k = torch.empty(num_tokens, heads, nope + rope, dtype=dtype, device=device)
    k_nope = torch.randn(num_tokens, heads, nope, dtype=dtype, device=device)
    k_rope = torch.randn(num_tokens, 1, rope, dtype=dtype, device=device)

    concat_ops.concat_mla_k(k, k_nope, k_rope)
    torch_concat_mla_k(k, k_nope, k_rope)

    fi_ms = np.median(
        bench_gpu_time(
            lambda: concat_ops.concat_mla_k(k, k_nope, k_rope), repeat_iters=num_iters
        )
    )
    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_concat_mla_k(k, k_nope, k_rope), repeat_iters=num_iters
        )
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", nargs="+", type=int, default=[128, 512, 2048, 8192])
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--nope", type=int, default=128)
    p.add_argument("--rope", type=int, default=64)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'tokens':>7} {'heads':>6} {'nope':>5} {'rope':>5} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for n in args.tokens:
        t_ms, fi_ms = bench_one(n, args.heads, args.nope, args.rope, dtype, args.num_iters)
        sp = t_ms / fi_ms
        speedups.append(sp)
        print(
            f"{n:>7} {args.heads:>6} {args.nope:>5} {args.rope:>5} "
            f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
        )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
