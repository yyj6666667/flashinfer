#!/usr/bin/env python3
"""Benchmark flashinfer.merge_states vs a torch eager reference.

Attention state merging combines K partial attention outputs weighted by
their log-sum-exp. Given (v, s) with v:[L, K, H, D], s:[L, K, H] we compute
    w  = softmax(s, dim=1)                    # [L, K, H]
    V  = sum(w[..., None] * v, dim=1)         # [L, H, D]
    S  = logsumexp(s, dim=1)                  # [L, H]
"""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_merge_states(v: torch.Tensor, s: torch.Tensor):
    s32 = s.to(torch.float32)
    w = torch.softmax(s32, dim=1)  # [L, K, H]
    V = (w.unsqueeze(-1) * v.to(torch.float32)).sum(dim=1).to(v.dtype)  # [L, H, D]
    S = torch.logsumexp(s32, dim=1)  # [L, H]
    return V, S


@torch.inference_mode()
def bench_one(seq_len: int, num_states: int, heads: int, head_dim: int, dtype, num_iters: int):
    device = torch.device("cuda")
    v = torch.randn(seq_len, num_states, heads, head_dim, dtype=dtype, device=device)
    s = torch.randn(seq_len, num_states, heads, dtype=torch.float32, device=device)

    _ = flashinfer.merge_states(v, s)
    _ = torch_merge_states(v, s)

    fi_ms = np.median(
        bench_gpu_time(lambda: flashinfer.merge_states(v, s), repeat_iters=num_iters)
    )
    torch_ms = np.median(
        bench_gpu_time(lambda: torch_merge_states(v, s), repeat_iters=num_iters)
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-lens", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--num-states", nargs="+", type=int, default=[4, 16, 64])
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'seq':>6} {'states':>7} {'heads':>6} {'hd':>5} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for L in args.seq_lens:
        for K in args.num_states:
            t_ms, fi_ms = bench_one(L, K, args.heads, args.head_dim, dtype, args.num_iters)
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{L:>6} {K:>7} {args.heads:>6} {args.head_dim:>5} "
                f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
