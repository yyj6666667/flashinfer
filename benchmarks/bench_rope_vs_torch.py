#!/usr/bin/env python3
"""Benchmark flashinfer.apply_rope vs a torch eager reference."""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    indptr: torch.Tensor,
    offsets: torch.Tensor,
    rope_theta: float = 1e4,
):
    nnz, nq, d = q.shape
    _, nk, _ = k.shape
    device = q.device
    half = d // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    batch = indptr.shape[0] - 1
    out_q = q.clone()
    out_k = k.clone()
    for b in range(batch):
        s = indptr[b].item()
        e = indptr[b + 1].item()
        off = offsets[b].item()
        pos = torch.arange(off, off + (e - s), dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)
        cos = freqs.cos().to(q.dtype)
        sin = freqs.sin().to(q.dtype)
        for t, nh in ((out_q, nq), (out_k, nk)):
            x = t[s:e]
            x1, x2 = x[..., :half], x[..., half:]
            c = cos.unsqueeze(1).expand(-1, nh, -1)
            si = sin.unsqueeze(1).expand(-1, nh, -1)
            t[s:e, ..., :half] = x1 * c - x2 * si
            t[s:e, ..., half:] = x1 * si + x2 * c
    return out_q, out_k


@torch.inference_mode()
def bench_one(batch: int, seq: int, nq: int, nk: int, d: int, dtype, num_iters: int):
    device = torch.device("cuda")
    nnz = batch * seq
    q = torch.randn(nnz, nq, d, dtype=dtype, device=device)
    k = torch.randn(nnz, nk, d, dtype=dtype, device=device)
    indptr = torch.arange(0, nnz + 1, seq, dtype=torch.int32, device=device)
    offsets = torch.zeros(batch, dtype=torch.int32, device=device)

    _ = flashinfer.apply_rope(q, k, indptr, offsets)
    _ = torch_apply_rope(q, k, indptr, offsets)

    fi_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.apply_rope(q, k, indptr, offsets), repeat_iters=num_iters
        )
    )
    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_apply_rope(q, k, indptr, offsets), repeat_iters=num_iters
        )
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--seqs", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'seq':>6} {'heads':>6} {'hd':>5} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for s in args.seqs:
            t_ms, fi_ms = bench_one(b, s, args.heads, args.heads, args.head_dim, dtype, args.num_iters)
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{b:>6} {s:>6} {args.heads:>6} {args.head_dim:>5} "
                f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
