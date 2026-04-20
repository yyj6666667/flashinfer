#!/usr/bin/env python3
"""Benchmark flashinfer.single_prefill_with_kv_cache vs torch SDPA."""

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_prefill(q, k, v, causal):
    # q: (qo_len, num_qo_heads, head_dim) -> (1, H, Lq, D)
    # k,v: (kv_len, num_kv_heads, head_dim) -> (1, H, Lk, D)
    q_ = q.unsqueeze(0).transpose(1, 2)
    k_ = k.unsqueeze(0).transpose(1, 2)
    v_ = v.unsqueeze(0).transpose(1, 2)
    num_qo = q_.shape[1]
    num_kv = k_.shape[1]
    if num_qo != num_kv:
        assert num_qo % num_kv == 0
        rep = num_qo // num_kv
        k_ = k_.repeat_interleave(rep, dim=1)
        v_ = v_.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q_, k_, v_, is_causal=causal).transpose(1, 2).squeeze(0)


@torch.inference_mode()
def bench_one(qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal, dtype, num_iters):
    device = "cuda"
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Correctness is already covered by tests/attention/test_single_prefill.py;
    # this bench only measures perf.
    _ = torch_prefill(q, k, v, causal)  # warm up torch path
    _ = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal)  # warm up flashinfer

    torch_ms = np.median(
        bench_gpu_time(lambda: torch_prefill(q, k, v, causal), repeat_iters=num_iters)
    )
    fi_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal),
            repeat_iters=num_iters,
        )
    )
    bytes_moved = (q.numel() + k.numel() + v.numel() + q.numel()) * q.element_size()
    fi_tbps = bytes_moved / (fi_ms * 1e-3) / 1e12
    return torch_ms, fi_ms, fi_tbps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qo-lens", nargs="+", type=int, default=[128, 512, 2048])
    p.add_argument("--kv-lens", nargs="+", type=int, default=[512, 2048, 8192])
    p.add_argument("--num-qo-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(
        f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype} | "
        f"qh={args.num_qo_heads} kvh={args.num_kv_heads} hd={args.head_dim} causal={args.causal}"
    )
    header = (
        f"{'qo_len':>7} {'kv_len':>7} {'torch (us)':>12} "
        f"{'flashinfer (us)':>17} {'speedup':>9} {'TB/s':>7}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for ql in args.qo_lens:
        for kl in args.kv_lens:
            if args.causal and ql > kl:
                continue
            t_ms, fi_ms, fi_tbps = bench_one(
                ql, kl, args.num_qo_heads, args.num_kv_heads, args.head_dim,
                args.causal, dtype, args.num_iters,
            )
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{ql:>7} {kl:>7} {t_ms * 1e3:>12.2f} "
                f"{fi_ms * 1e3:>17.2f} {sp:>8.2f}x {fi_tbps:>7.2f}"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
