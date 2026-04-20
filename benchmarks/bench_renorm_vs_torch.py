#!/usr/bin/env python3
"""Benchmark flashinfer top_k_renorm_probs / top_p_renorm_probs vs torch."""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_top_k_renorm_probs(probs: torch.Tensor, k: int) -> torch.Tensor:
    topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topk_idx, 1.0)
    out = probs * mask
    out = out / out.sum(dim=-1, keepdim=True)
    return out


@torch.inference_mode()
def torch_top_p_renorm_probs(probs: torch.Tensor, p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = sorted_probs.cumsum(dim=-1)
    mask = cum - sorted_probs > p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    out = torch.empty_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return out


@torch.inference_mode()
def bench_one(batch: int, vocab: int, dtype, num_iters: int):
    device = torch.device("cuda")
    probs = torch.softmax(torch.randn(batch, vocab, dtype=dtype, device=device), dim=-1)

    _ = flashinfer.top_k_renorm_probs(probs, top_k=50)
    _ = flashinfer.top_p_renorm_probs(probs, top_p=0.9)

    topk_fi = np.median(
        bench_gpu_time(
            lambda: flashinfer.top_k_renorm_probs(probs, top_k=50), repeat_iters=num_iters
        )
    )
    topk_t = np.median(
        bench_gpu_time(
            lambda: torch_top_k_renorm_probs(probs, 50), repeat_iters=num_iters
        )
    )
    topp_fi = np.median(
        bench_gpu_time(
            lambda: flashinfer.top_p_renorm_probs(probs, top_p=0.9), repeat_iters=num_iters
        )
    )
    topp_t = np.median(
        bench_gpu_time(
            lambda: torch_top_p_renorm_probs(probs, 0.9), repeat_iters=num_iters
        )
    )
    return [("top_k=50", topk_t, topk_fi), ("top_p=0.9", topp_t, topp_fi)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 32, 256])
    p.add_argument("--vocabs", nargs="+", type=int, default=[32000, 128000])
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'vocab':>7} {'variant':>10} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for v in args.vocabs:
            for name, t_ms, fi_ms in bench_one(b, v, dtype, args.num_iters):
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{b:>6} {v:>7} {name:>10} "
                    f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
                )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
