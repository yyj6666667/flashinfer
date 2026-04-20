#!/usr/bin/env python3
"""Benchmark flashinfer.sampling vs torch baseline.

Three common sampling routines measured:
  - sampling_from_probs (categorical)
  - top_k_sampling_from_probs
  - top_p_sampling_from_probs

Torch baselines mirror the naive eager reference.
"""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_sampling_from_probs(probs: torch.Tensor) -> torch.Tensor:
    # Equivalent to torch.multinomial(probs, 1).
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def torch_top_k_sampling(probs: torch.Tensor, top_k: int) -> torch.Tensor:
    topk_vals, topk_idx = torch.topk(probs, top_k, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(topk_vals, num_samples=1).squeeze(-1)
    return topk_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def torch_top_p_sampling(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = sorted_probs.cumsum(dim=-1)
    mask = cum - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def bench_one(batch: int, vocab: int, dtype: torch.dtype, num_iters: int):
    probs = torch.softmax(
        torch.randn(batch, vocab, dtype=dtype, device="cuda"), dim=-1
    )

    # Warm up.
    _ = flashinfer.sampling.sampling_from_probs(probs)
    _ = torch_sampling_from_probs(probs)

    base_torch = np.median(
        bench_gpu_time(lambda: torch_sampling_from_probs(probs), repeat_iters=num_iters)
    )
    base_fi = np.median(
        bench_gpu_time(lambda: flashinfer.sampling.sampling_from_probs(probs), repeat_iters=num_iters)
    )

    topk_torch = np.median(
        bench_gpu_time(lambda: torch_top_k_sampling(probs, 50), repeat_iters=num_iters)
    )
    topk_fi = np.median(
        bench_gpu_time(
            lambda: flashinfer.sampling.top_k_sampling_from_probs(probs, 50),
            repeat_iters=num_iters,
        )
    )

    topp_torch = np.median(
        bench_gpu_time(lambda: torch_top_p_sampling(probs, 0.9), repeat_iters=num_iters)
    )
    topp_fi = np.median(
        bench_gpu_time(
            lambda: flashinfer.sampling.top_p_sampling_from_probs(probs, 0.9),
            repeat_iters=num_iters,
        )
    )

    return [
        ("basic", base_torch, base_fi),
        ("top_k=50", topk_torch, topk_fi),
        ("top_p=0.9", topp_torch, topp_fi),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 32, 256])
    p.add_argument("--vocabs", nargs="+", type=int, default=[32000, 128000, 200000])
    p.add_argument("--dtype", choices=["float16", "float32"], default="float32")
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
            for variant, t_ms, fi_ms in bench_one(b, v, dtype, args.num_iters):
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{b:>6} {v:>7} {variant:>10} "
                    f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
                )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
