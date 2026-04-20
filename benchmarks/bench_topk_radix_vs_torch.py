#!/usr/bin/env python3
"""Benchmark flashinfer.top_k (radix) vs torch.topk.

flashinfer.top_k is pitched as a drop-in replacement for torch.topk
that wins on large vocabularies (>10k). This script spans small-ish
to LM-scale vocabs so we can see where the crossover is.
"""

import argparse
from typing import List

import numpy as np
import torch

from flashinfer.topk import top_k as fi_top_k
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_topk_sorted(x: torch.Tensor, k: int):
    return torch.topk(x, k, dim=-1, sorted=True)


@torch.inference_mode()
def bench_one(batch: int, vocab: int, k: int, dtype, num_iters: int):
    device = torch.device("cuda")
    x = torch.randn(batch, vocab, dtype=dtype, device=device)

    _ = fi_top_k(x, k=k, sorted=True)
    _ = torch_topk_sorted(x, k)

    fi_ms = np.median(
        bench_gpu_time(lambda: fi_top_k(x, k=k, sorted=True), repeat_iters=num_iters)
    )
    torch_ms = np.median(
        bench_gpu_time(lambda: torch_topk_sorted(x, k), repeat_iters=num_iters)
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 32, 256])
    p.add_argument("--vocabs", nargs="+", type=int, default=[4096, 32000, 128000])
    p.add_argument("--ks", nargs="+", type=int, default=[50])
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'vocab':>7} {'k':>4} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for v in args.vocabs:
            for k in args.ks:
                t_ms, fi_ms = bench_one(b, v, k, dtype, args.num_iters)
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{b:>6} {v:>7} {k:>4} "
                    f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
                )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
