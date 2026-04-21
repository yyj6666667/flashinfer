#!/usr/bin/env python3
"""Benchmark flashinfer.logits_processor.LogitsPipe vs torch eager equivalent.

Three common sampling pipelines:
  * Temperature -> Softmax -> TopK -> Sample
  * Temperature -> Softmax -> TopP -> Sample
  * Softmax -> MinP -> Sample
"""

import argparse
from typing import List

import numpy as np
import torch

from flashinfer.logits_processor import (
    LogitsPipe,
    MinP,
    Sample,
    Softmax,
    Temperature,
    TopK,
    TopP,
)
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_temp_softmax_topk_sample(logits: torch.Tensor, temp: float, k: int) -> torch.Tensor:
    probs = torch.softmax(logits / temp, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k, dim=-1)
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(topk_vals, num_samples=1).squeeze(-1)
    return topk_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def torch_temp_softmax_topp_sample(logits: torch.Tensor, temp: float, p: float) -> torch.Tensor:
    probs = torch.softmax(logits / temp, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cum = sorted_probs.cumsum(dim=-1)
    mask = cum - sorted_probs > p
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)
    return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def torch_softmax_minp_sample(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    top_prob, _ = probs.max(dim=-1, keepdim=True)
    threshold = top_prob * min_p
    probs = torch.where(probs >= threshold, probs, torch.zeros_like(probs))
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def bench_one(batch: int, vocab: int, dtype, num_iters: int):
    device = torch.device("cuda")
    logits = torch.randn(batch, vocab, dtype=dtype, device=device)

    pipe_tk = LogitsPipe([Temperature(), Softmax(), TopK(), Sample(deterministic=True)])
    pipe_tp = LogitsPipe([Temperature(), Softmax(), TopP(), Sample(deterministic=True)])
    pipe_mp = LogitsPipe([Softmax(), MinP(), Sample(deterministic=True)])

    # warmup
    _ = pipe_tk(logits, temperature=0.9, top_k=40)
    _ = pipe_tp(logits, temperature=0.9, top_p=0.9)
    _ = pipe_mp(logits, min_p=0.05)
    _ = torch_temp_softmax_topk_sample(logits, 0.9, 40)
    _ = torch_temp_softmax_topp_sample(logits, 0.9, 0.9)
    _ = torch_softmax_minp_sample(logits, 0.05)

    results = []
    fi_ms = np.median(
        bench_gpu_time(lambda: pipe_tk(logits, temperature=0.9, top_k=40), repeat_iters=num_iters)
    )
    t_ms = np.median(
        bench_gpu_time(
            lambda: torch_temp_softmax_topk_sample(logits, 0.9, 40), repeat_iters=num_iters
        )
    )
    results.append(("temp+topk", t_ms, fi_ms))

    fi_ms = np.median(
        bench_gpu_time(lambda: pipe_tp(logits, temperature=0.9, top_p=0.9), repeat_iters=num_iters)
    )
    t_ms = np.median(
        bench_gpu_time(
            lambda: torch_temp_softmax_topp_sample(logits, 0.9, 0.9), repeat_iters=num_iters
        )
    )
    results.append(("temp+topp", t_ms, fi_ms))

    fi_ms = np.median(
        bench_gpu_time(lambda: pipe_mp(logits, min_p=0.05), repeat_iters=num_iters)
    )
    t_ms = np.median(
        bench_gpu_time(
            lambda: torch_softmax_minp_sample(logits, 0.05), repeat_iters=num_iters
        )
    )
    results.append(("minp", t_ms, fi_ms))
    return results


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
