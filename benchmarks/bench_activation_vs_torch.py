#!/usr/bin/env python3
"""Benchmark flashinfer.activation vs torch eager silu/gelu gated variants."""

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


@torch.inference_mode()
def torch_gelu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d]) * x[..., d:]


@torch.inference_mode()
def torch_gelu_tanh_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.gelu(x[..., :d], approximate="tanh") * x[..., d:]


@torch.inference_mode()
def bench_one(batch: int, hidden: int, dtype: torch.dtype, num_iters: int):
    # hidden is the GATED hidden dim; x carries 2*hidden.
    x = torch.randn(batch, 2 * hidden, dtype=dtype, device="cuda")

    variants = [
        ("silu_and_mul", flashinfer.activation.silu_and_mul, torch_silu_and_mul),
        ("gelu_and_mul", flashinfer.activation.gelu_and_mul, torch_gelu_and_mul),
        ("gelu_tanh_and_mul", flashinfer.activation.gelu_tanh_and_mul, torch_gelu_tanh_and_mul),
    ]

    results = []
    for name, fi_fn, torch_fn in variants:
        _ = fi_fn(x)
        _ = torch_fn(x)
        torch_ms = np.median(bench_gpu_time(lambda: torch_fn(x), repeat_iters=num_iters))
        fi_ms = np.median(bench_gpu_time(lambda: fi_fn(x), repeat_iters=num_iters))
        results.append((name, torch_ms, fi_ms))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 128, 2048])
    p.add_argument("--hiddens", nargs="+", type=int, default=[4096, 11008, 14336])
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=50)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'hidden':>7} {'variant':>20} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for h in args.hiddens:
            for name, t_ms, fi_ms in bench_one(b, h, dtype, args.num_iters):
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{b:>6} {h:>7} {name:>20} "
                    f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
                )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
