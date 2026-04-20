#!/usr/bin/env python3
"""Benchmark flashinfer.quantization.packbits vs torch eager equivalent
(casting bool -> uint8 + bit-pack via gather/shift)."""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_packbits(x: torch.Tensor, bitorder: str) -> torch.Tensor:
    # x: bool, shape [N]; returns packed uint8 [ceil(N/8)]
    n = x.numel()
    pad = (-n) % 8
    if pad:
        x = torch.cat([x, torch.zeros(pad, dtype=x.dtype, device=x.device)])
    x = x.to(torch.uint8).view(-1, 8)
    if bitorder == "big":
        shifts = torch.arange(7, -1, -1, dtype=torch.uint8, device=x.device)
    else:
        shifts = torch.arange(0, 8, dtype=torch.uint8, device=x.device)
    return (x << shifts).sum(dim=-1, dtype=torch.uint8)


@torch.inference_mode()
def bench_one(n: int, bitorder: str, num_iters: int):
    device = torch.device("cuda")
    x = (torch.rand(n, device=device) < 0.5)
    _ = flashinfer.quantization.packbits(x, bitorder)
    _ = torch_packbits(x, bitorder)
    fi_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.quantization.packbits(x, bitorder), repeat_iters=num_iters
        )
    )
    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_packbits(x, bitorder), repeat_iters=num_iters
        )
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[1024, 131072, 1048576, 16777216])
    p.add_argument("--bitorders", nargs="+", default=["big", "little"])
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name()}")
    header = (
        f"{'n':>10} {'bitorder':>10} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for n in args.sizes:
        for bo in args.bitorders:
            t_ms, fi_ms = bench_one(n, bo, args.num_iters)
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{n:>10} {bo:>10} "
                f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
