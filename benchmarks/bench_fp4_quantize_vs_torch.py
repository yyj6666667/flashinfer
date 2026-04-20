#!/usr/bin/env python3
"""Benchmark flashinfer.fp4_quantize / nvfp4_quantize vs a torch eager
equivalent that computes per-block scales and clamps to an FP4 grid.

Only the timing is comparable — the flashinfer kernel packs 2 FP4
values per uint8, so output shapes differ from the naive reference.
"""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

FP4_E2M1_MAX = 6.0


@torch.inference_mode()
def torch_fp4_quantize_ref(x: torch.Tensor, global_scale: torch.Tensor, block_size: int = 16):
    """Naive torch reference: per-block amax -> block scale, clamp to [-6, 6],
    round to nearest 0.5 (FP4 E2M1 has 7 unique positive values: {0,0.5,1,1.5,2,3,4,6})."""
    m, n = x.shape
    x = x.to(torch.float32)
    blocks = x.view(m, n // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    block_scale = amax / FP4_E2M1_MAX * global_scale
    block_scale = torch.clamp(block_scale, min=1e-8)
    normalized = blocks / block_scale
    # Approximate FP4 E2M1 rounding via clamp + 0.5-step rounding (not
    # bit-exact, but computationally similar).
    packed = torch.clamp(torch.round(normalized * 2.0) / 2.0, -FP4_E2M1_MAX, FP4_E2M1_MAX)
    return packed, block_scale.squeeze(-1)


@torch.inference_mode()
def bench_one(fn_name: str, rows: int, cols: int, num_iters: int):
    device = torch.device("cuda")
    x = torch.randn(rows, cols, dtype=torch.bfloat16, device=device)
    global_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    if fn_name == "fp4_quantize":
        fi_fn = lambda: flashinfer.fp4_quantize(x, global_scale)
    elif fn_name == "nvfp4_quantize":
        fi_fn = lambda: flashinfer.nvfp4_quantize(x, global_scale)
    else:
        raise ValueError(fn_name)
    torch_fn = lambda: torch_fp4_quantize_ref(x, global_scale)

    _ = fi_fn()
    _ = torch_fn()

    fi_ms = np.median(bench_gpu_time(fi_fn, repeat_iters=num_iters))
    torch_ms = np.median(bench_gpu_time(torch_fn, repeat_iters=num_iters))
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="+", type=int, default=[128, 1024, 4096])
    p.add_argument("--cols", nargs="+", type=int, default=[512, 4096])
    p.add_argument("--fns", nargs="+", default=["fp4_quantize", "nvfp4_quantize"])
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name()}")
    header = (
        f"{'fn':>16} {'rows':>6} {'cols':>6} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for fn in args.fns:
        for r in args.rows:
            for c in args.cols:
                t_ms, fi_ms = bench_one(fn, r, c, args.num_iters)
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{fn:>16} {r:>6} {c:>6} "
                    f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
                )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
