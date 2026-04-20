#!/usr/bin/env python3
"""Benchmark flashinfer.single_decode_with_kv_cache vs torch SDPA reference."""

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_single_decode(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float
) -> torch.Tensor:
    # q: [1, H, D]; k, v: [kv_len, H_kv, D]
    # broadcast GQA: if H_kv < H, repeat_interleave along the head axis
    H, D = q.shape[-2], q.shape[-1]
    H_kv = k.shape[-2]
    if H_kv != H:
        rep = H // H_kv
        k = k.repeat_interleave(rep, dim=-2)
        v = v.repeat_interleave(rep, dim=-2)
    q_t = q.reshape(1, H, 1, D)                # [1, H, 1, D]
    k_t = k.transpose(0, 1).unsqueeze(0)        # [1, H, kv_len, D]
    v_t = v.transpose(0, 1).unsqueeze(0)        # [1, H, kv_len, D]
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=sm_scale, is_causal=False)
    return out.squeeze(0).squeeze(1)            # -> [H, D]


@torch.inference_mode()
def bench_one(kv_len: int, qh: int, kvh: int, hd: int, dtype, num_iters: int):
    device = torch.device("cuda")
    q = torch.randn(qh, hd, dtype=dtype, device=device)
    k = torch.randn(kv_len, kvh, hd, dtype=dtype, device=device)
    v = torch.randn(kv_len, kvh, hd, dtype=dtype, device=device)
    sm_scale = 1.0 / (hd ** 0.5)

    _ = flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="NHD")
    _ = torch_single_decode(q, k, v, sm_scale)

    fi_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="NHD"),
            repeat_iters=num_iters,
        )
    )
    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_single_decode(q, k, v, sm_scale), repeat_iters=num_iters
        )
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kv-lens", nargs="+", type=int, default=[512, 2048, 8192])
    p.add_argument("--qh", type=int, default=32)
    p.add_argument("--kvh", type=int, default=8)
    p.add_argument("--hd", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'kv_len':>7} {'qh':>4} {'kvh':>4} {'hd':>5} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for kv in args.kv_lens:
        t_ms, fi_ms = bench_one(kv, args.qh, args.kvh, args.hd, dtype, args.num_iters)
        sp = t_ms / fi_ms
        speedups.append(sp)
        print(
            f"{kv:>7} {args.qh:>4} {args.kvh:>4} {args.hd:>5} "
            f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
        )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
