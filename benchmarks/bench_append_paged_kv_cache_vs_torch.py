#!/usr/bin/env python3
"""Benchmark flashinfer.append_paged_kv_cache vs torch scatter equivalent."""

import argparse
from typing import List

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_append_paged(
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    page_size: int,
):
    # Naive scatter: for every row in the append arrays, compute target page/slot
    # from (batch_idx, position) and copy in-place.
    for i in range(k_in.shape[0]):
        b = batch_indices[i].item()
        pos = positions[i].item()
        page_idx = kv_indices[kv_indptr[b].item() + pos // page_size].item()
        slot = pos % page_size
        k_cache[page_idx, slot] = k_in[i]
        v_cache[page_idx, slot] = v_in[i]


@torch.inference_mode()
def bench_one(batch: int, append_len: int, kv_heads: int, head_dim: int, page_size: int, dtype, num_iters: int):
    device = torch.device("cuda")
    nnz = batch * append_len
    pages_per_req = (append_len + page_size - 1) // page_size
    total_pages = batch * pages_per_req + 4

    k_in = torch.randn(nnz, kv_heads, head_dim, dtype=dtype, device=device)
    v_in = torch.randn(nnz, kv_heads, head_dim, dtype=dtype, device=device)
    k_cache = torch.zeros(total_pages, page_size, kv_heads, head_dim, dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)

    batch_indices = torch.repeat_interleave(
        torch.arange(batch, dtype=torch.int32, device=device), append_len
    )
    positions = torch.arange(append_len, dtype=torch.int32, device=device).repeat(batch)
    kv_indptr = torch.arange(0, (batch + 1) * pages_per_req, pages_per_req, dtype=torch.int32, device=device)
    kv_indices = torch.arange(batch * pages_per_req, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((batch,), (append_len - 1) % page_size + 1, dtype=torch.int32, device=device)

    _ = flashinfer.append_paged_kv_cache(
        k_in, v_in, batch_indices, positions, (k_cache, v_cache),
        kv_indices, kv_indptr, kv_last_page_len, kv_layout="NHD",
    )
    _ = torch_append_paged(
        k_in, v_in, batch_indices, positions, k_cache, v_cache,
        kv_indices, kv_indptr, page_size,
    )

    fi_ms = np.median(
        bench_gpu_time(
            lambda: flashinfer.append_paged_kv_cache(
                k_in, v_in, batch_indices, positions, (k_cache, v_cache),
                kv_indices, kv_indptr, kv_last_page_len, kv_layout="NHD",
            ),
            repeat_iters=num_iters,
        )
    )
    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_append_paged(
                k_in, v_in, batch_indices, positions, k_cache, v_cache,
                kv_indices, kv_indptr, page_size,
            ),
            repeat_iters=num_iters,
        )
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--append-lens", nargs="+", type=int, default=[16, 64, 256])
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'append':>7} {'kvh':>4} {'hd':>5} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for a in args.append_lens:
            t_ms, fi_ms = bench_one(b, a, args.kv_heads, args.head_dim, args.page_size, dtype, args.num_iters)
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{b:>6} {a:>7} {args.kv_heads:>4} {args.head_dim:>5} "
                f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
