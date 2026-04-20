#!/usr/bin/env python3
"""Benchmark flashinfer.BatchDecodeWithPagedKVCacheWrapper vs torch SDPA.

Per-shape latency, speedup, and effective bandwidth. Torch baseline uses
torch.nn.functional.scaled_dot_product_attention over contiguous KV;
FlashInfer uses paged KV cache (page_size=16).
"""

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_decode(q, k, v):
    # q: (bsz, num_qo_heads, 1, head_dim)
    # k, v: (bsz, num_kv_heads, kv_len, head_dim)
    # Repeat KV heads to match qo heads (GQA).
    num_qo_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_qo_heads != num_kv_heads:
        assert num_qo_heads % num_kv_heads == 0
        rep = num_qo_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


@torch.inference_mode()
def bench_one(batch, kv_len, num_qo_heads, num_kv_heads, head_dim, dtype, num_iters):
    # Shapes.
    page_size = 16
    device = "cuda"

    # Build paged KV cache for flashinfer.
    seq_lens = torch.full((batch,), kv_len, device=device, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens.float() / page_size).int()
    kv_indptr = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int32), seq_lens_blocks.cumsum(0).int()]
    )
    num_blocks = kv_indptr[-1].item()
    last_page_len = seq_lens - (seq_lens_blocks - 1) * page_size

    q = torch.randn(batch, num_qo_heads, head_dim, dtype=dtype, device=device)
    kv_data = torch.randn(
        num_blocks, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD"
    )
    wrapper.plan(
        kv_indptr,
        torch.arange(num_blocks, device=device, dtype=torch.int32),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Correctness + baseline tensors. Rebuild contiguous K/V from pages.
    # Layout NHD: (num_blocks, 2, page_size, num_kv_heads, head_dim).
    k_cont = kv_data[:, 0].reshape(batch, kv_len, num_kv_heads, head_dim).transpose(1, 2)
    v_cont = kv_data[:, 1].reshape(batch, kv_len, num_kv_heads, head_dim).transpose(1, 2)
    q_torch = q.unsqueeze(2)  # (B, H, 1, D)

    ref = torch_decode(q_torch, k_cont, v_cont).squeeze(2)
    out = wrapper.run(q, kv_data)
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)

    torch_ms = np.median(
        bench_gpu_time(lambda: torch_decode(q_torch, k_cont, v_cont), repeat_iters=num_iters)
    )
    fi_ms = np.median(
        bench_gpu_time(lambda: wrapper.run(q, kv_data), repeat_iters=num_iters)
    )

    # Memory traffic: read Q + KV + write O.
    bytes_moved = (
        q.numel() * q.element_size()
        + kv_data.numel() * kv_data.element_size() * 2 / (2 * page_size * num_kv_heads * head_dim)  # approx actual KV read
        + q.numel() * q.element_size()
    )
    fi_tbps = bytes_moved / (fi_ms * 1e-3) / 1e12
    return torch_ms, fi_ms, fi_tbps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64])
    parser.add_argument("--kv-lens", nargs="+", type=int, default=[512, 2048, 8192])
    parser.add_argument("--num-qo-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--num-iters", type=int, default=30)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(
        f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype} | "
        f"qh={args.num_qo_heads} kvh={args.num_kv_heads} hd={args.head_dim}"
    )
    header = (
        f"{'batch':>6} {'kv_len':>7} {'torch (us)':>12} "
        f"{'flashinfer (us)':>17} {'speedup':>9} {'TB/s':>7}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for kl in args.kv_lens:
            t_ms, fi_ms, fi_tbps = bench_one(
                b, kl, args.num_qo_heads, args.num_kv_heads, args.head_dim,
                dtype, args.num_iters,
            )
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{b:>6} {kl:>7} {t_ms * 1e3:>12.2f} "
                f"{fi_ms * 1e3:>17.2f} {sp:>8.2f}x {fi_tbps:>7.2f}"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
