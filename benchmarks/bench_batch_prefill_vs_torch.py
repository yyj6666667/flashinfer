#!/usr/bin/env python3
"""Benchmark flashinfer.BatchPrefillWithPagedKVCacheWrapper vs torch SDPA.

All sequences in the batch share the same qo_len/kv_len for simplicity.
FlashInfer uses paged KV cache (page_size=16); torch uses contiguous.
"""

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_prefill(q_bhld, k_bhld, v_bhld, causal):
    # q, k, v: (B, H, L, D) already in SDPA layout.
    num_qo = q_bhld.shape[1]
    num_kv = k_bhld.shape[1]
    if num_qo != num_kv:
        rep = num_qo // num_kv
        k_bhld = k_bhld.repeat_interleave(rep, dim=1)
        v_bhld = v_bhld.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q_bhld, k_bhld, v_bhld, is_causal=causal)


@torch.inference_mode()
def bench_one(batch, qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim, causal, dtype, num_iters):
    device = "cuda"
    page_size = 16

    # Paged KV setup.
    seq_lens = torch.full((batch,), kv_len, device=device, dtype=torch.int32)
    seq_lens_blocks = torch.ceil(seq_lens.float() / page_size).int()
    kv_indptr = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.int32), seq_lens_blocks.cumsum(0).int()]
    )
    num_blocks = kv_indptr[-1].item()
    last_page_len = seq_lens - (seq_lens_blocks - 1) * page_size

    # qo_indptr (ragged): batch * qo_len per sequence.
    qo_indptr = torch.arange(0, (batch + 1) * qo_len, qo_len, device=device, dtype=torch.int32)

    q = torch.randn(batch * qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    kv_data = torch.randn(
        num_blocks, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        torch.arange(num_blocks, device=device, dtype=torch.int32),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    # Torch baseline: reshape paged KV into contiguous (B, H, L, D).
    k_cont = kv_data[:, 0].reshape(batch, kv_len, num_kv_heads, head_dim).transpose(1, 2)
    v_cont = kv_data[:, 1].reshape(batch, kv_len, num_kv_heads, head_dim).transpose(1, 2)
    q_bhld = q.reshape(batch, qo_len, num_qo_heads, head_dim).transpose(1, 2)

    # Warm up both paths.
    _ = torch_prefill(q_bhld, k_cont, v_cont, causal)
    _ = wrapper.run(q, kv_data)

    torch_ms = np.median(
        bench_gpu_time(lambda: torch_prefill(q_bhld, k_cont, v_cont, causal), repeat_iters=num_iters)
    )
    fi_ms = np.median(
        bench_gpu_time(lambda: wrapper.run(q, kv_data), repeat_iters=num_iters)
    )
    bytes_moved = q.numel() * q.element_size() * 2 + kv_data.numel() * kv_data.element_size()
    fi_tbps = bytes_moved / (fi_ms * 1e-3) / 1e12
    return torch_ms, fi_ms, fi_tbps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--qo-lens", nargs="+", type=int, default=[128, 512])
    p.add_argument("--kv-lens", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--num-qo-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(
        f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype} | "
        f"qh={args.num_qo_heads} kvh={args.num_kv_heads} hd={args.head_dim} causal={args.causal}"
    )
    header = (
        f"{'batch':>6} {'qo_len':>7} {'kv_len':>7} {'torch (us)':>12} "
        f"{'flashinfer (us)':>17} {'speedup':>9} {'TB/s':>7}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for ql in args.qo_lens:
            for kl in args.kv_lens:
                if args.causal and ql > kl:
                    continue
                try:
                    t_ms, fi_ms, fi_tbps = bench_one(
                        b, ql, kl, args.num_qo_heads, args.num_kv_heads, args.head_dim,
                        args.causal, dtype, args.num_iters,
                    )
                except Exception as e:
                    print(f"{b:>6} {ql:>7} {kl:>7}  SKIPPED: {type(e).__name__}")
                    continue
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{b:>6} {ql:>7} {kl:>7} {t_ms * 1e3:>12.2f} "
                    f"{fi_ms * 1e3:>17.2f} {sp:>8.2f}x {fi_tbps:>7.2f}"
                )

    print("=" * len(header))
    if speedups:
        print(
            f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
            f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
        )


if __name__ == "__main__":
    main()
