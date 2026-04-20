#!/usr/bin/env python3
"""Benchmark flashinfer.attention.BatchAttention (holistic attention) vs
a torch SDPA reference that reconstructs dense K/V per batch entry."""

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from flashinfer.attention import BatchAttention
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_ref(
    q, kv, qo_indptr, kv_indptr, kv_indices, kv_last_page_len, page_size,
    num_qo_heads, num_kv_heads, head_dim, causal, sm_scale,
):
    batch = qo_indptr.shape[0] - 1
    out = torch.empty(q.shape[0], num_qo_heads, head_dim, dtype=q.dtype, device=q.device)
    rep = num_qo_heads // num_kv_heads
    for b in range(batch):
        qo_s, qo_e = int(qo_indptr[b]), int(qo_indptr[b + 1])
        kv_s, kv_e = int(kv_indptr[b]), int(kv_indptr[b + 1])
        pages = kv_indices[kv_s:kv_e]
        last = int(kv_last_page_len[b])
        # Reconstruct dense KV for this request.
        parts_k, parts_v = [], []
        for i, pg in enumerate(pages.tolist()):
            take = last if i == len(pages) - 1 else page_size
            parts_k.append(kv[pg, 0, :take])
            parts_v.append(kv[pg, 1, :take])
        k_dense = torch.cat(parts_k, dim=0)            # [L, H_kv, D]
        v_dense = torch.cat(parts_v, dim=0)
        if rep > 1:
            k_dense = k_dense.repeat_interleave(rep, dim=-2)
            v_dense = v_dense.repeat_interleave(rep, dim=-2)
        q_t = q[qo_s:qo_e].transpose(0, 1).unsqueeze(0)        # [1, H, qo, D]
        k_t = k_dense.transpose(0, 1).unsqueeze(0)
        v_t = v_dense.transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=sm_scale, is_causal=causal)
        out[qo_s:qo_e] = o.squeeze(0).transpose(0, 1)
    return out


@torch.inference_mode()
def bench_one(batch, qo_len, kv_len, qh, kvh, hd, page_size, dtype, num_iters):
    device = torch.device("cuda")
    pages_per_req = (kv_len + page_size - 1) // page_size
    total_pages = batch * pages_per_req

    qo_indptr = torch.arange(0, (batch + 1) * qo_len, qo_len, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(0, (batch + 1) * pages_per_req, pages_per_req, dtype=torch.int32, device=device)
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last = torch.full((batch,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device=device)
    kv_len_arr = torch.full((batch,), kv_len, dtype=torch.int32, device=device)

    q = torch.randn(batch * qo_len, qh, hd, dtype=dtype, device=device)
    kv = torch.randn(total_pages, 2, page_size, kvh, hd, dtype=dtype, device=device)

    w = BatchAttention(kv_layout="NHD")
    w.plan(qo_indptr, kv_indptr, kv_indices, kv_len_arr,
           num_qo_heads=qh, num_kv_heads=kvh,
           head_dim_qk=hd, head_dim_vo=hd, page_size=page_size,
           causal=True, q_data_type=dtype, kv_data_type=dtype)
    sm_scale = 1.0 / (hd ** 0.5)

    _ = w.run(q, kv)
    _ = torch_ref(q, kv, qo_indptr, kv_indptr, kv_indices, kv_last, page_size,
                  qh, kvh, hd, True, sm_scale)

    fi_ms = np.median(bench_gpu_time(lambda: w.run(q, kv), repeat_iters=num_iters))
    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_ref(q, kv, qo_indptr, kv_indptr, kv_indices, kv_last,
                              page_size, qh, kvh, hd, True, sm_scale),
            repeat_iters=num_iters,
        )
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--qo-lens", nargs="+", type=int, default=[128, 512])
    p.add_argument("--kv-lens", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--qh", type=int, default=32)
    p.add_argument("--kvh", type=int, default=8)
    p.add_argument("--hd", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'qo':>5} {'kv':>6} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for qo in args.qo_lens:
            for kv in args.kv_lens:
                t_ms, fi_ms = bench_one(b, qo, kv, args.qh, args.kvh, args.hd,
                                        args.page_size, dtype, args.num_iters)
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{b:>6} {qo:>5} {kv:>6} "
                    f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
                )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
