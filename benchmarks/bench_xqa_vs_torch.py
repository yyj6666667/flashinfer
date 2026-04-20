#!/usr/bin/env python3
"""Benchmark flashinfer.xqa (single-query paged decode) vs torch SDPA.

XQA requires SM90/SM100/SM120/SM121 GPUs. On other arches the kernel
compiles to a guard that raises RuntimeError.
"""

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_single_token_decode(q, k, v, seq_lens, sm_scale):
    # q: [B, 1, H, D] ; k, v: [B, kv_len, H_kv, D] (already un-paged for the ref)
    B, _, H, D = q.shape
    H_kv = k.shape[-2]
    if H_kv != H:
        rep = H // H_kv
        k = k.repeat_interleave(rep, dim=-2)
        v = v.repeat_interleave(rep, dim=-2)
    out = torch.empty_like(q)
    for b in range(B):
        L = int(seq_lens[b].item())
        q_t = q[b, :, :, :].transpose(0, 1).unsqueeze(0)       # [1, H, 1, D]
        k_t = k[b, :L, :, :].transpose(0, 1).unsqueeze(0)      # [1, H, L, D]
        v_t = v[b, :L, :, :].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=sm_scale)
        out[b] = o.squeeze(0).transpose(0, 1)
    return out


@torch.inference_mode()
def bench_one(batch: int, kv_len: int, qh: int, kvh: int, hd: int, page_size: int, num_iters: int):
    device = torch.device("cuda")
    pages_per_req = (kv_len + page_size - 1) // page_size
    total_pages = batch * pages_per_req

    q = torch.randn(batch, 1, qh, hd, dtype=torch.float16, device=device)
    kcache = torch.randn(total_pages, page_size, kvh, hd, dtype=torch.float16, device=device)
    vcache = torch.randn_like(kcache)

    page_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(batch, -1)
    seq_lens = torch.full((batch,), kv_len, dtype=torch.int32, device=device)
    output = torch.empty_like(q)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    semaphores = torch.zeros(1024, dtype=torch.uint32, device=device)

    def run_fi():
        flashinfer.xqa(
            q, kcache, vcache, page_table, seq_lens, output, workspace, semaphores,
            num_kv_heads=kvh, page_size=page_size, kv_layout="NHD",
        )

    # torch ref needs k/v reconstructed in dense form
    k_dense = kcache.reshape(batch, pages_per_req * page_size, kvh, hd)
    v_dense = vcache.reshape(batch, pages_per_req * page_size, kvh, hd)
    sm_scale = 1.0 / (hd ** 0.5)
    def run_torch():
        return torch_single_token_decode(q, k_dense, v_dense, seq_lens, sm_scale)

    run_fi()
    run_torch()

    fi_ms = np.median(bench_gpu_time(run_fi, repeat_iters=num_iters))
    torch_ms = np.median(bench_gpu_time(run_torch, repeat_iters=num_iters))
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--kv-lens", nargs="+", type=int, default=[512, 2048, 8192])
    p.add_argument("--qh", type=int, default=8)
    p.add_argument("--kvh", type=int, default=8)
    p.add_argument("--hd", type=int, default=128)
    p.add_argument("--page-size", type=int, default=16)
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    print(f"Device: {torch.cuda.get_device_name()}")
    header = (
        f"{'batch':>6} {'kv_len':>7} {'qh':>4} {'kvh':>4} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for kv in args.kv_lens:
            t_ms, fi_ms = bench_one(b, kv, args.qh, args.kvh, args.hd, args.page_size, args.num_iters)
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{b:>6} {kv:>7} {args.qh:>4} {args.kvh:>4} "
                f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
