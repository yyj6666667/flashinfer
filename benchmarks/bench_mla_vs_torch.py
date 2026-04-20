#!/usr/bin/env python3
"""Benchmark flashinfer.mla BatchMLAPagedAttentionWrapper vs a naive torch
reference that reconstructs the full QK^T and softmax(QK^T)V."""

import argparse
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


@torch.inference_mode()
def torch_mla_ref(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    # q_nope: [B, H, D_ckv], q_pe: [B, H, D_kpe]
    # ckv:    [B*S, 1, D_ckv], kpe: [B*S, 1, D_kpe]  (page_size=1, 1 kv head)
    B, H, D_ckv = q_nope.shape
    S = ckv.shape[0] // B
    D_kpe = kpe.shape[-1]
    q = torch.cat([q_nope, q_pe], dim=-1).view(B, 1, H, D_ckv + D_kpe)
    kv = torch.cat([ckv.view(B, S, D_ckv), kpe.view(B, S, D_kpe)], dim=-1)
    k = kv.unsqueeze(2).expand(B, S, H, D_ckv + D_kpe)
    v = ckv.view(B, S, 1, D_ckv).expand(B, S, H, D_ckv)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=sm_scale, is_causal=False)
    return out.transpose(1, 2).reshape(B, H, D_ckv)


@torch.inference_mode()
def bench_one(batch: int, seq: int, heads: int, dtype: torch.dtype, num_iters: int):
    D_ckv, D_kpe = 512, 64
    device = torch.device("cuda")

    q_nope = torch.randn(batch, heads, D_ckv, dtype=dtype, device=device)
    q_pe = torch.randn(batch, heads, D_kpe, dtype=dtype, device=device)
    ckv = torch.randn(batch * seq, 1, D_ckv, dtype=dtype, device=device)
    kpe = torch.randn(batch * seq, 1, D_kpe, dtype=dtype, device=device)
    sm_scale = 1.0 / ((D_ckv + D_kpe) ** 0.5)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrap = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    q_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * seq
    kv_indices = torch.arange(batch * seq, dtype=torch.int32, device=device)
    kv_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    wrap.plan(
        q_indptr, kv_indptr, kv_indices, kv_lens,
        heads, D_ckv, D_kpe, 1, False, sm_scale, dtype, dtype,
    )

    _ = wrap.run(q_nope, q_pe, ckv, kpe)
    _ = torch_mla_ref(q_nope, q_pe, ckv, kpe, sm_scale)

    fi_ms = np.median(
        bench_gpu_time(lambda: wrap.run(q_nope, q_pe, ckv, kpe), repeat_iters=num_iters)
    )
    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_mla_ref(q_nope, q_pe, ckv, kpe, sm_scale),
            repeat_iters=num_iters,
        )
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    p.add_argument("--seqs", nargs="+", type=int, default=[512, 2048, 8192])
    p.add_argument("--heads", nargs="+", type=int, default=[16, 64])
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'batch':>6} {'seq':>6} {'heads':>6} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for s in args.seqs:
            for h in args.heads:
                t_ms, fi_ms = bench_one(b, s, h, dtype, args.num_iters)
                sp = t_ms / fi_ms
                speedups.append(sp)
                print(
                    f"{b:>6} {s:>6} {h:>6} "
                    f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
                )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
