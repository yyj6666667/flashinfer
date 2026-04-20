#!/usr/bin/env python3
"""Benchmark flashinfer.BlockSparseAttentionWrapper vs a torch dense reference
with an explicit attention mask.

The reference expands the BSR mask to a dense boolean mask and calls
F.scaled_dot_product_attention with `attn_mask=`. Competitive when sparsity
is low; becomes wasteful when the BSR is very sparse, so speedups grow with
sparsity.
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.testing.utils import bench_gpu_time


def _make_bsr(M: int, N: int, density: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random BSR (block_size 1x1) with given density."""
    torch.manual_seed(0)
    dense = torch.rand(M, N, device=device) < density
    indptr = torch.zeros(M + 1, dtype=torch.int32, device=device)
    indices_list = []
    for i in range(M):
        cols = torch.where(dense[i])[0].to(torch.int32)
        indices_list.append(cols)
        indptr[i + 1] = indptr[i] + cols.numel()
    indices = torch.cat(indices_list) if indices_list else torch.zeros(0, dtype=torch.int32, device=device)
    return indptr, indices, dense


@torch.inference_mode()
def torch_dense_masked(q, k, v, mask_bool, sm_scale):
    # q: [M, H, D]; k, v: [N, Hkv, D]; mask_bool: [M, N]
    H, D = q.shape[-2], q.shape[-1]
    Hkv = k.shape[-2]
    if Hkv != H:
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=-2)
        v = v.repeat_interleave(rep, dim=-2)
    q_t = q.transpose(0, 1).unsqueeze(0)  # [1, H, M, D]
    k_t = k.transpose(0, 1).unsqueeze(0)  # [1, H, N, D]
    v_t = v.transpose(0, 1).unsqueeze(0)
    # attn_mask=True means attend; float mask: 0=attend, -inf=mask
    attn_mask = torch.where(mask_bool, 0.0, float("-inf")).to(q.dtype).unsqueeze(0).unsqueeze(0)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=attn_mask, scale=sm_scale)
    return out.squeeze(0).transpose(0, 1)  # -> [M, H, D]


@torch.inference_mode()
def bench_one(M: int, N: int, density: float, qh: int, kvh: int, hd: int, dtype, num_iters: int):
    device = torch.device("cuda")
    indptr, indices, dense_mask = _make_bsr(M, N, density, device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    w = flashinfer.BlockSparseAttentionWrapper(workspace)
    w.plan(indptr, indices, M, N, 1, 1, qh, kvh, hd)

    q = torch.randn(M, qh, hd, dtype=dtype, device=device)
    k = torch.randn(N, kvh, hd, dtype=dtype, device=device)
    v = torch.randn(N, kvh, hd, dtype=dtype, device=device)
    sm_scale = 1.0 / (hd ** 0.5)

    _ = w.run(q, k, v)
    _ = torch_dense_masked(q, k, v, dense_mask, sm_scale)

    fi_ms = np.median(bench_gpu_time(lambda: w.run(q, k, v), repeat_iters=num_iters))
    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_dense_masked(q, k, v, dense_mask, sm_scale),
            repeat_iters=num_iters,
        )
    )
    return torch_ms, fi_ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", nargs="+", type=int, default=[512, 2048])
    p.add_argument("--densities", nargs="+", type=float, default=[0.1, 0.5])
    p.add_argument("--qh", type=int, default=32)
    p.add_argument("--kvh", type=int, default=8)
    p.add_argument("--hd", type=int, default=128)
    p.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    p.add_argument("--num-iters", type=int, default=30)
    args = p.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'M':>6} {'N':>6} {'density':>8} {'qh':>4} {'kvh':>4} "
        f"{'torch (us)':>12} {'flashinfer (us)':>17} {'speedup':>9}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for s in args.sizes:
        for d in args.densities:
            t_ms, fi_ms = bench_one(s, s, d, args.qh, args.kvh, args.hd, dtype, args.num_iters)
            sp = t_ms / fi_ms
            speedups.append(sp)
            print(
                f"{s:>6} {s:>6} {d:>8.2f} {args.qh:>4} {args.kvh:>4} "
                f"{t_ms * 1e3:>12.2f} {fi_ms * 1e3:>17.2f} {sp:>8.2f}x"
            )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
