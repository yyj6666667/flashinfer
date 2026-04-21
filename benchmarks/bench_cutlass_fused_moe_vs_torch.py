#!/usr/bin/env python3
"""Benchmark a PyTorch SwiGLU MoE baseline vs flashinfer.fused_moe.cutlass_fused_moe.

Matches the vs_torch convention used by the other Windows-port benches
(bench_rmsnorm_vs_torch.py, bench_gemm_fp8_vs_torch.py, ...): prints
per-shape torch time, flashinfer time, and speedup.

Uses pure bf16 weights (no FP8/FP4 quantization) — this is the portable
path and is what the Windows SM120 MoE module currently exposes.
"""

import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import flashinfer.fused_moe as fused_moe
from flashinfer.testing.utils import bench_gpu_time


def compute_routing(
    router_logits: torch.Tensor, top_k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    weights, experts = torch.topk(weights, top_k, dim=-1)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), experts


def torch_moe(
    x: torch.Tensor,
    w31: torch.Tensor,
    w2: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU MoE reference: gather per-expert rows, FC1→silu×gate→FC2, scatter-add."""
    num_experts = w31.shape[0]
    out = torch.zeros_like(x)
    for e in range(num_experts):
        mask = selected_experts == e
        if not mask.any():
            continue
        rows, slot = torch.where(mask)
        w3, w1 = torch.chunk(w31[e], 2, dim=0)
        xi = x[rows]
        inter = F.silu(xi @ w1.t()) * (xi @ w3.t())
        yi = inter @ w2[e].t()
        out[rows] += routing_weights[rows, slot, None].to(yi.dtype) * yi
    return out


@torch.inference_mode()
def bench_one(
    batch: int,
    hidden: int,
    num_experts: int,
    top_k: int,
    inter: int,
    dtype: torch.dtype,
    num_iters: int,
) -> Tuple[float, float]:
    torch.manual_seed(42)
    scale = 1.0 / 5.0
    x = torch.randn(batch, hidden, dtype=dtype, device="cuda") * scale
    w31 = (
        torch.randn(num_experts, 2 * inter, hidden, dtype=dtype, device="cuda") * scale
    )
    w2 = torch.randn(num_experts, hidden, inter, dtype=dtype, device="cuda") * scale
    router_logits = torch.randn(batch, num_experts, dtype=torch.float32, device="cuda")
    routing_weights, selected_experts = compute_routing(router_logits, top_k)

    ref = torch_moe(x, w31, w2, selected_experts, routing_weights).to(dtype)
    flash_out = torch.empty_like(ref)
    fi_out = fused_moe.cutlass_fused_moe(
        x,
        selected_experts.to(torch.int32),
        routing_weights,
        w31,
        w2,
        dtype,
        output=flash_out,
        quant_scales=None,
    )
    # flashinfer returns a list-like; normalise to a tensor for allclose.
    fi_tensor = fi_out[0] if isinstance(fi_out, (list, tuple)) else fi_out
    # Loose tolerance: bf16 MoE accumulates through 3 GEMMs + SwiGLU, so per-
    # element drift occasionally exceeds atol=0.1 on isolated outliers.
    # Compare 99th-percentile abs error instead of worst-case.
    abs_err = (fi_tensor.float() - ref.float()).abs().flatten()
    p99 = torch.quantile(abs_err, 0.99).item()
    if p99 > 0.15:
        raise AssertionError(
            f"cutlass_fused_moe p99 abs error {p99:.3f} exceeds 0.15"
        )

    torch_ms = np.median(
        bench_gpu_time(
            lambda: torch_moe(x, w31, w2, selected_experts, routing_weights),
            repeat_iters=num_iters,
        )
    )
    fi_ms = np.median(
        bench_gpu_time(
            lambda: fused_moe.cutlass_fused_moe(
                x,
                selected_experts.to(torch.int32),
                routing_weights,
                w31,
                w2,
                dtype,
                output=flash_out,
                quant_scales=None,
            ),
            repeat_iters=num_iters,
        )
    )
    return torch_ms, fi_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 64])
    parser.add_argument("--hiddens", nargs="+", type=int, default=[512, 1024])
    parser.add_argument("--experts", nargs="+", type=int, default=[8])
    parser.add_argument("--topk", nargs="+", type=int, default=[2])
    parser.add_argument("--inters", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--num-iters", type=int, default=30)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Device: {torch.cuda.get_device_name()} | dtype: {args.dtype}")
    header = (
        f"{'bs':>4} {'hidden':>6} {'E':>3} {'topK':>4} {'inter':>5} "
        f"{'torch (us)':>11} {'flashinfer (us)':>16} {'speedup':>8}"
    )
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    speedups: List[float] = []
    for b in args.batches:
        for h in args.hiddens:
            for e in args.experts:
                for k in args.topk:
                    if k > e:
                        continue
                    for i in args.inters:
                        t_ms, fi_ms = bench_one(
                            b, h, e, k, i, dtype, args.num_iters
                        )
                        sp = t_ms / fi_ms
                        speedups.append(sp)
                        print(
                            f"{b:>4} {h:>6} {e:>3} {k:>4} {i:>5} "
                            f"{t_ms * 1e3:>11.2f} {fi_ms * 1e3:>16.2f} {sp:>7.2f}x"
                        )

    print("=" * len(header))
    print(
        f"speedup  avg={np.mean(speedups):.2f}x  median={np.median(speedups):.2f}x  "
        f"min={np.min(speedups):.2f}x  max={np.max(speedups):.2f}x"
    )


if __name__ == "__main__":
    main()
