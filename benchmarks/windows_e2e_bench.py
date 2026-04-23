"""End-to-end Windows perf validation for sglang.

Compares a running sglang server (flashinfer backend) against a HuggingFace
transformers eager baseline on the same machine and same model. Intended
to answer the question "is the Windows port fast enough to be useful?"
with concrete numbers rather than wishful thinking.

The output CSV lands in ``benchmarks/results/windows_e2e_<model_tag>.csv``
and a human-readable summary prints to stdout.

Usage (on fi-win, with sglang already running on localhost:30000):

    python benchmarks/windows_e2e_bench.py \\
        --model-path C:/models/Qwen2.5-0.5B-Instruct \\
        --model-tag qwen2_5_0_5b \\
        --sglang-url http://127.0.0.1:30000 \\
        --hf-baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class BenchResult:
    backend: str           # "sglang_flashinfer" or "hf_eager"
    phase: str             # "prefill" or "decode" or "e2e"
    seq_len_prompt: int
    seq_len_gen: int
    batch_size: int
    iters: int
    median_ms: float
    p95_ms: float
    tokens_per_s: float
    notes: str = ""


def time_block(fn, iters: int, warmup: int = 2):
    """Return (times_ms_list, result) for ``iters`` calls to ``fn``; discards warmups."""
    for _ in range(warmup):
        fn()
    times = []
    result = None
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times, result


def bench_sglang(
    url: str,
    prompt: str,
    max_new_tokens: int,
    iters: int,
    warmup: int = 2,
    batch: int = 1,
    temperature: float = 0.0,
) -> list[float]:
    """Drive /generate; return per-call latencies in ms.

    batch>1 sent as a single request with `n` = batch (sglang semantics).
    """
    import json as _json

    body = {
        "text": prompt if batch == 1 else [prompt] * batch,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
    }
    raw = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url + "/generate",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def call():
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()

    times, _ = time_block(call, iters, warmup)
    return times


def bench_hf(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    iters: int,
    warmup: int = 2,
    dtype: str = "bfloat16",
) -> list[float]:
    """Lower bound: plain HF transformers eager (no torch.compile, no flashinfer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    ).to("cuda").eval()

    enc = tok(prompt, return_tensors="pt").to("cuda")

    @torch.inference_mode()
    def call():
        return model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0
        )

    times, _ = time_block(call, iters, warmup)
    del model
    torch.cuda.empty_cache()
    return times


def fmt_median_p95(times: list[float]) -> tuple[float, float]:
    times_sorted = sorted(times)
    p95_idx = max(0, int(len(times) * 0.95) - 1)
    return statistics.median(times), times_sorted[p95_idx]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-tag", required=True, help="Used to name the output CSV.")
    p.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    p.add_argument("--hf-baseline", action="store_true",
                   help="Also run HF eager as a bottom baseline.")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument(
        "--prompts",
        default="short,medium,long",
        help="Comma-separated prompt categories (short=~10tok, medium=~200, long=~1500).",
    )
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument(
        "--out-dir", default="benchmarks/results",
    )
    args = p.parse_args()

    prompts = {
        "short":  "The capital of France is",
        "medium": "Explain the theory of relativity in plain terms. " * 5,
        "long":   "Summarize the plot of each Shakespeare play below. " * 60,
    }
    selected = [s.strip() for s in args.prompts.split(",") if s.strip()]

    results: list[BenchResult] = []

    for tag in selected:
        prompt = prompts[tag]
        prompt_tok_est = len(prompt.split())  # rough
        print(f"=== prompt={tag!r} (~{prompt_tok_est}w) max_new_tokens={args.new_tokens} ===")

        # sglang
        sg_times = bench_sglang(
            args.sglang_url, prompt, args.new_tokens, args.iters, args.warmup
        )
        med, p95 = fmt_median_p95(sg_times)
        tps = args.new_tokens / (med / 1000.0)
        results.append(BenchResult(
            backend="sglang_flashinfer",
            phase="e2e",
            seq_len_prompt=prompt_tok_est,
            seq_len_gen=args.new_tokens,
            batch_size=1,
            iters=args.iters,
            median_ms=med,
            p95_ms=p95,
            tokens_per_s=tps,
            notes="",
        ))
        print(f"  sglang_flashinfer  : median {med:7.1f} ms, p95 {p95:7.1f} ms  ({tps:5.1f} tok/s)")

        if args.hf_baseline:
            hf_times = bench_hf(
                args.model_path, prompt, args.new_tokens, args.iters, args.warmup, args.dtype
            )
            med, p95 = fmt_median_p95(hf_times)
            tps = args.new_tokens / (med / 1000.0)
            results.append(BenchResult(
                backend="hf_eager",
                phase="e2e",
                seq_len_prompt=prompt_tok_est,
                seq_len_gen=args.new_tokens,
                batch_size=1,
                iters=args.iters,
                median_ms=med,
                p95_ms=p95,
                tokens_per_s=tps,
                notes="transformers eager, same dtype",
            ))
            print(f"  hf_eager (baseline): median {med:7.1f} ms, p95 {p95:7.1f} ms  ({tps:5.1f} tok/s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"windows_e2e_{args.model_tag}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    print(f"\nwritten {out}")

    # Speedup summary
    by_phase = {}
    for r in results:
        by_phase.setdefault((r.seq_len_prompt, r.seq_len_gen), {})[r.backend] = r
    print("\n=== speedup summary (sglang vs hf_eager) ===")
    for k, bmap in by_phase.items():
        sg = bmap.get("sglang_flashinfer")
        hf = bmap.get("hf_eager")
        if sg and hf:
            speedup = hf.median_ms / sg.median_ms
            print(f"  prompt_tok={k[0]:>5} gen={k[1]:>3}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
