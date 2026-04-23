# Windows Port Performance Report

**Host**: fi-win · RTX 5060 (SM 12.0, 8.55 GB VRAM) · CUDA 12.9 · MSVC 14.44 · torch 2.11.0+cu128 · Python 3.12.8
**Date**: 2026-04-23
**sglang branch**: `windows` + `scripts/windows/patches/sglang/apply_sglang_patches.py`
**flashinfer branch**: `windows` (commit `1f251780`) + `scripts/windows/patches/apply_tvm_ffi_patches.py`

---

## TL;DR

| Workload | sglang+flashinfer (tok/s) | HF transformers eager (tok/s) | Speedup |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct, bf16, short prompt, 64 gen | **131.9** | 63.1 | **2.09×** |
| Qwen2.5-0.5B-Instruct, bf16, medium prompt, 64 gen | **132.2** | 54.7 | **2.42×** |
| Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 (CPU-offload 2 GB) | ~0.1 tok/s | (baseline not run — model doesn't fit) | N/A |

**Conclusion**: on consumer Blackwell (SM120, 8GB VRAM) with dense bf16 Qwen,
the Windows port delivers **~2.1–2.4× throughput over HF eager** on the same
hardware. This is the floor — the comparison is against HF's deliberately
slow baseline; real-world useful comparisons (torch.compile, vLLM, TensorRT-LLM
Windows) weren't exercised here but would show smaller deltas. On the 8GB-VRAM
MoE path the speedup claim is **not meaningful** because CPU offload dominates
(~10s per token); the port works end-to-end but is bottlenecked by PCIe bandwidth
swapping experts. This part of the story changes as soon as the model fits
in VRAM (RTX 5070/5080/5090 16-32GB; Qwen1.5-MoE int4 needs ~7GB + KV cache).

---

## Method

1. **Baseline**: HuggingFace `transformers` eager generation (`AutoModelForCausalLM.from_pretrained(..., torch_dtype=bf16).to("cuda")` + `.generate(do_sample=False)`). No torch.compile, no flash-attn. This is what a user gets with zero optimization — a useful *floor*.

2. **Test rig**: `sglang.launch_server` with `--dtype bfloat16 --disable-cuda-graph --mem-fraction-static 0.85`, `/generate` via HTTP POST. Driven by `benchmarks/windows_e2e_bench.py` (part of this repo).

3. **Measured**: median + p95 wall time over 5 iterations (+ 2 warmups). `tokens_per_s = max_new_tokens / median_ms`. For small models on single-request paths, e2e latency is dominated by decode; prefill is a small fixed overhead.

4. **Prompt set**:
   - `short`  — "The capital of France is" (5 tokens)
   - `medium` — ~40-word instruction template

5. **Correctness check**: every run executed a warmup generate and sanity-checked the output string before timing. For Qwen2.5-0.5B `"The three primary colors are"` returns `"red, green, and blue."` on both paths — identical text.

---

## Qwen2.5-0.5B-Instruct (dense, bf16)

Raw CSV: `benchmarks/results/windows_e2e_qwen2_5_0_5b.csv`

| backend | prompt tok | gen | median (ms) | p95 (ms) | tok/s | notes |
|---|---:|---:|---:|---:|---:|---|
| sglang_flashinfer | 5 | 64 | 485 | 506 | **131.9** | |
| hf_eager | 5 | 64 | 1015 | 1095 | 63.1 | transformers eager, same dtype |
| sglang_flashinfer | 40 | 64 | 484 | 489 | **132.2** | |
| hf_eager | 40 | 64 | 1170 | 1188 | 54.7 | transformers eager, same dtype |

Decode-throughput ratio holds at **~132 tok/s** regardless of prompt length (prefill is a small fraction at 5-40 tokens on a 0.5B model). HF's number degrades slightly with longer prompts — HF's eager path recomputes full attention without KV cache reuse between generate calls.

### Reading the numbers

- 132 tok/s at batch 1 on a RTX 5060 for a 0.5B model is modest in absolute terms. For comparison, published Linux sglang benchmarks on an RTX 4090 show 300+ tok/s for the same model — so the Windows port is leaving a factor of 2-3× on the table vs. tuned Linux, but *not* vs. the alternative on the same machine (HF eager at 63 tok/s).
- The 132 tok/s ceiling is likely dominated by **`--disable-cuda-graph`** (cold launch forces it off for correctness during first validation) and by the JIT-compiled kernels not being tuned for SM120 tile sizes. See `docs/windows/KNOWN_ISSUES.md` 🟡-PARTIAL entries for the SM120-specific shape/workspace caveats.

---

## Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 (MoE, int4) — two paths compared

### Path 1: sglang with `--cpu-offload-gb 2` (naïve expert swapping over PCIe)

End-to-end pipeline works (correctness verified via one generation producing
`"The three primary colors are red, green, and blue."` prefix), but the
5060's 8GB VRAM cannot hold the ~7 GB int4 weights plus the KV cache + MoE
activations simultaneously. `--cpu-offload-gb` frees VRAM by shuffling MoE
expert weights between CPU RAM and GPU VRAM on every forward pass; each
forward pays the PCIe transfer cost.

**Steady state**: ~0.10 tok/s (9.7 s / token).

### Path 2: sglang + kt-kernel AVX2 (experts stay resident on CPU)

The ktransformers windows-branch kt-kernel module builds on Windows with
MSVC 14.44 + CUDA 12.9 (see `scripts/windows/patches/ktransformers/README.md`
for the build recipe). Sglang's built-in `kt_ep_wrapper` drives it via the
`--kt-*` family of server args. With `--kt-num-gpu-experts 0` all 60 experts
× 24 MoE layers live on CPU; the GPU only does attention + gate + output.

Measured on fi-win (Intel Core Ultra 9 285, 24 CPU cores, 20 used via
`--kt-cpuinfer 20`), 3-run steady state:

| Run | e2e (s) | tokens | tok/s |
|---|---:|---:|---:|
| 1 | 64.49 | 23 | 0.36 |
| 2 | 59.39 | 23 | **0.39** |
| 3 | 59.54 | 23 | **0.39** |

Prompt = `"The capital of France is"`, `max_new_tokens=32`, `temperature=0`.
Generated text degrades past ~10 tokens (same on both paths — checkpoint
precision issue, not kt-specific).

### Summary

| Backend (same Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4, same hardware) | tok/s | ratio |
|---|---:|---:|
| sglang `--cpu-offload-gb 2` | 0.10 | 1.0× |
| **sglang + kt-kernel AVX2 (`--kt-method GPTQ_INT4`)** | **0.39** | **3.9×** |

**Interpretation**: 0.39 tok/s is still slow in absolute terms — on a 24-core
Ultra 9 with AVX2 we'd expect closer to 1-2 tok/s for Qwen1.5-MoE-A2.7B
(2.7B active params at int4 ≈ 1.4 GB of per-token compute). The 3.9× gap
between the two paths reflects the raw cost of PCIe expert swapping vs
keeping experts resident in CPU RAM — that cost is avoidable, and the
remaining headroom (kt-kernel vs theoretical peak) likely comes from
single-NUMA-node thread pool config, non-tuned prefill chunk size, and
room to move a few hot experts to GPU via `--kt-num-gpu-experts > 0`.

For the Windows port decision: **kt-kernel on Windows works end-to-end and
is 4× better than the pure-PCIe alternative on the same consumer hardware**.
Scaling past that requires either more VRAM (RTX 5070 16GB / 5080 16GB /
5090 32GB so experts can overflow to GPU) or profiling the kt-kernel side
to find where 1-2 tok/s is being left on the table.

### Legacy first-run number (superseded)

Prior cold-path number of 239 s / 24 tokens was the `--cpu-offload-gb` path
**including** first-request JIT compile of the Marlin and MoE Marlin
kernels (~90-120 s of the 239 s). After that, steady-state was the 0.10
tok/s figure above.

**What we're NOT yet measuring** (all blocked on larger consumer VRAM):
- Qwen1.5-MoE Int4 throughput at no CPU offload (needs ≥12 GB VRAM for weights alone)
- Qwen3-30B-A3B in any format (30 GB bf16; ~7.5 GB int4 still won't fit 8 GB with KV cache)
- Realistic MoE prompt/batch throughput on SM120

Recommendation: repeat this section on a 16+ GB card (RTX 5070Ti 16GB / 5080 16GB / 5090 32GB) before drawing perf conclusions for the MoE consumer story. The code path is proven; the hardware ceiling on fi-win is the only blocker.

---

## Raw reproducibility

One-shot bench against a running sglang on fi-win:

```powershell
# launcher is at C:\flashinfer\.bench_session.ps1, from this repo's tmp-like path
powershell -File C:\flashinfer\.bench_session.ps1 `
    -ModelPath C:/models/Qwen2.5-0.5B-Instruct `
    -ModelTag qwen2_5_0_5b `
    -Dtype bfloat16
```

That script:
1. applies vcvars + CUDA env
2. starts sglang with the given model
3. polls `/model_info` until 200
4. sends one warmup request
5. runs `benchmarks/windows_e2e_bench.py --hf-baseline` for the timing sweep
6. kills the sglang process

---

## Caveats / what this report does NOT claim

- **No absolute Linux-vs-Windows comparison**: we don't have an RTX 5060 on Linux to diff against. Our in-absolute-terms numbers (132 tok/s) likely trail a Linux reference; we'd want that data before claiming the Windows port is "as fast as Linux".
- **No torch.compile baseline**: the `hf_eager` baseline here is the slowest reasonable comparison. `torch.compile(model)` often recovers 40-70% of the gap to optimized LLM runtimes and should be added as a second baseline.
- **No perf for kernels individually**: the whole-request e2e bench mixes prefill + sampling + HTTP. Kernel-level perf lives in `benchmarks/bench_*_vs_torch.py` files (covered in `benchmarks/results/windows_port_speedup.csv`).
- **No long-context or concurrent-request workloads**: context 64 tokens, batch 1. Under prefix caching / pagedattention / batched decode, the ratios will differ.
- **Single seed, 5 iterations**: variance within ~5%; not statistically hard numbers.

All of the above are in-scope for a later, production-grade perf pass.
