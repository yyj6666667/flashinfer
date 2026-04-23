# sglang Windows-port patches

These four patches apply to the **sglang** repository (separate from
FlashInfer) and are required for MoE models (Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
and kin) to load + run on Windows. Run `apply_sglang_patches.py` from a
FlashInfer clone to (re)apply them to an sglang checkout; the script is
idempotent and uses narrow anchor-based replacement, so minor upstream
churn around the anchors is tolerated.

End-to-end validation (2026-04-23, fi-win / RTX 5060): Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
launches, /model_info returns 200, /generate produces coherent tokens
for the prompt "The three primary colors are" → " red, green, and blue.".

The four patches:

## 1. `kt_ep_wrapper.py` — fallback to JIT `gptq_marlin_repack`

**Target**: `sglang/srt/layers/moe/kt_ep_wrapper.py`

**Problem**: top-level `from sgl_kernel import gptq_marlin_repack` fails
on Windows because the pre-built `sgl_kernel` wheel skipped that kernel
(FA3 / gptq_marlin_repack are deferred due to MSVC template issues).
The failed import cascades: every MoE model file (qwen2_moe, qwen3_moe,
mixtral, olmoe, phimoe, …) transitively imports this wrapper and gets
dropped at module-import time with the cryptic
`Ignore import error … cannot import name 'gptq_marlin_repack' from 'sgl_kernel'`.

**Fix**: fall back to `sglang.jit_kernel.gptq_marlin_repack`, which
compiles through tvm-ffi (fully working on Windows after the cuda-link
patch in `flashinfer/scripts/windows/patches/apply_tvm_ffi_patches.py`).

## 2. `qwen2_moe.py` — skip extra GPTQ bias tensors in expert mapping

**Target**: `sglang/srt/models/qwen2_moe.py` (the `load_weights` method)

**Problem**: Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 checkpoints contain per-expert
`down_proj.bias` / `gate_proj.bias` / `up_proj.bias` tensors
(~4392 of them). The `stacked_params_mapping` branch already skips these
via "if name.endswith('.bias') and name not in params_dict: continue";
the `expert_params_mapping` branch didn't, so it KeyErrors on
`model.layers.0.mlp.experts.w2_bias`.

**Fix**: mirror the same bias-skip guard in the expert branch.

## 3. `gptq_marlin.cuh` — break the else-if chain that trips MSVC C1061

**Target**: `sglang/jit_kernel/csrc/gemm/marlin/gptq_marlin.cuh`

**Problem**: the `_GET_IF` macro expands (via `COMMON_GET_IF`,
`FP4_GET_IF`, `BIGGROUP_GET_IF`, `ACT_GET_IF`, `FZP_GET_IF`) into
~200 chained `else if (...) { kernel = ... }` clauses inside
`get_marlin_kernel`. MSVC's C1061 "nesting too deep" parse-tree limit
triggers at ~128–256 chained else-ifs, blocking the JIT compile with
`fatal error C1061` at line 412.

**Fix**: change the macro from `else if` to plain `if`. The conditions
are mutually exclusive on `q_type + thread_*_blocks + group_blocks + ...`
so at most one fires — the if-chain-vs-independent-if distinction is
semantically equivalent, but the latter has per-statement parse-tree
depth 1 instead of 200.

## 4. `moe_wna16_marlin.cuh` — same pattern, same fix

**Target**: `sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh`

Same `_GET_IF` macro, same C1061 nesting-too-deep failure at line 456
during the first MoE forward pass. Applying the same `else if` → `if`
transformation unblocks the MoE-specific Marlin kernel JIT.

## Persistence

These patches should go upstream to sglang as PRs. Until then,
`apply_sglang_patches.py` is the way to reproduce this state on a
fresh sglang clone.
