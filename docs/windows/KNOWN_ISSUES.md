# Windows Port — Known Issues & Blockers (SM120 consumer Blackwell)

Last verified host: fi-win / RTX 5060 / SM120 / CUDA 12.9 / MSVC 14.44 / torch 2.11.0+cu128.
If the behavior below changes on newer toolchains, update this file.

Items are categorized as:

- **🔴 BLOCKED** — kernel is unreachable on Windows SM120; needs upstream fix or a non-trivial refactor.
- **🟡 PARTIAL** — kernel works in a reduced envelope (some shapes / some modes only).
- **🟢 RESOLVED** — fixed on this branch; listed for historical traceability.
- **⚪ BY DESIGN** — not supported on SM120 by upstream decision; torch fallback is expected.

---

## 🔴 BLOCKED

### `fused_moe` (TRT-LLM fused_moe path)
- **nvcc ICE**: `cutlass_fused_moe_kernels.cuh:1565` "could not lookup variable in map!"
- **MSVC CRT cascade**: C2641 on `__vcrt_assert_va_start_is_not_reference` in `corecrt_wstdio.h`
- Already fixed before these: `cudaUtils.h` C99 designated init, deep_gemm operator keywords via `<ciso646>`
- **Workaround**: sglang should pick `cutlass_fused_moe` (which works) instead of `fused_moe`.

### `pod` attention
- MSVC resolves `using PrefillAttentionVariant/DecodeAttentionVariant` incorrectly across nested `DISPATCH_MASK_MODE` switch cases: the `static constexpr bool use_custom_mask_p/d` declared inside inner switch-case propagates a stale value, so `pod.cu` emits symbols that don't match the `kernel_inst` instantiations.
- **LNK1120** on `DefaultAttention<use_custom_mask,...>` template arg mismatch.
- Likely MSVC bug with nested switch+constexpr+lambda; no simple macro rewrite works.
- **Fix path**: refactor `DISPATCH_context` into a template helper (not trivial).

### `mamba` (selective_state_update)
Multiple MSVC issues surface after we got past the C++20 lambda template-params (added `-std=c++20` to module cflags):
- `uint` POSIX typedef (Linux-only)
- `__align__(N)` below default alignment
- Array of void from `std::conditional<..., void, char>[]`
- Narrowing `long long -> unsigned long long` in list-init

Too many per-file edits to take on in this pass.

### `xqa_mla`
- Fixed before hitting the hard wall: `tensorMap.h` missing `<cstdint>`, `mla_sm120.cu` missing `<string>`.
- **Remaining**: `utils.h(115)/(122)` C2719 "128-aligned parameter will not be aligned" on `Vec<T,size_>::fill(T val) / filled(T val)`. MSVC refuses to pass a heavily-aligned type by value; GCC/Clang accept it.
- **Fix path**: rework `Vec` value-parameter API into const-ref or tag-dispatched overloads — non-trivial on shared xqa code.

### `gdn_decode` / `gdn_prefill`
- Python modules import fine but calling `gated_delta_rule_decode` / `chunk_gated_delta_rule` resolves `run_nontranspose_decode` / cuTe kernel handle to None: the cute-DSL kernels (`flashinfer/gdn_kernels/*.py`) import `cutlass.cute` which is shipped by the `nvidia-cutlass-dsl` PyPI wheel.
- That wheel **has no Windows build**; `requirements.txt` already gates it out (linux-only).
- **Fix path**: none actionable on our side; wait for upstream to publish a Windows wheel.

### `mm_mxfp8` / `bmm_mxfp8`
- `cudaErrorMisalignedAddress` in the CUTLASS mxfp8 groupwise kernel on every `(m, n, k)` tried.
- Unlike `fp8` (where routing through the grouped kernel with `num_groups=1` worked), there is no equivalent fallback.
- **Fix path**: upstream alignment fix in `mxfp8_gemm_cutlass_sm120.cu`.

### `mm_fp4`
- `cudaErrorMisalignedAddress` in CUTLASS fp4 kernel.
- Same category as `mm_mxfp8` / `nvfp4_quantize` — SM120 CUTLASS block-scaled kernels share an alignment issue.

### `cudnn_batch_decode`
- `cudnn-frontend 1.22.1` Python wheel raises `Unable to load any libcudart.so.*` on Windows — the wheel's Windows ABI glue is incomplete.
- **Not a flashinfer bug**; our cudnn path is now lazy (import on first use + one-shot warn), so consumer users are unaffected as long as they don't explicitly request the cudnn backend.
- **Fix path**: upstream `nvidia-cudnn-frontend` Windows wheel fix.

---

## 🟡 PARTIAL

### `BatchAttention` (holistic / persistent paged attention)
- JIT + plan + run all pass on 5060.
- The cooperative-launch "persistent paged attention" kernel overflows the **30-SM budget** on non-trivial configs → "too many blocks in cooperative launch".
- Smoke works at small configs; bench across full shape matrix is skipped.

### `nvfp4_quantize`
- Smoke at `(128, 512)` passes.
- `cudaErrorMisalignedAddress` on shapes > ~1024 cols on SM120 / CUDA 12.9.
- Root cause same family as `mm_mxfp8` / `mm_fp4`.

### `attention_sink`
- `BatchPrefillWithPagedKVCacheWrapper.run(..., sinks=tensor)` runs to completion on default fa2 path.
- But the `sinks` argument is **silently ignored** (`max|out_no_sinks - out_with_sinks| = 0.0000`).
- API surface exists on Windows; effective use requires FA3 / dedicated sink variant, which isn't on the port target list.

### `group_gemm_fp8`
- Limited to `num_groups=1` by an existing flashinfer backend guard (`has correctness issues for num_groups > 1 on SM120/121`).
- Not Windows-specific but matters here because our `gemm_fp8_nt_groupwise` SM120 fallback routes single-matrix through this grouped kernel.

### `gemm_fp8_nt_blockscaled` (MN mode)
- K-mode works (`max|err|=0` at tested shapes).
- MN-mode previously yielded `rel>1` error via our `group_gemm` routing.
- **Now auto-fixed** (see RESOLVED section) — auto-transposes scales to K-mode on SM12x.

---

## 🟢 RESOLVED (for history / traceability)

| Item | Resolved by |
|---|---|
| `gemm_fp8_nt_groupwise(cutlass)` on SM120 | Route single-matrix case through grouped CUTLASS kernel with `num_groups=1` (commit `fc7722bb`). Non-grouped SM120 kernel still has upstream alignment bug but is no longer reachable. |
| `mm_fp8` on SM120 | Route through CUTLASS groupwise fallback (commit `9259ca97`): `prepare_low_latency_gemm_weights` becomes a no-op on SM12x returning raw (n,k) fp8; `mm_fp8` constructs degenerate per-block scales and calls `gemm_fp8_nt_groupwise(backend='cutlass')`. Trade-off: generic groupwise, not TRTLLM's sm100f low-latency micro-kernel — perf is bf16-baseline-ish on small shapes, reaches parity/beats only on `m*n*k >= ~512*4096*4096`. |
| `blockscaled` MN-mode on SM120 | `gemm_fp8_nt_blockscaled` now auto-transposes `a_scale` / `b_scale` to K-mode on SM12x (commit `d325d1c9`). MN and K modes both return `max|err|<0.005` at `m,n,k up to 1024`. |
| sglang scheduler subprocess `check_cuda_arch` crash | `_normalize_cuda_arch(12,0)` on CUDA 12.8 now falls back to `compute_120a` (commit `bb24abc3`). |
| `import flashinfer` loading `cudnn64_9.dll` on Windows | Lazy cudnn import + one-shot `RuntimeWarning` on failure (commit `8437a17e`). |
| `FI_CUBIN_EXPORT` / `FI_RESTRICT` scattered across headers | Consolidated into `include/flashinfer/_compat.h` (commit `49e7e25d`). |
| tvm-ffi Windows branch missing `-gencode` and `cudart.lib` | Local patch script at `scripts/windows/patches/apply_tvm_ffi_patches.py` (commit `6ca26a63`). |
| sglang MoE models not loadable on Windows (Qwen1.5/2/3-MoE, Mixtral, OLMoE, etc.) | Four-patch bundle at `scripts/windows/patches/sglang/apply_sglang_patches.py`: (1) lazy fallback for `kt_ep_wrapper.py`'s hard `sgl_kernel.gptq_marlin_repack` import, (2) skip GPTQ per-expert bias tensors in `qwen2_moe.py` load_weights, (3)+(4) rewrite `_GET_IF` macro in `gptq_marlin.cuh` / `moe_wna16_marlin.cuh` from `else if` to `if` to avoid MSVC C1061. Validated end-to-end on fi-win 2026-04-23: Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 generates coherent tokens. |

---

## ⚪ BY DESIGN (not a blocker — torch fallback is expected)

### `mm_bf16` / `bmm_bf16` on SM120
The `@backend_requirement` decorator explicitly rejects cutlass/tgv/cudnn backends at `cc=120` — users are expected to fall back to `torch.matmul` / `torch.bmm`.

---

## How to file a new blocker

Add a new entry under the appropriate section with:
- **What fails** (1-line summary)
- **Exact error signature** (error code, file:line, stack fragment)
- **Workaround** / `**Fix path**`
- **Date + host** the finding was made on

Mirror any cross-link to a CSV row in `benchmarks/results/windows_port_speedup.csv` so bench results stay attributable.
