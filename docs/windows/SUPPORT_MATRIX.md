# Windows Support Matrix

This is the officially-tested combination for the `windows` branch of
FlashInfer. Deviations **may** work (the port hasn't been artificially
locked), but everything below is what's been validated end-to-end with
sglang + Qwen generation as of 2026-04-23.

## Tested combination

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX 5060 (SM 12.0, consumer Blackwell) |
| CUDA Runtime (torch wheel) | 12.8 (`torch==2.11.0+cu128`) |
| CUDA Toolkit (nvcc on PATH) | 12.9 |
| MSVC (cl.exe) | 14.44.35207 (VS 2022 Community 17.14) |
| Windows SDK | shipped with VS 2022 BuildTools |
| Python | 3.12.8 (Python.org installer, not Store alias) |
| Host OS | Windows 10/11 x64 |
| Windows Update | **Disabled** (see Operational section) |

## Supported SM targets

| SM | nvcc gencode | Minimum CUDA | Rationale |
|----|--------------|--------------|-----------|
| `sm_120a` | `compute_120a,code=sm_120a` | 12.8 | Consumer Blackwell (RTX 50-series); **fallback target on cu128 torch wheels** |
| `sm_120f` | `compute_120f,code=sm_120f` | 12.9 | Consumer Blackwell with extra PTX feature set |
| `sm_121a` | `compute_121a,code=sm_121a` | 12.9 | DGX Spark — not code-compatible with `sm_120`, ships its own cubin |

The fallback from `sm_120f` → `sm_120a` on CUDA 12.8 is what lets spawn
subprocesses (sglang's scheduler) still find a valid target when `nvcc`
isn't on PATH and `torch.version.cuda="12.8"`. See
`flashinfer/compilation_context.py::_normalize_cuda_arch`.

## Deliberately out-of-scope on Windows today

| Thing | Why | Alternative |
|-------|-----|-------------|
| MLA (DeepSeek multi-head latent attention) | Upstream kernel only targets SM10/11 | Use a non-MLA model (Qwen/Llama/etc.) |
| `cutlass-dsl` Python wheel | No Windows build from upstream | Kernels that need cute-DSL (`gdn_*`) are gated off in `requirements.txt` |
| `cudnn-frontend` attention backend | Windows wheel's ABI glue is broken (`Unable to load libcudart.so.*`) | Attention falls back to FA2 / CUTLASS automatically |
| `fused_moe` (TRT-LLM variant) | MSVC cascade + nvcc ICE (see `KNOWN_ISSUES.md`) | Use `cutlass_fused_moe` — it builds and runs on SM120 |
| `mamba` / `pod` / `xqa_mla` | MSVC non-trivial issues (see `KNOWN_ISSUES.md`) | Non-applicable for Qwen-series workloads |
| `mm_mxfp8` / `bmm_mxfp8` / `mm_fp4` | SM120 CUTLASS alignment bug upstream | Use `mm_fp8` / bf16 path |
| `gptq_marlin_repack` kernel | sgl-kernel Windows build skipped it | Can't load GPTQ-quantized models; use AWQ / bf16 / fp8 |
| FA3 attention | Similar — build skipped | FA2 is default; effectively covers the same models |

## Operational notes

### CUDA_PATH + PATH requirement
Any process (including spawn subprocesses of sglang) that may trigger
JIT compilation needs:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
```

and the MSVC dev environment applied (via `vcvarsall.bat x64` captured
into env — see `scripts/windows/` for the reference launcher).

### Windows Update
Auto-update has been known to reboot the machine mid-session and drop
SSH / Tailscale for hours. Disabled on fi-win via:

- `Set-Service wuauserv -StartupType Disabled`
- `Set-Service UsoSvc -StartupType Disabled`
- `HKLM\SYSTEM\CurrentControlSet\Services\WaaSMedicSvc\Start = 4` (registry, since `Set-Service` denied)
- `HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU\NoAutoUpdate = 1`

### tvm-ffi runtime patch
The installed `apache-tvm-ffi` wheel's `cpp/extension.py` has two
Windows-only gaps (missing `-gencode` and missing `cudart.lib`).
Apply the local patch script after **every** `pip install apache-tvm-ffi`:

```powershell
python scripts\windows\patches\apply_tvm_ffi_patches.py
```

The script is idempotent; safe to re-run on an already-patched file.

## When this matrix is wrong

Open a new entry in `docs/windows/KNOWN_ISSUES.md` or update this file
directly. Prefer updating over adding "addendum" files.
