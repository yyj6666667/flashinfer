# ktransformers → sglang end-to-end on Windows

Verified 2026-04-23 on fi-win (RTX 5060 / SM120 / CUDA 12.9 / MSVC 14.44 /
Intel Core Ultra 9 285 24C / 64 GB RAM / torch 2.11.0+cu128). Delivers
**3.9× steady-state decode throughput** on Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
vs sglang's naive `--cpu-offload-gb` on the same hardware:

| Backend | tok/s (steady state, 3-run median) |
|---|---:|
| sglang `--cpu-offload-gb 2` | 0.10 |
| **sglang + kt-kernel AVX2 (kt_ep)** | **0.39** |

## One-shot bring-up (assumes flashinfer + sglang already built on fi-win)

1. **Clone ktransformers windows branch** on fi-win:
   ```powershell
   git clone --branch windows https://github.com/yyj6666667/ktransformers.git C:\ktransformers
   cd C:\ktransformers
   git submodule update --init --recursive --depth 1 third_party/llama.cpp third_party/pybind11
   ```

2. **Fetch hwloc Windows prebuilt** (on a machine with fast github access),
   scp to fi-win, extract:
   ```bash
   # on local machine
   curl -L -o /tmp/hwloc-win64.zip "https://download.open-mpi.org/release/hwloc/v2.11/hwloc-win64-build-2.11.2.zip"
   scp /tmp/hwloc-win64.zip fi-win:C:/hwloc-win64.zip
   # then on fi-win
   powershell -File scripts\windows\patches\ktransformers\install_hwloc_prebuilt.ps1
   ```
   Skips the vcpkg hwloc-from-source build chain (needs MSYS2 autotools,
   which hits slow GitHub downloads from Chinese networks).

3. **Build kt-kernel** on fi-win:
   ```powershell
   powershell -File scripts\windows\patches\ktransformers\build_kt_kernel.ps1
   ```
   The script sets:
   - `CPUINFER_CPU_INSTRUCT=AVX2`
   - `CPUINFER_USE_CUDA=1`
   - `CMAKE_ARGS=-DHWLOC_INCLUDE_DIR=C:/hwloc/include -DHWLOC_LIBRARY=C:/hwloc/lib/libhwloc.lib -DCMAKE_CUDA_ARCHITECTURES=120-real`
   - Falls through to `pip install C:\ktransformers\kt-kernel --no-build-isolation`.

4. **Copy libhwloc-15.dll next to the .pyd** so `import kt_kernel` doesn't
   fail with "DLL load failed". The .pyd lands in a `Release\`
   subdir (MSBuild multi-config output):
   ```powershell
   powershell -File scripts\windows\patches\ktransformers\copy_hwloc_dll.ps1
   ```

5. **Launch sglang with `--kt-*` flags** for end-to-end MoE:
   ```powershell
   powershell -File scripts\windows\patches\ktransformers\launch_sglang_kt.ps1
   ```
   Key args (inside the script):
   ```
   --kt-cpuinfer 20
   --kt-method GPTQ_INT4
   --kt-num-gpu-experts 0
   --kt-weight-path C:/models/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4
   --kt-threadpool-count 1
   --kt-numa-nodes 0
   --dtype float16           # Qwen1.5-MoE-Int4 scales are fp16
   --disable-cuda-graph
   --mem-fraction-static 0.7 # experts are on CPU so much of VRAM is free
   ```

## Sharp corners / why each step exists

### vcpkg hwloc build was abandoned
vcpkg wants to build hwloc from source, which pulls in PowerShell 7.5.4
(110 MB), 7zip 26.00, ninja, and MSYS2's autoconf/automake/libtool/perl
tarballs from `mirror.msys2.org` — all of which get slow/refused from the
fi-win network. We pre-fetched PS7 + 7zip + ninja successfully but the
MSYS2 chain is the blocker. Switching to open-mpi's prebuilt
`hwloc-win64-build-2.11.2.zip` (~7 MB, served by `download.open-mpi.org`
which is fast from a non-Chinese box) avoids the whole MSYS2 dance.

### CUDA architecture override
kt-kernel's `CMakeLists.txt` defaults `CMAKE_CUDA_ARCHITECTURES` to
`80;86;89;90` (Ampere→Hopper). SM120 Blackwell requires `120-real`
(the `-real` suffix forces sm_120 SASS without PTX so the cubin is
guaranteed to execute; no forward-compat PTX JIT).

### libhwloc-15.dll placement
MSVC's `pybind11_add_module` on a multi-config generator emits the .pyd
into `{Release,Debug,...}/` relative to the kt_kernel package dir. The
`_cpu_detect.py::load_extension` helper already `os.add_dll_directory()`s
the .pyd's directory, but not the parent, so the hwloc runtime DLL must
live next to the .pyd (i.e. inside `Release/`) — not beside `__init__.py`.

### Known quality issue (not kt-specific)
The Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4 checkpoint at the current pip index
generates a coherent first clause then degrades to gibberish past ~10
tokens (observed in both the sglang `--cpu-offload` path and the kt path).
This is consistent with an fp16 precision or chat-template issue in the
checkpoint itself, not a kt-kernel bug. A cleaner bf16 or AWQ Qwen MoE
may produce better text quality at equivalent throughput.
