# Windows runtime patches applied to external packages

These patches must be applied to pip-installed packages after each (re)install
in the `.venv`. They are symptom fixes for known upstream gaps that show up
only on Windows. File upstream issues and link them here as they're reported.

## tvm_ffi cpp/extension.py — Windows cuda link gap

**Symptom**: `sgl_kernel` JIT builds (used by sglang) fail at nvcc with
`__grid_constant__ only for compute_70+`, and at link with
`LNK2019: unresolved cudaLaunchKernel / cudaGetErrorString / cudaRegister...`.

**Root cause**: in `tvm_ffi/cpp/extension.py` the Windows branch of the
platform switch is missing two lines that the Linux branch has:

1. `default_cuda_cflags += [_get_cuda_target()]` — adds `-gencode=arch=...`
   based on `TVM_FFI_CUDA_ARCH_LIST` or `nvidia-smi`. Without it nvcc
   targets sm_52 by default.
2. `default_ldflags += ["-L<cuda_home>/lib64", "-lcudart"]` —
   on Windows the equivalent is `/LIBPATH:"<cuda_home>/lib/x64"` +
   `cudart.lib`. Must be quoted because the canonical CUDA Toolkit install
   path contains spaces.

**Apply**: run `python apply_tvm_ffi_patches.py` after `pip install tvm-ffi`.

## Persistence

When tvm-ffi ships a proper fix (or we send a PR), this patch will be retired.
Track via the tvm-ffi issue URL noted at the top of each patch file.
