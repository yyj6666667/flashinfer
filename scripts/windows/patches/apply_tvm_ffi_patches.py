"""Apply Windows-specific fixes to the pip-installed ``tvm_ffi`` package.

Run this once after every ``pip install`` (or ``pip upgrade``) that touches
``apache-tvm-ffi``::

    python scripts/windows/patches/apply_tvm_ffi_patches.py

The script is idempotent — re-running it on an already-patched ``extension.py``
leaves the file untouched and exits 0.

Background
----------

``tvm_ffi/cpp/extension.py`` has a platform switch for Windows vs Linux. The
Linux branch appends two things to the default compile/link flags that the
Windows branch forgets:

1. ``default_cuda_cflags += [_get_cuda_target()]`` — emits the
   ``-gencode=arch=...`` flag derived from ``TVM_FFI_CUDA_ARCH_LIST`` (or
   ``nvidia-smi``). Without it nvcc silently targets ``sm_52`` on Windows
   and any kernel that uses ``__grid_constant__`` (≥CC 7.0) fails at compile
   time.

2. ``default_ldflags += ["-L<cuda>/lib64", "-lcudart"]`` — links the CUDA
   runtime. Without it every ``cudaLaunchKernel`` / ``cudaGetErrorString``
   reference fails link (LNK2019). The Windows equivalent is
   ``/LIBPATH:"<cuda>/lib/x64"`` + ``cudart.lib``. The path needs quotes
   because the canonical CUDA Toolkit install directory contains spaces
   (``C:\\Program Files\\NVIDIA GPU Computing Toolkit\\...``).

Both gaps were verified on tvm-ffi 0.1.x shipping in torch 2.11.0+cu128
editable installs used by ``flashinfer`` on Windows as of 2026-04-23.

When upstream tvm-ffi ships a fix, this script can be retired.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

GENCODE_MARKER = "WINDOWS_GENCODE_FIX"
CUDART_MARKER = "WINDOWS_CUDART_LINK_FIX"

GENCODE_BEFORE = (
    '        default_cuda_cflags = ["-Xcompiler=/std:c++20", "-Xcompiler=/O2"]  # '
    "SGL_KERNEL_WIN_PATCH: each host flag needs its own -Xcompiler token on Windows\n"
    "        default_ldflags = [\n"
)

GENCODE_AFTER = (
    '        default_cuda_cflags = ["-Xcompiler=/std:c++20", "-Xcompiler=/O2"]  # '
    "SGL_KERNEL_WIN_PATCH: each host flag needs its own -Xcompiler token on Windows\n"
    "        # WINDOWS_GENCODE_FIX: the Linux branch below appends _get_cuda_target()\n"
    "        # onto default_cuda_cflags, but the Windows branch never did — so nvcc\n"
    "        # defaulted to sm_52 and any kernel using __grid_constant__ (CC>=7.0)\n"
    "        # failed at compile time. Mirror the Linux behavior here.\n"
    "        if with_cuda:\n"
    "            default_cuda_cflags += [_get_cuda_target()]\n"
    "        default_ldflags = [\n"
)

CUDART_BEFORE = (
    "        default_ldflags = [\n"
    '            "/DLL",\n'
    '            f"/LIBPATH:{tvm_ffi_lib_path}",\n'
    '            f"{tvm_ffi_lib_name}.lib",\n'
    "        ]\n"
    "    else:\n"
)

CUDART_AFTER = (
    "        default_ldflags = [\n"
    '            "/DLL",\n'
    '            f"/LIBPATH:{tvm_ffi_lib_path}",\n'
    '            f"{tvm_ffi_lib_name}.lib",\n'
    "        ]\n"
    "        # WINDOWS_CUDART_LINK_FIX: the Linux branch links against cudart via\n"
    '        # "-L<cuda>/lib64 -lcudart"; the Windows branch forgot, so every\n'
    "        # CUDA API call (cudaLaunchKernel, cudaGetErrorString, etc.) failed\n"
    "        # at link with LNK2019. Add cudart.lib with the Windows lib path —\n"
    "        # quoted because the canonical CUDA Toolkit install path has spaces\n"
    '        # ("C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\...").\n'
    "        if with_cuda:\n"
    "            _cuda_lib_x64 = str(Path(_find_cuda_home()) / \"lib\" / \"x64\")\n"
    "            default_ldflags += [\n"
    "                f'/LIBPATH:\"{_cuda_lib_x64}\"',\n"
    '                "cudart.lib",\n'
    "            ]\n"
    "    else:\n"
)


def locate_extension_py() -> Path:
    spec = importlib.util.find_spec("tvm_ffi")
    if spec is None or spec.origin is None:
        raise SystemExit(
            "tvm_ffi not importable; install via `pip install apache-tvm-ffi` first."
        )
    ext = Path(spec.origin).parent / "cpp" / "extension.py"
    if not ext.is_file():
        raise SystemExit(f"Unexpected layout: {ext} does not exist.")
    return ext


def apply_one(text: str, marker: str, before: str, after: str, label: str) -> tuple[str, bool]:
    if marker in text:
        print(f"[skip] {label}: already patched")
        return text, False
    if before not in text:
        raise SystemExit(
            f"[error] {label}: anchor not found. "
            f"tvm-ffi layout changed upstream; please update this script."
        )
    if text.count(before) > 1:
        raise SystemExit(
            f"[error] {label}: anchor matches {text.count(before)} locations; "
            "refusing to guess. Please refine the anchor."
        )
    print(f"[apply] {label}")
    return text.replace(before, after, 1), True


def main() -> int:
    if sys.platform != "win32":
        print("This patch is Windows-only; nothing to do.")
        return 0
    ext = locate_extension_py()
    print(f"Target : {ext}")
    original = ext.read_text(encoding="utf-8")
    patched = original
    patched, changed1 = apply_one(
        patched, GENCODE_MARKER, GENCODE_BEFORE, GENCODE_AFTER, "gencode"
    )
    patched, changed2 = apply_one(
        patched, CUDART_MARKER, CUDART_BEFORE, CUDART_AFTER, "cudart"
    )
    if changed1 or changed2:
        ext.write_text(patched, encoding="utf-8")
        print(f"[done]  wrote {ext}")
    else:
        print("[done]  nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
