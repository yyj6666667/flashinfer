# Adapted from https://github.com/pytorch/pytorch/blob/v2.7.0/torch/utils/cpp_extension.py

import functools
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from packaging.version import Version
from pathlib import Path
from typing import List, Optional

import tvm_ffi
import torch

from . import env as jit_env
from ..compilation_context import CompilationContext

logger = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# Platform-dependent constants
if IS_WINDOWS:
    SHARED_LIB_EXT = ".dll"
    OBJ_EXT = ".obj"
    CUDA_OBJ_EXT = ".cuda.obj"
    DEFAULT_CXX = "cl.exe"
else:
    SHARED_LIB_EXT = ".so"
    OBJ_EXT = ".o"
    CUDA_OBJ_EXT = ".cuda.o"
    DEFAULT_CXX = "c++"


def parse_env_flags(env_var_name) -> List[str]:
    env_flags = os.environ.get(env_var_name)
    if env_flags:
        try:
            import shlex

            return shlex.split(env_flags)
        except ValueError as e:
            logger.warning(
                "Could not parse %s with shlex: %s. Falling back to simple split.",
                env_var_name,
                e,
            )
            return env_flags.split()
    return []


def _ninja_escape_path(p) -> str:
    # Ninja treats ':', ' ', and '$' as syntactic in build/default lines.
    # Windows absolute paths like ``C:\...`` will otherwise be parsed as a
    # target named ``C`` followed by rule ``\...``, blowing up on the first
    # ``build`` statement emitted for a JIT op.
    return str(p).replace("$", "$$").replace(":", "$:").replace(" ", "$ ")


def _get_glibcxx_abi_build_flags() -> List[str]:
    # _GLIBCXX_USE_CXX11_ABI is a libstdc++ macro; MSVC's STL is unaffected, so
    # we omit the define on Windows to avoid spurious warnings.
    if IS_WINDOWS:
        return []
    return [
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    ]


def _define_flag(macro: str) -> str:
    """Return a compiler-specific preprocessor define flag.

    On Windows we emit unix-style ``-D`` (not MSVC-style ``/D``) because
    nvcc 12.6 on Windows mis-parses commands that contain ``/D<name>=<value>``
    and raises a misleading::

        nvcc fatal : A single input file is required for a non-link phase
                     when an outputfile is specified

    Both cl.exe (via nvcc host forwarding) and nvcc itself accept ``-D``,
    so switching to unix-style is the safe universal form.
    """
    return f"-D{macro}"


def _include_flag(path) -> str:
    """Return a compiler-specific include directive.

    Emits ``-I<path>`` on both platforms for the same reason as
    ``_define_flag``: nvcc 12.6 on Windows mis-parses ``/I<path>`` in
    combination with the rest of our argv and aborts. cl.exe accepts ``-I``
    equivalently to ``/I``, so this is a safe universal form.
    """
    if IS_WINDOWS:
        return f"-I{path}"
    return f"-isystem {path}"


def _user_include_flag(path) -> str:
    return f"-I{path}"


@functools.cache
def get_cuda_path() -> str:
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home is not None:
        return cuda_home
    # Use shutil.which to locate nvcc cross-platform (handles .exe on Windows).
    nvcc_exe = "nvcc.exe" if IS_WINDOWS else "nvcc"
    nvcc_full_path = shutil.which(nvcc_exe)
    if nvcc_full_path is not None:
        # nvcc is at <cuda_home>/bin/nvcc[.exe], so go up two levels.
        cuda_home = os.path.dirname(os.path.dirname(nvcc_full_path))
    else:
        if IS_WINDOWS:
            # On Windows, the canonical installation directory carries the
            # CUDA version suffix (e.g. v12.4); without CUDA_PATH set we
            # cannot reliably guess it, so fail with an actionable message.
            raise RuntimeError(
                "Could not find nvcc.exe on PATH. "
                "Please set the CUDA_PATH environment variable to your CUDA "
                "Toolkit install directory (e.g. "
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4)."
            )
        cuda_home = "/usr/local/cuda"  # This default value is from: https://github.com/pytorch/pytorch/blob/ceb11a584d6b3fdc600358577d9bf2644f88def9/torch/utils/cpp_extension.py#L115
        if not os.path.exists(cuda_home):
            raise RuntimeError(
                f"Could not find nvcc and default {cuda_home=} doesn't exist"
            )
    return cuda_home


def _nvcc_bin_path(cuda_home: str) -> str:
    """Return the absolute path to the nvcc executable for ``cuda_home``."""
    nvcc_name = "nvcc.exe" if IS_WINDOWS else "nvcc"
    return os.path.join(cuda_home, "bin", nvcc_name)


def _cuda_lib_dirs(cuda_home: str) -> List[str]:
    """Return the platform-specific CUDA library search directories.

    On Windows the import libraries live under ``lib/x64``; on Linux they are
    under ``lib64`` (with the additional ``stubs`` directory used to satisfy
    the libcuda link dependency in container builds).
    """
    if IS_WINDOWS:
        return [os.path.join(cuda_home, "lib", "x64")]
    return [
        os.path.join(cuda_home, "lib64"),
        os.path.join(cuda_home, "lib64", "stubs"),
    ]


@functools.cache
def get_cuda_version() -> Version:
    # Try to query nvcc for CUDA version; if nvcc is unavailable, fall back to torch.version.cuda
    try:
        cuda_home = get_cuda_path()
        nvcc = _nvcc_bin_path(cuda_home)
        txt = subprocess.check_output([nvcc, "--version"], text=True)
        matches = re.findall(r"release (\d+\.\d+),", txt)
        if not matches:
            raise RuntimeError(
                f"Could not parse CUDA version from nvcc --version output: {txt}"
            )
        return Version(matches[0])
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as e:
        # NOTE(Zihao): when nvcc is unavailable, fall back to torch.version.cuda
        if torch.version.cuda is None:
            raise RuntimeError(
                "nvcc not found and PyTorch is not built with CUDA support. "
                "Could not determine CUDA version."
            ) from e
        return Version(torch.version.cuda)


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def get_system_includes(cuda_home: str) -> List:
    """Get list of system include directories."""
    system_includes = [
        sysconfig.get_path("include"),
        "$cuda_home/include",
        "$cuda_home/include/cccl",
        tvm_ffi.libinfo.find_include_path(),
        tvm_ffi.libinfo.find_dlpack_include_path(),
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]
    system_includes.append(jit_env.SPDLOG_INCLUDE_DIR.resolve())

    if cuda_home == "/usr":
        # NOTE: this will resolve to /usr/include, which will mess up includes. See #1793
        system_includes.remove("$cuda_home/include")

    return system_includes


def build_common_cflags(
    cuda_home: str,
    extra_include_dirs: Optional[List[Path]] = None,
) -> List[str]:
    """Build common compilation flags."""
    system_includes = get_system_includes(cuda_home)

    common_cflags = []
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        common_cflags.append(_define_flag("Py_LIMITED_API=0x03090000"))
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            common_cflags.append(_user_include_flag(extra_dir.resolve()))
    for sys_dir in system_includes:
        common_cflags.append(_include_flag(sys_dir))

    return common_cflags


def build_cflags(
    common_cflags: List[str],
    extra_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build C++ compilation flags."""
    cflags: List[str] = ["$common_cflags"]
    if IS_WINDOWS:
        # MSVC: enable C++ exceptions and link against the multi-threaded DLL
        # CRT (matching the CPython runtime), which is what extension modules
        # need. Position-independent code is implicit on x64.
        cflags += ["/EHsc", "/MD", "/bigobj", "/permissive-"]
    else:
        cflags.append("-fPIC")
    if extra_cflags is not None:
        cflags += extra_cflags

    env_extra_cflags = parse_env_flags("FLASHINFER_EXTRA_CFLAGS")
    if env_extra_cflags is not None:
        cflags += env_extra_cflags

    return cflags


def build_cuda_cflags(
    common_cflags: List[str],
    extra_cuda_cflags: Optional[List[str]] = None,
) -> List[str]:
    """Build CUDA compilation flags."""
    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    cuda_cflags += [
        "$common_cflags",
        "--expt-relaxed-constexpr",
    ]
    if IS_WINDOWS:
        # Forward host-compiler options for MSVC. /EHsc enables C++ exceptions,
        # /MD links the DLL CRT, /bigobj raises the per-object section limit
        # (some heavily-templated CUTLASS kernels exceed the default).
        cuda_cflags += [
            "-Xcompiler=/EHsc",
            "-Xcompiler=/MD",
            "-Xcompiler=/bigobj",
        ]
    else:
        cuda_cflags.append("--compiler-options=-fPIC")
    cuda_version = get_cuda_version()
    # enable -static-global-template-stub when cuda version >= 12.8
    if cuda_version >= Version("12.8"):
        cuda_cflags += [
            "-static-global-template-stub=false",
        ]

    cpp_ext_initial_compilation_context = CompilationContext()
    global_flags = cpp_ext_initial_compilation_context.get_nvcc_flags_list()
    if extra_cuda_cflags is not None:
        # Check if module provides architecture flags
        module_has_gencode = any(
            flag.startswith("-gencode=") for flag in extra_cuda_cflags
        )

        if module_has_gencode:
            # Use module's architecture flags, but keep global non-architecture flags
            global_non_arch_flags = [
                flag for flag in global_flags if not flag.startswith("-gencode=")
            ]
            cuda_cflags += global_non_arch_flags + extra_cuda_cflags
        else:
            # No module architecture flags, use both global and module flags
            cuda_cflags += global_flags + extra_cuda_cflags
    else:
        # No module flags, use global flags
        cuda_cflags += global_flags

    env_extra_cuda_cflags = parse_env_flags("FLASHINFER_EXTRA_CUDAFLAGS")
    if env_extra_cuda_cflags is not None:
        cuda_cflags += env_extra_cuda_cflags

    return cuda_cflags


def _build_link_flags(cuda_home: str) -> List[str]:
    """Construct the platform-specific linker flags for the JIT shared object.

    The Linux toolchain uses GNU/clang-style flags (``-shared``, ``-L``, ``-l``)
    while the Windows MSVC link.exe consumes ``/DLL`` and ``/LIBPATH:`` plus
    bare ``.lib`` filenames. Note that on Windows there is no ``stubs/``
    directory and we link against ``cuda.lib`` from the import library set.
    """
    if IS_WINDOWS:
        ldflags = ["/DLL", "/nologo"]
        for d in _cuda_lib_dirs("$cuda_home"):
            # Avoid backslash-escaped quote handling: $cuda_home is already
            # rendered into the ninja file as an absolute path.
            ldflags.append(f"/LIBPATH:{d}")
        ldflags += ["cudart.lib", "cuda.lib"]
        return ldflags

    return [
        "-shared",
        "-L$cuda_home/lib64",
        "-L$cuda_home/lib64/stubs",
        "-lcudart",
        "-lcuda",
    ]


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
) -> str:
    cuda_home = get_cuda_path()
    common_cflags = build_common_cflags(cuda_home, extra_include_dirs)
    cflags = build_cflags(common_cflags, extra_cflags)
    cuda_cflags = build_cuda_cflags(common_cflags, extra_cuda_cflags)

    ldflags = _build_link_flags(cuda_home)

    env_extra_ldflags = parse_env_flags("FLASHINFER_EXTRA_LDFLAGS")
    if env_extra_ldflags is not None:
        ldflags += env_extra_ldflags

    if extra_ldflags is not None:
        ldflags += extra_ldflags

    cxx = os.environ.get("CXX", DEFAULT_CXX)
    nvcc = os.environ.get(
        "FLASHINFER_NVCC",
        "$cuda_home/bin/nvcc.exe" if IS_WINDOWS else "$cuda_home/bin/nvcc",
    )

    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "cflags = " + join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "ldflags = " + join_multiline(ldflags),
        "",
    ]

    if IS_WINDOWS:
        # MSVC writes dependency info to stderr via /showIncludes; ninja's
        # ``deps = msvc`` mode parses that stream. No depfile is produced.
        #
        # NOTE: cuda_compile intentionally does NOT set deps=msvc or forward
        # /showIncludes via -Xcompiler. Empirically, on CUDA 12.6 Windows
        # nvcc, having -Xcompiler=/showIncludes in the argv mis-sets its
        # parser state so that any following /-prefixed token (/DPy_LIMITED_API,
        # /I<path>, ...) is mis-classified, and the build fails with the
        # misleading fatal:
        #   "A single input file is required for a non-link phase when an
        #    outputfile is specified"
        # JIT cache invalidation happens at the JitSpec hash level, so we do
        # not need ninja-level incremental dep tracking for .cu inputs.
        # The host .cpp rule still uses /showIncludes because cl.exe accepts
        # it directly and is not subject to the nvcc parser quirk.
        lines += [
            "msvc_deps_prefix = Note: including file:",
            "",
            "rule compile",
            "  command = $cxx /nologo /showIncludes $cflags /c $in /Fo$out $post_cflags",
            "  deps = msvc",
            "",
            "rule cuda_compile",
            "  command = $nvcc $cuda_cflags -c $in -o $out $cuda_post_cflags",
            "",
        ]
    else:
        lines += [
            "rule compile",
            "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
            "",
            "rule cuda_compile",
            "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
            "  depfile = $out.d",
            "  deps = gcc",
            "",
        ]

    # Add nvcc linking rule for device code
    if needs_device_linking:
        if IS_WINDOWS:
            # nvcc on Windows accepts -Xlinker to forward MSVC-style options.
            link_cmd = "$nvcc -shared $in $ldflags -o $out"
        else:
            link_cmd = "$nvcc -shared $in $ldflags -o $out"
        lines.extend(
            [
                "rule nvcc_link",
                f"  command = {link_cmd}",
                "",
            ]
        )
    else:
        if IS_WINDOWS:
            # Invoke link.exe directly so we do not depend on the host C++
            # compiler driver understanding /DLL semantics.
            link_cmd = "link.exe /nologo /DLL $in $ldflags /OUT:$out"
        else:
            link_cmd = "$cxx $in $ldflags -o $out"
        lines.extend(
            [
                "rule link",
                f"  command = {link_cmd}",
                "",
            ]
        )

    # Use absolute paths for outputs so ninja files work with any workdir
    # This enables isolated workdirs for runtime JIT (avoiding .ninja_log races)
    # while still supporting subninja for parallel AOT builds
    output_dir = jit_env.FLASHINFER_JIT_DIR / name

    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = CUDA_OBJ_EXT if is_cuda else OBJ_EXT
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = f"{source.parent.name}_{source.stem}{object_suffix}"
        obj = _ninja_escape_path((output_dir / obj_name).resolve())
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {_ninja_escape_path(source.resolve())}")

    lines.append("")
    link_rule = "nvcc_link" if needs_device_linking else "link"
    output_lib = _ninja_escape_path((output_dir / f"{name}{SHARED_LIB_EXT}").resolve())
    lines.append(f"build {output_lib}: {link_rule} " + " ".join(objects))
    lines.append(f"default {output_lib}")
    lines.append("")

    return "\n".join(lines)


def _get_num_workers() -> Optional[int]:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers()
    if num_workers is not None:
        command += ["-j", str(num_workers)]

    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
