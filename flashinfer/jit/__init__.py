"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ctypes
import functools
import os

# Re-export
from . import cubin_loader
from . import env as env
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from .attention import gen_cudnn_fmha_module as gen_cudnn_fmha_module
from .attention import gen_batch_attention_module as gen_batch_attention_module
from .attention import gen_batch_decode_mla_module as gen_batch_decode_mla_module
from .attention import gen_batch_decode_module as gen_batch_decode_module
from .attention import gen_batch_mla_module as gen_batch_mla_module
from .attention import gen_batch_prefill_module as gen_batch_prefill_module
from .attention import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .attention import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .attention import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .attention import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .attention import gen_fmha_cutlass_sm100a_module as gen_fmha_cutlass_sm100a_module
from .attention import gen_batch_pod_module as gen_batch_pod_module
from .attention import gen_pod_module as gen_pod_module
from .attention import gen_single_decode_module as gen_single_decode_module
from .attention import gen_single_prefill_module as gen_single_prefill_module
from .attention import get_batch_attention_uri as get_batch_attention_uri
from .attention import get_batch_decode_mla_uri as get_batch_decode_mla_uri
from .attention import get_batch_decode_uri as get_batch_decode_uri
from .attention import get_batch_mla_uri as get_batch_mla_uri
from .attention import get_batch_prefill_uri as get_batch_prefill_uri
from .attention import get_pod_uri as get_pod_uri
from .attention import get_single_decode_uri as get_single_decode_uri
from .attention import get_single_prefill_uri as get_single_prefill_uri
from .attention import gen_trtllm_gen_fmha_module as gen_trtllm_gen_fmha_module
from .attention import gen_fmha_v2_module as gen_fmha_v2_module
from .attention import (
    gen_trtllm_fmha_v2_sm120_module as gen_trtllm_fmha_v2_sm120_module,
)
from .core import JitSpec as JitSpec
from .core import JitSpecStatus as JitSpecStatus
from .core import JitSpecRegistry as JitSpecRegistry
from .core import jit_spec_registry as jit_spec_registry
from .core import build_jit_specs as build_jit_specs
from .core import clear_cache_dir as clear_cache_dir
from .core import gen_jit_spec as gen_jit_spec
from .core import MissingJITCacheError as MissingJITCacheError
from .core import sm90a_nvcc_flags as sm90a_nvcc_flags
from .core import sm100a_nvcc_flags as sm100a_nvcc_flags
from .core import sm100f_nvcc_flags as sm100f_nvcc_flags
from .core import sm103a_nvcc_flags as sm103a_nvcc_flags
from .core import sm110a_nvcc_flags as sm110a_nvcc_flags
from .core import sm120a_nvcc_flags as sm120a_nvcc_flags
from .core import sm120f_nvcc_flags as sm120f_nvcc_flags
from .core import sm121a_nvcc_flags as sm121a_nvcc_flags
from .core import current_compilation_context as current_compilation_context
from .cubin_loader import setup_cubin_loader
from .comm import gen_comm_alltoall_module as gen_comm_alltoall_module
from .comm import gen_trtllm_mnnvl_comm_module as gen_trtllm_mnnvl_comm_module
from .comm import gen_trtllm_comm_module as gen_trtllm_comm_module
from .comm import gen_vllm_comm_module as gen_vllm_comm_module
from .comm import gen_moe_alltoall_module as gen_moe_alltoall_module
from .dsv3_optimizations import (
    gen_dsv3_router_gemm_module as gen_dsv3_router_gemm_module,
)
from .dsv3_optimizations import (
    gen_dsv3_fused_routing_module as gen_dsv3_fused_routing_module,
)
from .tinygemm2 import gen_tinygemm2_module as gen_tinygemm2_module
from .moe_utils import gen_moe_utils_module as gen_moe_utils_module
from .fp4_kv_dequantization import (
    gen_fp4_kv_dequantization_module as gen_fp4_kv_dequantization_module,
)
from .fp4_kv_quantization import (
    gen_fp4_kv_quantization_module as gen_fp4_kv_quantization_module,
)


def _preload_cudart() -> None:
    """Preload libcudart so JIT-built modules can resolve CUDA runtime symbols.

    On Linux we promote libcudart to ``RTLD_GLOBAL`` so that downstream JIT
    modules link against the same instance. Windows has no notion of
    ``RTLD_GLOBAL`` in ctypes — DLL search is governed by the loader's
    directory list — so we simply ensure ``cudart64_*.dll`` is reachable via
    ``CUDA_PATH``/``CUDA_LIB_PATH`` (or already on PATH) and load it.
    """
    import sys
    import glob

    if sys.platform == "win32":
        # Allow the user to point at an explicit CUDA bin directory; otherwise
        # rely on CUDA_PATH (set by the CUDA installer) and finally on PATH.
        candidate_dirs = []
        env_dir = os.environ.get("CUDA_LIB_PATH")
        if env_dir:
            candidate_dirs.append(env_dir)
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            candidate_dirs.append(os.path.join(cuda_path, "bin"))

        # Python 3.8+ on Windows ignores PATH for DLL resolution; explicitly
        # register any candidate directories so cudart is discoverable when
        # later loaded by tvm_ffi-loaded modules.
        for d in candidate_dirs:
            if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(d)
                except (OSError, FileNotFoundError):
                    pass

        # cudart on Windows is named cudart64_<major>.dll (e.g. cudart64_12.dll).
        for d in candidate_dirs + [""]:
            pattern = os.path.join(d, "cudart64_*.dll") if d else "cudart64_*.dll"
            matches = sorted(glob.glob(pattern))
            if matches:
                try:
                    ctypes.CDLL(matches[-1])
                except OSError:
                    pass
                return
        # Fall through silently: tvm_ffi will surface a clear error if cudart
        # cannot be located when the first JIT module is loaded.
        return

    cuda_lib_path = os.environ.get(
        "CUDA_LIB_PATH", "/usr/local/cuda/targets/x86_64-linux/lib/"
    )
    if os.path.exists(f"{cuda_lib_path}/libcudart.so.12"):
        ctypes.CDLL(f"{cuda_lib_path}/libcudart.so.12", mode=ctypes.RTLD_GLOBAL)


_preload_cudart()
