"""Apply Windows-specific runtime patches to a local sglang checkout.

Usage::

    python scripts/windows/patches/sglang/apply_sglang_patches.py --sglang C:\\sglang

Re-running on an already-patched checkout is a no-op (idempotent).

Each patch uses anchor-based text replacement rather than a unified diff
so minor upstream churn around the anchor doesn't break the patch.
The anchors are kept narrow and the replacement includes a marker
comment so this script can detect an already-patched file.

If the anchor can't be found, the script aborts with a clear message —
upstream moved and this script needs updating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

# --- patch 1: kt_ep_wrapper.py — fallback to JIT gptq_marlin_repack --------

P1_REL = "python/sglang/srt/layers/moe/kt_ep_wrapper.py"
P1_MARKER = "Fall back to the pure-JIT"
P1_BEFORE = (
    "if is_cuda():\n"
    "    from sgl_kernel import gptq_marlin_repack\n"
)
P1_AFTER = (
    "if is_cuda():\n"
    "    # On Linux, sgl_kernel ships a compiled gptq_marlin_repack; on Windows\n"
    "    # the sgl_kernel wheel skipped building it (FA3 / marlin_repack were\n"
    "    # deferred due to MSVC template issues). Fall back to the pure-JIT\n"
    "    # implementation under sglang.jit_kernel, which goes through tvm-ffi\n"
    "    # (fully working on Windows after the cuda-link patch in\n"
    "    # flashinfer/scripts/windows/patches/apply_tvm_ffi_patches.py).\n"
    "    try:\n"
    "        from sgl_kernel import gptq_marlin_repack\n"
    "    except ImportError:\n"
    "        from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack\n"
)

# --- patch 2: qwen2_moe.py — skip GPTQ bias tensors in expert mapping ------

P2_REL = "python/sglang/srt/models/qwen2_moe.py"
P2_MARKER = "Mirror the bias-skip that the stacked-weight branch"
P2_BEFORE = (
    "                for mapping in expert_params_mapping:\n"
    "                    param_name, weight_name, expert_id, shard_id = mapping\n"
    "                    if weight_name not in name:\n"
    "                        continue\n"
    "                    name = name.replace(weight_name, param_name)\n"
    "                    param = params_dict[name]\n"
    "                    weight_loader = param.weight_loader\n"
)
P2_AFTER = (
    "                for mapping in expert_params_mapping:\n"
    "                    param_name, weight_name, expert_id, shard_id = mapping\n"
    "                    if weight_name not in name:\n"
    "                        continue\n"
    "                    name = name.replace(weight_name, param_name)\n"
    "                    # GPTQ checkpoints emit per-expert bias tensors\n"
    "                    # (down_proj.bias / gate_proj.bias / up_proj.bias) that\n"
    "                    # FusedMoE's fused layout doesn't have a slot for.\n"
    "                    # Mirror the bias-skip that the stacked-weight branch\n"
    "                    # already applies at line ~834 above.\n"
    "                    if name.endswith(\".bias\") and name not in params_dict:\n"
    "                        break\n"
    "                    if name not in params_dict:\n"
    "                        break\n"
    "                    param = params_dict[name]\n"
    "                    weight_loader = param.weight_loader\n"
)

# --- patch 3 + 4: Marlin .cuh — break else-if chain to avoid MSVC C1061 ----

P3_REL = "python/sglang/jit_kernel/csrc/gemm/marlin/gptq_marlin.cuh"
P4_REL = "python/sglang/jit_kernel/csrc/gemm/marlin_moe/moe_wna16_marlin.cuh"
MARLIN_MARKER = "else-if chain that exceeded MSVC's C1061"
MARLIN_BEFORE = (
    "#define _GET_IF(                                                                                                       \\\n"
    "    W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \\\n"
    "  else if (                                                                                                            \\\n"
)
MARLIN_AFTER = (
    "// NOTE: was `else if (...) { ... }` — broken into an independent `if`\n"
    "// statement. The _GET_IF macros expand to ~200 chained clauses; the\n"
    "// else-if chain that exceeded MSVC's C1061 \"nesting too deep\" parse-\n"
    "// tree limit. Conditions are mutually exclusive on q_type + thread_*_\n"
    "// blocks + group_blocks + num_threads + is_zp_float so at most one\n"
    "// fires — semantically equivalent, parse-tree depth 1 per statement.\n"
    "#define _GET_IF(                                                                                                       \\\n"
    "    W_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, M_BLOCK_SIZE_8, GROUP_BLOCKS, NUM_THREADS, IS_ZP_FLOAT) \\\n"
    "  if (                                                                                                                 \\\n"
)

PATCHES: list[Tuple[str, str, str, str, str]] = [
    ("kt_ep_wrapper", P1_REL, P1_MARKER, P1_BEFORE, P1_AFTER),
    ("qwen2_moe_bias_skip", P2_REL, P2_MARKER, P2_BEFORE, P2_AFTER),
    ("gptq_marlin_c1061", P3_REL, MARLIN_MARKER, MARLIN_BEFORE, MARLIN_AFTER),
    ("moe_wna16_marlin_c1061", P4_REL, MARLIN_MARKER, MARLIN_BEFORE, MARLIN_AFTER),
]


def apply_one(name: str, path: Path, marker: str, before: str, after: str) -> bool:
    if not path.is_file():
        print(f"[skip] {name}: target file not found at {path}")
        return False
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"[skip] {name}: already patched")
        return False
    if before not in text:
        print(
            f"[error] {name}: anchor not found in {path.name}; "
            "sglang may have been updated upstream — please refresh the anchor.",
            file=sys.stderr,
        )
        return False
    if text.count(before) > 1:
        print(
            f"[error] {name}: anchor matches {text.count(before)} locations in "
            f"{path.name}; refusing to guess.",
            file=sys.stderr,
        )
        return False
    path.write_text(text.replace(before, after, 1), encoding="utf-8")
    print(f"[apply] {name}")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sglang", default="C:/sglang",
                   help="Path to sglang checkout root.")
    args = p.parse_args()

    root = Path(args.sglang)
    if not root.is_dir():
        print(f"--sglang {root} does not exist", file=sys.stderr)
        return 1
    print(f"sglang root: {root}")

    any_changed = False
    for name, rel, marker, before, after in PATCHES:
        if apply_one(name, root / rel, marker, before, after):
            any_changed = True
    if any_changed:
        print("[done] at least one patch applied")
    else:
        print("[done] nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
