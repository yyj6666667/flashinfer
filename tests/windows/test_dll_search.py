"""`import flashinfer` must not trigger cudnn DLL load on Windows.

This was regressed repeatedly before commit ``8437a17e`` (cudnn lazy
import): every ``import flashinfer`` on a consumer Windows box paid the
cost of ``os.add_dll_directory(torch/lib)`` + ``import cudnn`` inside
submodules, which either:

1. Succeeded but slowed import by a meaningful amount, or
2. Raised an uninformative OSError that was silently caught, hiding
   a real environment misconfiguration from the user.

This test catches regressions from anyone who reintroduces an eager
cudnn import via a new submodule.
"""
from __future__ import annotations

import importlib
import sys
import warnings


def test_import_flashinfer_does_not_import_cudnn():
    """Neither `cudnn` nor `flashinfer.cudnn._lazy._cudnn` may be resolved
    simply by `import flashinfer`.
    """
    # Reset state so we observe a clean import.
    for name in list(sys.modules):
        if name == "cudnn" or name.startswith("flashinfer"):
            del sys.modules[name]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import flashinfer  # noqa: F401

    cudnn_module_present = "cudnn" in sys.modules
    assert not cudnn_module_present, (
        "import flashinfer pulled in the 'cudnn' top-level module; this "
        "re-introduces the DLL-load-at-import-time problem. If you added a "
        "new cudnn-using submodule, gate its import via "
        "flashinfer.cudnn._lazy.get_cudnn() inside the call site instead of "
        "a top-level `import cudnn`."
    )

    # Also — no RuntimeWarning from our lazy machinery should fire at
    # plain import time. The warning is only legitimate when someone
    # actively invokes a cudnn-backed API.
    for w in caught:
        if issubclass(w.category, RuntimeWarning) and "cudnn" in str(w.message):
            raise AssertionError(
                f"RuntimeWarning about cudnn fired at import time: {w.message}"
            )


def test_flashinfer_cudnn_is_imported_without_touching_cudnn_module():
    """Importing the `flashinfer.cudnn` package itself should also not
    trigger the wheel. It only re-exports the lazy helpers + entry points;
    the actual `cudnn` module is resolved lazily on first call.
    """
    for name in list(sys.modules):
        if name == "cudnn" or name.startswith("flashinfer"):
            del sys.modules[name]

    import flashinfer.cudnn  # noqa: F401

    assert "cudnn" not in sys.modules


def test_lazy_cudnn_helper_is_callable():
    """Smoke: the lazy helper exists and is importable independently."""
    from flashinfer.cudnn._lazy import get_cudnn, is_available

    assert callable(get_cudnn)
    assert callable(is_available)
    # Calling is_available() IS allowed to trigger the lazy import (that's
    # the whole point). We don't assert the return value — it depends on
    # whether cudnn is installed and whether the Windows DLL search makes
    # it loadable in this environment.
