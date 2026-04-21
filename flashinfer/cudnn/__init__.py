import os
import shutil
import sys

# On Windows, the `cudnn` pip wheel (nvidia-cudnn-frontend 1.22.1) has
# two Windows-broken code paths we work around here:
#
# 1) At `import cudnn`, the Python layer does
#        ctypes.windll.LoadLibrary("cudnn64_9.dll")
#    but the DLL ships with PyTorch under `<torch>/lib` and is not on
#    the default DLL search path. We prepend `<torch>/lib` with
#    `os.add_dll_directory`.
#
# 2) Inside the wheel's native pyd (`_compiled_module.cp312-win_amd64.pyd`)
#    the CUDA runtime loader uses the Linux-style filenames
#    `libcudart.so.12` / `libcudart.so.13` / `libcudart.so.*`. Those
#    calls trip `Unable to load any libcudart.so.* library` on Windows,
#    because `cudart64_12.dll` is what's present. Windows
#    `LoadLibrary` searches by exact filename regardless of extension,
#    so we materialise aliases in a flashinfer-managed cache dir and add
#    that dir to the DLL search path. Cheap (one 600 KB file copy the
#    first time flashinfer.cudnn is imported).
#
# Both workarounds are no-ops on Linux.
if sys.platform == "win32":
    try:
        import torch

        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)

        # Provide libcudart.so.12 / libcudart.so.13 / libcudart.so aliases
        # directly inside torch/lib (we already added it to the DLL search
        # path above). Placing them next to torch's real DLLs makes them
        # visible regardless of the specific LoadLibrary flag combination
        # the wheel's native loader chose. libcuda.so.1 maps to the driver.
        _cudart_src = os.path.join(_torch_lib, "cudart64_12.dll")
        if os.path.isfile(_cudart_src):
            for _alias in ("libcudart.so.12", "libcudart.so.13", "libcudart.so"):
                _dst = os.path.join(_torch_lib, _alias)
                if not os.path.isfile(_dst):
                    try:
                        shutil.copy2(_cudart_src, _dst)
                    except OSError:
                        pass
        _nvcuda = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvcuda.dll")
        if os.path.isfile(_nvcuda):
            _dst = os.path.join(_torch_lib, "libcuda.so.1")
            if not os.path.isfile(_dst):
                try:
                    shutil.copy2(_nvcuda, _dst)
                except OSError:
                    pass
    except Exception:
        pass

from .decode import cudnn_batch_decode_with_kv_cache
from .prefill import cudnn_batch_prefill_with_kv_cache
