import os
import sys

# On Windows, the `cudnn` pip wheel (nvidia-cudnn-frontend) does
# `ctypes.windll.LoadLibrary("cudnn64_9.dll")` at import time. That DLL
# ships with the PyTorch wheel under <torch>/lib and is NOT on the default
# Windows DLL search path. Add it before any `import cudnn` downstream so
# the frontend can find the runtime.  No-op on Linux.
if sys.platform == "win32":
    try:
        import torch

        _torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(_torch_lib):
            os.add_dll_directory(_torch_lib)
    except Exception:
        pass

from .decode import cudnn_batch_decode_with_kv_cache
from .prefill import cudnn_batch_prefill_with_kv_cache
