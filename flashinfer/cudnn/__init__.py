"""flashinfer.cudnn — cuDNN-backed attention kernels.

The ``cudnn`` wheel (nvidia-cudnn-frontend) is imported lazily on first
call via :mod:`flashinfer.cudnn._lazy`; ``import flashinfer.cudnn`` does
NOT touch cuDNN. Consumer-grade Windows users who rely on FA2/CUTLASS
never pay the DLL-search cost.
"""

from ._lazy import get_cudnn as get_cudnn
from ._lazy import is_available as is_available
from .decode import cudnn_batch_decode_with_kv_cache
from .prefill import cudnn_batch_prefill_with_kv_cache
