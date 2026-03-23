import ctypes
import platform
from pathlib import Path

import torch

pkg_dir = Path(__file__).parent
pattern = "*.pyd" if platform.system() == "Windows" else "*.so"
lib_files = list(pkg_dir.glob(pattern))
if not lib_files:
    raise RuntimeError(f"No extension binary found in {pkg_dir} matching {pattern}")

with torch._ops.dl_open_guard():
    ctypes.CDLL(str(lib_files[0]))

from .ops import fused_grouped_topk
from .ops import grouped_topk
__all__ = ["fused_grouped_topk","grouped_topk"]
