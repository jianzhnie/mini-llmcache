# SPDX-License-Identifier: Apache-2.0
"""Device backend selection.

The same code runs on NVIDIA GPUs (``torch.cuda``) and Ascend NPUs
(``torch.npu``): everything device-specific goes through the ``DEV``
namespace below (streams, events, device context, synchronization).
"""
from typing import Any

import importlib

import torch


def _pick_backend() -> Any:
    """Return the device module to use: ``torch.npu`` or ``torch.cuda``."""
    # Ascend NPU: importing torch_npu registers itself as a torch platform
    # plugin (the import is a deliberate side effect).
    try:
        importlib.import_module("torch_npu")
        if torch.npu.device_count() > 0:
            return torch.npu
    except (ImportError, AttributeError, RuntimeError):
        pass
    if torch.cuda.is_available():
        return torch.cuda
    raise RuntimeError(
        "mini_llmcache needs a CUDA GPU or an Ascend NPU, "
        "but neither is available")


DEV: Any = _pick_backend()
