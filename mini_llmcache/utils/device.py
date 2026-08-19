# SPDX-License-Identifier: Apache-2.0
"""Device backend selection.

The same code runs on NVIDIA GPUs (``torch.cuda``), Ascend NPUs
(``torch.npu``) and other accelerators: everything device-specific goes
through the namespace returned by :func:`get_device_module`.
"""

import torch
from functools import lru_cache
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

def get_device_type() -> str:
    """Get the accelerator device type available on this system.

    Returns:
        One of: 'npu', 'cuda', 'xpu', 'mlu', 'musa', 'mps', 'cpu'
    """
    device_type = None
    if is_torch_npu_available():
        device_type = "npu"
    elif is_torch_cuda_available():
        device_type = "cuda"
    elif is_torch_xpu_available():
        device_type = "xpu"
    elif is_torch_mlu_available():
        device_type = "mlu"
    elif is_torch_musa_available():
        device_type = "musa"
    elif is_torch_mps_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    return device_type


def set_device(device: torch.device) -> None:
    """Set the current device for the accelerator."""
    dtype = device.type
    if dtype == "npu":
        torch.npu.set_device(device)
    elif dtype == "cuda":
        torch.cuda.set_device(device)
    elif dtype == "xpu":
        torch.xpu.set_device(device)
    elif dtype == "mlu":
        torch.mlu.set_device(device)
    elif dtype == "musa":
        torch.musa.set_device(device)
    elif dtype == "mps":
        torch.mps.set_device(device)
    else:
        raise ValueError(f"unsupported device type: {dtype}")


@lru_cache(maxsize=1)
def get_device_module():
    """Get the torch device namespace (torch.npu / torch.cuda / ...) for
    the active accelerator.  Cached: the accelerator never changes during
    a process lifetime.

    Raises:
        RuntimeError: when no accelerator is available.
    """
    device_type = get_device_type()
    if device_type == "cpu":
        raise RuntimeError(
            "mini_llmcache needs a CUDA GPU or an Ascend NPU, "
            "but neither is available")
    return getattr(torch, device_type)


def is_device_available() -> bool:
    """Return True when an accelerator (NPU/CUDA/XPU/...) is usable."""
    try:
        get_device_module().current_device()
        return True
    except (RuntimeError, ImportError, AttributeError):
        return False
