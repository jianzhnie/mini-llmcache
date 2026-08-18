# SPDX-License-Identifier: Apache-2.0
"""Device backend selection.

The same code runs on NVIDIA GPUs (torch.cuda) and Ascend NPUs (torch.npu):
everything device-specific goes through the ``DEV`` namespace below.
"""
import torch


def _pick_backend():
    # Ascend NPU: torch_npu registers itself as a torch platform plugin.
    try:
        import torch_npu  # noqa: F401

        if torch.npu.device_count() > 0:
            return torch.npu
    except (ImportError, AttributeError, RuntimeError):
        pass
    if torch.cuda.is_available():
        return torch.cuda
    raise RuntimeError(
        "mini_llmcache needs a CUDA GPU or an Ascend NPU, "
        "but neither is available")


DEV = _pick_backend()
