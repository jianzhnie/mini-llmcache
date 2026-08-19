# SPDX-License-Identifier: Apache-2.0
"""GPU KV cache layout detection and normalization.

Different vLLM builds store the KV cache in different tensor shapes.
``normalize()`` rewrites them all to a common block-first layout where
dim 0 indexes blocks, so the transfer pipelines can treat every tensor
uniformly.
"""

import enum

import torch


class GPUKVFormat(enum.StrEnum):
    """vLLM layout names (``NL`` layers are the outer list dimension)."""

    FA_BLOCK_FIRST = "NL_X_NB_TWO_BS_NH_HS"
    FA_KV_FIRST = "NL_X_TWO_NB_BS_NH_HS"
    FA_SPLIT = "NL_X_NB_BS_NH_HS"
    MLA = "NL_X_NB_BS_HS"


def detect(shape: torch.Size, num_blocks: int | None) -> GPUKVFormat:
    """Identify the layout from the shape of one KV tensor.

    ``num_blocks`` may be None (e.g. TP>1 in vllm 0.23): detection then
    falls back to shape-only heuristics.  Note the block dim may be
    padded beyond num_blocks (vllm-ascend views raw buffers), so
    FA_KV_FIRST is decided by shape[0] == 2 alone.
    """
    if (
        len(shape) == 5
        and shape[1] == 2
        and (num_blocks is None or shape[0] == num_blocks)
    ):
        return GPUKVFormat.FA_BLOCK_FIRST
    if len(shape) == 5 and shape[0] == 2:
        return GPUKVFormat.FA_KV_FIRST
    if len(shape) == 4:
        # vllm-ascend keeps K and V in separate tensors, each
        # (num_blocks, block_size, num_kv_heads, head_size).
        return GPUKVFormat.FA_SPLIT
    if len(shape) == 3 and (num_blocks is None or shape[0] == num_blocks):
        return GPUKVFormat.MLA
    raise ValueError(f"unsupported KV layout {tuple(shape)}")


def normalize(
    tensors: list[torch.Tensor], num_blocks: int | None
) -> list[torch.Tensor]:
    """Return tensors in block-first layout (dim 0 = block index)."""
    fmt = detect(tensors[0].shape, num_blocks)
    if fmt is GPUKVFormat.FA_KV_FIRST:
        return [t.movedim(1, 0) for t in tensors]
    return list(tensors)
