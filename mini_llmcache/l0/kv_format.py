# SPDX-License-Identifier: Apache-2.0
import enum


class GPUKVFormat(enum.StrEnum):
    FA_BLOCK_FIRST = "NL_X_NB_TWO_BS_NH_HS"
    FA_KV_FIRST = "NL_X_TWO_NB_BS_NH_HS"
    FA_SPLIT = "NL_X_NB_BS_NH_HS"
    MLA = "NL_X_NB_BS_HS"


def detect(shape, num_blocks):
    # Note: the block dim may be padded beyond num_blocks (vllm-ascend views
    # raw buffers), so only shape[0] == 2 decides FA_KV_FIRST below.
    if len(shape) == 5 and shape[0] == num_blocks and shape[1] == 2:
        return GPUKVFormat.FA_BLOCK_FIRST
    if len(shape) == 5 and shape[0] == 2:
        return GPUKVFormat.FA_KV_FIRST
    if len(shape) == 4 and shape[0] >= num_blocks:
        # vllm-ascend keeps K and V in separate tensors, each
        # (num_blocks, block_size, num_kv_heads, head_size).
        return GPUKVFormat.FA_SPLIT
    if len(shape) == 3 and shape[0] == num_blocks:
        return GPUKVFormat.MLA
    raise ValueError(f"unsupported KV layout {tuple(shape)}")


def normalize(tensors, num_blocks):
    fmt = detect(tensors[0].shape, num_blocks)
    if fmt is GPUKVFormat.FA_KV_FIRST:
        return [t.movedim(1, 0) for t in tensors]
    return list(tensors)
