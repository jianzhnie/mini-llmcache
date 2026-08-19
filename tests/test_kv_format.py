# SPDX-License-Identifier: Apache-2.0
"""Tests for KV layout detection and normalization."""
import pytest

from mini_llmcache.l0.kv_format import GPUKVFormat, detect, normalize


def test_detect_fa_block_first():
    # (num_blocks, 2, block_size, nkv_heads, head_size)
    assert detect((1024, 2, 128, 8, 128), 1024) is GPUKVFormat.FA_BLOCK_FIRST


def test_detect_fa_kv_first():
    # (2, num_blocks, block_size, nkv_heads, head_size)
    assert detect((2, 2048, 128, 8, 128), 2048) is GPUKVFormat.FA_KV_FIRST


def test_detect_fa_kv_first_with_padded_blocks():
    # vllm-ascend pads the block dim; detection must not require equality.
    assert detect((2, 4032, 128, 8, 128), 3500) is GPUKVFormat.FA_KV_FIRST


def test_detect_fa_split():
    # vllm-ascend K/V: (num_blocks, block_size, nkv_heads, head_size)
    assert detect((4032, 128, 8, 128), 3500) is GPUKVFormat.FA_SPLIT


def test_detect_mla():
    # MLA layout: (num_blocks, block_size, head_size)
    assert detect((1024, 64, 576), 1024) is GPUKVFormat.MLA


def test_detect_unknown_layout_raises():
    with pytest.raises(ValueError, match="unsupported KV layout"):
        detect((10, 20), 1024)


def test_normalize_moves_kv_first_to_block_first():
    import torch

    kv_first = torch.zeros(2, 4, 3, 2, 2)
    (normalized,) = normalize([kv_first], num_blocks=4)
    assert normalized.shape == (4, 2, 3, 2, 2)


def test_normalize_leaves_block_first_layouts_untouched():
    import torch

    t = torch.zeros(4, 2, 3, 2, 2)
    (normalized,) = normalize([t], num_blocks=4)
    assert normalized is t
