# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-process GPU<->host transfer engine.

These run only when a CUDA GPU or Ascend NPU is available (in the
vllm-ascend container they exercise the real NPU path); elsewhere the
module is skipped.
"""
import pytest
import torch


def _device_available() -> bool:
    try:
        from mini_llmcache.utils.device import get_device_module

        get_device_module().current_device()
        return True
    except (RuntimeError, ImportError, AttributeError):
        return False


pytestmark = pytest.mark.skipif(not _device_available(),
                                reason="no CUDA GPU or Ascend NPU available")

from mini_llmcache.l0.transfer import KVTransfer  # noqa: E402
from mini_llmcache.utils.device import get_device_module  # noqa: E402

DEV = get_device_module()


def make_transfer(num_layers=2, num_blocks=8, block_shape=(4, 3),
                  block_size=2, chunk_size=4):
    """Build a KVTransfer over small fake KV tensors.

    block_nbytes = 4*3*2 = 24 bytes; blocks_per_chunk = 2;
    chunk_nbytes = 2 layers * 2 blocks * 24 = 96 bytes.
    """
    kv_caches = [
        torch.empty(num_blocks, *block_shape, dtype=torch.float16,
                    device=DEV.current_device())
        for _ in range(num_layers)
    ]
    return KVTransfer(kv_caches, block_size, chunk_size, rank=0), kv_caches


def fill_data(kv_caches, block_ids):
    """Deterministic content per block."""
    for b in block_ids:
        for layer, kv in enumerate(kv_caches):
            kv[b] = float(b + layer)


def test_to_host_from_host_roundtrip():
    transfer, kv_caches = make_transfer()
    block_ids = [0, 1, 4, 5]  # two chunks
    fill_data(kv_caches, block_ids)

    chunks, elapsed = transfer.to_host(block_ids)
    assert len(chunks) == 2
    assert all(len(c) == transfer.chunk_nbytes for c in chunks)
    assert elapsed >= 0

    # Corrupt the source, then restore it from the bytes.
    for kv in kv_caches:
        kv.fill_(0)
    elapsed = transfer.from_host(chunks, block_ids)
    assert elapsed >= 0

    for b in block_ids:
        for layer, kv in enumerate(kv_caches):
            assert kv[b][0, 0].item() == pytest.approx(float(b + layer))


def test_from_host_skips_first_blocks():
    transfer, kv_caches = make_transfer()
    block_ids = [0, 1, 4, 5]
    fill_data(kv_caches, block_ids)
    chunks, _ = transfer.to_host(block_ids)

    # First chunk: block 0 was computed locally, so only block 1 is loaded.
    for kv in kv_caches:
        kv.fill_(0)
    transfer.from_host(chunks, block_ids, skip_blocks=1)
    for layer, kv in enumerate(kv_caches):
        assert kv[0][0, 0].item() == 0.0  # untouched
        assert kv[1][0, 0].item() == pytest.approx(float(1 + layer))
        assert kv[4][0, 0].item() == pytest.approx(float(4 + layer))


def test_to_host_requires_whole_chunks():
    transfer, _ = make_transfer()
    with pytest.raises(AssertionError):
        transfer.to_host([0, 1, 2])  # 3 blocks: not a multiple of 2
