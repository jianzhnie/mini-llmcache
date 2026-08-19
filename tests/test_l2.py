# SPDX-License-Identifier: Apache-2.0
"""Tests for the L2 adapters."""
import pytest
import torch

from mini_llmcache.l1.memory import MemoryObj
from mini_llmcache.l2.base import L2Adapter
from mini_llmcache.l2.fs import FSAdapter
from mini_llmcache.l2.mock import MockAdapter
from mini_llmcache.protocol import ChunkKey


def key(i: int) -> ChunkKey:
    return ChunkKey(bytes([i]) * 32, "test/model", 0)


def mem_obj(content: bytes) -> MemoryObj:
    t = torch.empty(len(content), dtype=torch.uint8)
    t.numpy()[:] = list(content)
    return MemoryObj(t, 0)


def preload(adapter: L2Adapter, keys, contents) -> None:
    adapter.submit_store(keys, [mem_obj(c) for c in contents]).result()


class TestMockAdapter:
    def test_store_lookup_load_roundtrip(self):
        adapter = MockAdapter()
        keys = [key(1), key(2), key(3)]
        contents = [b"a" * 8, b"b" * 8, b"c" * 8]
        preload(adapter, keys, contents)
        assert adapter.submit_lookup(keys).result() == 3
        objs = [mem_obj(b"\x00" * 8) for _ in keys]
        assert adapter.submit_load(keys, objs).result() == 3
        assert [bytes(o.byte_array) for o in objs] == contents

    def test_lookup_counts_only_contiguous_prefix(self):
        adapter = MockAdapter()
        preload(adapter, [key(1), key(3)], [b"x" * 4, b"y" * 4])
        assert adapter.submit_lookup([key(1), key(2), key(3)]).result() == 1

    def test_load_stops_at_first_missing_chunk(self):
        adapter = MockAdapter()
        preload(adapter, [key(1)], [b"x" * 4])
        objs = [mem_obj(b"\x00" * 4) for _ in range(2)]
        assert adapter.submit_load([key(1), key(2)], objs).result() == 1


class TestFSAdapter:
    def test_store_lookup_load_roundtrip(self, tmp_path):
        adapter = FSAdapter(str(tmp_path))
        keys = [key(1), key(2)]
        contents = [b"p" * 16, b"q" * 16]
        preload(adapter, keys, contents)
        assert adapter.submit_lookup(keys).result() == 2
        objs = [mem_obj(b"\x00" * 16) for _ in keys]
        assert adapter.submit_load(keys, objs).result() == 2
        assert [bytes(o.byte_array) for o in objs] == contents

    def test_missing_chunk_stops_load_and_counts_hits(self, tmp_path):
        adapter = FSAdapter(str(tmp_path))
        preload(adapter, [key(1)], [b"p" * 16])
        objs = [mem_obj(b"\x00" * 16), mem_obj(b"\x00" * 16)]
        assert adapter.submit_load([key(1), key(2)], objs).result() == 1


class FailingAdapter(L2Adapter):
    def store(self, keys, objs):
        raise OSError("disk on fire")

    def lookup(self, keys):
        return 0

    def load(self, keys, objs):
        return 0


def test_adapter_worker_propagates_exceptions():
    """A failing adapter must raise on the future, not die silently."""
    adapter = FailingAdapter()
    with pytest.raises(OSError, match="disk on fire"):
        adapter.submit_store([key(1)], [mem_obj(b"x" * 4)]).result()
    # The worker thread survived: subsequent tasks still complete.
    assert adapter.submit_lookup([key(1)]).result() == 0
