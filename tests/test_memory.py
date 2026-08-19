# SPDX-License-Identifier: Apache-2.0
"""Tests for the L1 pool allocator."""

from mini_llmcache.l1.memory import PoolAllocator


def test_allocate_returns_distinct_slices():
    pool = PoolAllocator(1024)
    objs = pool.allocate(4, 256)
    assert objs is not None
    offsets = {obj.offset for obj in objs}
    assert offsets == {0, 256, 512, 768}
    assert all(obj.nbytes == 256 for obj in objs)


def test_freed_segments_are_reused():
    pool = PoolAllocator(1024)
    objs = pool.allocate(4, 256)
    pool.free([objs[1], objs[3]])
    again = pool.allocate(2, 256)
    # The two freed offsets come back first (LIFO from the free list).
    assert [o.offset for o in again] == [768, 256]


def test_exhaustion_returns_none_and_keeps_free_list():
    pool = PoolAllocator(512)
    first = pool.allocate(2, 256)
    assert pool.allocate(1, 256) is None
    pool.free(first)
    assert pool.allocate(1, 256) is not None


def test_usage_accounting():
    pool = PoolAllocator(1024)
    assert pool.usage() == (0, 1024)
    objs = pool.allocate(2, 256)
    assert pool.usage() == (512, 1024)
    pool.free(objs)
    assert pool.usage() == (0, 1024)


def test_memory_obj_byte_view_roundtrip():
    pool = PoolAllocator(64)
    (obj,) = pool.allocate(1, 64)
    obj.byte_array[:] = b"\xab" * 64
    assert bytes(obj.byte_array) == b"\xab" * 64
