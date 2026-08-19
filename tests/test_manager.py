# SPDX-License-Identifier: Apache-2.0
"""Tests for the L1 manager's lock protocol."""

from mini_llmcache.l1.manager import L1Manager, Listener
from mini_llmcache.l1.memory import PoolAllocator
from mini_llmcache.protocol import ChunkKey


def key(i: int) -> ChunkKey:
    return ChunkKey(bytes([i]), "m", 0)


class RecordingListener(Listener):
    def __init__(self):
        self.events = []

    def on_created(self, keys):
        self.events.append(("created", list(keys)))

    def on_write_finished(self, keys):
        self.events.append(("write_finished", list(keys)))

    def on_touched(self, keys):
        self.events.append(("touched", list(keys)))

    def on_removed(self, keys):
        self.events.append(("removed", list(keys)))


def make_l1(capacity: int = 4096) -> L1Manager:
    return L1Manager(PoolAllocator(capacity))


def test_write_lock_hides_entry_until_finished():
    l1 = make_l1()
    keys = [key(1)]
    reserved = l1.reserve_write(keys, 64)
    assert reserved[keys[0]] is not None
    # Not readable while write-locked.
    assert l1.reserve_read(keys)[keys[0]] is None
    l1.finish_write(keys)
    assert l1.reserve_read(keys)[keys[0]] is not None


def test_reserve_write_existing_key_returns_none():
    l1 = make_l1()
    keys = [key(1)]
    l1.reserve_write(keys, 64)
    l1.finish_write(keys)
    reserved = l1.reserve_write(keys, 64)
    assert reserved[keys[0]] is None


def test_read_hold_blocks_eviction_until_released():
    l1 = make_l1()
    keys = [key(1)]
    l1.reserve_write(keys, 64)
    l1.finish_write(keys)
    assert l1.is_evictable(keys[0])
    l1.reserve_read(keys)
    assert not l1.is_evictable(keys[0])
    assert l1.delete(keys) == 0  # still held
    l1.finish_read(keys)
    assert l1.delete(keys) == 1


def test_temporary_entry_is_freed_when_read_count_hits_zero():
    l1 = make_l1()
    keys = [key(1)]
    l1.reserve_write(keys, 64, is_temporary=True)
    l1.finish_write_and_reserve_read(keys)  # prefetch path
    assert keys[0] in l1.entries
    l1.finish_read(keys)
    assert keys[0] not in l1.entries
    assert l1.usage()[0] == 0


def test_prefix_read_reservation_counts_contiguous_hits():
    l1 = make_l1()
    keys = [key(1), key(2), key(3)]
    for k in keys[:2]:
        l1.reserve_write([k], 64)
        l1.finish_write([k])
    # key(1) and key(2) exist; key(3) does not -> prefix hit is 2.
    assert l1.reserve_read_prefix(keys) == 2
    l1.finish_read(keys[:2])


def test_delete_force_removes_even_locked_entries():
    l1 = make_l1()
    keys = [key(1)]
    l1.reserve_write(keys, 64)
    assert l1.delete(keys, force=True) == 1
    assert keys[0] not in l1.entries


def test_listener_notifications():
    l1 = make_l1()
    listener = RecordingListener()
    l1.register_listener(listener)
    keys = [key(1)]
    l1.reserve_write(keys, 64)
    l1.finish_write(keys)
    l1.reserve_read(keys)
    l1.finish_read(keys)
    l1.delete(keys)
    kinds = [e[0] for e in listener.events]
    assert kinds == ["created", "write_finished", "touched", "removed"]
