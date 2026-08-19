# SPDX-License-Identifier: Apache-2.0
"""Tests for lookup/prefetch orchestration against a mock L2."""
import torch

from mini_llmcache.hasher import chunk_hashes
from mini_llmcache.l1.manager import L1Manager
from mini_llmcache.l1.memory import MemoryObj, PoolAllocator
from mini_llmcache.l1.prefetch_controller import PrefetchController
from mini_llmcache.l2.mock import MockAdapter
from mini_llmcache.protocol import ChunkKey
from conftest import wait_until

CHUNK_SIZE = 4
CHUNK_NBYTES = 16
MODEL = "test-model"


def keys_for(tokens: list[int]) -> list[ChunkKey]:
    return [ChunkKey(h, MODEL, 0)
            for h in chunk_hashes(tokens, CHUNK_SIZE)]


def make_controller(l2: MockAdapter) -> tuple[PrefetchController, L1Manager]:
    l1 = L1Manager(PoolAllocator(1 << 16))
    controller = PrefetchController(l1, [l2])
    return controller, l1


def preload_l2(l2: MockAdapter, keys: list[ChunkKey]) -> None:
    objs = [MemoryObj(torch.empty(CHUNK_NBYTES, dtype=torch.uint8), 0)
            for _ in keys]
    l2.submit_store(keys, objs).result()


def test_l2_hits_resolve_and_release_cleanly():
    tokens = list(range(12))  # 3 chunks
    keys = keys_for(tokens)
    l2 = MockAdapter()
    preload_l2(l2, keys)
    controller, l1 = make_controller(l2)

    controller.start_session("rid", keys, CHUNK_NBYTES)
    wait_until(lambda: controller.query("rid") is not None)
    l1_hits, l2_hits, gbps = controller.query("rid")
    assert (l1_hits, l2_hits) == (0, 3)
    assert gbps >= 0

    # All three chunks came from L2 as temporary entries; releasing the
    # session must free them and forget the request.
    controller.end_session("rid")
    assert controller.query("rid") is None
    assert l1.usage()[0] == 0


def test_l1_prefix_hits_are_counted_first():
    tokens = list(range(12))
    keys = keys_for(tokens)
    l2 = MockAdapter()
    controller, l1 = make_controller(l2)

    # Pre-seed the first chunk into L1 as a resident (non-temporary) entry.
    l1.reserve_write(keys[:1], CHUNK_NBYTES)
    l1.finish_write(keys[:1])

    controller.start_session("rid", keys, CHUNK_NBYTES)
    wait_until(lambda: controller.query("rid") is not None)
    l1_hits, l2_hits, _ = controller.query("rid")
    assert (l1_hits, l2_hits) == (1, 0)  # rest is missing from L2 as well

    controller.end_session("rid")
    # The resident chunk survives; nothing else is allocated.
    assert l1.usage()[0] == CHUNK_NBYTES


def test_progressive_release_first():
    tokens = list(range(12))
    keys = keys_for(tokens)
    l2 = MockAdapter()
    preload_l2(l2, keys)
    controller, l1 = make_controller(l2)

    controller.start_session("rid", keys, CHUNK_NBYTES)
    wait_until(lambda: controller.query("rid") is not None)
    # All three temporary entries are held (read-locked).
    assert l1.usage()[0] == 3 * CHUNK_NBYTES

    controller.release_first("rid", 2)
    assert len(controller.held["rid"]) == 1

    controller.end_session("rid")
    assert l1.usage()[0] == 0


def test_unresolved_query_returns_none():
    l2 = MockAdapter()
    controller, _ = make_controller(l2)
    assert controller.query("never-started") is None
