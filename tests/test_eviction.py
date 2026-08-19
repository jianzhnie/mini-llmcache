# SPDX-License-Identifier: Apache-2.0
"""Tests for the LRU policy."""

from mini_llmcache.l1.eviction import LRUPolicy
from mini_llmcache.protocol import ChunkKey


def key(i: int) -> ChunkKey:
    return ChunkKey(bytes([i]), "m", 0)


def test_order_tracks_recency():
    policy = LRUPolicy()
    k1, k2, k3 = key(1), key(2), key(3)
    policy.on_created([k1])
    policy.on_created([k2])
    policy.on_created([k3])
    assert list(policy.order) == [k1, k2, k3]
    # Touching k1 moves it to the end (most recent).
    policy.on_touched([k1])
    assert list(policy.order) == [k2, k3, k1]


def test_get_victims_takes_least_recent_and_eligible():
    policy = LRUPolicy()
    k1, k2, k3 = key(1), key(2), key(3)
    for k in (k1, k2, k3):
        policy.on_created([k])
    victims = policy.get_victims(0.5, eligible=lambda k: k == k1)
    assert victims == [k1]


def test_on_removed_forgets_keys():
    policy = LRUPolicy()
    k1, k2 = key(1), key(2)
    policy.on_created([k1, k2])
    policy.on_removed([k1])
    assert list(policy.order) == [k2]
