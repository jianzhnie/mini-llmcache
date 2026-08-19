# SPDX-License-Identifier: Apache-2.0
"""LRU eviction: when L1 fills past a watermark, throw out the cold tail."""

import threading
import time
from collections import OrderedDict

from mini_llmcache.l1.manager import L1Manager, Listener
from mini_llmcache.protocol import ChunkKey

WATERMARK = 0.8
EVICTION_RATIO = 0.2
INTERVAL = 1.0


class LRUPolicy(Listener):
    """Keeps chunks in recency order; recency updates on create/touch."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.order: OrderedDict[ChunkKey, None] = OrderedDict()

    def on_created(self, keys: list[ChunkKey]) -> None:
        with self.lock:
            for key in reversed(keys):
                self.order[key] = None
                self.order.move_to_end(key)

    on_touched = on_created

    def on_removed(self, keys: list[ChunkKey]) -> None:
        with self.lock:
            for key in keys:
                self.order.pop(key, None)

    def get_victims(self, ratio: float, eligible: callable) -> list[ChunkKey]:
        """Return up to ``ratio`` of the least-recently-used eligible keys."""
        with self.lock:
            target = max(1, int(len(self.order) * ratio))
            victims = []
            for key in self.order:
                if len(victims) >= target:
                    break
                if eligible(key):
                    victims.append(key)
            return victims


class EvictionController:
    """Background thread: once a second, evict above the watermark."""

    def __init__(self, l1: L1Manager, policy: LRUPolicy):
        self.l1 = l1
        self.policy = policy
        l1.register_listener(policy)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self) -> None:
        while True:
            time.sleep(INTERVAL)
            used, total = self.l1.usage()
            if used / total < WATERMARK:
                continue
            victims = self.policy.get_victims(EVICTION_RATIO, self.l1.is_evictable)
            self.l1.delete(victims)
