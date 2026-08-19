# SPDX-License-Identifier: Apache-2.0
"""Background write-back: mirror every finished L1 write into all L2s."""
import queue
import threading

from mini_llmcache.l1.manager import L1Manager, Listener
from mini_llmcache.l2.base import L2Adapter
from mini_llmcache.protocol import ChunkKey


class StoreController(Listener):
    def __init__(self, l1: L1Manager, l2s: list[L2Adapter]):
        self.l1 = l1
        self.l2s = l2s
        self.pending: queue.SimpleQueue = queue.SimpleQueue()
        l1.register_listener(self)
        threading.Thread(target=self.run, daemon=True).start()

    def on_write_finished(self, keys: list[ChunkKey]) -> None:
        self.pending.put(list(keys))

    def run(self) -> None:
        while True:
            keys = self.pending.get()
            reserved = self.l1.reserve_read(keys)
            locked = [key for key in keys if reserved[key] is not None]
            if not locked:
                continue  # everything was evicted before we got to it
            objs = [reserved[key] for key in locked]
            for future in [l2.submit_store(locked, objs) for l2 in self.l2s]:
                future.result()
            self.l1.finish_read(locked)
