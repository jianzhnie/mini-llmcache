# SPDX-License-Identifier: Apache-2.0
"""Lookup / prefetch orchestration.

On LOOKUP, a worker thread checks L1 for a contiguous prefix hit, asks
every L2 adapter for the missing chunks, pulls those into L1 as temporary
entries, and finally "resolves" the request: the hit keys stay read-locked
(``held``) until the connector releases them (FREE_LOOKUP_LOCKS / RETRIEVE
/ END_SESSION), so they cannot be evicted while the request needs them.
"""

import queue
import threading
import time

from mini_llmcache.l1.manager import L1Manager
from mini_llmcache.l1.memory import MemoryObj
from mini_llmcache.l2.base import L2Adapter
from mini_llmcache.protocol import ChunkKey


class PrefetchController:
    def __init__(self, l1: L1Manager, l2s: list[L2Adapter]):
        self.l1 = l1
        self.l2s = l2s
        self.lock = threading.Lock()
        self.jobs: queue.SimpleQueue = queue.SimpleQueue()
        #: request_id -> None (pending) or (l1_hits, l2_hits, l2_gbps)
        self.hits: dict[str, tuple[int, int, float] | None] = {}
        #: request_id -> {ChunkKey: None} held read locks
        self.held: dict[str, dict[ChunkKey, None]] = {}
        threading.Thread(target=self.run, daemon=True).start()

    def run(self) -> None:
        while True:
            self.prefetch(*self.jobs.get())

    def prefetch(
        self, request_id: str, keys: list[ChunkKey], chunk_nbytes: int | None
    ) -> None:
        """Resolve one lookup: L1 prefix hits + L2 loads."""
        l1_hits = self.l1.reserve_read_prefix(keys)
        missing = keys[l1_hits:]
        if chunk_nbytes is None:
            # No engine registered for this deployment yet: fail open with
            # zero hits rather than trying to allocate unknown-size chunks.
            self.resolve(request_id, keys[:l1_hits], [], 0.0)
            return
        l2_hits = self.lookup_l2s(missing)
        load_keys, objs = self.reserve_load(
            missing, max(l2_hits, default=0), chunk_nbytes
        )
        start = time.perf_counter()
        loaded = self.load_from_l2s(load_keys, objs, l2_hits)
        elapsed = time.perf_counter() - start
        self.l1.finish_write_and_reserve_read(load_keys[:loaded])
        self.l1.delete(load_keys[loaded:], force=True)
        gbps = loaded * chunk_nbytes / elapsed / 1e9 if loaded else 0.0
        self.resolve(request_id, keys[:l1_hits], load_keys[:loaded], gbps)

    def lookup_l2s(self, missing: list[ChunkKey]) -> list[int]:
        """Per-adapter prefix hit counts for the missing chunks."""
        if not missing:
            return []
        futures = [l2.submit_lookup(missing) for l2 in self.l2s]
        return [future.result() for future in futures]

    def reserve_load(
        self, missing: list[ChunkKey], l2_hits: int, chunk_nbytes: int
    ) -> tuple[list[ChunkKey], list[MemoryObj]]:
        """Allocate temporary L1 slots for up to ``l2_hits`` chunks."""
        load_keys, objs = [], []
        for key in missing[:l2_hits]:
            obj = self.l1.reserve_write([key], chunk_nbytes, is_temporary=True)[key]
            if obj is None:
                break  # pool full — load only what we could allocate
            load_keys.append(key)
            objs.append(obj)
        return load_keys, objs

    def load_from_l2s(
        self, load_keys: list[ChunkKey], objs: list[MemoryObj], l2_hits: list[int]
    ) -> int:
        """Fill chunks from L2s, best-hit adapter first. Returns count."""
        if not load_keys:
            return 0
        loaded = 0
        for i in sorted(range(len(self.l2s)), key=lambda i: -l2_hits[i]):
            if loaded == len(load_keys) or l2_hits[i] <= loaded:
                break
            loaded += (
                self.l2s[i].submit_load(load_keys[loaded:], objs[loaded:]).result()
            )
        return loaded

    def resolve(
        self,
        request_id: str,
        l1_keys: list[ChunkKey],
        loaded_keys: list[ChunkKey],
        gbps: float,
    ) -> None:
        """Publish the result and hold read locks on every hit chunk.

        If the session already ended (connector raced us), just release.
        """
        hit_keys = l1_keys + loaded_keys
        with self.lock:
            if request_id in self.hits:
                self.hits[request_id] = (len(l1_keys), len(loaded_keys), gbps)
                self.held[request_id] = dict.fromkeys(hit_keys)
                return
        self.l1.finish_read(hit_keys)

    def start_session(
        self, request_id: str, keys: list[ChunkKey], chunk_nbytes: int | None
    ) -> None:
        """Register a lookup as pending and enqueue its prefetch job."""
        with self.lock:
            self.hits[request_id] = None
        self.jobs.put((request_id, keys, chunk_nbytes))

    def query(self, request_id: str) -> tuple[int, int, float] | None:
        """Return ``(l1_hits, l2_hits, l2_gbps)`` or None if still pending."""
        with self.lock:
            return self.hits.get(request_id)

    def release(self, request_id: str, keys: list[ChunkKey] | None = None) -> None:
        """Drop held read locks (all of them when ``keys`` is None)."""
        with self.lock:
            held = self.held.get(request_id, {})
            if keys is None:
                keys = list(held)
            released = [key for key in keys if key in held]
            for key in released:
                del held[key]
        self.l1.finish_read(released)

    def release_first(self, request_id: str, count: int) -> None:
        """Drop the first ``count`` held locks (progressive unlock)."""
        with self.lock:
            keys = list(self.held.get(request_id, {}))[:count]
        self.release(request_id, keys)

    def end_session(self, request_id: str) -> None:
        """Release everything and forget the request."""
        self.release(request_id)
        with self.lock:
            self.hits.pop(request_id, None)
            self.held.pop(request_id, None)
