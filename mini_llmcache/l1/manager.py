# SPDX-License-Identifier: Apache-2.0
"""L1 cache manager: entries, reference counting, and the lock protocol.

Lock protocol (the heart of cache safety):

- **write lock** — a chunk being written is invisible to readers until
  ``finish_write``.
- **read count** — a chunk being read (or held for a pending RETRIEVE) is
  counted so the evictor never frees it mid-flight.
- **temporary entries** — chunks pulled in from L2 by the prefetcher exist
  only to serve one lookup; once their read count drops to zero they are
  freed immediately instead of becoming resident cache.
"""

import threading
from dataclasses import dataclass

from mini_llmcache.l1.memory import MemoryObj, PoolAllocator
from mini_llmcache.protocol import ChunkKey


class Listener:
    """Observer of L1 lifecycle events (LRU policy, store controller...)."""

    def on_created(self, _keys: list[ChunkKey]) -> None: ...

    def on_write_finished(self, _keys: list[ChunkKey]) -> None: ...

    def on_touched(self, _keys: list[ChunkKey]) -> None: ...

    def on_removed(self, _keys: list[ChunkKey]) -> None: ...


@dataclass
class Entry:
    obj: MemoryObj
    write_locked: bool = False
    read_count: int = 0
    is_temporary: bool = False

    def readable(self) -> bool:
        return not self.write_locked

    def unlocked(self) -> bool:
        """True when nothing references the entry (evictable)."""
        return not self.write_locked and self.read_count == 0


class L1Manager:
    def __init__(self, allocator: PoolAllocator):
        self.allocator = allocator
        self.lock = threading.Lock()
        self.entries: dict[ChunkKey, Entry] = {}
        self.listeners: list[Listener] = []

    def register_listener(self, listener: Listener) -> None:
        self.listeners.append(listener)

    def notify(self, event: str, keys: list[ChunkKey]) -> None:
        for listener in self.listeners:
            getattr(listener, event)(keys)

    def reserve_write(
        self, keys: list[ChunkKey], chunk_nbytes: int, is_temporary: bool = False
    ) -> dict[ChunkKey, MemoryObj | None]:
        """Allocate and write-lock any missing keys.

        Keys already present are left untouched and map to ``None`` in the
        result.  If the pool cannot satisfy the request the result is all
        ``None`` (nothing is reserved).
        """
        with self.lock:
            result: dict[ChunkKey, MemoryObj | None] = dict.fromkeys(keys)
            missing = [key for key in keys if key not in self.entries]
            objs = self.allocator.allocate(len(missing), chunk_nbytes)
            if objs is None:
                return result
            for key, obj in zip(missing, objs, strict=False):
                self.entries[key] = Entry(
                    obj, write_locked=True, is_temporary=is_temporary
                )
                result[key] = obj
            return result

    def finish_write(self, keys: list[ChunkKey]) -> None:
        """Release the write locks; announce the new chunks."""
        with self.lock:
            for key in keys:
                self.entries[key].write_locked = False
            self.notify("on_created", keys)
            self.notify("on_write_finished", keys)

    def finish_write_and_reserve_read(self, keys: list[ChunkKey]) -> None:
        """Write-lock release plus an immediate read hold (prefetch path)."""
        with self.lock:
            for key in keys:
                entry = self.entries[key]
                entry.write_locked = False
                entry.read_count += 1
            self.notify("on_created", keys)

    def reserve_read(self, keys: list[ChunkKey]) -> dict[ChunkKey, MemoryObj | None]:
        """Take a read hold on every readable key (None for missing/locked)."""
        with self.lock:
            result: dict[ChunkKey, MemoryObj | None] = {}
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or not entry.readable():
                    result[key] = None
                else:
                    entry.read_count += 1
                    result[key] = entry.obj
            return result

    def reserve_read_prefix(self, keys: list[ChunkKey]) -> int:
        """Take read holds on the contiguous readable prefix of ``keys``.

        Returns the number of held chunks (the L1 prefix hit count).
        """
        with self.lock:
            hit = 0
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or not entry.readable():
                    break
                hit += 1
            for key in keys[:hit]:
                self.entries[key].read_count += 1
            return hit

    def read(self, keys: list[ChunkKey]) -> list[MemoryObj]:
        """Return the objects for ``keys``.

        Caller must already hold read locks (see ``reserve_read``); keys
        are expected to exist.
        """
        with self.lock:
            return [self.entries[key].obj for key in keys]

    def finish_read(self, keys: list[ChunkKey]) -> None:
        """Drop read holds; temporary entries are freed when count hits 0."""
        freed, touched = [], []
        with self.lock:
            for key in keys:
                entry = self.entries[key]
                entry.read_count -= 1
                if entry.read_count == 0 and entry.is_temporary:
                    self.allocator.free([entry.obj])
                    del self.entries[key]
                    freed.append(key)
                else:
                    touched.append(key)
            if touched:
                self.notify("on_touched", touched)
            if freed:
                self.notify("on_removed", freed)

    def is_evictable(self, key: ChunkKey) -> bool:
        entry = self.entries.get(key)
        return entry is not None and entry.unlocked()

    def delete(self, keys: list[ChunkKey], force: bool = False) -> int:
        """Remove unlocked entries (or any entry when ``force``).

        Returns the number actually deleted.
        """
        with self.lock:
            deleted = []
            for key in keys:
                entry = self.entries.get(key)
                if entry is None or (not entry.unlocked() and not force):
                    continue
                self.allocator.free([entry.obj])
                del self.entries[key]
                deleted.append(key)
            if deleted:
                self.notify("on_removed", deleted)
            return len(deleted)

    def usage(self) -> tuple[int, int]:
        with self.lock:
            return self.allocator.usage()
