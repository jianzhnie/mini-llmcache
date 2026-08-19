# SPDX-License-Identifier: Apache-2.0
"""Base class for pluggable L2 (secondary storage) adapters.

Each adapter owns a worker thread that runs ``store``/``lookup``/``load``
serially, so adapters do not need to be thread-safe themselves.  Subclasses
implement the three operations; the worker catches their exceptions and
surfaces them on the returned futures.
"""
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any, Callable

from mini_llmcache.l1.memory import MemoryObj
from mini_llmcache.protocol import ChunkKey


class L2Adapter(ABC):
    def __init__(self) -> None:
        self.tasks: queue.SimpleQueue = queue.SimpleQueue()
        threading.Thread(target=self.run, daemon=True).start()

    # ---- interface implemented by subclasses ----

    @abstractmethod
    def store(self, keys: list[ChunkKey], objs: list[MemoryObj]) -> None:
        """Persist ``objs`` under their ``keys``."""

    @abstractmethod
    def lookup(self, keys: list[ChunkKey]) -> int:
        """Number of chunks of ``keys`` found (as a prefix)."""

    @abstractmethod
    def load(self, keys: list[ChunkKey], objs: list[MemoryObj]) -> int:
        """Fill ``objs`` from storage; return the number actually loaded."""

    # ---- async submission ----

    def submit_store(self, keys: list[ChunkKey],
                     objs: list[MemoryObj]) -> Future:
        return self.submit(self.store, keys, objs)

    def submit_lookup(self, keys: list[ChunkKey]) -> Future:
        return self.submit(self.lookup, keys)

    def submit_load(self, keys: list[ChunkKey],
                    objs: list[MemoryObj]) -> Future:
        return self.submit(self.load, keys, objs)

    def submit(self, fn: Callable, *args: Any) -> Future:
        future: Future = Future()
        self.tasks.put((future, fn, args))
        return future

    def run(self) -> None:
        while True:
            future, fn, args = self.tasks.get()
            try:
                future.set_result(fn(*args))
            except Exception as exc:  # noqa: BLE001 — surfaced to callers
                future.set_exception(exc)
