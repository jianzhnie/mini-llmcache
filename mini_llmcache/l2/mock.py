# SPDX-License-Identifier: Apache-2.0
"""In-memory L2 adapter, useful for tests and quick experiments."""
from mini_llmcache.l2.base import L2Adapter
from mini_llmcache.l1.memory import MemoryObj
from mini_llmcache.protocol import ChunkKey


class MockAdapter(L2Adapter):
    def __init__(self) -> None:
        self.chunks: dict[ChunkKey, bytes] = {}
        super().__init__()

    def store(self, keys: list[ChunkKey], objs: list[MemoryObj]) -> None:
        for key, obj in zip(keys, objs):
            self.chunks[key] = bytes(obj.byte_array)

    def lookup(self, keys: list[ChunkKey]) -> int:
        hits = 0
        for key in keys:
            if key not in self.chunks:
                break
            hits += 1
        return hits

    def load(self, keys: list[ChunkKey], objs: list[MemoryObj]) -> int:
        loaded = 0
        for key, obj in zip(keys, objs):
            data = self.chunks.get(key)
            if data is None:
                break
            obj.byte_array[:] = data
            loaded += 1
        return loaded
