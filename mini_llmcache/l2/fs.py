# SPDX-License-Identifier: Apache-2.0
"""Filesystem L2 adapter: one file per chunk under ``base_path``.

File name encodes identity: ``model@rank@hash.data``.  Writes go through
a temp file + atomic rename so a crash never leaves a half-written chunk
that a later lookup would trust.
"""
import os
from pathlib import Path

from mini_llmcache.l2.base import L2Adapter
from mini_llmcache.l1.memory import MemoryObj
from mini_llmcache.protocol import ChunkKey


def filename(key: ChunkKey) -> str:
    return (f"{key.model.replace('/', '-')}@{key.rank:x}"
            f"@{key.chunk_hash.hex()}.data")


class FSAdapter(L2Adapter):
    def __init__(self, base_path: str):
        self.base = Path(base_path)
        self.tmp = self.base / "tmp"
        self.tmp.mkdir(parents=True, exist_ok=True)
        super().__init__()

    def store(self, keys: list[ChunkKey], objs: list[MemoryObj]) -> None:
        for key, obj in zip(keys, objs):
            final = self.base / filename(key)
            if final.exists():
                continue
            tmp = self.tmp / final.name
            tmp.write_bytes(obj.byte_array)
            os.replace(tmp, final)

    def lookup(self, keys: list[ChunkKey]) -> int:
        """Count the contiguous prefix of ``keys`` that exists on disk."""
        hits = 0
        for key in keys:
            if not (self.base / filename(key)).exists():
                break
            hits += 1
        return hits

    def load(self, keys: list[ChunkKey], objs: list[MemoryObj]) -> int:
        """Fill ``objs`` from disk; stop at the first missing chunk."""
        loaded = 0
        for key, obj in zip(keys, objs):
            path = self.base / filename(key)
            if not path.exists():
                break
            with open(path, "rb") as f:
                if f.readinto(obj.byte_array) != obj.nbytes:
                    break
            loaded += 1
        return loaded
