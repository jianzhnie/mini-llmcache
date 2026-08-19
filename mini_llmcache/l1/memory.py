# SPDX-License-Identifier: Apache-2.0
"""L1 memory: a single big pinned host buffer carved into fixed-size chunks.

``PoolAllocator`` owns one pinned ``torch`` byte tensor and hands out
``MemoryObj`` slices of a given size.  Freed slices are recorded per size
and reused before the bump pointer moves forward.  Pinned (page-locked)
memory lets the vLLM process DMA straight into these buffers.
"""
from dataclasses import dataclass

import torch


@dataclass
class MemoryObj:
    """One allocated slice of the L1 pool."""

    tensor: torch.Tensor
    offset: int

    @property
    def byte_array(self) -> memoryview:
        """Writable byte view of the slice (for copying data in/out)."""
        return memoryview(self.tensor.numpy())

    @property
    def nbytes(self) -> int:
        return self.tensor.numel()


class PoolAllocator:
    def __init__(self, capacity_bytes: int):
        self.pool = torch.empty(capacity_bytes, dtype=torch.uint8,
                                pin_memory=True)
        self.brk = 0
        #: Free segments per chunk size, as offsets into the pool.
        self.free_segments: dict[int, list[int]] = {}
        self.used_bytes = 0

    def allocate(self, count: int, nbytes: int) -> list[MemoryObj] | None:
        """Allocate ``count`` slices of ``nbytes`` each.

        Returns ``None`` when the pool cannot satisfy the request (the
        already-collected slices are returned to the free list).
        """
        free = self.free_segments.setdefault(nbytes, [])
        offsets = []
        while len(offsets) < count and free:
            offsets.append(free.pop())
        while len(offsets) < count and self.brk + nbytes <= self.pool.numel():
            offsets.append(self.brk)
            self.brk += nbytes
        if len(offsets) < count:
            free.extend(offsets)
            return None
        self.used_bytes += count * nbytes
        return [MemoryObj(self.pool[o:o + nbytes], o) for o in offsets]

    def free(self, objs: list[MemoryObj]) -> None:
        for obj in objs:
            self.used_bytes -= obj.nbytes
            self.free_segments[obj.nbytes].append(obj.offset)

    def usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)``."""
        return self.used_bytes, self.pool.numel()
