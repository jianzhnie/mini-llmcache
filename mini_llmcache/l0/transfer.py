# SPDX-License-Identifier: Apache-2.0
"""In-process GPU<->host transfer engine (runs inside the vLLM process).

``KVTransfer`` holds direct references to the engine's KV tensors and
moves blocks between the GPU and pinned host memory with a two-stream,
three-buffer pipeline: while one chunk is being copied, the previous one
is still being consumed by the kernel stream.  Only bytes cross the
process boundary (over ZMQ) — the cache server never touches the GPU.
"""
import threading
from concurrent.futures import Future

import numpy as np
import torch

from mini_llmcache.utils.device import get_device_module

#: Active accelerator namespace; raises a clear error on CPU-only machines.
DEV = get_device_module()


class DeviceFuture:
    """Tracks a transfer that was submitted to the cache server.

    ``future`` resolves when the message round-trip completes;
    ``done_event`` (when given) is set by a worker thread once the
    GPU-side scatter into the KV cache has finished as well.
    """

    def __init__(self, future: Future, device_index: int,
                 done_event: threading.Event | None = None):
        self.future = future
        self.device_index = device_index
        self.done_event = done_event

    def done(self) -> bool:
        """True once the message arrived AND the GPU work is complete."""
        if not self.future.done():
            return False
        if self.done_event is not None:
            return self.done_event.is_set()
        return True


class Pipeline:
    """Two streams + three staging buffers: overlap kernel work and copies.

    ``kernel_stream`` runs the block gather/scatter kernels (``index_select``
    / ``index_copy_``); ``copy_stream`` runs the DMA between the device
    staging buffers and the pinned host buffers.  ``ready``/``free`` events
    pair the two per staging slot, so slot i's kernel work overlaps slot
    i-1's copy.
    """

    def __init__(self, kv_caches: list[torch.Tensor], blocks_per_chunk: int,
                 block_nbytes: int, chunk_nbytes: int):
        device = kv_caches[0].device
        self.lock = threading.Lock()
        self.kernel_stream = DEV.Stream(device)
        self.copy_stream = DEV.Stream(device)
        self.staging = torch.empty(3, chunk_nbytes, dtype=torch.uint8,
                                   device=device)
        self.cpu_bufs = [torch.empty(chunk_nbytes, dtype=torch.uint8,
                                     pin_memory=True) for _ in range(3)]
        self.ready = [DEV.Event() for _ in range(3)]
        self.free = [DEV.Event() for _ in range(3)]
        #: Per staging slot, per layer: a typed view of the chunk bytes.
        self.layer_views: list[list[torch.Tensor]] = []
        for buf in self.staging:
            chunk = buf.view(len(kv_caches), blocks_per_chunk, block_nbytes)
            views = []
            for layer, kv in enumerate(kv_caches):
                per_block_shape = kv.shape[1:]
                views.append(chunk[layer].view(kv.dtype)
                             .view(blocks_per_chunk, *per_block_shape))
            self.layer_views.append(views)


class KVTransfer:
    def __init__(self, kv_caches: list[torch.Tensor], block_size: int,
                 chunk_size: int, rank: int):
        self.rank = rank
        self.kv_caches = kv_caches
        self.device: torch.device = kv_caches[0].device
        self.num_layers = len(kv_caches)
        self.blocks_per_chunk = chunk_size // block_size
        self.block_nbytes = kv_caches[0][0].nbytes
        self.chunk_nbytes = (self.num_layers * self.blocks_per_chunk
                             * self.block_nbytes)
        self.d2h = Pipeline(kv_caches, self.blocks_per_chunk,
                            self.block_nbytes, self.chunk_nbytes)
        self.h2d = Pipeline(kv_caches, self.blocks_per_chunk,
                            self.block_nbytes, self.chunk_nbytes)

    def to_host(self, block_ids: list[int]) -> tuple[list[bytes], float]:
        """Copy the given blocks out of the GPU KV cache into bytes."""
        assert len(block_ids) % self.blocks_per_chunk == 0, \
            "block_ids must fill whole chunks"
        n_chunks = len(block_ids) // self.blocks_per_chunk
        block_ids_t = torch.tensor(block_ids, dtype=torch.long,
                                   device=self.device)
        pipe = self.d2h
        with pipe.lock, DEV.device(self.device):
            # Freeze all device work before reading the KV blocks
            # (portable replacement for a producer event).
            DEV.synchronize()
            begin = DEV.Event(enable_timing=True)
            begin.record(pipe.kernel_stream)
            for i in range(n_chunks):
                buf = i % 3
                ids = block_ids_t[i * self.blocks_per_chunk:
                                  (i + 1) * self.blocks_per_chunk]
                with DEV.stream(pipe.kernel_stream):
                    pipe.kernel_stream.wait_event(pipe.free[buf])
                    for layer, kv_cache in enumerate(self.kv_caches):
                        torch.index_select(kv_cache, 0, ids,
                                           out=pipe.layer_views[buf][layer])
                    pipe.ready[buf].record(pipe.kernel_stream)
                with DEV.stream(pipe.copy_stream):
                    pipe.copy_stream.wait_event(pipe.ready[buf])
                    pipe.cpu_bufs[buf].copy_(pipe.staging[buf],
                                             non_blocking=True)
                    pipe.free[buf].record(pipe.copy_stream)
            end = DEV.Event(enable_timing=True)
            end.record(pipe.copy_stream)
            pipe.copy_stream.synchronize()
            chunks = [pipe.cpu_bufs[i % 3].numpy().tobytes()
                      for i in range(n_chunks)]
        return chunks, begin.elapsed_time(end) / 1e3

    def from_host(self, chunks: list[bytes], block_ids: list[int],
                  skip_blocks: int = 0) -> float:
        """Scatter the given bytes back into the GPU KV cache.

        ``skip_blocks`` blocks of the first chunk are left untouched (the
        connector has already computed that prefix locally).
        """
        assert len(block_ids) % self.blocks_per_chunk == 0, \
            "block_ids must fill whole chunks"
        block_ids_t = torch.tensor(block_ids, dtype=torch.long,
                                   device=self.device)
        pipe = self.h2d
        with pipe.lock, DEV.device(self.device):
            begin = DEV.Event(enable_timing=True)
            begin.record(pipe.copy_stream)
            for i, blob in enumerate(chunks):
                buf = i % 3
                if i >= 3:
                    # The previous use of this host buffer must be fully
                    # consumed before we overwrite it with new data.
                    pipe.free[buf].synchronize()
                # Direct CPU-side byte copy (avoids torch's read-only-buffer
                # warning that torch.frombuffer triggers).
                pipe.cpu_bufs[buf].numpy()[:] = np.frombuffer(blob,
                                                              dtype=np.uint8)
                skip = skip_blocks if i == 0 else 0
                ids = block_ids_t[i * self.blocks_per_chunk + skip:
                                  (i + 1) * self.blocks_per_chunk]
                with DEV.stream(pipe.copy_stream):
                    pipe.copy_stream.wait_event(pipe.free[buf])
                    pipe.staging[buf].copy_(pipe.cpu_bufs[buf],
                                            non_blocking=True)
                    pipe.ready[buf].record(pipe.copy_stream)
                with DEV.stream(pipe.kernel_stream):
                    pipe.kernel_stream.wait_event(pipe.ready[buf])
                    for layer, kv_cache in enumerate(self.kv_caches):
                        kv_cache.index_copy_(
                            0, ids, pipe.layer_views[buf][layer][skip:])
                    pipe.free[buf].record(pipe.kernel_stream)
            end = DEV.Event(enable_timing=True)
            end.record(pipe.kernel_stream)
            # Guarantee the scatter is complete before the caller marks
            # this transfer as finished (vLLM will read these blocks).
            pipe.kernel_stream.synchronize()
        return begin.elapsed_time(end) / 1e3
