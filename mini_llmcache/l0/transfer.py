# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import Future

import torch

from mini_llmcache.l0.device import DEV


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

    def done(self):
        if not self.future.done():
            return False
        if self.done_event is not None:
            return self.done_event.is_set()
        return True


class Pipeline:
    """Two streams + three staging buffers: overlap kernel work and copies.

    Runs inside the vLLM process — KV tensors are referenced directly, no
    cross-process memory sharing is needed (which Ascend NPU cannot do).
    """

    def __init__(self, kv_caches, blocks_per_chunk, block_nbytes,
                 chunk_nbytes):
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
        self.layer_views = []
        for buf in self.staging:
            chunk = buf.view(len(kv_caches), blocks_per_chunk, block_nbytes)
            views = []
            for layer, kv in enumerate(kv_caches):
                per_block_shape = kv.shape[1:]
                views.append(chunk[layer].view(kv.dtype)
                             .view(blocks_per_chunk, *per_block_shape))
            self.layer_views.append(views)


class KVTransfer:
    def __init__(self, kv_caches, block_size, chunk_size, rank):
        self.rank = rank
        self.kv_caches = kv_caches
        self.device = kv_caches[0].device
        self.num_layers = len(kv_caches)
        self.blocks_per_chunk = chunk_size // block_size
        self.block_nbytes = kv_caches[0][0].nbytes
        self.chunk_nbytes = (self.num_layers * self.blocks_per_chunk
                             * self.block_nbytes)
        self.d2h = Pipeline(kv_caches, self.blocks_per_chunk,
                            self.block_nbytes, self.chunk_nbytes)
        self.h2d = Pipeline(kv_caches, self.blocks_per_chunk,
                            self.block_nbytes, self.chunk_nbytes)

    def to_host(self, block_ids) -> tuple[list[bytes], float]:
        """Copy the given blocks out of the GPU KV cache into bytes."""
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

    def from_host(self, chunks: list[bytes], block_ids, skip_blocks=0) -> float:
        """Scatter the given bytes back into the GPU KV cache."""
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
                pipe.cpu_bufs[buf].copy_(torch.frombuffer(blob,
                                                          dtype=torch.uint8))
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
