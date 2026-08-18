# SPDX-License-Identifier: Apache-2.0
import threading
from concurrent.futures import Future

import torch

from mini_llmcache.l0.device import DEV
from mini_llmcache.l1.memory import MemoryObj


def export_kv_caches(tensors):
    from torch.multiprocessing.reductions import reduce_tensor

    return [reduce_tensor(t) for t in tensors]


def import_kv_caches(ipc_handles):
    return [rebuild(*args) for rebuild, args in ipc_handles]


class DeviceFuture:
    """Tracks a transfer submitted to the cache server.

    The server synchronizes its transfer streams before replying, so once the
    message future resolves the GPU-side work is guaranteed visible.  (The
    original cross-process CUDA event trick is not portable: Ascend NPU does
    not support ``Event(interprocess=True)``.)
    """

    def __init__(self, future: Future, device_index: int):
        self.future = future
        self.device_index = device_index

    def done(self):
        return self.future.done()


class Pipeline:
    """Two streams + three staging buffers: overlap kernel work and copies."""

    def __init__(self, kv_caches, blocks_per_chunk, block_nbytes,
                 chunk_nbytes):
        device = kv_caches[0].device
        self.lock = threading.Lock()
        self.kernel_stream = DEV.Stream(device)
        self.copy_stream = DEV.Stream(device)
        self.staging = torch.empty(3, chunk_nbytes, dtype=torch.uint8,
                                   device=device)
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
    def __init__(self, ipc_handles, block_size, chunk_size, rank):
        self.rank = rank
        self.kv_caches = import_kv_caches(ipc_handles)
        self.device = self.kv_caches[0].device
        self.num_layers = len(self.kv_caches)
        self.blocks_per_chunk = chunk_size // block_size
        self.block_nbytes = self.kv_caches[0][0].nbytes
        self.chunk_nbytes = (self.num_layers * self.blocks_per_chunk
                             * self.block_nbytes)
        self.d2h = Pipeline(self.kv_caches, self.blocks_per_chunk,
                            self.block_nbytes, self.chunk_nbytes)
        self.h2d = Pipeline(self.kv_caches, self.blocks_per_chunk,
                            self.block_nbytes, self.chunk_nbytes)

    def store(self, block_ids, objs: list[MemoryObj], producer_event_handle=None):
        block_ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        pipe = self.d2h
        with pipe.lock, DEV.device(self.device):
            if producer_event_handle is not None:
                pipe.kernel_stream.wait_event(producer_event_handle)
            begin = DEV.Event(enable_timing=True)
            begin.record(pipe.kernel_stream)
            for i, obj in enumerate(objs):
                buf = i % 3
                block_ids_in_chunk = block_ids[
                    i * self.blocks_per_chunk:(i + 1) * self.blocks_per_chunk]
                with DEV.stream(pipe.kernel_stream):
                    pipe.kernel_stream.wait_event(pipe.free[buf])
                    for layer, kv_cache in enumerate(self.kv_caches):
                        torch.index_select(kv_cache, 0, block_ids_in_chunk,
                                           out=pipe.layer_views[buf][layer])
                    pipe.ready[buf].record(pipe.kernel_stream)
                with DEV.stream(pipe.copy_stream):
                    pipe.copy_stream.wait_event(pipe.ready[buf])
                    obj.tensor.copy_(pipe.staging[buf], non_blocking=True)
                    pipe.free[buf].record(pipe.copy_stream)
            end = DEV.Event(enable_timing=True)
            end.record(pipe.copy_stream)
            completion = DEV.Event()
            completion.record(pipe.copy_stream)
        # Wait until every block is in host memory before replying.
        completion.synchronize()
        return begin.elapsed_time(end) / 1e3

    def load(self, block_ids, objs: list[MemoryObj], producer_event_handle=None,
             skip_blocks=0):
        block_ids = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        pipe = self.h2d
        with pipe.lock, DEV.device(self.device):
            if producer_event_handle is not None:
                pipe.copy_stream.wait_event(producer_event_handle)
            begin = DEV.Event(enable_timing=True)
            begin.record(pipe.copy_stream)
            for i, obj in enumerate(objs):
                buf = i % 3
                skip = skip_blocks if i == 0 else 0
                selected = self.blocks_per_chunk - skip
                if selected <= 0:
                    continue
                block_ids_in_chunk = block_ids[
                    i * self.blocks_per_chunk + skip:
                    (i + 1) * self.blocks_per_chunk]
                with DEV.stream(pipe.copy_stream):
                    pipe.copy_stream.wait_event(pipe.free[buf])
                    pipe.staging[buf].copy_(obj.tensor, non_blocking=True)
                    pipe.ready[buf].record(pipe.copy_stream)
                with DEV.stream(pipe.kernel_stream):
                    pipe.kernel_stream.wait_event(pipe.ready[buf])
                    for layer, kv_cache in enumerate(self.kv_caches):
                        kv_cache.index_copy_(0, block_ids_in_chunk,
                                             pipe.layer_views[buf][layer][skip:])
                    pipe.free[buf].record(pipe.kernel_stream)
            end = DEV.Event(enable_timing=True)
            end.record(pipe.kernel_stream)
            completion = DEV.Event()
            completion.record(pipe.kernel_stream)
        # Wait until every block is back in the GPU KV cache before replying.
        completion.synchronize()
        return begin.elapsed_time(end) / 1e3
