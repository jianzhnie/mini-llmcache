# SPDX-License-Identifier: Apache-2.0
"""The MiniConnector: vLLM's KVConnector V1 hook implementation.

vLLM calls these hooks from its scheduler/worker at fixed points; the
connector answers three questions: how many prompt tokens can be skipped
(via the cache server's LOOKUP/QUERY), which blocks to load back into the
KV cache (RETRIEVE), and which computed blocks to save (STORE).  GPU<->host
copies happen here in the vLLM process (see l0/transfer.py); the cache
server only ever receives bytes.
"""

import threading
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

from mini_llmcache.l0 import kv_format
from mini_llmcache.l0.transfer import DeviceFuture, KVTransfer
from mini_llmcache.mq import MQClient
from mini_llmcache.protocol import (
    AckPayload,
    FreeLocksPayload,
    LoadStoreOp,
    LookupPayload,
    QueryPayload,
    RegisterPayload,
    Req,
    ReqMeta,
    TransferPayload,
)
from mini_llmcache.utils.device import get_device_module

#: Active accelerator namespace; raises a clear error on CPU-only machines.
device_module = get_device_module()


@dataclass
class RequestTracker:
    """Per-request state, threaded through the vLLM hook callbacks.

    The token "waterline" view:
    ``[0, vllm_hits)`` computed locally, ``[vllm_hits, lmcache_hits)`` must
    be loaded back from the cache (RETRIEVE), the rest is computed and then
    stored (STORE).
    """

    token_ids: list[int]
    lmcache_hits: int = 0
    vllm_hits: int = 0
    block_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    num_stored_tokens: int = 0
    lookup_resolved: bool = False
    load_pending: bool = False
    alloc_seen: bool = False


@dataclass
class MiniConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]


class MiniConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config: Any, role: Any, kv_cache_config: Any = None):
        super().__init__(vllm_config, role, kv_cache_config)
        self.vllm_config = vllm_config
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.client = MQClient(
            "tcp://{}:{}".format(
                extra.get("mini.host", "localhost"), extra.get("mini.port", 5555)
            )
        )
        # Fail fast with a clear message if the cache server is unreachable.
        try:
            self.chunk_size = self.client.call(Req.GET_CHUNK_SIZE, timeout=30)
        except Exception as exc:
            raise RuntimeError(
                "cannot reach the mini-llmcache cache server "
                f"(mini.host={extra.get('mini.host', 'localhost')}, "
                f"mini.port={extra.get('mini.port', 5555)}): {exc}"
            ) from exc
        self.block_size = vllm_config.cache_config.block_size
        assert self.chunk_size % self.block_size == 0, (
            "chunk_size must be a multiple of vLLM's block_size"
        )
        self.model = vllm_config.model_config.model
        self.world_size = vllm_config.parallel_config.tensor_parallel_size
        self.instance_id = uuid.uuid4().int & ((1 << 63) - 1)
        self.trackers: dict[str, RequestTracker] = {}
        self.store_futures: dict[str, list[DeviceFuture]] = {}
        self.load_futures: dict[str, DeviceFuture] = {}
        self.pending_sends: set[str] = set()
        self.transfer: KVTransfer | None = None

    # ---- op generation -------------------------------------------------

    def generate_retrieve_op(self, tracker: RequestTracker) -> LoadStoreOp | None:
        """One RETRIEVE op covering the cached tokens we haven't computed."""
        if not tracker.load_pending:
            return None
        tracker.load_pending = False
        start = tracker.vllm_hits // self.chunk_size * self.chunk_size
        end = tracker.lmcache_hits
        return LoadStoreOp(
            tracker.token_ids[:end],
            tracker.block_ids[start // self.block_size : end // self.block_size],
            start,
            end,
            tracker.vllm_hits - start,
        )

    def generate_store_op(self, tracker: RequestTracker) -> LoadStoreOp | None:
        """One STORE op for the newly computed whole chunks, if any."""
        available = min(
            len(tracker.token_ids),
            len(tracker.block_ids) * self.block_size,
            tracker.num_computed_tokens,
        )
        num_chunks = (available - tracker.num_stored_tokens) // self.chunk_size
        if num_chunks <= 0:
            return None
        start = tracker.num_stored_tokens
        end = start + num_chunks * self.chunk_size
        tracker.num_stored_tokens = end
        return LoadStoreOp(
            tracker.token_ids[:end],
            tracker.block_ids[start // self.block_size : end // self.block_size],
            start,
            end,
        )

    # ---- vLLM hooks ----------------------------------------------------

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        """Tell vLLM how many prompt tokens the cache has pre-computed.

        First call submits LOOKUP (async); later calls poll
        QUERY_PREFETCH_STATUS.  Returns (n, True) to skip n tokens,
        (0, False) for "no hits", (None, True) while still pending.
        """
        tracker = self.trackers.get(request.request_id)
        if tracker is None:
            tracker = RequestTracker(request.all_token_ids)
            self.trackers[request.request_id] = tracker
            prompt = request.prompt_token_ids
            # Whole chunks only; chunk_hashes ignores a trailing partial
            # chunk anyway, so no need to drop one here.
            num_lookup = len(prompt) // self.chunk_size * self.chunk_size
            self.client.submit(
                Req.LOOKUP,
                LookupPayload(
                    request.request_id, prompt[:num_lookup], self.model, self.world_size
                ),
            )
        elif tracker.lookup_resolved:
            return 0, False
        try:
            hits = self.client.call(
                Req.QUERY_PREFETCH_STATUS,
                QueryPayload(request.request_id, self.world_size),
                timeout=30,
            )
        except Exception:
            tracker.lookup_resolved = True
            return 0, False
        if hits is None:
            return None, True
        tracker.lookup_resolved = True
        tracker.lmcache_hits = hits
        tracker.vllm_hits = num_computed_tokens
        tracker.num_stored_tokens = hits
        if hits > num_computed_tokens:
            return hits - num_computed_tokens, True
        return 0, False

    def update_state_after_alloc(
        self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Record allocated blocks and free lookup locks we won't need."""
        if not (tracker := self.trackers.get(request.request_id)):
            return
        group_blocks = blocks.get_block_ids()[0]
        tracker.block_ids.extend(group_blocks[len(tracker.block_ids) :])
        if num_external_tokens > 0:
            tracker.load_pending = True
        if tracker.alloc_seen:
            return
        tracker.alloc_seen = True
        lmcache_chunks = tracker.lmcache_hits // self.chunk_size
        free_until = (
            min(tracker.vllm_hits // self.chunk_size, lmcache_chunks)
            if tracker.load_pending
            else lmcache_chunks
        )
        if free_until:
            self.client.submit(
                Req.FREE_LOOKUP_LOCKS,
                FreeLocksPayload(request.request_id, free_until, self.world_size),
            )

    def build_connector_meta(self, scheduler_output: Any) -> MiniConnectorMetadata:
        """Update trackers from the scheduler output and queue transfer ops."""
        metas = []
        for req_id, tracker in self.trackers.items():
            op = self.generate_retrieve_op(tracker)
            if op is not None:
                metas.append(ReqMeta(req_id, Req.RETRIEVE, op))
        for new_req in scheduler_output.scheduled_new_reqs:
            tracker = self.trackers.get(new_req.req_id)
            if tracker is not None:
                tracker.block_ids = list(new_req.block_ids[0])
                tracker.num_computed_tokens = new_req.num_computed_tokens
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            tracker = self.trackers.get(req_id)
            if tracker is None:
                continue
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                if req_id in cached.resumed_req_ids:
                    # Preemption: the block table was rebuilt from scratch.
                    tracker.block_ids = list(new_blocks[0])
                else:
                    tracker.block_ids.extend(new_blocks[0])
            tracker.num_computed_tokens = cached.num_computed_tokens[i]
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            tracker = self.trackers.get(req_id)
            if tracker is None:
                continue
            tracker.num_computed_tokens += scheduled
            op = self.generate_store_op(tracker)
            if op is not None:
                metas.append(ReqMeta(req_id, Req.STORE, op))
        return MiniConnectorMetadata(metas)

    def request_finished(
        self, request: Any, _block_ids: list[int]
    ) -> tuple[bool, None]:
        self.trackers.pop(request.request_id, None)
        self.client.submit(Req.END_SESSION, request.request_id)
        return True, None

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Build the local transfer engine and announce cache geometry.

        Upstream vLLM passes one fused tensor per layer; vllm-ascend passes
        a (k_cache, v_cache) tuple per layer.  Flatten either way — each
        tensor becomes one "layer" for the transfer pipelines.  The tensors
        stay in this process: all GPU<->host copies happen here, the cache
        server only ever sees bytes (Ascend NPU cannot share device memory
        across processes, matching upstream LMCache-Ascend).
        """
        layers = []
        for _, cache in sorted(kv_caches.items()):
            if isinstance(cache, tuple):
                layers.extend(cache)
            else:
                layers.append(cache)
        views = kv_format.normalize(
            layers, self.vllm_config.cache_config.num_gpu_blocks
        )
        self.transfer = KVTransfer(
            views, self.block_size, self.chunk_size, get_tensor_model_parallel_rank()
        )
        self.client.call(
            Req.REGISTER_KV_CACHE,
            RegisterPayload(
                instance_id=self.instance_id,
                model=self.model,
                rank=self.transfer.rank,
                world_size=self.world_size,
                block_size=self.block_size,
                chunk_nbytes=self.transfer.chunk_nbytes,
            ),
            timeout=30,
        )

    def submit_transfers(self, req: Req) -> dict[str, DeviceFuture]:
        """Execute the queued ops for one Req type (STORE or RETRIEVE)."""
        ops = [m for m in self._get_connector_metadata().requests if m.req == req]
        if not ops:
            return {}
        if self.transfer is None:
            raise RuntimeError("register_kv_caches was not called")
        device = device_module.current_device()
        out: dict[str, DeviceFuture] = {}
        for m in ops:
            if req == Req.STORE:
                # D2H copy happens here in the vLLM worker; the cache server
                # only receives the resulting bytes.
                chunks, elapsed = self.transfer.to_host(m.op.block_ids)
                nbytes = len(chunks) * self.transfer.chunk_nbytes
                future = self.client.submit(
                    req,
                    TransferPayload(
                        m.request_id,
                        self.instance_id,
                        m.op,
                        chunks=chunks,
                        elapsed=elapsed,
                        nbytes=nbytes,
                    ),
                )
                out[m.request_id] = DeviceFuture(future, device)
            else:  # RETRIEVE: fetch bytes, then scatter them into the KV
                # cache in a worker thread so the forward isn't blocked.
                future = self.client.submit(
                    req, TransferPayload(m.request_id, self.instance_id, m.op)
                )
                done = threading.Event()
                threading.Thread(
                    target=self._finish_load,
                    args=(m.request_id, m.op, future, done),
                    daemon=True,
                ).start()
                out[m.request_id] = DeviceFuture(future, device, done_event=done)
        return out

    def _finish_load(
        self, request_id: str, op: LoadStoreOp, future: Future, done: threading.Event
    ) -> None:
        """Worker thread: scatter the fetched bytes into the KV cache."""
        try:
            device_module.set_device(self.transfer.device.index)
            chunks = future.result()
            if chunks is None:
                raise RuntimeError("cache server returned no chunks")
            elapsed = self.transfer.from_host(
                chunks,
                op.block_ids,
                skip_blocks=op.skip_first_n_tokens // self.block_size,
            )
            nbytes = len(chunks) * self.transfer.chunk_nbytes
            self.client.submit(
                Req.TRANSFER_ACK, AckPayload(request_id, "RETRIEVE", nbytes, elapsed)
            )
        except Exception as exc:
            print(f"RETRIEVE failed rid={request_id}: {exc}", flush=True)
        finally:
            done.set()

    def start_load_kv(self, _forward_context: Any, **_kwargs: Any) -> None:
        # Called before the forward pass; loads run in worker threads.
        self.load_futures.update(self.submit_transfers(Req.RETRIEVE))

    def wait_for_save(self) -> None:
        # Called after the forward pass; the D2H copy happens inline.
        for req_id, future in self.submit_transfers(Req.STORE).items():
            self.store_futures.setdefault(req_id, []).append(future)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Report requests whose async store/load has fully completed."""
        self.pending_sends |= finished_req_ids
        for req_id, futures in list(self.store_futures.items()):
            live = [f for f in futures if not f.done()]
            if live:
                self.store_futures[req_id] = live
            else:
                del self.store_futures[req_id]
        finished_sending = {
            req_id for req_id in self.pending_sends if req_id not in self.store_futures
        }
        self.pending_sends -= finished_sending
        finished_recving = {
            req_id for req_id, future in self.load_futures.items() if future.done()
        }
        for req_id in finished_recving:
            del self.load_futures[req_id]
        return finished_sending, finished_recving

    def wait_for_layer_load(self, _layer_name: str) -> None:
        """Unused: this connector takes the whole-chunk transfer route
        (build_connector_meta + submit_transfers), not layer-by-layer."""
        pass

    def save_kv_layer(
        self, _layer_name: str, _kv_layer: Any, _attn_metadata: Any, **_kwargs: Any
    ) -> None:
        """Unused: see wait_for_layer_load."""
        pass

    def shutdown(self) -> None:
        self.client.call(Req.UNREGISTER_KV_CACHE, self.instance_id)
        self.client.close()
