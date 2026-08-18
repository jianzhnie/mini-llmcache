# SPDX-License-Identifier: Apache-2.0
import argparse
import json
import threading
from dataclasses import dataclass

from mini_llmcache.hasher import chunk_hashes
from mini_llmcache.l2.fs import FSAdapter
from mini_llmcache.l2.mock import MockAdapter
from mini_llmcache.mq import MQServer
from mini_llmcache.protocol import ChunkKey, Req
from mini_llmcache.storage_manager import StorageManager


@dataclass
class EngineInstance:
    rank: int
    model: str
    world_size: int
    block_size: int
    chunk_nbytes: int


class CacheServer:
    def __init__(self, bind_url, chunk_size, capacity_bytes, l2s=()):
        self.chunk_size = chunk_size
        self.sm = StorageManager(capacity_bytes, l2s)
        self.mq = MQServer(bind_url)
        self.register_lock = threading.Lock()
        self.deployments: dict[tuple[str, int], int] = {}
        self.instances: dict[int, EngineInstance] = {}
        for req, fn in [
            (Req.GET_CHUNK_SIZE, self.get_chunk_size),
            (Req.REGISTER_KV_CACHE, self.register),
            (Req.UNREGISTER_KV_CACHE, self.unregister),
            (Req.LOOKUP, self.lookup),
            (Req.QUERY_PREFETCH_STATUS, self.query),
            (Req.STORE, self.store),
            (Req.RETRIEVE, self.retrieve),
            (Req.FREE_LOOKUP_LOCKS, self.free_locks),
            (Req.END_SESSION, self.end_session),
            (Req.TRANSFER_ACK, self.ack),
        ]:
            self.mq.register(req, fn)

    def start(self):
        self.mq.start()

    def get_chunk_size(self, payload):
        return self.chunk_size

    def register(self, payload):
        with self.register_lock:
            self.deployments[(payload.model, payload.world_size)] = \
                payload.chunk_nbytes
            self.instances[payload.instance_id] = EngineInstance(
                payload.rank, payload.model, payload.world_size,
                payload.block_size, payload.chunk_nbytes)
        print(f"REGISTER {payload.model} rank {payload.rank}/"
              f"{payload.world_size} (chunk={payload.chunk_nbytes >> 20} MiB, "
              f"{len(self.instances)} engines)", flush=True)
        return True

    def unregister(self, instance_id):
        with self.register_lock:
            return self.instances.pop(instance_id, None) is not None

    def keys_for(self, hashes, model, ranks):
        return [ChunkKey(h, model, r) for h in hashes for r in ranks]

    def op_keys(self, op, instance):
        hashes = chunk_hashes(op.token_ids[:op.end], self.chunk_size)
        return self.keys_for(hashes[op.start // self.chunk_size:],
                             instance.model, [instance.rank])

    def lookup(self, payload):
        chunk_nbytes = self.deployments.get((payload.model, payload.world_size))
        keys = [] if chunk_nbytes is None else self.keys_for(
            chunk_hashes(payload.token_ids, self.chunk_size),
            payload.model, range(payload.world_size))
        self.sm.prefetch.start_session(payload.request_id, keys, chunk_nbytes)
        return True

    def query(self, payload):
        resolved = self.sm.prefetch.query(payload.request_id)
        if resolved is None:
            return None
        l1_keys, l2_keys, _ = resolved
        return ((l1_keys + l2_keys) // payload.world_size) * self.chunk_size

    def free_locks(self, payload):
        self.sm.prefetch.release_first(
            payload.request_id, payload.num_chunks * payload.world_size)
        return True

    def end_session(self, request_id):
        self.sm.prefetch.end_session(request_id)
        return True

    def store(self, payload):
        op = payload.op
        instance = self.instances[payload.instance_id]
        keys = self.op_keys(op, instance)
        reserved = self.sm.l1.reserve_write(keys, instance.chunk_nbytes)
        written, objs, blobs = [], [], []
        for chunk, key in enumerate(keys):
            if reserved[key] is not None:
                written.append(key)
                objs.append(reserved[key])
                blobs.append(payload.chunks[chunk])
        for obj, blob in zip(objs, blobs):
            obj.byte_array[:] = blob
        self.sm.l1.finish_write(written)
        nbytes = len(objs) * instance.chunk_nbytes
        elapsed = payload.elapsed
        gbps = nbytes / elapsed / 1e9 if elapsed > 0 else 0.0
        print(f"STORE rid={payload.request_id} tokens [{op.start}, {op.end}) "
              f"L0->L1 {gbps:.1f} GB/s", flush=True)
        return True

    def retrieve(self, payload):
        op = payload.op
        instance = self.instances[payload.instance_id]
        keys = self.op_keys(op, instance)
        l1_keys, l2_keys, l2_gbps = self.sm.prefetch.query(payload.request_id)
        objs = self.sm.l1.read(keys)
        chunks = [bytes(obj.byte_array) for obj in objs]
        self.sm.prefetch.release(payload.request_id, keys)
        world_size = instance.world_size
        print(f"RETRIEVE rid={payload.request_id} tokens [{op.start}, {op.end}) "
              f"hit L1={l1_keys // world_size} L2={l2_keys // world_size} | "
              f"L2->L1 {l2_gbps:.1f} GB/s | {len(chunks)} chunks sent",
              flush=True)
        return chunks

    def ack(self, payload):
        gbps = payload.nbytes / payload.elapsed / 1e9 \
            if payload.elapsed > 0 else 0.0
        print(f"{payload.kind} rid={payload.request_id} "
              f"L1->L0 {gbps:.1f} GB/s", flush=True)
        return True


REGISTRY = {"fs": FSAdapter, "mock": MockAdapter}


def build_l2s(specs):
    l2s = []
    for spec in map(json.loads, specs):
        l2s.append(REGISTRY[spec.pop("type")](**spec))
    return l2s


def serve(bind_url, chunk_size, capacity_bytes, l2_specs=()):
    l2s = build_l2s(l2_specs)
    CacheServer(bind_url, chunk_size, capacity_bytes, l2s).start()
    print(f"mini cache server up and ready on {bind_url} "
          f"(chunk_size={chunk_size}, {capacity_bytes >> 30} GB L1, "
          f"{len(l2s)} L2 adapters)", flush=True)
    threading.Event().wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--l1-size-gb", type=float, default=20.0)
    parser.add_argument("--l2-adapter", action="append", default=[])
    args = parser.parse_args()
    serve(f"tcp://{args.host}:{args.port}", args.chunk_size,
          int(args.l1_size_gb * (1 << 30)), args.l2_adapter)


if __name__ == "__main__":
    main()
