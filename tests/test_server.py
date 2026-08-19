# SPDX-License-Identifier: Apache-2.0
"""End-to-end test of the cache server over a real ZMQ connection.

The server is pure CPU, so this exercises the full handler stack
(register -> lookup -> store -> retrieve -> end session) without any
GPU or vLLM involvement.
"""

import pytest
from conftest import wait_until

from mini_llmcache.mq import MQClient
from mini_llmcache.protocol import (
    FreeLocksPayload,
    LoadStoreOp,
    LookupPayload,
    QueryPayload,
    RegisterPayload,
    Req,
    TransferPayload,
)
from mini_llmcache.server import CacheServer

CHUNK_SIZE = 16
CHUNK_NBYTES = 64
MODEL = "test-model"


def start_server(free_port, l2s=None):
    server = CacheServer(f"tcp://127.0.0.1:{free_port}", CHUNK_SIZE, 1 << 20, l2s or [])
    server.start()
    return server


def register_instance(client, instance_id=1):
    assert (
        client.call(
            Req.REGISTER_KV_CACHE,
            RegisterPayload(
                instance_id=instance_id,
                model=MODEL,
                rank=0,
                world_size=1,
                block_size=8,
                chunk_nbytes=CHUNK_NBYTES,
            ),
            timeout=10,
        )
        is True
    )


def make_op(tokens, block_ids, start, end):
    return LoadStoreOp(tokens, block_ids, start, end)


def test_full_cycle_store_then_retrieve(free_port):
    server = start_server(free_port)
    client = MQClient(f"tcp://127.0.0.1:{free_port}")
    try:
        assert client.call(Req.GET_CHUNK_SIZE, None) == CHUNK_SIZE
        register_instance(client)

        tokens = list(range(16))
        op = make_op(tokens, [0, 1], 0, 16)
        rid = "req-1"

        # LOOKUP against an empty cache resolves to zero hits.
        client.submit(Req.LOOKUP, LookupPayload(rid, tokens, MODEL, 1))
        wait_until(
            lambda: client.call(Req.QUERY_PREFETCH_STATUS, QueryPayload(rid, 1))
            is not None
        )
        assert client.call(Req.QUERY_PREFETCH_STATUS, QueryPayload(rid, 1)) == 0

        # STORE one chunk of bytes, then RETRIEVE it back verbatim.
        payload = TransferPayload(
            rid,
            1,
            op,
            chunks=[b"\x5a" * CHUNK_NBYTES],
            elapsed=0.01,
            nbytes=CHUNK_NBYTES,
        )
        assert client.call(Req.STORE, payload) is True
        assert client.call(Req.RETRIEVE, TransferPayload(rid, 1, op)) == [
            b"\x5a" * CHUNK_NBYTES
        ]

        # Freeing locks and ending the session are no-ops that succeed.
        assert client.call(Req.FREE_LOOKUP_LOCKS, FreeLocksPayload(rid, 1, 1)) is True
        assert client.call(Req.END_SESSION, rid) is True
        assert client.call(Req.UNREGISTER_KV_CACHE, 1) is True
    finally:
        client.close()
        server.mq.close()


def test_lookup_hits_after_store(free_port):
    """A second lookup for the same tokens must report a full hit."""
    server = start_server(free_port)
    client = MQClient(f"tcp://127.0.0.1:{free_port}")
    try:
        register_instance(client)
        tokens = list(range(16))
        op = make_op(tokens, [0, 1], 0, 16)

        rid1, rid2 = "req-1", "req-2"
        client.submit(Req.LOOKUP, LookupPayload(rid1, tokens, MODEL, 1))
        client.call(
            Req.STORE,
            TransferPayload(
                rid1,
                1,
                op,
                chunks=[b"\x11" * CHUNK_NBYTES],
                elapsed=0.01,
                nbytes=CHUNK_NBYTES,
            ),
        )

        client.submit(Req.LOOKUP, LookupPayload(rid2, tokens, MODEL, 1))
        wait_until(
            lambda: client.call(Req.QUERY_PREFETCH_STATUS, QueryPayload(rid2, 1))
            is not None
        )
        assert (
            client.call(Req.QUERY_PREFETCH_STATUS, QueryPayload(rid2, 1)) == CHUNK_SIZE
        )
    finally:
        client.close()
        server.mq.close()


def test_unregistered_instance_is_rejected(free_port):
    server = start_server(free_port)
    client = MQClient(f"tcp://127.0.0.1:{free_port}")
    try:
        op = make_op(list(range(16)), [0, 1], 0, 16)
        with pytest.raises(KeyError):
            client.call(Req.STORE, TransferPayload("rid", 42, op, chunks=[b"x" * 64]))
    finally:
        client.close()
        server.mq.close()
