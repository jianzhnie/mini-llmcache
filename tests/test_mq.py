# SPDX-License-Identifier: Apache-2.0
"""Tests for the ZMQ RPC layer."""

import pytest

from mini_llmcache.mq import MQClient, MQServer
from mini_llmcache.protocol import Req

ECHO = Req.GET_CHUNK_SIZE
FAIL = Req.END_SESSION


@pytest.fixture
def client_server(free_port):
    server = MQServer(f"tcp://127.0.0.1:{free_port}")

    def echo(payload):
        return payload

    def fail(payload):
        raise ValueError("boom")

    server.register(ECHO, echo)
    server.register(FAIL, fail)
    server.start()
    client = MQClient(f"tcp://127.0.0.1:{free_port}")
    yield client, server
    client.close()
    server.close()


def test_call_roundtrip(client_server):
    client, _ = client_server
    assert client.call(ECHO, {"hello": [1, 2, 3]}) == {"hello": [1, 2, 3]}


def test_async_submit_returns_future(client_server):
    client, _ = client_server
    future = client.submit(ECHO, "async")
    assert future.result() == "async"


def test_handler_exception_is_raised_on_client(client_server):
    client, _ = client_server
    with pytest.raises(ValueError, match="boom"):
        client.call(FAIL, None)


def test_unknown_request_raises_on_client(client_server):
    client, _ = client_server
    with pytest.raises(KeyError):
        client.call(Req.STORE, None)


def test_call_timeout_raises_when_server_is_missing(free_port):
    client = MQClient(f"tcp://127.0.0.1:{free_port}")
    try:
        with pytest.raises(TimeoutError):
            # Nothing is listening; the call must not hang forever.
            client.call(ECHO, None, timeout=1.0)
    finally:
        client.close()
