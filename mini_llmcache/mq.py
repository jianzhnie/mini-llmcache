# SPDX-License-Identifier: Apache-2.0
"""A tiny request/reply RPC layer over ZeroMQ.

Each endpoint owns one socket plus a dedicated I/O thread.  Sends are
queued and flushed via a pipe wakeup; replies are matched to pending
``Future``s by an incrementing request id.  Server-side handler
exceptions are shipped back and re-raised on the client's future.
"""
import itertools
import os
import pickle
import queue
import threading
from concurrent.futures import Future
from typing import Any, Callable

import zmq

from mini_llmcache.protocol import Req


class SocketLoop:
    """Single-threaded event loop over one ZMQ socket + one wakeup pipe."""

    def __init__(self, sock: zmq.Socket):
        self.sock = sock
        self.send_queue: queue.SimpleQueue = queue.SimpleQueue()
        self.wake_read, self.wake_write = os.pipe()
        self.stopped = False
        self.thread = threading.Thread(target=self.run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def send(self, frames: list[bytes]) -> None:
        """Queue ``frames`` for delivery from the I/O thread."""
        self.send_queue.put(frames)
        os.write(self.wake_write, b"x")

    def close(self) -> None:
        self.stopped = True
        os.write(self.wake_write, b"x")
        self.thread.join()
        self.sock.close()
        os.close(self.wake_read)
        os.close(self.wake_write)

    def run(self) -> None:
        poller = zmq.Poller()
        poller.register(self.sock, zmq.POLLIN)
        poller.register(self.wake_read, zmq.POLLIN)
        while not self.stopped:
            ready = dict(poller.poll())
            if self.wake_read in ready:
                os.read(self.wake_read, 4096)
                while not self.send_queue.empty():
                    self.sock.send_multipart(self.send_queue.get())
            if self.sock in ready:
                self.on_recv(self.sock.recv_multipart())

    def on_recv(self, frames: list[bytes]) -> None:
        raise NotImplementedError


class MQClient(SocketLoop):
    """Dealer-side RPC client; ``call`` blocks for the reply."""

    def __init__(self, server_url: str):
        sock = zmq.Context.instance().socket(zmq.DEALER)
        sock.connect(server_url)
        super().__init__(sock)
        self.uid_counter = itertools.count()
        self.pending: dict[int, Future] = {}
        self.start()

    def submit(self, req: Req, payload: Any = None) -> Future:
        """Send ``req`` and return a future for the reply."""
        uid, fut = next(self.uid_counter), Future()
        self.pending[uid] = fut
        self.send([pickle.dumps((uid, req, payload))])
        return fut

    def call(self, req: Req, payload: Any = None,
             timeout: float | None = None) -> Any:
        """Send ``req`` and block for the reply.

        Raises whatever exception the server handler raised; ``timeout``
        (seconds) guards against a dead or unreachable server.
        """
        return self.submit(req, payload).result(timeout=timeout)

    def on_recv(self, frames: list[bytes]) -> None:
        uid, ok, value = pickle.loads(frames[0])
        if ok and value is None and len(frames) > 1:
            # Large byte payloads arrive as separate frames (see MQServer).
            value = [bytes(f) for f in frames[1:]]
        fut = self.pending.pop(uid, None)
        if fut is None:
            return  # unknown/stale reply — ignore
        if ok:
            fut.set_result(value)
        else:
            fut.set_exception(value)


class MQServer(SocketLoop):
    """Router-side RPC server; each request is handled in its own thread."""

    def __init__(self, bind_url: str):
        sock = zmq.Context.instance().socket(zmq.ROUTER)
        sock.bind(bind_url)
        super().__init__(sock)
        self.handlers: dict[Req, Callable] = {}

    def register(self, req: Req, fn: Callable) -> None:
        self.handlers[req] = fn

    def on_recv(self, frames: list[bytes]) -> None:
        identity, blob = frames
        threading.Thread(target=self.handle, args=(identity, blob),
                         daemon=True).start()

    def handle(self, identity: bytes, blob: bytes) -> None:
        """Run the handler and ship back ``(uid, ok, result-or-exception)``.

        A ``list[bytes]`` result (RETRIEVE chunks) travels as separate ZMQ
        frames so the hot path avoids pickling hundreds of megabytes.
        """
        uid = None
        try:
            uid, req, payload = pickle.loads(blob)
            result = self.handlers[req](payload)
            if _is_chunk_list(result):
                frames = [identity, pickle.dumps((uid, True, None))]
                frames.extend(result)
            else:
                frames = [identity, pickle.dumps((uid, True, result))]
            self.send(frames)
        except Exception as exc:  # noqa: BLE001 — surfaced to the client
            self.send([identity, pickle.dumps((uid, False, exc))])


def _is_chunk_list(result: Any) -> bool:
    return (isinstance(result, list) and bool(result)
            and all(isinstance(x, bytes) for x in result))
