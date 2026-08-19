# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures and helpers."""
import socket
import time

import pytest


@pytest.fixture
def free_port() -> int:
    """An ephemeral localhost port (released before the test uses it)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> None:
    """Poll ``predicate`` until it returns truthy or the timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError("condition not met within timeout")
