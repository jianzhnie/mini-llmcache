# SPDX-License-Identifier: Apache-2.0
"""Shared utilities: prefix hashing and accelerator backend selection."""
from mini_llmcache.utils.device import (
    get_device_module,
    get_device_type,
    is_device_available,
    set_device,
)
from mini_llmcache.utils.hasher import NONE_HASH, chunk_hashes, hash_one

__all__ = [
    "NONE_HASH",
    "chunk_hashes",
    "get_device_module",
    "get_device_type",
    "hash_one",
    "is_device_available",
    "set_device",
]
