# SPDX-License-Identifier: Apache-2.0
"""Prefix-chained BLAKE3 hashing of token id chunks.

A prompt is split into fixed-size chunks (``chunk_size`` tokens).  Each
chunk's hash is computed over the hash of the *previous* chunk plus the
chunk's own tokens, so hash[i] uniquely identifies the full token prefix
up to chunk i.  This turns "does the cache contain this prefix?" into a
single dictionary lookup per chunk.
"""
import struct

import blake3


def hash_one(prefix_hash: bytes, tokens: list[int]) -> bytes:
    """Hash one chunk given the hash of the preceding prefix."""
    h = blake3.blake3(prefix_hash)
    h.update(struct.pack(f">{len(tokens)}I", *tokens))
    return h.digest()


#: Hash of the empty prefix; the chain starts here.
NONE_HASH: bytes = hash_one((0).to_bytes(8, byteorder="big", signed=True), (0,))


def chunk_hashes(token_ids: list[int], chunk_size: int) -> list[bytes]:
    """Return the prefix-chained hash of every whole chunk in ``token_ids``.

    A trailing partial chunk (fewer than ``chunk_size`` tokens) is ignored:
    only complete chunks are cached.
    """
    hashes = []
    prefix_hash = NONE_HASH
    for i in range(0, len(token_ids) - len(token_ids) % chunk_size, chunk_size):
        prefix_hash = hash_one(prefix_hash, token_ids[i : i + chunk_size])
        hashes.append(prefix_hash)
    return hashes
