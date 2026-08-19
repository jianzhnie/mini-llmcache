# SPDX-License-Identifier: Apache-2.0
"""Tests for prefix-chained chunk hashing."""
from mini_llmcache.hasher import chunk_hashes


def test_chunk_alignment_and_partial_tail():
    tokens = list(range(10))
    hashes = chunk_hashes(tokens, chunk_size=4)
    # 10 tokens -> two whole chunks; the 2-token tail is ignored.
    assert len(hashes) == 2


def test_prefix_chaining_is_content_dependent():
    """The same prefix hashes identically regardless of what follows."""
    prefix = list(range(16))
    a = chunk_hashes(prefix + [100, 101], chunk_size=4)
    b = chunk_hashes(prefix + [999, 999], chunk_size=4)
    assert a == b


def test_chunk_hash_depends_on_its_own_prefix():
    """hash[i] folds in the full prefix: same first chunk hashes equal,
    but every later chunk changes once an earlier chunk differs."""
    a = chunk_hashes(list(range(16)), chunk_size=4)
    b = chunk_hashes(list(range(4)) + [9, 9, 9, 9] + list(range(8, 16)),
                     chunk_size=4)
    assert a[0] == b[0]  # identical first chunk
    assert a[1:] != b[1:]  # prefix differs from chunk 1 onward


def test_empty_and_short_inputs():
    assert chunk_hashes([], chunk_size=4) == []
    assert chunk_hashes([1, 2, 3], chunk_size=4) == []


def test_deterministic():
    tokens = list(range(32))
    assert chunk_hashes(tokens, chunk_size=8) == chunk_hashes(tokens, 8)
