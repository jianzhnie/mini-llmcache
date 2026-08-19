# SPDX-License-Identifier: Apache-2.0
"""Wire protocol: message types and payload dataclasses shared by the
vLLM connector and the cache server (serialized with pickle over ZMQ).
"""
import enum
from dataclasses import dataclass, field


class Req(enum.IntEnum):
    """Request types understood by the cache server."""

    GET_CHUNK_SIZE = 1
    REGISTER_KV_CACHE = 2
    UNREGISTER_KV_CACHE = 3
    LOOKUP = 4
    QUERY_PREFETCH_STATUS = 5
    RETRIEVE = 6
    STORE = 7
    FREE_LOOKUP_LOCKS = 8
    END_SESSION = 9
    TRANSFER_ACK = 10


@dataclass(frozen=True)
class ChunkKey:
    """Identity of one cacheable chunk: content hash + model + TP rank."""

    chunk_hash: bytes
    model: str
    rank: int


@dataclass
class LoadStoreOp:
    """One chunk-aligned load/store operation over a token range.

    ``token_ids`` is the request's token list up to ``end``;
    ``block_ids`` maps the token range onto GPU KV blocks;
    ``skip_first_n_tokens`` marks a prefix of the first chunk that must
    not be overwritten (it was already computed locally).
    """

    token_ids: list[int]
    block_ids: list[int]
    start: int
    end: int
    skip_first_n_tokens: int = 0


@dataclass
class ReqMeta:
    """A queued transfer for one request, as carried by connector metadata."""

    request_id: str
    req: Req
    op: LoadStoreOp


@dataclass
class RegisterPayload:
    """Sent once per engine to announce its cache geometry."""

    instance_id: int
    model: str
    rank: int
    world_size: int
    block_size: int
    chunk_nbytes: int


@dataclass
class LookupPayload:
    """Async prefetch request for a prompt prefix."""

    request_id: str
    token_ids: list[int]
    model: str
    world_size: int


@dataclass
class TransferPayload:
    """A STORE or RETRIEVE message.

    STORE carries ``chunks`` (the bytes copied out of the GPU in the vLLM
    process) plus the connector-side timing; RETRIEVE is sent empty and the
    server replies with the chunk bytes.
    """

    request_id: str
    instance_id: int
    op: LoadStoreOp
    chunks: list[bytes] = field(default_factory=list)
    elapsed: float = 0.0  # connector-side transfer seconds
    nbytes: int = 0


@dataclass
class AckPayload:
    """Fire-and-forget transfer completion notice (for logging)."""

    request_id: str
    kind: str  # "STORE" | "RETRIEVE"
    nbytes: int
    elapsed: float


@dataclass
class QueryPayload:
    """Poll for prefetch resolution."""

    request_id: str
    world_size: int


@dataclass
class FreeLocksPayload:
    """Release the first ``num_chunks`` held lookup locks."""

    request_id: str
    num_chunks: int
    world_size: int
