# SPDX-License-Identifier: Apache-2.0
"""Break down the warm-path latency by talking to the cache server directly.

Registers a probe instance (with the real chunk size so the L1 deployment
table stays intact), times LOOKUP resolution and a full RETRIEVE roundtrip
for a prefix the live engine has already cached.

    python benchmarks/probe.py --tokenizer /path/to/model \
        --model /path/to/model --chunk-mb 32 --prompt-chunks 12
"""
import argparse
import time
from typing import Any

from mini_llmcache.mq import MQClient
from mini_llmcache.protocol import (
    LoadStoreOp,
    LookupPayload,
    QueryPayload,
    RegisterPayload,
    Req,
    TransferPayload,
)
from mini_llmcache.utils.hasher import chunk_hashes
from transformers import AutoTokenizer

PROBE_INSTANCE = 2**63 - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model", required=True,
                        help="exact model name the engine registered with")
    parser.add_argument("--chunk-mb", type=float, required=True,
                        help="chunk size in MiB (from the REGISTER log line)")
    parser.add_argument("--prompt", required=True,
                        help="prompt already cached by the running engine")
    parser.add_argument("--world-size", type=int, default=1)
    args: Any = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    client = MQClient("tcp://127.0.0.1:45881")
    client.call(Req.REGISTER_KV_CACHE, RegisterPayload(
        instance_id=PROBE_INSTANCE, model=args.model, rank=0,
        world_size=args.world_size, block_size=128,
        chunk_nbytes=int(args.chunk_mb * (1 << 20))), timeout=10)

    # Try both tokenization conventions; vLLM uses the raw prompt ids.
    for add_special in (False, True):
        tokens = tok(args.prompt, add_special_tokens=add_special)["input_ids"]
        rid = f"probe-{add_special}"
        client.call(Req.LOOKUP, LookupPayload(rid, tokens, args.model,
                                              args.world_size), timeout=10)
        t0 = time.perf_counter()
        while True:
            hits = client.call(Req.QUERY_PREFETCH_STATUS,
                               QueryPayload(rid, args.world_size), timeout=10)
            if hits is not None:
                break
        lookup_ms = (time.perf_counter() - t0) * 1e3
        print(f"LOOKUP resolve (add_special={add_special}): "
              f"{lookup_ms:.1f} ms, hits={hits} tokens")
        if hits == 0:
            client.call(Req.END_SESSION, rid)
            continue

        op = LoadStoreOp(tokens, list(range(1, 1 + hits // 128)),
                         0, hits, 0)
        t0 = time.perf_counter()
        chunks = client.call(Req.RETRIEVE, TransferPayload(
            rid, PROBE_INSTANCE, op), timeout=120)
        ms = (time.perf_counter() - t0) * 1e3
        # END_SESSION only after RETRIEVE: it releases the prefetch-held
        # locks, which frees temporary (L2-loaded) entries.
        client.call(Req.END_SESSION, rid)
        total_mb = sum(len(c) for c in chunks) / 1e6
        print(f"RETRIEVE: {len(chunks)} chunks, {total_mb:.0f} MB in "
              f"{ms:.1f} ms -> {total_mb / ms * 1e3:.0f} MB/s "
              f"(roundtrip incl. serialize + ZMQ + server)")
        break

    client.call(Req.UNREGISTER_KV_CACHE, PROBE_INSTANCE, timeout=10)
    client.close()


if __name__ == "__main__":
    main()
