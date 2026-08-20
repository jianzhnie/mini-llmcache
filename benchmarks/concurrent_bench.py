# SPDX-License-Identifier: Apache-2.0
"""Scenario 7 (manual §7): concurrent clients sharing one prefix.

Sends ``workers`` requests at once (all sharing the same long prefix,
each with a distinct suffix), repeated for ``rounds`` rounds.  Reports
the total wall time, the per-request TTFT distribution (p50/p90), and
the hit evidence from the cache server log.

    python benchmarks/concurrent.py --url http://localhost:8000 \\
        --model Qwen/Qwen3-32B --tokenizer /path/to/model \\
        --server-log /tmp/mini-server.log --workers 4 --rounds 5
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer, PreTrainedTokenizer

from benchmarks.bench import p50, p90, server_hits
from benchmarks.datasets import DOCUMENTS, repeat_to_tokens

MAX_OUTPUT_TOKENS = 24
CHUNK_TOKENS = 256


def complete(url: str, model: str, prompt: str) -> tuple[float, float, dict]:
    """One streaming completion; returns (ttft, total, {"id","text"})."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "prompt": prompt,
        "stream": True,
    }
    t0 = time.perf_counter()
    text, rid, ttft = "", "", None
    with requests.post(f"{url}/v1/completions", json=payload, timeout=600,
                       stream=True) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if not rid:
                rid = chunk.get("id", "")
            piece = chunk["choices"][0].get("text", "")
            if ttft is None and piece:
                ttft = time.perf_counter() - t0
            text += piece
    return ttft or (time.perf_counter() - t0), time.perf_counter() - t0, {
        "id": rid,
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--server-log", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--chunks", type=int, default=12)
    args: Any = parser.parse_args()

    tok: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    server_log = Path(args.server_log) if args.server_log else None
    prefix = repeat_to_tokens(tok, DOCUMENTS["birds"],
                              args.chunks * CHUNK_TOKENS)
    prompts = [prefix + f" Question {i}. Answer:" for i in range(args.workers)]

    # Warm the prefix once (serial) so concurrent rounds hit L1.
    complete(args.url, args.model, prompts[0])
    print(f"== 场景 7 · 并发共享前缀({args.workers} workers × "
          f"{args.rounds} rounds,前缀 {args.chunks * CHUNK_TOKENS} tokens)==")

    round_times, all_ttfts, hit_count, total_reqs = [], [], 0, 0
    for rnd in range(args.rounds):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(
                lambda p: complete(args.url, args.model, p), prompts))
        round_times.append(time.perf_counter() - t0)
        for ttft, _, resp in results:
            all_ttfts.append(ttft)
            total_reqs += 1
            if "hit L1=" in server_hits(server_log, resp):
                hit_count += 1
        print(f"  round {rnd + 1}: {round_times[-1]:.2f}s")

    serial_estimate = sum(round_times)  # rounds are serial; workers concurrent
    print(f"  每轮并发耗时 p50={p50(round_times):.2f}s "
          f"p90={p90(round_times):.2f}s")
    print(f"  单请求 TTFT p50={p50(all_ttfts):.3f}s "
          f"p90={p90(all_ttfts):.3f}s")
    print(f"  命中率 {hit_count}/{total_reqs}(应全部命中)")
    print(f"  参考:串行同数量(场景 5 口径)约 "
          f"{serial_estimate:.1f}s,并发为 {sum(round_times):.1f}s"
          f"——并发加速 {serial_estimate / sum(round_times):.1f}x")


if __name__ == "__main__":
    main()
