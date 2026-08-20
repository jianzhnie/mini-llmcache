# SPDX-License-Identifier: Apache-2.0
"""Benchmark mini-llmcache speedups against a running vLLM server.

Run inside the environment where the cache server and vllm serve are up:

    python benchmarks/bench.py \\
        --url http://localhost:8000 \\
        --model Qwen/Qwen3-0.6B \\
        --tokenizer /path/to/model/dir \\
        --server-log /tmp/mini-server.log

The script drives four datasets (see benchmarks/datasets.py and
benchmarks/datasets_hf.py) and prints per-request latency plus the cache
server's own RETRIEVE hit evidence (grepped from the server log by request
id).  Requests are streamed so time-to-first-token — the part the cache
actually accelerates (prefill + transfer) — can be separated from decode
time (a fixed cost on both sides).  Temperature 0 keeps outputs
deterministic so cached and uncached runs can be compared byte for byte.

For honest numbers, start with a FRESH cache server (empty L1 and L2).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer, PreTrainedTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.datasets import (
    CHUNK_TOKENS,
    SUFFIXES,
    make_exact_repeat,
    make_no_reuse,
    make_shared_prefix,
)
from benchmarks.datasets_hf import load_100

MAX_OUTPUT_TOKENS = 24
LONG_CHUNKS = 12  # 3072-token prompts make prefill time meaningful


def complete(url: str, model: str, prompt: str) -> tuple[float, float, dict]:
    """POST one completion; returns (ttft_s, total_s, {"id", "text"})."""
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "prompt": prompt,
        "stream": True,
    }
    t0 = time.perf_counter()
    text, rid, ttft = "", "", None
    with requests.post(
        f"{url}/v1/completions", json=payload, timeout=600, stream=True
    ) as resp:
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
    total = time.perf_counter() - t0
    return ttft or total, total, {"id": rid, "text": text}


def server_hits(server_log: Path | None, response: dict) -> str:
    """Extract the RETRIEVE hit line for this request from the server log."""
    if server_log is None or not server_log.exists():
        return ""
    rid = response.get("id", "")
    ack = ""
    for line in reversed(server_log.read_text(errors="ignore").splitlines()):
        if "RETRIEVE" in line and rid in line:
            # Prefer the line carrying the hit counts over the ack line.
            if "hit L" in line:
                return line.split("] ", 1)[-1]
            ack = line.split("] ", 1)[-1]
    return ack or "(no RETRIEVE — computed from scratch)"


def pct(values: list[float], q: float) -> float:
    """Percentile of a sample (linear interpolation, numpy-free)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def p50(values: list[float]) -> float:
    return pct(values, 0.5)


def p90(values: list[float]) -> float:
    return pct(values, 0.9)


def summarize(name: str, cold: list[float], hot: list[float], hits: str,
              consistent: bool = True) -> None:
    """Print the manual's summary row: samples, p50/p90, speedup."""
    speedup = p50(cold) / p50(hot) if hot and p50(hot) > 0 else 0.0
    print(f"{name}: cold n={len(cold)} p50={p50(cold):.3f}s p90={p90(cold):.3f}s | "
          f"hot n={len(hot)} p50={p50(hot):.3f}s p90={p90(hot):.3f}s | "
          f"speedup {speedup:.2f}x | {hits} | consistent={'✓' if consistent else '✗'}")
    return speedup


def print_row(
    name: str, phase: str, ttft: float, total: float, hits: str, speedup: str = "-"
) -> None:
    print(
        f"{name:<14} {phase:<6} ttft {ttft:7.3f}s  total {total:7.3f}s  "
        f"{speedup:<7} {hits}"
    )


def run_exact_repeat(url: str, model: str, ds: dict, server_log: Path | None) -> None:
    (prompt,) = ds["prompts"]
    n = ds["tokens"] // CHUNK_TOKENS
    print(f"\n== 场景 1 · 完全重复命中({ds['tokens']} tokens,{n} chunks)==")
    cold_ttft, cold_total, cold = complete(url, model, prompt)
    print_row(
        "exact_repeat", "cold", cold_ttft, cold_total, server_hits(server_log, cold)
    )
    warm_ttft, warm_total, warm = complete(url, model, prompt)
    print_row(
        "exact_repeat",
        "warm",
        warm_ttft,
        warm_total,
        server_hits(server_log, warm),
        f"{cold_ttft / warm_ttft:.1f}x",
    )
    same = cold["text"] == warm["text"]
    print(
        f"  输出一致性: {'一致 ✓' if same else '不一致 ✗'}  "
        f"TTFT 加速 {cold_ttft / warm_ttft:.1f}x"
    )


def run_shared_prefix(url: str, model: str, ds: dict, server_log: Path | None) -> None:
    prompts = ds["prompts"]
    print(
        f"\n== 场景 2 · 共享前缀({ds['prefix_tokens']} token 前缀,"
        f"{len(prompts)} 个不同后缀)=="
    )
    ttft, total, resp = complete(url, model, prompts[0])
    print_row("prefix-s0", "cold", ttft, total, server_hits(server_log, resp))
    warm_ttfts = []
    for i, prompt in enumerate(prompts[1:], start=1):
        ttft, total, resp = complete(url, model, prompt)
        warm_ttfts.append(ttft)
        print_row(f"prefix-s{i}", "warm", ttft, total, server_hits(server_log, resp))
    cold_ttft, cold_total, resp = complete(url, model, ds["baseline"])
    print_row("baseline", "cold", cold_ttft, cold_total, server_hits(server_log, resp))
    avg = sum(warm_ttfts) / len(warm_ttfts)
    print(
        f"  前缀命中平均 TTFT {avg:.3f}s vs 同长度冷启动 {cold_ttft:.3f}s "
        f"→ 加速 {cold_ttft / avg:.1f}x"
    )


def run_no_reuse(url: str, model: str, ds: dict, server_log: Path | None) -> None:
    print(
        f"\n== 场景 3 · 无复用基线({ds['tokens']} tokens × {len(ds['prompts'])} 条)=="
    )
    ttfts = []
    for i, prompt in enumerate(ds["prompts"]):
        ttft, total, resp = complete(url, model, prompt)
        ttfts.append(ttft)
        print_row(f"no_reuse-{i}", "cold", ttft, total, server_hits(server_log, resp))
    print(f"  冷启动平均 TTFT {sum(ttfts) / len(ttfts):.3f}s")


def run_hf100(
    url: str, model: str, tok: PreTrainedTokenizer, server_log: Path | None
) -> None:
    """100-sample SQuAD-style dataset: 20 contexts x 5 questions.

    The first question of each context computes and stores the context;
    the other four hit its prefix.  Speedup is measured against an
    estimate of what recomputing everything would cost (5x the measured
    cold TTFT per context).
    """
    prompts, group_ids = load_100(tok)
    n_groups = max(group_ids) + 1
    print(
        f"\n== 场景 4 · 100 条 SQuAD 式数据集({len(prompts)} prompts,"
        f"{n_groups} 组共享上下文)=="
    )
    cold_ttfts, warm_ttfts = [], []
    hit_prompts = 0
    seen_groups: set[int] = set()
    t0 = time.perf_counter()
    for i, prompt in enumerate(prompts):
        ttft, _, resp = complete(url, model, prompt)
        hits = server_hits(server_log, resp)
        if group_ids[i] not in seen_groups:  # first question of its context
            seen_groups.add(group_ids[i])
            cold_ttfts.append(ttft)
        else:
            warm_ttfts.append(ttft)
            if "hit L1=" in hits:
                hit_prompts += 1
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(prompts)} 完成")
    total = time.perf_counter() - t0
    cold_sum = sum(cold_ttfts)
    warm_sum = sum(warm_ttfts)
    estimated = 5 * cold_sum
    actual_ttft = cold_sum + warm_sum
    print(
        f"  冷启动(每组首问): {len(cold_ttfts)} 次 TTFT 共 {cold_sum:.2f}s,"
        f"平均 {cold_sum / len(cold_ttfts):.3f}s"
    )
    print(
        f"  前缀命中(其余 {len(warm_ttfts)} 问): TTFT 共 {warm_sum:.2f}s,"
        f"平均 {warm_sum / len(warm_ttfts):.3f}s"
    )
    print(
        f"  TTFT 合计: 缓存 {actual_ttft:.2f}s vs 全部重算预估 "
        f"{estimated:.2f}s → 加速 {estimated / actual_ttft:.2f}x"
    )
    print(
        f"  墙钟总耗时(含 decode){total:.2f}s | "
        f"命中率 {hit_prompts}/{len(warm_ttfts)} 条前缀命中"
    )


def chunk_mb(server_log: Path | None) -> float | None:
    """Parse the chunk size (MiB) from the REGISTER line of the server log."""
    if server_log is None or not server_log.exists():
        return None
    for line in server_log.read_text(errors="ignore").splitlines():
        if "REGISTER" in line and "chunk=" in line:
            return float(line.split("chunk=")[1].split(" ")[0])
    return None


def run_throughput(
    url: str,
    model: str,
    tok: PreTrainedTokenizer,
    server_log: Path | None,
    n_requests: int = 20,
) -> None:
    """Throughput scenario: N requests sharing one long prefix.

    The first request computes and stores the prefix; the rest replay it.
    Speedup = N x (measured cold time of one same-length request) over the
    actual total wall time — the metric that matters for serving many
    requests against a shared context.
    """
    from benchmarks.datasets import DOCUMENTS, repeat_to_tokens

    prefix = repeat_to_tokens(tok, DOCUMENTS["birds"], LONG_CHUNKS * 256)
    prompts = [prefix + " " + SUFFIXES[i % len(SUFFIXES)] for i in range(n_requests)]
    cold_prompt = repeat_to_tokens(
        tok, DOCUMENTS["rocks"], LONG_CHUNKS * 256 + len(tok.encode(SUFFIXES[0]))
    )

    print(
        f"\n== 场景 5 · 吞吐测试(共享 {LONG_CHUNKS} chunk 前缀 × {n_requests} 请求)=="
    )
    cold_ttft, cold_total, _ = complete(url, model, cold_prompt)
    print(f"  同长度冷启动参考: TTFT {cold_ttft:.3f}s / total {cold_total:.3f}s")

    # Warm the prefix once, then measure the shared-prefix requests.
    complete(url, model, prompts[0])
    ttfts, totals = [], []
    for i, prompt in enumerate(prompts[1:]):
        ttft, total, resp = complete(url, model, prompt)
        ttfts.append(ttft)
        totals.append(total)
        hits = server_hits(server_log, resp)
        if i < 4:
            print(f"  req {i+1:2d} 命中   {hits}")
    sum_ttft = sum(ttfts)
    sum_total = sum(totals)
    est_ttft = (n_requests - 1) * cold_ttft
    est_total = (n_requests - 1) * cold_total
    print(
        f"  累计 TTFT: 缓存 {sum_ttft:.2f}s vs 无缓存预估 {est_ttft:.2f}s "
        f"→ TTFT 加速 {est_ttft / sum_ttft:.1f}x(prefill 口径)"
    )
    print(
        f"  累计墙钟: 缓存 {sum_total:.2f}s vs 无缓存预估 {est_total:.2f}s "
        f"→ 吞吐加速 {est_total / sum_total:.1f}x(含 decode)"
    )
    mb = chunk_mb(server_log)
    if mb:
        moved = (n_requests - 1) * LONG_CHUNKS * mb
        print(f"  缓存路径搬运 {moved / 1024:.2f} GiB(chunk={mb} MiB)")


def main() -> None:
    global LONG_CHUNKS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--tokenizer", required=True, help="model dir for tokenizing dataset prompts"
    )
    parser.add_argument(
        "--server-log", default=None, help="cache server log file for hit evidence"
    )
    parser.add_argument("--prefix-chunks", type=int, default=LONG_CHUNKS)
    parser.add_argument(
        "--chunks", type=int, default=LONG_CHUNKS,
        help="prompt length in 256-token chunks (default 12)")
    parser.add_argument(
        "--scenarios", default="1,2,3,4,5", help="comma-separated scenario ids to run"
    )
    parser.add_argument("--throughput-n", type=int, default=20)
    args: Any = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    server_log = Path(args.server_log) if args.server_log else None
    LONG_CHUNKS = args.chunks
    wanted = {s.strip() for s in args.scenarios.split(",")}

    # One-time warmup: the first NPU prefill is 30-40x slower; without this
    # the first measured request would dominate every scenario.
    print("== 模型 warmup(一次性,不计入结果)==")
    complete(args.url, args.model, "Hello. ")

    if "1" in wanted:
        run_exact_repeat(
            args.url,
            args.model,
            make_exact_repeat(tok, n_chunks=LONG_CHUNKS, doc="ships"),
            server_log,
        )
    if "2" in wanted:
        run_shared_prefix(
            args.url,
            args.model,
            make_shared_prefix(tok, args.prefix_chunks, doc="birds"),
            server_log,
        )
    if "3" in wanted:
        run_no_reuse(
            args.url,
            args.model,
            make_no_reuse(
                tok,
                n_chunks=LONG_CHUNKS,
                docs=("fungi", "weather", "insects", "plants"),
            ),
            server_log,
        )
    if "4" in wanted:
        run_hf100(args.url, args.model, tok, server_log)
    if "5" in wanted:
        run_throughput(args.url, args.model, tok, server_log, args.throughput_n)


if __name__ == "__main__":
    main()
