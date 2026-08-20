# SPDX-License-Identifier: Apache-2.0
"""Full benchmark matrix runner for mini-llmcache.

This script manages the cache server and vLLM, then runs the Cartesian
product described by docs/prompt.md:

    model x chunk-size x L1 size x L2 size x prompt length x scenario

Example:

    python benchmarks/sweep.py \\
        --models 'Qwen/Qwen3-0.6B,Qwen/Qwen3-8B,Qwen/Qwen3-32B@2' \\
        --tokenizer Qwen/Qwen3-0.6B \\
        --chunk-sizes 128,256,512,1024 \\
        --l1-gbs 4,8,16,32 \\
        --l2-gbs 4,8,16,32,64,128 \\
        --prompt-tokens 1024,2048,4096,8192,16384,32768 \\
        --scenarios 1,2,3,4,5 \\
        --out-dir /tmp/mini-sweep-results

Model specs use:

    model_path[=served_name][@tensor_parallel_size]

For example:

    /models/Qwen3-32B=Qwen/Qwen3-32B@2

Note: the current filesystem L2 adapter does not enforce a capacity quota.
The l2_gb axis is recorded and gets its own clean directory per run, but it
does not change eviction behavior unless the adapter grows a capacity option.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.client import MAX_OUTPUT_TOKENS, complete, server_hits
from benchmarks.datasets import (
    DOCUMENTS,
    SUFFIXES,
    repeat_to_tokens,
)
from benchmarks.datasets_hf import load_100

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = Path("/tmp/mini-sweep-results")
DEFAULT_CACHE_PORT = 45881
DEFAULT_VLLM_PORT = 8000
DEFAULT_BLOCK_SIZE = 128
DEFAULT_MODEL_ROOT = Path("/home/jianzhnie/llmtuner/hfhub/models/Qwen")
SCENARIO_NAMES = {
    "1": "exact_repeat",
    "2": "shared_prefix",
    "3": "no_reuse",
    "4": "squad100",
    "5": "throughput",
}
CSV_FIELDS = [
    "status",
    "model_path",
    "served_name",
    "tensor_parallel_size",
    "chunk_size",
    "l1_gb",
    "l2_gb",
    "prompt_tokens",
    "scenario_id",
    "scenario",
    "cold_ttft",
    "hot_ttft",
    "ttft_speedup",
    "cold_total",
    "hot_total",
    "total_speedup",
    "hit_rate",
    "output_equal",
    "elapsed_s",
    "error",
]
SCENARIO_IDS = {name: sid for sid, name in SCENARIO_NAMES.items()}
HIT_RE = re.compile(r"hit L1=(\d+) L2=(\d+)")


@dataclass(frozen=True)
class ModelSpec:
    model_path: str
    served_name: str
    tensor_parallel_size: int


@dataclass(frozen=True)
class RunConfig:
    model: ModelSpec
    chunk_size: int
    l1_gb: float
    l2_gb: float
    prompt_tokens: int
    scenario: str


def env() -> dict[str, str]:
    e = dict(os.environ)
    e["PYTHONPATH"] = str(REPO) + ":" + e.get("PYTHONPATH", "")
    return e


def parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_csv_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def parse_scenarios(raw: str) -> list[str]:
    out = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        for sid, name in SCENARIO_NAMES.items():
            if item in (sid, name):
                out.append(sid)
                break
        else:
            raise ValueError(f"unknown scenario: {item}")
    return out


def parse_model_spec(raw: str) -> ModelSpec:
    item = raw.strip()
    if not item:
        raise ValueError("empty model spec")
    tensor_parallel_size = 1
    if "@" in item:
        item, tp = item.rsplit("@", 1)
        tensor_parallel_size = int(tp)
    if "=" in item:
        model_path, served_name = item.split("=", 1)
    else:
        model_path = item
        served_name = item
    return ModelSpec(model_path, served_name, tensor_parallel_size)


def parse_models(raw: str) -> list[ModelSpec]:
    return [parse_model_spec(x) for x in raw.split(",") if x.strip()]


def slug(value: str) -> str:
    return (
        value.replace("/", "-")
        .replace(":", "-")
        .replace("@", "-")
        .replace("=", "-")
        .replace(" ", "_")
    )


def run_key(cfg: RunConfig) -> tuple[Any, ...]:
    return (
        cfg.model.model_path,
        cfg.model.served_name,
        cfg.model.tensor_parallel_size,
        cfg.chunk_size,
        cfg.l1_gb,
        cfg.l2_gb,
        cfg.prompt_tokens,
        cfg.scenario,
    )


def result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    scenario = row.get("scenario_id") or SCENARIO_IDS.get(row["scenario"], row["scenario"])
    return (
        row["model_path"],
        row["served_name"],
        int(row["tensor_parallel_size"]),
        int(row["chunk_size"]),
        float(row["l1_gb"]),
        float(row["l2_gb"]),
        int(row["prompt_tokens"]),
        scenario,
    )


def completed_keys(jsonl: Path) -> set[tuple[Any, ...]]:
    if not jsonl.exists():
        return set()
    keys = set()
    for line in jsonl.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") in {"ok", "skipped"}:
            keys.add(result_key(row))
    return keys


def stop_all() -> None:
    subprocess.run(["pkill", "-f", "mini_llmcache.serv[e]r"], check=False)
    subprocess.run(["pkill", "-f", "vllm serv[e]"], check=False)
    time.sleep(2)


def wait_for_log(path: Path, needle: str, timeout_s: float, error_needles: tuple[str, ...]) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        text = path.read_text(errors="ignore") if path.exists() else ""
        if needle in text:
            return
        if any(err in text for err in error_needles):
            raise RuntimeError(text[-4000:])
        time.sleep(0.5)
    tail = path.read_text(errors="ignore")[-4000:] if path.exists() else ""
    raise TimeoutError(f"timed out waiting for {needle!r} in {path}\n{tail}")


def start_server(
    cfg: RunConfig,
    server_log: Path,
    l2_dir: Path,
    cache_port: int,
    cache_bind_url: str | None,
) -> None:
    shutil.rmtree(l2_dir, ignore_errors=True)
    l2_dir.mkdir(parents=True, exist_ok=True)
    bind_args = (
        ["--bind-url", cache_bind_url]
        if cache_bind_url
        else ["--port", str(cache_port)]
    )
    cmd = [
        "python3",
        "-m",
        "mini_llmcache.server",
        *bind_args,
        "--chunk-size",
        str(cfg.chunk_size),
        "--l1-size-gb",
        str(cfg.l1_gb),
        "--l2-adapter",
        json.dumps({"type": "fs", "base_path": str(l2_dir)}),
    ]
    with server_log.open("w") as out:
        subprocess.Popen(cmd, cwd=REPO, env=env(), stdout=out, stderr=subprocess.STDOUT)
    wait_for_log(server_log, "up and ready", 60, ("Traceback", "Address already in use"))


def start_vllm(
    cfg: RunConfig,
    vllm_log: Path,
    vllm_port: int,
    cache_port: int,
    cache_bind_url: str | None,
    max_model_len: int,
    extra_args: list[str],
) -> None:
    extra_config = (
        {"mini.url": cache_bind_url}
        if cache_bind_url
        else {"mini.port": cache_port}
    )
    kv_config = {
        "kv_connector": "MiniConnector",
        "kv_connector_module_path": "mini_llmcache.integration.vllm_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": extra_config,
    }
    cmd = [
        "vllm",
        "serve",
        cfg.model.model_path,
        "--served-model-name",
        cfg.model.served_name,
        "--enforce-eager",
        "--max-model-len",
        str(max_model_len),
        "--no-enable-prefix-caching",
        "--kv-transfer-config",
        json.dumps(kv_config),
        "--port",
        str(vllm_port),
    ]
    if cfg.model.tensor_parallel_size > 1:
        cmd += ["--tensor-parallel-size", str(cfg.model.tensor_parallel_size)]
    cmd += extra_args
    with vllm_log.open("w") as out:
        subprocess.Popen(cmd, cwd=REPO, env=env(), stdout=out, stderr=subprocess.STDOUT)
    wait_for_log(
        vllm_log,
        "startup complete",
        900,
        ("Traceback", "EngineCore failed", "RuntimeError"),
    )


def metric_row(cfg: RunConfig, status: str, **values: Any) -> dict[str, Any]:
    row = {
        "status": status,
        "model_path": cfg.model.model_path,
        "served_name": cfg.model.served_name,
        "tensor_parallel_size": cfg.model.tensor_parallel_size,
        "chunk_size": cfg.chunk_size,
        "l1_gb": cfg.l1_gb,
        "l2_gb": cfg.l2_gb,
        "prompt_tokens": cfg.prompt_tokens,
        "scenario_id": cfg.scenario,
        "scenario": SCENARIO_NAMES[cfg.scenario],
    }
    row.update(values)
    return row


def hit_count(server_log: Path, responses: list[dict[str, Any]]) -> int:
    hits = 0
    for resp in responses:
        line = server_hits(server_log, resp)
        match = HIT_RE.search(line)
        if match and (int(match.group(1)) + int(match.group(2))) > 0:
            hits += 1
    return hits


def measure_exact_repeat(
    url: str,
    served_name: str,
    tok: PreTrainedTokenizer,
    prompt_tokens: int,
    server_log: Path,
) -> dict[str, Any]:
    prompt = repeat_to_tokens(tok, DOCUMENTS["ships"], prompt_tokens)
    cold_ttft, cold_total, cold = complete(url, served_name, prompt)
    hot_ttft, hot_total, hot = complete(url, served_name, prompt)
    return {
        "cold_ttft": cold_ttft,
        "hot_ttft": hot_ttft,
        "ttft_speedup": cold_ttft / hot_ttft,
        "cold_total": cold_total,
        "hot_total": hot_total,
        "total_speedup": cold_total / hot_total,
        "hit_rate": hit_count(server_log, [hot]),
        "output_equal": cold["text"] == hot["text"],
    }


def measure_shared_prefix(
    url: str,
    served_name: str,
    tok: PreTrainedTokenizer,
    prompt_tokens: int,
    server_log: Path,
) -> dict[str, Any]:
    prefix = repeat_to_tokens(tok, DOCUMENTS["birds"], prompt_tokens)
    prompts = [prefix + " " + suffix for suffix in SUFFIXES[:4]]
    suffix_tokens = len(tok.encode(SUFFIXES[0]))
    baseline = repeat_to_tokens(tok, DOCUMENTS["rocks"], prompt_tokens + suffix_tokens)

    complete(url, served_name, prompts[0])
    hot_ttfts, hot_totals, responses = [], [], []
    for prompt in prompts[1:]:
        ttft, total, resp = complete(url, served_name, prompt)
        hot_ttfts.append(ttft)
        hot_totals.append(total)
        responses.append(resp)
    cold_ttft, cold_total, _ = complete(url, served_name, baseline)
    hot_ttft = sum(hot_ttfts) / len(hot_ttfts)
    hot_total = sum(hot_totals) / len(hot_totals)
    return {
        "cold_ttft": cold_ttft,
        "hot_ttft": hot_ttft,
        "ttft_speedup": cold_ttft / hot_ttft,
        "cold_total": cold_total,
        "hot_total": hot_total,
        "total_speedup": cold_total / hot_total,
        "hit_rate": hit_count(server_log, responses) / len(responses),
    }


def measure_no_reuse(
    url: str,
    served_name: str,
    tok: PreTrainedTokenizer,
    prompt_tokens: int,
) -> dict[str, Any]:
    docs = ("fungi", "weather", "insects", "plants")
    prompts = [repeat_to_tokens(tok, DOCUMENTS[doc], prompt_tokens) for doc in docs]
    ttfts, totals = [], []
    for prompt in prompts:
        ttft, total, _ = complete(url, served_name, prompt)
        ttfts.append(ttft)
        totals.append(total)
    return {
        "cold_ttft": sum(ttfts) / len(ttfts),
        "cold_total": sum(totals) / len(totals),
        "hit_rate": 0,
    }


def measure_squad100(
    url: str,
    served_name: str,
    tok: PreTrainedTokenizer,
    prompt_tokens: int,
    server_log: Path,
) -> dict[str, Any]:
    prompts, group_ids = load_100(tok, context_tokens=prompt_tokens)
    cold_ttfts, cold_totals, hot_ttfts, hot_totals = [], [], [], []
    responses = []
    seen_groups: set[int] = set()
    for i, prompt in enumerate(prompts):
        ttft, total, resp = complete(url, served_name, prompt)
        if group_ids[i] in seen_groups:
            hot_ttfts.append(ttft)
            hot_totals.append(total)
            responses.append(resp)
        else:
            seen_groups.add(group_ids[i])
            cold_ttfts.append(ttft)
            cold_totals.append(total)
    estimated_cold_ttft = 5 * sum(cold_ttfts)
    estimated_cold_total = 5 * sum(cold_totals)
    actual_ttft = sum(cold_ttfts) + sum(hot_ttfts)
    actual_total = sum(cold_totals) + sum(hot_totals)
    return {
        "cold_ttft": estimated_cold_ttft / len(prompts),
        "hot_ttft": actual_ttft / len(prompts),
        "ttft_speedup": estimated_cold_ttft / actual_ttft,
        "cold_total": estimated_cold_total / len(prompts),
        "hot_total": actual_total / len(prompts),
        "total_speedup": estimated_cold_total / actual_total,
        "hit_rate": hit_count(server_log, responses) / len(responses)
        if responses
        else 0,
    }


def measure_throughput(
    url: str,
    served_name: str,
    tok: PreTrainedTokenizer,
    prompt_tokens: int,
    server_log: Path,
    n_requests: int,
) -> dict[str, Any]:
    prefix = repeat_to_tokens(tok, DOCUMENTS["plants"], prompt_tokens)
    prompts = [prefix + " " + SUFFIXES[i % len(SUFFIXES)] for i in range(n_requests)]
    cold_prompt = repeat_to_tokens(
        tok,
        DOCUMENTS["weather"],
        prompt_tokens + len(tok.encode(SUFFIXES[0])),
    )
    cold_ttft, cold_total, _ = complete(url, served_name, cold_prompt)
    complete(url, served_name, prompts[0])
    hot_ttfts, hot_totals, responses = [], [], []
    for prompt in prompts[1:]:
        ttft, total, resp = complete(url, served_name, prompt)
        hot_ttfts.append(ttft)
        hot_totals.append(total)
        responses.append(resp)
    estimated_ttft = (n_requests - 1) * cold_ttft
    estimated_total = (n_requests - 1) * cold_total
    actual_ttft = sum(hot_ttfts)
    actual_total = sum(hot_totals)
    return {
        "cold_ttft": cold_ttft,
        "hot_ttft": actual_ttft / len(hot_ttfts) if hot_ttfts else 0.0,
        "ttft_speedup": estimated_ttft / actual_ttft,
        "cold_total": cold_total,
        "hot_total": actual_total / len(hot_totals) if hot_totals else 0.0,
        "total_speedup": estimated_total / actual_total,
        "hit_rate": hit_count(server_log, responses) / len(responses)
        if responses
        else 0,
    }


def measure_scenario(
    cfg: RunConfig,
    tok: PreTrainedTokenizer,
    url: str,
    server_log: Path,
    throughput_n: int,
) -> dict[str, Any]:
    complete(url, cfg.model.served_name, "Hello. ")
    if cfg.scenario == "1":
        return measure_exact_repeat(url, cfg.model.served_name, tok, cfg.prompt_tokens, server_log)
    if cfg.scenario == "2":
        return measure_shared_prefix(url, cfg.model.served_name, tok, cfg.prompt_tokens, server_log)
    if cfg.scenario == "3":
        return measure_no_reuse(url, cfg.model.served_name, tok, cfg.prompt_tokens)
    if cfg.scenario == "4":
        return measure_squad100(url, cfg.model.served_name, tok, cfg.prompt_tokens, server_log)
    if cfg.scenario == "5":
        return measure_throughput(
            url,
            cfg.model.served_name,
            tok,
            cfg.prompt_tokens,
            server_log,
            throughput_n,
        )
    raise ValueError(f"unknown scenario id: {cfg.scenario}")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_csv(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def build_matrix(
    models: list[ModelSpec],
    chunk_sizes: list[int],
    l1_gbs: list[float],
    l2_gbs: list[float],
    prompt_tokens: list[int],
    scenarios: list[str],
) -> list[RunConfig]:
    configs = []
    for model, chunk, l1, l2, prompt_len, scenario in itertools.product(
        models, chunk_sizes, l1_gbs, l2_gbs, prompt_tokens, scenarios
    ):
        configs.append(RunConfig(model, chunk, l1, l2, prompt_len, scenario))
    return configs


def invalid_reason(cfg: RunConfig, block_size: int) -> str | None:
    if cfg.chunk_size < block_size:
        return f"chunk_size {cfg.chunk_size} is smaller than block_size {block_size}"
    if cfg.chunk_size % block_size != 0:
        return f"chunk_size {cfg.chunk_size} is not a multiple of block_size {block_size}"
    if cfg.prompt_tokens < cfg.chunk_size:
        return "prompt_tokens is smaller than chunk_size, so no cache chunk can hit"
    if cfg.model.tensor_parallel_size < 1:
        return "tensor_parallel_size must be >= 1"
    return None


def max_model_len(prompt_tokens: int, floor: int) -> int:
    return max(floor, prompt_tokens + MAX_OUTPUT_TOKENS + 256)


def run_one(
    cfg: RunConfig,
    args: Any,
    tok: PreTrainedTokenizer,
    jsonl: Path,
    csv_path: Path,
) -> dict[str, Any]:
    run_name = (
        f"{slug(cfg.model.served_name)}_tp{cfg.model.tensor_parallel_size}_"
        f"chunk{cfg.chunk_size}_l1{cfg.l1_gb:g}_l2{cfg.l2_gb:g}_"
        f"tok{cfg.prompt_tokens}_{SCENARIO_NAMES[cfg.scenario]}"
    )
    run_dir = args.out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    server_log = run_dir / "server.log"
    vllm_log = run_dir / "vllm.log"
    l2_dir = args.l2_root / run_name
    url = f"http://localhost:{args.vllm_port}"
    started = time.perf_counter()

    stop_all()
    try:
        start_server(cfg, server_log, l2_dir, args.cache_port, args.cache_bind_url)
        start_vllm(
            cfg,
            vllm_log,
            args.vllm_port,
            args.cache_port,
            args.cache_bind_url,
            max_model_len(cfg.prompt_tokens, args.max_model_len_floor),
            args.vllm_arg,
        )
        metrics = measure_scenario(cfg, tok, url, server_log, args.throughput_n)
        row = metric_row(cfg, "ok", **metrics)
    except Exception as exc:
        row = metric_row(cfg, "failed", error=str(exc)[-4000:])
        if args.fail_fast:
            raise
    finally:
        stop_all()
    row["elapsed_s"] = time.perf_counter() - started
    append_jsonl(jsonl, row)
    append_csv(csv_path, row)
    return row


def print_summary(row: dict[str, Any], index: int, total: int) -> None:
    prefix = (
        f"[{index}/{total}] {row['status']} {row['served_name']} "
        f"tp={row['tensor_parallel_size']} chunk={row['chunk_size']} "
        f"L1={row['l1_gb']}GB L2={row['l2_gb']}GB "
        f"tokens={row['prompt_tokens']} scenario={row['scenario']}"
    )
    if row["status"] != "ok":
        print(prefix + f" error={row.get('error', '')[:160]}", flush=True)
        return
    speedup = row.get("ttft_speedup")
    hit_rate = row.get("hit_rate")
    print(
        prefix
        + f" TTFT_x={speedup:.2f} hit_rate={hit_rate}"
        + f" elapsed={row['elapsed_s']:.1f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=(
            f"{DEFAULT_MODEL_ROOT / 'Qwen3-0.6B'}=Qwen/Qwen3-0.6B,"
            f"{DEFAULT_MODEL_ROOT / 'Qwen3-8B'}=Qwen/Qwen3-8B,"
            f"{DEFAULT_MODEL_ROOT / 'Qwen3-32B'}=Qwen/Qwen3-32B@2,"
            f"{DEFAULT_MODEL_ROOT / 'Qwen3-32B'}=Qwen/Qwen3-32B@4,"
            f"{DEFAULT_MODEL_ROOT / 'Qwen3-32B'}=Qwen/Qwen3-32B@8"
        ),
        help="comma-separated model_path[=served_name][@tp] specs",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer path/name; defaults to the first model_path",
    )
    parser.add_argument("--chunk-sizes", default="64,128,256,512,1024")
    parser.add_argument("--l1-gbs", default="4,8,16,32")
    parser.add_argument("--l2-gbs", default="4,8,16,32,64,128")
    parser.add_argument(
        "--prompt-tokens",
        default="1024,2048,4096,8192,16384,32768",
    )
    parser.add_argument("--scenarios", default="1,2,3,4,5")
    parser.add_argument("--throughput-n", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--l2-root", type=Path, default=Path("/tmp/mini-l2-sweep"))
    parser.add_argument("--cache-port", type=int, default=DEFAULT_CACHE_PORT)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument(
        "--cache-bind-url",
        default=None,
        help="full cache bind url, e.g. ipc:///tmp/mini-cache.sock",
    )
    parser.add_argument("--max-model-len-floor", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--vllm-arg", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="record invalid combinations as skipped instead of dropping them",
    )
    args: Any = parser.parse_args()
    scenario_ids = parse_scenarios(args.scenarios)
    if "5" in scenario_ids and args.throughput_n < 2:
        raise ValueError("--throughput-n must be >= 2 when scenario 5 is enabled")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.l2_root.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.models)
    tokenizer_name = args.tokenizer or models[0].model_path
    matrix = build_matrix(
        models,
        parse_csv_ints(args.chunk_sizes),
        parse_csv_floats(args.l1_gbs),
        parse_csv_floats(args.l2_gbs),
        parse_csv_ints(args.prompt_tokens),
        scenario_ids,
    )
    jsonl = args.out_dir / "results.jsonl"
    csv_path = args.out_dir / "results.csv"
    done = completed_keys(jsonl) if args.resume else set()

    valid_matrix = []
    skipped_rows = []
    invalid_count = 0
    for cfg in matrix:
        reason = invalid_reason(cfg, args.block_size)
        if reason and not args.include_invalid:
            invalid_count += 1
            continue
        if reason:
            invalid_count += 1
            skipped_rows.append(metric_row(cfg, "skipped", error=reason))
        else:
            valid_matrix.append(cfg)

    print(
        f"matrix: {len(matrix)} total, {len(valid_matrix)} runnable, "
        f"{invalid_count} invalid, {len(skipped_rows)} recorded skipped, "
        f"{len(done)} already complete"
    )
    print(
        "note: l2_gb is an axis in the result set, but FSAdapter has no "
        "capacity quota yet."
    )
    if args.dry_run:
        for cfg in valid_matrix[:20]:
            print(run_key(cfg))
        if len(valid_matrix) > 20:
            print(f"... {len(valid_matrix) - 20} more")
        return

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    for row in skipped_rows:
        if result_key(row) in done:
            continue
        append_jsonl(jsonl, row)
        append_csv(csv_path, row)
    runnable = [cfg for cfg in valid_matrix if run_key(cfg) not in done]
    for i, cfg in enumerate(runnable, start=1):
        row = run_one(cfg, args, tok, jsonl, csv_path)
        print_summary(row, i, len(runnable))

    rows = [
        json.loads(line)
        for line in jsonl.read_text(errors="ignore").splitlines()
        if line.strip()
    ]
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("ttft_speedup")]
    if ok_rows:
        best = max(ok_rows, key=lambda r: r["ttft_speedup"])
        print(
            "\nbest TTFT: "
            f"{best['served_name']} tp={best['tensor_parallel_size']} "
            f"chunk={best['chunk_size']} L1={best['l1_gb']}GB "
            f"L2={best['l2_gb']}GB tokens={best['prompt_tokens']} "
            f"scenario={best['scenario']} speedup={best['ttft_speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
