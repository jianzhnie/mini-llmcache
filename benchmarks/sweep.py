# SPDX-License-Identifier: Apache-2.0
"""Configuration sweep: chunk-size x L1-size, measured end-to-end.

Runs INSIDE the container and manages the cache server / vLLM processes
itself (vLLM restarts only when the chunk size changes, because the
connector reads it once at startup).

    python benchmarks/sweep.py \\
        --model-path /path/to/model --served-name Qwen/Qwen3-8B \\
        --tokenizer /path/to/model \\
        --chunk-sizes 128,256,512,1024 --l1-gbs 4,8,16

Metric per config: the throughput scenario (10 requests sharing one
3072-token prefix) — cold time of one same-length request vs the average
warm request, plus the L1 hit count from the server log.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from benchmarks.bench import complete
from benchmarks.datasets import DOCUMENTS, repeat_to_tokens
from transformers import AutoTokenizer

REPO = "/home/jianzhnie/llmtuner/llm/mini-llmcache"
SERVER_LOG = "/tmp/mini-sweep-server.log"
VLLM_LOG = "/tmp/mini-sweep-vllm.log"
PORT = 8000
CACHE_PORT = 45881
PROMPT_TOKENS = 3072
N_WARM = 10


def env() -> dict:
    e = dict(os.environ)
    e["PYTHONPATH"] = REPO + ":" + e.get("PYTHONPATH", "")
    return e


def stop_all() -> None:
    subprocess.run(["pkill", "-f", "mini_llmcache.serv[e]r"], check=False)
    subprocess.run(["pkill", "-f", "vllm serv[e]"], check=False)
    time.sleep(2)


def start_server(chunk_size: int, l1_gb: float) -> None:
    for attempt in range(3):
        subprocess.run(["pkill", "-f", "mini_llmcache.serv[e]r"], check=False)
        time.sleep(3)  # let the old process release its port and memory
        subprocess.run(["rm", "-rf", "/tmp/mini-l2-sweep"], check=False)
        with open(SERVER_LOG, "w") as out:
            subprocess.Popen(
                ["python3", "-m", "mini_llmcache.server", "--port",
                 str(CACHE_PORT), "--chunk-size", str(chunk_size),
                 "--l1-size-gb", str(l1_gb), "--l2-adapter",
                 '{"type": "fs", "base_path": "/tmp/mini-l2-sweep"}'],
                cwd=REPO, env=env(), stdout=out, stderr=subprocess.STDOUT)
        for _ in range(50):
            with open(SERVER_LOG) as f:
                if "up and ready" in f.read():
                    return
            time.sleep(0.2)
        print(f"  [warn] server start attempt {attempt + 1} failed: "
              f"{Path(SERVER_LOG).read_text(errors='ignore')[-300:]}")
    raise RuntimeError("cache server failed to start")


def start_vllm(model_path: str, served_name: str) -> None:
    subprocess.run(["pkill", "-f", "vllm serv[e]"], check=False)
    time.sleep(2)
    with open(VLLM_LOG, "w") as out:
        subprocess.Popen(
            ["vllm", "serve", model_path, "--served-model-name", served_name,
             "--enforce-eager", "--max-model-len", "4096",
             "--no-enable-prefix-caching", "--kv-transfer-config",
             ('{"kv_connector": "MiniConnector", "kv_connector_module_path": '
              '"mini_llmcache.integration.vllm_connector", "kv_role": '
              '"kv_both", "kv_connector_extra_config": {"mini.port": '
              f'{CACHE_PORT}}}}}'),
             "--port", str(PORT)],
            cwd=REPO, env=env(), stdout=out, stderr=subprocess.STDOUT)
    for _ in range(600):
        with open(VLLM_LOG) as f:
            log = f.read()
            if "startup complete" in log:
                return
            if "Traceback" in log or "EngineCore failed" in log:
                raise RuntimeError("vllm failed: " + log[-2000:])
        time.sleep(1)
    raise RuntimeError("vllm did not start in time")


def measure(served_name: str, tok, model: str) -> dict:
    """Throughput metric for the current config; returns a result row."""
    prefix = repeat_to_tokens(tok, DOCUMENTS["birds"], PROMPT_TOKENS)
    cold_prompt = repeat_to_tokens(tok, DOCUMENTS["rocks"], PROMPT_TOKENS)

    url = f"http://localhost:{PORT}"
    complete(url, served_name, "Hello. ")  # one-time warmup
    _, cold_total, _ = complete(url, served_name, cold_prompt)

    warm_times = []
    for i in range(N_WARM):
        prompt = prefix + f" Question number {i}. Answer:"
        _, total, _ = complete(url, served_name, prompt)
        warm_times.append(total)
    warm_avg = sum(warm_times) / len(warm_times)
    # The first warm request computed the prefix (cold); drop it.
    return {
        "cold_total": cold_total,
        "warm_avg": warm_avg,
        "warm_median": sorted(warm_times[1:])[len(warm_times[1:]) // 2]
        if len(warm_times) > 1 else warm_avg,
        "speedup": cold_total / warm_avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-name", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--chunk-sizes", default="128,256,512,1024")
    parser.add_argument("--l1-gbs", default="8")
    args: Any = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    chunk_sizes = [int(x) for x in args.chunk_sizes.split(",")]
    l1_gbs = [float(x) for x in args.l1_gbs.split(",")]

    stop_all()
    rows = []
    print(f"{'chunk':>6} {'L1(GB)':>6} {'cold(s)':>8} {'warm_avg(s)':>11} "
          f"{'speedup':>8}")
    try:
        for chunk in chunk_sizes:
            for l1 in l1_gbs:
                # Order matters: the connector reaches the cache server
                # during vLLM startup, so the server must come up first.
                # Both restart per config (a new server has no REGISTER
                # state, which a running vLLM cannot re-send).
                start_server(chunk, l1)
                start_vllm(args.model_path, args.served_name)
                row = measure(args.served_name, tok, args.model_path)
                row.update(chunk=chunk, l1_gb=l1)
                rows.append(row)
                print(f"{chunk:>6} {l1:>6} {row['cold_total']:>8.3f} "
                      f"{row['warm_avg']:>11.3f} {row['speedup']:>7.2f}x")
                subprocess.run(["pkill", "-f", "vllm serv[e]"], check=False)
                time.sleep(2)
    finally:
        stop_all()

    best = max(rows, key=lambda r: r["speedup"])
    print(f"\n最佳配置: chunk-size={best['chunk']}, L1={best['l1_gb']}GB "
          f"(加速 {best['speedup']:.2f}x)")


if __name__ == "__main__":
    main()
