# mini-llmcache

**English** | [中文](README_zh.md)

mini-llmcache is a pedagogical [LMCache](https://github.com/LMCache/LMCache) reimplementation: a standalone KV-cache server for vLLM that stores hashed prompt chunks and replays matching prefixes to skip prefill. Measured on Ascend 910: **up to 19.4× on cold starts** (incl. the first-request NPU warmup penalty), **0.8–1.1× steady-state TTFT** on a 32B model with shared prefixes, 1.4× on multi-turn chains — and honestly ~1× or less on small models / short prompts.

About 870 lines of Python, no heavy dependencies (`pyzmq`, `blake3`, `torch`): the core ideas of LLM KV-cache sharing — prefix hashing, two-tier caching, reference-counted locks, pipelined transfers — in readable form.

- **Dual platform**: NVIDIA GPU (`torch.cuda`) and Ascend NPU (`torch.npu`) via a tiny device-abstraction layer. NPU adaptation notes: [docs/ascend_env.md](docs/ascend_env.md).
- **Two-tier storage**: L1 pinned host memory, L2 pluggable adapters (filesystem included; write your own in ~20 lines).
- **Architecture aligned with upstream LMCache-Ascend**: all GPU⇄CPU copies happen inside the vLLM process; only bytes cross the wire, so the cache server is pure CPU.

## Quick Start

```bash
# 1. Start the cache server (8 GB L1, L2 on disk under /tmp/mini-l2)
python -m mini_llmcache.server --port 45881 --l1-size-gb 8 \
    --l2-adapter '{"type": "fs", "base_path": "/tmp/mini-l2"}'
```

```bash
# 2. Serve a model with vLLM, attaching the MiniConnector
vllm serve Qwen/Qwen3-0.6B --enforce-eager \
    --max-model-len 4096 --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "MiniConnector",
                           "kv_connector_module_path": "mini_llmcache.integration.vllm_connector",
                           "kv_role": "kv_both",
                           "kv_connector_extra_config": {"mini.port": 45881}}'
```

```bash
# 3. Send a request with a long repeated prefix (the second one hits the cache)
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "{
    \"model\": \"Qwen/Qwen3-0.6B\", \"temperature\": 0, \"max_tokens\": 24,
    \"prompt\": \"$(python3 -c "print('A field guide to the birds of North America. ' * 80)")\"}"
```

Cache server output (measured on Ascend 910 + Qwen3-0.6B):

```
mini cache server up and ready on tcp://127.0.0.1:45881 (chunk_size=256, 8 GB L1, 1 L2 adapters)
REGISTER Qwen/Qwen3-0.6B rank 0/1 (chunk=28 MiB, 1 engines)
STORE    rid=cmpl-... tokens [0, 768) L0->L1 4.7 GB/s          ← 1st call: computed and cached
RETRIEVE rid=cmpl-... tokens [0, 768) hit L1=3 L2=0 | 3 chunks ← 2nd call: replayed, prefill skipped
```

> The first STORE and RETRIEVE need a warmup (30–40× slower); measure throughput from the second run.
>
> **Ascend NPU note**: the vllm-ascend image ships CANN paths in `$PYTHONPATH` — prepend the repo path
> (`PYTHONPATH=repo:$PYTHONPATH python3 -m mini_llmcache.server ...`) instead of overwriting it, or the
> engine process fails with `No module named 'acl'`. See [docs/ascend_env.md](docs/ascend_env.md).

## Measured Results

End-to-end on Ascend 910 (fresh server, model warmed up first). TTFT = time to
first token — the part the cache accelerates (prefill + transfer); decode is a
fixed cost on both sides.

| Model | Scenario | Cold TTFT | Warm TTFT | Speedup | Notes |
|---|---|---|---|---|---|
| Qwen3-0.6B | cold start → warm hit (768t) | 18.07 s | **0.93 s** | **19.4×** | prefill fully skipped |
| Qwen3-0.6B | L2 persistence (full restart) | — | 0.97 s | — | chunks prefetched from disk (L2→L1 4.5 GB/s) |
| Qwen3-0.6B | 3072t exact repeat | 0.41 s | 0.53 s | 0.8× | 11/11 chunks hit, byte-identical output |
| Qwen3-8B | 3072t exact repeat | 0.58 s | 0.81 s | 0.7× | 11/11 chunks hit, byte-identical output |
| Qwen3-32B (TP2) | 2560t prefix, 20 reqs | 0.62 s | 0.88 s | **0.8–1.0× TTFT** | all L1 hits; per-request transfer (~320 MB) ≈ saved prefill |
| Qwen3-32B (TP2) | multi-turn chain (6 rounds) | 0.82 s | 0.57 s | **1.44× overall** | per-round 0.94–1.54×; decays as the chain grows |
| Qwen3-32B (TP2) | concurrent, 4×5 shared prefix | — | TTFT p50 1.86 s | 20/20 hits | transfer contention inflates per-request TTFT ~2.8× |
| Qwen3-32B (TP2) | 8192t exact repeat (tcp) | 2.10 s | 1.86 s | 1.1× | 29/29 chunks hit |
| Qwen3-32B (TP2) | 8192t exact repeat (**ipc://**) | 2.10 s | 1.42 s | 1.1× | transport −24% vs tcp |
| Qwen3-32B (TP2) | **16384t exact repeat** | 3.65 s | 4.18 s | 0.87× | 59/59 hits but 3.8 GB/req transfer; 8 GB L1 holds only ~2 reqs — eviction jitter |
| Qwen3-32B (TP2) | sweep 最优:chunk=256, L1=8GB | 0.82 s | 0.60 s | **1.36×** | 全矩阵 8 配置中最优 |

**Optimal-config validation (Qwen3-32B TP2, chunk=256 / L1=8GB)** — all five
benchmark scenarios on a fresh server:

| Scenario | Cold TTFT | Hot TTFT | Speedup | Evidence |
|---|---|---|---|---|
| Exact repeat 3072t | 0.576 s | 0.570 s | 1.01× | 11/11 L1 hits, byte-identical output (clean env) |
| Shared prefix 3072t | 0.633 s | 0.584 s | 1.08× | 10/10 L1 hits |
| No reuse (baseline) | 0.546 s | — | — | cold reference (10 samples) |
| 100 SQuAD-style (1KB contexts) | 0.202 s | 0.298 s | 0.72× | 80/80 (100%) hits; short-context transfer dominates |
| Throughput, 20 shared-prefix reqs | 0.62 s | 0.88 s | 0.8× TTFT / 1.0× wall | all L1 hits |

At 3072t the cache roughly breaks even (0.92–1.08×): the per-request transfer
(~320 MB) costs about as much as the prefill it saves. Longer prompts do not
help — 16384t transfers 3.8 GB/req and an 8 GB L1 holds only ~2 requests, so
eviction jitter makes warm requests slower than cold (0.87×). The gains that
survive statistical re-measurement: multi-turn chains (1.44× overall — the
transfer amortizes across rounds), and concurrent clients (correctness holds,
20/20 hits, though transfer contention inflates per-request TTFT ~2.8×).

**Why small models show ~1×**: prefill on a 0.6B/8B model is very cheap, so the
cache path's fixed costs (transfer + H2D scatter, ~0.2–0.4 s) cancel the
savings. The cache wins where prefill dominates: cold starts (19.4×,
largely the first-request warmup penalty), disk-persistent reuse across
restarts, and shared-prefix serving on larger models (1.0–1.6× TTFT on 32B).
The binding constraint everywhere is transfer bandwidth — multi-turn chains
and concurrent clients both erode the gain once the moved bytes outweigh the
prefill saved.

## Benchmarks

A reproducible benchmark suite lives in `benchmarks/` (six scenarios in
`bench.py` — exact repeat, shared prefix, no-reuse baseline, 100 SQuAD-style
samples, throughput, multi-turn incremental — plus `concurrent_bench.py` for
the concurrency scenario). Design and per-scenario data requirements are
specified in [docs/benchmark_design.md](docs/benchmark_design.md). Run it against a live pair with a **fresh cache server**:

```bash
python benchmarks/bench.py --url http://localhost:8000 --model Qwen/Qwen3-0.6B \
    --tokenizer /path/to/model/dir --server-log /tmp/mini-server.log
```

Measured on Ascend 910 + Qwen3-0.6B (3072-token prompts; TTFT = time to first token):

| Scenario | Cold TTFT | Warm TTFT | Speedup | Notes |
|---|---|---|---|---|
| Exact repeat (12 chunks) | 0.413 s | 0.527 s | 0.8× | 11/11 chunks hit, output **byte-identical** to cold |
| Shared prefix (12 chunks + 4 suffixes) | 0.344 s | 0.369 s | 0.9× | prefix hits, only the suffix is computed |
| No reuse (4 distinct prompts) | 0.347 s avg | — | — | cold baseline |
| 100 SQuAD-style samples (20 contexts × 5 questions) | 0.132 s avg | 0.147 s avg | ~0.9× TTFT | **80/80 (100%) prefix hits** |

**Why warm ≈ cold here:** prefill on a 0.6B model is very cheap (~0.35 s for 3072
tokens), so the cache path's fixed costs (chunk serialization + ZMQ transfer +
H2D scatter, ~0.2-0.4 s) roughly cancel the savings. The cache still wins in the
regimes it was designed for: cold-start (19.4×, first table), disk-persistent
reuse across restarts, and larger models — prefill cost grows with model size
while transfer cost grows only with KV bytes.

## Documentation

| Doc | Contents |
|---|---|
| [docs/mini_llmcache_guide.md](docs/mini_llmcache_guide.md) | Complete walkthrough (in Chinese): overview → execution flow → MiniConnector breakdown → getting-started map |
| [docs/ascend_env.md](docs/ascend_env.md) | Ascend NPU adaptation record + full test report (in Chinese) |
| [docs/benchmark_design.md](docs/benchmark_design.md) | Benchmark design: 8-scenario system, metrics, config matrix, execution flow |

## Tests

50 pytest cases cover the hasher, pool allocator, L1 lock protocol, LRU, L2 adapters, ZMQ RPC (including
error propagation), prefetch orchestration, a full server round-trip, and a device-gated GPU⇄host transfer
round-trip (runs where a GPU/NPU is present):

```bash
pip install pytest
python -m pytest tests/
```
