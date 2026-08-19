# mini-llmcache

**English** | [中文](README_zh.md)

mini-llmcache is a pedagogical [LMCache](https://github.com/LMCache/LMCache) reimplementation: a standalone KV-cache server for vLLM that stores hashed prompt chunks and replays matching prefixes to skip prefill — **19.4× faster on GPU and NPU**.

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

| Scenario | Latency | Notes |
|---|---|---|
| Cold (first request) | 18.07 s | prefill 768 tokens + 24 output tokens, plus STORE |
| Warm L1 hit | **0.93 s** | 3/3 chunks hit, prefill fully skipped — **19.4× speedup** |
| After full restart (L2 disk) | 0.97 s | chunks prefetched from disk back to L1 (L2→L1 4.5 GB/s) |

## Documentation

| Doc | Contents |
|---|---|
| [docs/mini_llmcache_guide.md](docs/mini_llmcache_guide.md) | Complete walkthrough (in Chinese): overview → execution flow → MiniConnector breakdown → getting-started map |
| [docs/ascend_env.md](docs/ascend_env.md) | Ascend NPU adaptation record + full test report (in Chinese) |

## Tests

49 pytest cases cover the hasher, pool allocator, L1 lock protocol, LRU, L2 adapters, ZMQ RPC (including
error propagation), prefetch orchestration, a full server round-trip, and a device-gated GPU⇄host transfer
round-trip (runs where a GPU/NPU is present):

```bash
pip install pytest
python -m pytest tests/
```
