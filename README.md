# mini-llmcache

mini-llmcache 是 [LMCache](https://github.com/LMCache/LMCache) 的迷你教学版， 实现了一个 **LLM KV Cache 共享系统**，为 vLLM 挂载一个独立的 KV Cache 缓存服务器, 把 LLM 算过的 prompt"草稿"按 256 token 切块哈希后存进 L1 内存 / L2 磁盘,下次遇到相同前缀直接取回、跳过 prefill(实测 19.4× 加速), 支持 NVIDIA GPU(torch.cuda)与 Ascend NPU(torch.npu)双平台。

## Quick Start

```bash
# 1. 起缓存服务器(L1 8GB,L2 落盘到 /tmp/mini-l2)
python -m mini_llmcache.server --port 45881 --l1-size-gb 8 \
    --l2-adapter '{"type": "fs", "base_path": "/tmp/mini-l2"}'
```

```bash
# 2. 起 vLLM,挂上 MiniConnector
vllm serve Qwen/Qwen3-0.6B --enforce-eager \
    --max-model-len 4096 --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "MiniConnector",
                           "kv_connector_module_path": "mini_llmcache.integration.vllm_connector",
                           "kv_role": "kv_both",
                           "kv_connector_extra_config": {"mini.port": 45881}}'
```

```bash
# 3. 发一个带超长重复前缀的请求(第二遍就会命中缓存)
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "{
    \"model\": \"Qwen/Qwen3-0.6B\", \"temperature\": 0, \"max_tokens\": 24,
    \"prompt\": \"$(python3 -c "print('A field guide to the birds of North America. ' * 80)")\"}"
```

cache server 输出(Ascend 910 + Qwen3-0.6B 实测):

```
mini cache server up and ready on tcp://127.0.0.1:45881 (chunk_size=256, 8 GB L1, 1 L2 adapters)
REGISTER Qwen/Qwen3-0.6B rank 0/1 (chunk=28 MiB, 1 engines)
STORE    rid=cmpl-... tokens [0, 768) L0->L1 4.7 GB/s          ← 第一遍:算完存起来
RETRIEVE rid=cmpl-... tokens [0, 768) hit L1=3 L2=0 | 3 chunks ← 第二遍:直接搬回来,prefill 全跳过
```

> 注意:首次 STORE/RETRIEVE 需要 warmup,会慢 30~40 倍,取吞吐数据用第二轮。

## 实测成绩单

| 场景 | 延迟 | 说明 |
|---|---|---|
| 冷启动(首次请求) | 18.07 s | prefill 768 token + 生成 24 token,顺带 STORE 入库 |
| 热命中(L1) | **0.93 s** | 命中 3/3 chunk,prefill 完全跳过,**加速 19.4×** |
| 全量重启后(L2 磁盘) | 0.97 s | 从磁盘预取回 L1 再回填 GPU,L2→L1 4.5 GB/s |

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/mini_llmcache_guide.md](docs/mini_llmcache_guide.md) | 完全拆解:总览 + 执行流程 + MiniConnector 副驾拆解 + 上手地图 |
| [docs/ascend_env.md](docs/ascend_env.md) | Ascend NPU 适配记录 + 完整测试报告 |
