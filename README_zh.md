# mini-llmcache

[English](README.md) | **中文**

mini-llmcache 是 [LMCache](https://github.com/LMCache/LMCache) 的迷你教学版,实现了一个 **LLM KV Cache 共享系统**:为 vLLM 挂载一个独立的 KV Cache 缓存服务器,把 LLM 算过的 prompt"草稿"按 256 token 切块哈希后存进 L1 内存 / L2 磁盘,下次遇到相同前缀直接取回、跳过 prefill —— **GPU 与 NPU 上实测加速 19.4×**,支持 NVIDIA GPU(torch.cuda)与 Ascend NPU(torch.npu)双平台。

全部代码约 870 行 Python,只依赖 `pyzmq`、`blake3`、`torch`:前缀哈希、两级缓存、引用计数锁、流水线搬运——LLM KV Cache 共享的核心思想都以可读的形式呈现。

- **双平台**:通过轻量设备抽象层同时支持 NVIDIA GPU 与 Ascend NPU。NPU 适配细节见 [docs/ascend_env.md](docs/ascend_env.md)。
- **两级存储**:L1 为 pinned 主机内存,L2 为可插拔 adapter(自带文件系统实现,约 20 行即可自写一个)。
- **与官方 LMCache-Ascend 对齐的架构**:GPU⇄CPU 拷贝全部发生在 vLLM 进程内,跨进程只传字节,缓存服务器为纯 CPU。

## 快速开始

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

> 首次 STORE/RETRIEVE 需要 warmup,会慢 30~40 倍,取吞吐数据用第二轮。
>
> **Ascend NPU 注意**:vllm-ascend 镜像的 `$PYTHONPATH` 自带 CANN 路径,启动时必须**前置追加**仓库路径
> (`PYTHONPATH=repo:$PYTHONPATH python3 -m mini_llmcache.server ...`),用 `=` 覆盖会导致引擎进程报
> `No module named 'acl'`。详见 [docs/ascend_env.md](docs/ascend_env.md)。

## 实测成绩单

| 场景 | 延迟 | 说明 |
|---|---|---|
| 冷启动(首次请求) | 18.07 s | prefill 768 token + 生成 24 token,顺带 STORE 入库 |
| 热命中(L1) | **0.93 s** | 命中 3/3 chunk,prefill 完全跳过,**加速 19.4×** |
| 全量重启后(L2 磁盘) | 0.97 s | 从磁盘预取回 L1 再回填 GPU(L2→L1 4.5 GB/s) |

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/mini_llmcache_guide.md](docs/mini_llmcache_guide.md) | 完全拆解:总览 + 执行流程 + MiniConnector 副驾拆解 + 上手地图 |
| [docs/ascend_env.md](docs/ascend_env.md) | Ascend NPU 适配记录 + 完整测试报告 |

## 测试

49 个 pytest 用例覆盖:哈希、内存池、L1 锁协议、LRU、L2 adapter、ZMQ RPC(含异常回传)、预取编排、
服务器全流程往返,以及设备门控的 GPU⇄CPU 搬运往返测试(有 GPU/NPU 时运行):

```bash
pip install pytest
python -m pytest tests/
```
