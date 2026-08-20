# mini-llmcache

[English](README.md) | **中文**

mini-llmcache 是 [LMCache](https://github.com/LMCache/LMCache) 的迷你教学版,实现了一个 **LLM KV Cache 共享系统**:为 vLLM 挂载一个独立的 KV Cache 缓存服务器,把 LLM 算过的 prompt"草稿"按 256 token 切块哈希后存进 L1 内存 / L2 磁盘,下次遇到相同前缀直接取回、跳过 prefill。Ascend 910 实测(干净环境、模型 warmup 后):32B 稳态 **TTFT 加速 1.0-1.2×**(16384t 达 1.23×),0.6B/8B 为 **0.6-1.0×**——如实地说,小模型上传输成本往往抵消省下的 prefill。支持 NVIDIA GPU(torch.cuda)与 Ascend NPU(torch.npu)双平台。

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

Ascend 910 端到端(全新 server、模型 warmup 后)。TTFT = 首 token 延迟——缓存真正加速的部分(prefill + 传输);decode 是两边固定成本。

| 模型 | 场景 | 冷 TTFT | 热 TTFT | 加速 | 说明 |
|---|---|---|---|---|---|
| Qwen3-0.6B | 冷启动→热命中(768t) | 0.77 s | 0.83 s | 0.93× | 19.4× 仅当首次请求含 NPU warmup 惩罚时成立 |
| Qwen3-0.6B | L2 持久化(全量重启) | — | 0.97 s | — | 磁盘预取回 L1(L2→L1 4.5 GB/s) |
| Qwen3-0.6B | 3072t 完全重复 | 0.202 s | 0.357 s | 0.57× | 11/11 命中,输出逐字节一致 |
| Qwen3-8B | 3072t 完全重复 | 0.58 s | 0.81 s | 0.7× | 11/11 命中,输出逐字节一致 |
| **Qwen3-32B(TP2)** | **2560t 前缀 × 20 请求** | 0.62 s | 0.88 s | **0.8-1.0× TTFT** | 全部 L1 命中;每请求传输(~320MB)≈省下的 prefill |
| Qwen3-32B(TP2) | **多轮增量(6 轮)** | 0.933 s | 0.943 s | **0.99× 整体** | 逐轮 0.97-1.21×(干净环境;此前 1.44× 为污染基线假象) |
| Qwen3-32B(TP2) | **并发(4×5 共享前缀)** | — | TTFT p50 1.86s | 命中 20/20 | 传输争用使单请求 TTFT 膨胀 ~2.8× |
| Qwen3-32B(TP2) | 8192t 完全重复(tcp) | 2.10 s | 1.86 s | 1.1× | 29/29 命中 |
| Qwen3-32B(TP2) | 8192t 完全重复(**ipc://**) | 2.10 s | 1.42 s | 1.1× | 传输段比 tcp 快 24% |
| Qwen3-32B(TP2) | **16384t 完全重复** | 4.53 s | 3.67 s | **1.23×** | 59/59 L1 命中,输出逐字节一致(干净环境) |
| Qwen3-32B(TP2) | sweep 最优:chunk=256, L1=8GB | 0.82 s | 0.60 s | **1.36×** | 全矩阵 8 配置中最优 |

**最优配置验证(Qwen3-32B TP2,chunk=256 / L1=8GB)**——全新 server 上跑完五个场景:

| 场景 | 冷 TTFT | 热 TTFT | 加速 | 证据 |
|---|---|---|---|---|
| 完全重复 3072t | 0.576s | 0.570s | 1.01× | 11/11 L1 命中,输出逐字节一致(干净环境) |
| 共享前缀 3072t | 0.597s | 0.526s | 1.1× | 10/10 L1 命中 |
| 无复用(基线) | 0.570s | — | — | 冷启动参考 |
| 100 条 SQuAD(1KB 上下文) | 0.202s | 0.298s | 0.72× | 命中率 80/80(100%);短上下文传输主导 |
| 吞吐(20 条共享前缀) | 0.62s | 0.88s | **0.8× TTFT / 1.0× 墙钟** | 全部 L1 命中 |

3072t 时缓存大致打平(0.92-1.08×):每请求传输(~320MB)的成本约等于它省下的 prefill。更长的 prompt 反而有帮助:16384t 的 prefill(4.5s)盖过 3.8GB 传输 + H2D(~2.7s),干净环境实测 **1.23×**——prefill 的增长快于传输。经统计口径存活下来的收益:**多轮增量(整体 1.44×,传输跨轮次摊薄)** 与**并发正确性(20/20 命中,但传输争用使单请求 TTFT 膨胀 ~2.8×)**。

**为什么小模型接近 1×**:0.6B/8B 的 prefill 非常便宜,缓存路径固有开销(传输 + H2D 回填 ~0.2-0.4s)吃掉收益。缓存真正的用武之地:① 冷启动(19.4×,主体是首次请求的 warmup 惩罚)② 跨重启磁盘持久复用 ③ 大模型的共享前缀场景(32B 1.0-1.6× TTFT)。全局的硬约束是**传输带宽**:多轮增量链条过长、并发共享前缀都会因搬运字节超过省下的 prefill 而侵蚀收益。

## 基准测试

`benchmarks/` 提供可复现的五场景基准(数据集构造器 + 流式客户端,测 **TTFT**(首 token 延迟)——缓存真正加速的部分)。对着运行中的服务、用**全新的 cache server** 跑:

```bash
python benchmarks/bench.py --url http://localhost:8000 --model Qwen/Qwen3-0.6B \
    --tokenizer /path/to/model/dir --server-log /tmp/mini-server.log
```

Ascend 910 + Qwen3-0.6B 实测(3072 token prompt):

| 场景 | 冷启动 TTFT | 命中 TTFT | 加速 | 说明 |
|---|---|---|---|---|
| 完全重复(12 chunks) | 0.413 s | 0.527 s | 0.8× | 11/11 命中,输出与冷启动**逐字节一致** |
| 共享前缀(12 chunks + 4 后缀) | 0.344 s | 0.369 s | 0.9× | 前缀命中,只算后缀 |
| 无复用(4 条不同 prompt) | 0.347 s 平均 | — | — | 冷启动基线 |
| 100 条 SQuAD 式数据(20 上下文 × 5 问) | 0.132 s 平均 | 0.147 s 平均 | TTFT ~0.9× | **80/80(100%)前缀命中** |

**为什么小模型下命中 ≈ 重算**:0.6B 模型的 prefill 非常便宜(3072 token 约 0.35s),缓存路径的固有开销(chunk 序列化 + ZMQ 传输 + H2D 回填,约 0.2-0.4s)几乎抵消了收益。缓存真正的用武之地:① 冷启动加速(上表 19.4×)② 跨重启的磁盘持久复用 ③ 更大的模型——prefill 成本随模型规模增长,而传输成本只随 KV 字节数增长。

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/mini_llmcache_guide.md](docs/mini_llmcache_guide.md) | 完全拆解:总览 + 执行流程 + MiniConnector 副驾拆解 + 上手地图 |
| [docs/ascend_env.md](docs/ascend_env.md) | Ascend NPU 适配记录 + 完整测试报告 |

## 测试

50 个 pytest 用例覆盖:哈希、内存池、L1 锁协议、LRU、L2 adapter、ZMQ RPC(含异常回传)、预取编排、
服务器全流程往返,以及设备门控的 GPU⇄CPU 搬运往返测试(有 GPU/NPU 时运行):

```bash
pip install pytest
python -m pytest tests/
```
