# mini-llmcache 5 场景基准报告

日期：2026-08-20

## 1. 运行配置

| 项 | 值 |
|---|---|
| 容器 | `vllm-ascend-env` |
| 模型 | `Qwen/Qwen3-32B` |
| Tensor Parallel | 2 |
| chunk-size | 256 |
| L1 | 8 GB |
| L2 | `/tmp/mini-l2` |
| prompt 长度 | 3072 token（`--chunks 12`） |
| 前缀长度 | 3072 token（`--prefix-chunks 12`） |
| 场景数 | 5 |
| 吞吐请求数 | 20 |

运行命令：

```bash
docker exec vllm-ascend-env bash -lc '
cd /home/jianzhnie/llmtuner/llm/mini-llmcache &&
PYTHONPATH=/home/jianzhnie/llmtuner/llm/mini-llmcache:$PYTHONPATH \
python benchmarks/bench.py \
  --url http://localhost:8000 \
  --model Qwen/Qwen3-32B \
  --tokenizer /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-32B \
  --server-log /tmp/mini-server.log \
  --chunks 12 \
  --prefix-chunks 12 \
  --scenarios 1,2,3,4,5 \
  --throughput-n 20
'
```

## 2. 总结

| 场景 | 结果 |
|---|---|
| 1 完全重复 | 1.0x TTFT，输出一致 |
| 2 共享前缀 | 0.9x TTFT |
| 3 无复用 | 冷启动基线 |
| 4 SQuAD 式 100 条 | 0.85x TTFT |
| 5 吞吐 | 1.2x TTFT，1.1x 墙钟 |

结论：

- 3072 token、32B TP=2 下，缓存收益已经能在吞吐场景里跑到 1x 以上。
- 完全重复场景下没有明显 TTFT 优势，说明 decode 和传输开销仍然占比较高。
- 短上下文的 SQuAD 式数据仍然被传输成本压住，整体低于 1x。

## 3. 详细结果

### 场景 1：完全重复命中

| 指标 | 值 |
|---|---|
| 冷 TTFT | 0.823s |
| 热 TTFT | 0.808s |
| 加速 | 1.0x |
| 命中证据 | `RETRIEVE ... hit L1=11 L2=0 | 11 chunks sent` |
| 输出一致 | 是 |

### 场景 2：共享前缀

| 指标 | 值 |
|---|---|
| 冷参考 TTFT | 0.761s |
| 热平均 TTFT | 0.812s |
| 加速 | 0.9x |
| 命中证据 | 每个热请求均 `hit L1=10 L2=0` |

### 场景 3：无复用

| 指标 | 值 |
|---|---|
| 平均冷 TTFT | 0.738s |
| 单条结果 | 0.704s / 0.760s / 0.772s / 0.714s |

### 场景 4：SQuAD 式 100 条

| 指标 | 值 |
|---|---|
| 冷启动（20 次首问） | 4.82s，总平均 0.241s |
| 前缀命中（80 次） | 23.45s，总平均 0.293s |
| TTFT 合计 | 28.27s |
| 全部重算预估 | 24.10s |
| TTFT 加速 | 0.85x |
| 墙钟总耗时 | 226.36s |
| 命中率 | 80/80 |

### 场景 5：吞吐

| 指标 | 值 |
|---|---|
| 冷参考 TTFT | 0.869s |
| 冷参考 total | 2.885s |
| 缓存累计 TTFT | 13.53s |
| 无缓存预估 TTFT | 16.51s |
| TTFT 加速 | 1.2x |
| 缓存累计墙钟 | 51.59s |
| 无缓存预估墙钟 | 54.82s |
| 墙钟加速 | 1.1x |
| 搬运量 | 7.12 GiB |

## 4. 日志说明

- 场景 1 命中 11/11 chunks，输出一致。
- 场景 2 命中 10/10 chunks。
- 场景 4 使用了离线本地构造数据集。
  - 原因：容器里 `datasets` 包没有可用的 `load_dataset` 接口。
  - 这不影响 benchmark 结果的执行，只影响数据来源。

## 5. 结论

当前配置下，最明显的收益来自吞吐场景；短上下文和完全重复场景都还没把传输与 decode 成本压下去。对这个模型规模，缓存价值主要集中在共享前缀批量请求。
