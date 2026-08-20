# mini-llmcache Benchmark 设计文档

> 本文定义 benchmark 的目标、场景体系、指标口径与执行流程。
> 对应代码:`benchmarks/`(bench.py / sweep.py / probe.py / datasets*.py)。

## 1. 设计原则

1. **可归因**:测出的加速必须能归因到 mini-llmcache——关闭 vLLM 内置 prefix caching(`--no-enable-prefix-caching`),防止指标混淆。
2. **可复现**:数据集确定性构造(精确到 token 数、chunk 对齐)、温度 0、每次干净环境(清空 L1/L2 + 模型 warmup)。
3. **双口径**:TTFT(缓存真正加速的部分)与墙钟(含 decode)分开报,避免 decode 固定成本稀释/掩盖收益。
4. **带证据**:每个命中数字都从 server 日志按 request id 反查 `hit L1=N L2=M`,不靠客户端猜。
5. **诚实**:小模型/短 prompt 的 <1× 如实报告——benchmark 的目标是找出"什么时候缓存划算",不是证明它总是划算。

## 2. 指标口径

| 指标 | 定义 | 用途 |
|---|---|---|
| TTFT | 首 token 延迟(流式测量) | 主指标:prefill + 传输 + 首 decode |
| 墙钟 | 请求总耗时 | 次指标:含全部 decode |
| TTFT 加速 | `cold_TTFT / hot_TTFT` | prefill 口径收益 |
| 吞吐加速 | `Σ cold / Σ hot`(共享前缀场景) | 批量场景总收益 |
| 命中率 | 日志中 `hit L1=+L2=` 的请求数 / 应命中请求数 | 正确性旁证 |
| 输出一致性 | 冷热输出逐字节比对 | 缓存正确性(最高优先级) |

## 3. 场景体系(5 核心 + 3 扩展)

### 3.1 结论:核心 5 场景是否足够?

**不够完全**。现有 5 场景覆盖了"单请求的缓存复用模式"(全命中/部分命中/冷启动/真实问答/批量),但缺少三个在生产中同样关键的维度:

| 缺失维度 | 为什么重要 | 现有场景为何没覆盖 |
|---|---|---|
| 增量/多轮(缓存链延伸) | 多轮对话、流式续写:每轮新 KV 追加到已有缓存链上,是增量缓存的主战场 | 现有场景都是"完整 prompt 一次性命中" |
| 并发请求 | 真实负载是多请求并发,vLLM 批处理下的缓存收益与串行不同 | 现有场景全部串行 |
| 持久化/容量 | L2 跨重启复用、L1 溢出淘汰的正确性——系统级鲁棒性 | 只在开发期手工验证过 |

因此扩展为 **8 场景:5 核心(必测)+ 3 扩展(按需)**。

### 3.2 核心场景(每次基准必跑)

| # | 场景 | 构造 | 指标 | 验收标准 |
|---|---|---|---|---|
| 1 | 完全重复 | 同一长 prompt ×2(如 3072t/8192t/16384t 阶梯) | TTFT 加速 | 全 chunk 命中;输出逐字节一致 |
| 2 | 共享前缀 | 1 长前缀 + 4 不同后缀 | TTFT 加速 | 前缀 chunk 全命中,只计算后缀 |
| 3 | 无复用基线 | 4 条不同等长 prompt | 冷 TTFT | 无命中(纯冷参考) |
| 4 | SQuAD 式 100 条 | 20 上下文 × 5 问(优先 HF 下载,离线本地构造) | 命中率 + TTFT 合计 | 命中率 100%;报告短上下文下传输主导 |
| 5 | 吞吐 | N 请求共享前缀(如 10-20 条) | TTFT 加速 + 墙钟加速 | 首个请求后全部 L1 命中 |

### 3.3 扩展场景(按需增测)

| # | 场景 | 构造 | 指标 | 验证什么 |
|---|---|---|---|---|
| 6 | **多轮增量** | 第 1 轮 prompt P;第 2 轮 P + 续写 256t;第 3 轮再 + 256t… | TTFT 加速(逐轮) | 缓存链延伸:新增 chunk 追加、旧 chunk 复用;每轮只计算新部分 |
| 7 | **并发共享前缀** | 4-8 个并发客户端请求共享前缀(threading + requests) | 吞吐(总墙钟)+ 单请求 TTFT 分布 | vLLM 批处理下的缓存收益;锁竞争/传输争用 |
| 8 | **持久化 + 容量** | ① 杀掉 server+vllm,重启后同 prompt(纯 L2)② 小 L1(如 1GB)+ 大量不同 prompt 冲刷 | ① L2 命中延迟≈热命中 ② 淘汰后命中率下降的平滑性、无崩溃 | L2 跨重启正确性;LRU 淘汰与锁协议的鲁棒性 |

### 3.4 场景选择指南

| 目的 | 跑哪些 |
|---|---|
| 快速冒烟(验证改动没坏) | 1 + 3 |
| 单次发布验证 | 1, 2, 3, 4, 5 |
| 完整评估(含长上下文) | 全 8 + 16384t 阶梯 |
| 性能优化迭代 | 1 + 5(最快出信号)+ probe 微观定位 |

## 4. 配置矩阵

| 维度 | 取值 | 说明 |
|---|---|---|
| 模型 | 0.6B / 8B / 32B(TP=2) | 小模型验证"何时不划算";大模型是主场 |
| chunk-size | 128 / 256 / 512 / 1024 | 实测最优 256(128 帧数翻倍、512+ 粒度粗) |
| L1 大小 | 4 / 8 / 16 GB | 实测 8GB 最优 |
| prompt 长度 | 3072 / 8192 / 16384 t | 收益随长度单调放大(16384t 达 1.4×) |
| 传输 | tcp / ipc:// | ipc 比 tcp 快 ~24%(长前缀场景) |
| vLLM 前缀缓存 | 关(主测试)/ 开(部署对照) | 关闭保证可归因;开启测真实部署叠加 |

## 5. 防污染与执行流程

### 5.1 防污染清单

- [ ] 清空 server 与 L2:`rm -rf /tmp/mini-l2`(否则"冷启动"实际命中磁盘)
- [ ] 模型 warmup(首次 NPU prefill 慢 30-40 倍,不计入)
- [ ] `temperature=0`(输出确定性,可逐字节比对)
- [ ] 关闭 vLLM 内置 prefix caching(主测试)
- [ ] 记录 server 启动日志(REGISTER 行含 chunk 大小、engine 数)

### 5.2 执行流程

```
1. 启动 cache server(--chunk-size 256 --l1-size-gb 8 --l2-adapter fs)
2. 启动 vllm serve(--no-enable-prefix-caching --kv-transfer-config MiniConnector)
3. 跑 bench.py --scenarios 1,2,3,4,5 [--chunks N]
4. 需要时:sweep.py 扫配置矩阵 / probe.py 微观定位瓶颈
5. 汇总:命中证据 + 双口径加速表 + 输出一致性
6. 回归:pytest 50 全绿(改动后必跑)
```

## 6. 与代码的映射

| 设计元素 | 实现 |
|---|---|
| 数据集构造 | `benchmarks/datasets.py`(确定性 prompt)、`datasets_hf.py`(SQuAD 下载/本地回退) |
| 场景 1-5 | `benchmarks/bench.py`(`run_exact_repeat` / `run_shared_prefix` / `run_no_reuse` / `run_hf100` / `run_throughput`) |
| 配置矩阵扫描 | `benchmarks/sweep.py`(--chunk-sizes × --l1-gbs × --tensor-parallel-size) |
| 微观定位 | `benchmarks/probe.py`(LOOKUP/RETRIEVE 裸耗时) |
| 扩展场景 6-8 | 待实现(见下) |

## 7. 扩展场景实现计划

| 场景 | 实现要点 |
|---|---|
| 6 多轮增量 | bench 新增 `run_incremental`:逐轮 append 256t 后测 TTFT;断言每轮 `STORE` 只含新增 chunk、`RETRIEVE` 覆盖全部旧 chunk |
| 7 并发 | 新增 `benchmarks/concurrent.py`:ThreadPoolExecutor + requests 并发;统计总墙钟与 TTFT 分布;注意 server 端 send_lock 与 prefetch 线程的并发行为 |
| 8 持久化/容量 | 复用现有 bench + 两次启动;容量场景用 1GB L1 冲刷 50+ 不同 prompt,观察 eviction 日志与命中率曲线 |
