# mini-llmcache NPU 环境测试报告

> 测试日期:2026-08-18
> 环境:Docker 镜像 `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3`(容器 `vllm-ascend-env`)
> 硬件:3× Ascend 910 NPU,Python 3.12.13,torch 2.10.0 + torch_npu 2.10.0.post2,vLLM 0.23.0
> 模型:Qwen/Qwen3-0.6B(`/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B`)

## 一、测试结论

| 项目 | 结果 |
|---|---|
| 模型加载 | ✅ 通过(权重加载 2.4s,API 服务正常启动) |
| 推理 | ✅ 通过(HTTP 200,输出正确) |
| 缓存写入(STORE) | ✅ 通过(L0→L1 4.7 GB/s,L2 落盘 85 MB) |
| 缓存命中(RETRIEVE) | ✅ 通过(命中 3/3 chunk,**延迟 18.07s → 0.93s,加速 19.4×**) |
| L2 持久化 + 预取 | ✅ 通过(全量重启后从磁盘找回,hit L1=0 L2=3,L2→L1 4.5 GB/s) |

## 二、测试过程

测试方式:cache server 与 vllm serve 均运行在容器内,发送同一段 768 token 的重复长 prompt(`"A field guide to the birds of North America. " × 80`,恰好 3 个 chunk)。

### 1. 冷启动(第一轮)

```
HTTP 200 | total 18.07s
STORE    rid=cmpl-81afb... tokens [0, 768) L0->L1 4.7 GB/s
```

prefill 全部 768 token + 生成 24 token,新算的 3 个 chunk 存入 L1 并异步落盘到 `/tmp/mini-l2`(3 × 28 MiB 文件)。

### 2. 热命中(第二轮,同一 prompt)

```
HTTP 200 | total 0.93s
RETRIEVE rid=cmpl-a43ef... tokens [0, 768) hit L1=3 L2=0 | L2->L1 0.0 GB/s | 3 chunks sent
RETRIEVE rid=cmpl-a43ef... L1->L0 4.1 GB/s
```

3 个 chunk 全部命中 L1,prefill 完全跳过,**19.4× 加速**,两次输出逐字一致。

### 3. L2 持久化(杀掉 server + vllm,重启后再次请求)

```
HTTP 200 | total 0.97s
RETRIEVE rid=cmpl-b0c0f... tokens [0, 768) hit L1=0 L2=3 | L2->L1 4.5 GB/s | 3 chunks sent
RETRIEVE rid=cmpl-b0c0f... L1->L0 4.2 GB/s
```

L1 清空后,预取线程从磁盘把 3 个 chunk 读回(L2→L1 4.5 GB/s),再送回 GPU。预取条目为临时条目、用完即弃,因此第二次仍从磁盘读取——符合设计。

## 三、发现并修复的 Bug

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 引擎进程 `ModuleNotFoundError: No module named 'acl'` | 启动命令 `PYTHONPATH=` 赋值覆盖了镜像自带的 CANN 路径(`/usr/local/Ascend/ascend-toolkit/latest/python/site-packages`) | 改为前置追加:`PYTHONPATH=repo:$PYTHONPATH` |
| 2 | 全链路 `torch.cuda.*` 在 NPU 不可用 | 原代码只面向 NVIDIA CUDA | 新增 `l0/device.py` 设备抽象,自动选择 `torch.npu` / `torch.cuda` |
| 3 | `Event(interprocess=True)` 报错:驱动不支持 | Ascend 驱动/CANN 不支持跨进程 CUDA event IPC | 删除跨进程 event 握手,改用显式 `synchronize()` |
| 4 | `register_kv_caches` 崩溃:`'tuple' object has no attribute 'shape'` | vllm-ascend 每层 KV 是 `(k_cache, v_cache)` tuple(上游 CUDA 是融合单张量) | 按 tuple 展平,每个张量视为一个"层" |
| 5 | `unsupported KV layout (4032, 128, 8, 128)` | Ascend K/V 各自独立为 4-D `(num_blocks, block_size, nkv_heads, head_size)`,且 block 数有 padding | `kv_format.py` 新增 `FA_SPLIT` 4-D layout 检测 |
| 6 | 服务器导入 IPC 句柄失败:`halMemImportFromShareableHandle failed` | **NPU 驱动无法跨进程共享设备内存**(对照官方 LMCache-Ascend 后确认:官方同样不在进程间共享显存) | **架构重构**:搬运全部移到 vLLM 进程内(connector 直接持有本地 KV 张量,GPU⇄CPU 拷贝在进程内完成),跨进程只走 ZMQ 传字节;cache server 退化为纯 CPU 内存管理 |

## 四、与官方 LMCache-Ascend 的架构对照

修复 6 是本次最关键的重构,方案直接参照官方仓库:

| | mini-llmcache(修复后) | LMCache-Ascend 官方 |
|---|---|---|
| KV 张量归属 | 留在 vLLM 进程,connector 持有引用 | 同左 |
| GPU⇄CPU 搬运 | vLLM 进程内(双 CUDA/NPU stream + 3 缓冲流水线) | 同左(`NpuConnector` 的 from_gpu/to_gpu) |
| 跨进程传输 | ZMQ 传 bytes(localhost) | Mooncake/RDMA 或 TCP |
| 缓存服务器 | 纯 CPU:L1 内存池 + L2 磁盘 | 同构(另有 GPU 版本) |

## 五、已知限制

1. 跨进程数据走 ZMQ + pickle,吞吐受限于序列化拷贝(本机测试 4~6 GB/s,可接受);官方用 RDMA 可达更高
2. `wait_for_save` / RETRIEVE 完成后各有一次设备级 `synchronize()`,牺牲少量流水线重叠换取跨平台正确性
3. 未测试 TP > 1、多引擎并发、请求抢占(preemption)场景
4. 首次 STORE/RETRIEVE 有 NPU warmup(本测试冷启动 18s 主要含首次 prefill)
