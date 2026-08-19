# mini-llmcache NPU 环境测试报告

> 测试日期:2026-08-18
> 环境:Docker 镜像 `quay.io/ascend/vllm-ascend:v0.23.0rc1-a3`(容器 `vllm-ascend-env`)
> 硬件:3× Ascend 910 NPU,Python 3.12.13,torch 2.10.0 + torch_npu 2.10.0.post2,vLLM 0.23.0
> 模型:Qwen/Qwen3-0.6B(`/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B`)
> 相关阅读:[完全拆解(总览 + 执行流程 + 副驾)](mini_llmcache_guide.md)

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

| # | 现象 | 根因 | 修复 | 代码位置 |
|---|---|---|---|---|
| 1 | 引擎进程 `ModuleNotFoundError: No module named 'acl'` | 启动命令 `PYTHONPATH=` 赋值覆盖了镜像自带的 CANN 路径(`/usr/local/Ascend/ascend-toolkit/latest/python/site-packages`) | 改为前置追加:`PYTHONPATH=repo:$PYTHONPATH` | 操作层面,非代码 |
| 2 | 全链路 `torch.cuda.*` 在 NPU 不可用 | 原代码只面向 NVIDIA CUDA | 新增 `l0/device.py` 设备抽象,自动选择 `torch.npu` / `torch.cuda` | `mini_llmcache/l0/device.py`(全文,26 行 `DEV = _pick_backend()`) |
| 3 | `Event(interprocess=True)` 报错:驱动不支持 | Ascend 驱动/CANN 不支持跨进程 CUDA event IPC | 删除跨进程 event 握手,改用显式 `synchronize()` | `mini_llmcache/l0/transfer.py`(`to_host`/`from_host`) |
| 4 | `register_kv_caches` 崩溃:`'tuple' object has no attribute 'shape'` | vllm-ascend 每层 KV 是 `(k_cache, v_cache)` tuple(上游 CUDA 是融合单张量) | 按 tuple 展平,每个张量视为一个"层" | `mini_llmcache/integration/vllm_connector.py:171` `register_kv_caches` |
| 5 | `unsupported KV layout (4032, 128, 8, 128)` | Ascend K/V 各自独立为 4-D `(num_blocks, block_size, nkv_heads, head_size)`,且 block 数有 padding | `kv_format.py` 新增 `FA_SPLIT` 4-D layout 检测 | `mini_llmcache/l0/kv_format.py:8,22` |
| 6 | 服务器导入 IPC 句柄失败:`halMemImportFromShareableHandle failed` | **NPU 驱动无法跨进程共享设备内存**(对照官方 LMCache-Ascend 后确认:官方同样不在进程间共享显存) | **架构重构**:搬运全部移到 vLLM 进程内(connector 直接持有本地 KV 张量,GPU⇄CPU 拷贝在进程内完成),跨进程只走 ZMQ 传字节;cache server 退化为纯 CPU 内存管理 | 4 个文件,详见[第五节](#五第-6-项重构代码位置明细) |

## 四、与官方 LMCache-Ascend 的架构对照

修复 6 是本次最关键的重构,方案直接参照官方仓库:

| | mini-llmcache(修复后) | LMCache-Ascend 官方 |
|---|---|---|
| KV 张量归属 | 留在 vLLM 进程,connector 持有引用 | 同左 |
| GPU⇄CPU 搬运 | vLLM 进程内(双 CUDA/NPU stream + 3 缓冲流水线) | 同左(`NpuConnector` 的 from_gpu/to_gpu) |
| 跨进程传输 | ZMQ 传 bytes(localhost) | Mooncake/RDMA 或 TCP |
| 缓存服务器 | 纯 CPU:L1 内存池 + L2 磁盘 | 同构(另有 GPU 版本) |

## 五、第 6 项重构:代码位置明细

### 5.1 改动前后对比

```
改动前(CUDA 设计):                          改动后(NPU 设计,对齐官方 LMCache-Ascend):
┌─────────────┐   IPC 共享显存   ┌─────────┐   ┌─────────────┐   ZMQ 传 bytes  ┌─────────┐
│ vLLM 进程   │◄═══════════════►│ 服务器  │   │ vLLM 进程   │═══════════════►│ 服务器  │
│ 只负责计算  │                 │ 直读/写 │   │ 计算+搬运    │               │ 纯 CPU  │
│             │                 │ vLLM显存│   │ GPU⇄CPU 拷贝 │               │ L1/L2 管理│
└─────────────┘                 └─────────┘   └─────────────┘               └─────────┘
```

### 5.2 四个文件的改动明细

**① `mini_llmcache/l0/transfer.py` — 搬运引擎重写(改动最大)**

| 位置 | 改动 |
|---|---|
| `transfer.py:62` `KVTransfer` | 改动前:构造时 `import_kv_caches(ipc_handles)` 在服务器进程重建 vLLM 的显存张量;改动后:直接持有 vLLM 进程内本地张量引用,不再有任何跨进程共享 |
| `transfer.py:77` `to_host()` | 新增:GPU→CPU 拷贝(D2H),返回 `list[bytes]` 和耗时——原来此工作在服务器进程做 |
| `transfer.py:111` `from_host()` | 新增:bytes→GPU 回填(H2D,`index_copy_` + `skip_blocks`)——原来也在服务器进程做 |
| `transfer.py:10` `DeviceFuture` | 增加 `done_event`:`done()` 要求"消息回执 + GPU 回填线程完成"双条件 |

双流三缓冲流水线(`Pipeline`,`transfer.py:32`)原样保留,只是运行位置从服务器挪进 vLLM 进程。

**② `mini_llmcache/integration/vllm_connector.py` — 搬运的发起方**

- `vllm_connector.py:171` `register_kv_caches`:不再 `export_kv_caches` 导出 IPC 句柄,
  改为本地构建 `self.transfer = KVTransfer(views, ...)`(186 行),注册时只发元数据 `chunk_nbytes`
- `vllm_connector.py:197` `submit_transfers`:
  - STORE 分支(208 行):**先在本进程 `to_host()` 拷出 bytes**,再随 `TransferPayload` 发给服务器
  - RETRIEVE 分支:异步提交请求,并起一个后台线程
- `vllm_connector.py:227` `_finish_load`:后台线程拿到服务器返回的 bytes 后,
  **在本进程 `from_host()` 回填进 KV 显存**,完成后发 `TRANSFER_ACK`、`done.set()`
  让 vLLM 知道可以读这些 block

**③ `mini_llmcache/server.py` — 服务器退化为纯 CPU**

- `server.py:16` `EngineInstance`:删掉 `transfer: KVTransfer` 字段,只留 `chunk_nbytes` 等元数据
- `server.py:100` `store`:不再调 `transfer.store()`,改为把 payload 里的 bytes 直接写入 L1 内存对象(`obj.byte_array[:] = blob`)
- `server.py:121` `retrieve`:不再调 `transfer.load()`,改为把 L1 对象序列化成 bytes 返回
- `server.py:136` `ack`:新增 `TRANSFER_ACK` 处理器,打印 connector 侧实测的 L1→L0 吞吐

**④ `mini_llmcache/protocol.py` — 消息结构跟着变**

- `protocol.py:61` `TransferPayload`:`event_handle` 字段换成 `chunks: list[bytes]` + `elapsed` + `nbytes`
- `protocol.py:16` 新增 `TRANSFER_ACK = 10`;`protocol.py:71` 新增 `AckPayload`
- `protocol.py:43` `RegisterPayload`:`ipc_handles` 字段换成 `chunk_nbytes`

### 5.3 重构后的数据通路

```
STORE:    vLLM 算完 → 进程内 D2H(双流流水线) → bytes → ZMQ → 服务器写 L1 → 后台落盘 L2
RETRIEVE: 服务器读 L1/L2 → bytes → ZMQ → 后台线程 H2D 回填 vLLM 显存 → done 事件 → vLLM 使用
```

## 六、已知限制

1. 跨进程数据走 ZMQ + pickle,吞吐受限于序列化拷贝(本机测试 4~6 GB/s,可接受);官方用 RDMA 可达更高
2. `wait_for_save` / RETRIEVE 完成后各有一次设备级 `synchronize()`,牺牲少量流水线重叠换取跨平台正确性
3. 未测试 TP > 1、多引擎并发、请求抢占(preemption)场景
4. 首次 STORE/RETRIEVE 有 NPU warmup(本测试冷启动 18s 主要含首次 prefill)
