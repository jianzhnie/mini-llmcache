# vllm_connector.py 拆解 — vLLM 的"外挂副驾"

> 目标文件:`lmcache-mini/lmcache_mini/integration/vllm_connector.py`(232 行)
> 它没有 main、从不单独运行——vLLM 启动时按 `--kv-transfer-config` 反射加载它,
> 然后在自己调度循环的固定位置,一遍遍调用它的"钩子"。

## 1. 它在整个系统里的位置

缓存服务器(仓库)只会被动响应;谁来决定"何时存、何时取、能跳多少"?就是这个文件。
它像给 vLLM 装的**副驾**:vLLM 专心开车(调度、计算),副驾负责跟仓库联络。

vLLM 提供的接口叫 `KVConnectorBase_V1`——本质是一张钩子清单,`MiniConnector` 把每个钩子填上自己的逻辑:

| vLLM 钩子 | vLLM 什么时候叫 | 副驾干什么 |
|---|---|---|
| `register_kv_caches` | 启动时 | 把 KV 张量变成 IPC 句柄,注册到服务器 |
| `get_num_new_matched_tokens` | 每个调度步,反复问 | 首次发 LOOKUP;之后轮询预取结果,报告"可跳过几个 token" |
| `update_state_after_alloc` | 给请求分好 block 后 | 记下 block 编号;标记"要搬货";发 FREE_LOOKUP_LOCKS |
| `build_connector_meta` | 每个调度步 | 读调度结果更新账本,开出 RETRIEVE / STORE 搬运单 |
| `start_load_kv` | 开始本轮计算**前** | 提交 RETRIEVE 单(搬货与计算并行) |
| `wait_for_save` | 本轮计算**后** | 提交 STORE 单 |
| `get_finished` | 调度循环查询 | 报告哪些请求搬完 / 存完了 |
| `request_finished` | 请求结束时 | 撕账本,发 END_SESSION |
| `shutdown` | 引擎关闭 | 注销会员卡,挂断电话 |

## 2. RequestTracker — 每个请求一本流水账

钩子会被 vLLM 反复调用,connector 自己没机会"记住"请求状态,于是给每个请求建一个账本(`vllm_connector.py:32`):

```
token 坐标轴: 0 ───────── vllm_hits ───────── lmcache_hits ─────────── 结尾
               │ vLLM 已算 │ 仓库有,要搬回来  │ 两边都没有,算完再存 │
               │           │   (RETRIEVE)     │      (STORE)        │
               └─ 首块重叠处用 skip_first_n_tokens 跳过,防止覆盖 ─────┘
```

关键字段:

- `lmcache_hits` 仓库说命中多少 token;`vllm_hits` vLLM 自己已算到哪——**两数之差就是要搬回来的部分**
- `num_stored_tokens` 已存进仓库多少——**存过的绝不再存**
- `block_ids` 请求占用的 GPU block 编号表,搬运单全靠它
- `lookup_resolved` / `load_pending` / `alloc_seen` 三个状态位,防止重复干活

## 3. 核心钩子的工作方式

### get_num_new_matched_tokens — "帮我算好了吗?"(line 93)

vLLM 每步都来问一次,回答分三种:

| 返回值 | 意思 |
|---|---|
| `(None, True)` | 仓库还没查完,下一步再来问 |
| `(n, True)` | 前 n 个 token 外部已算好,跳过! |
| `(0, False)` | 查完了,没有便宜可占,自己全算 |

第一次被问:发 **LOOKUP**(异步,发完就走),然后照常轮询;之后每次:打 `QUERY_PREFETCH_STATUS` 问预取线程查好没。注意是**轮询**而非阻塞等待——vLLM 一边 prefill 一边问,查到为止。

### update_state_after_alloc — "block 分好了"(line 117)

vLLM 分完 block 后调它,副驾做三件事:

1. 把新分配的 block 编号追加进账本
2. 有外部 token(`num_external_tokens > 0`)?→ 标记 `load_pending`,该搬货了
3. 首次分配时算好"哪些预取锁可以解":vLLM 自己会算的部分锁先放掉,要搬回来的部分**继续押着**——`FREE_LOOKUP_LOCKS`

### build_connector_meta — "开搬运单"(line 134)

每步调度结束,副驾读 `scheduler_output`(谁新来了、谁续算、谁这轮算了多少 token),更新账本,开出两张单:

- **RETRIEVE 单**(`generate_retrieve_op`,line 67):只开给 `load_pending` 的请求,范围 = [vllm_hits 向下取整到块边界, lmcache_hits);首块中 vLLM 已算的部分用 `skip_first_n_tokens` 标出,避免覆盖
- **STORE 单**(`generate_store_op`,line 78):三个上限取最小——① 请求总 token 数 ② block 装得下的 ③ vLLM 实际算完的;只存整块,存过的不再存

单子打包成 `MiniConnectorMetadata`(requests 列表)交给 vLLM。

### 两张单何时送出?

- `start_load_kv`(line 200):**计算前**提交 RETRIEVE——搬货和算尾巴并行,互不耽误
- `wait_for_save`(line 203):**计算后**提交 STORE——货算齐了才入库

## 4. 三个巧妙的设计

### ① DeviceFuture 双保险(`l0/transfer.py:26`)

服务器回消息("我干完了")≠ 数据真的就绪了。副驾要的是 **GPU 事件**:服务器写完显存后 record 一个 interprocess CUDA event,把句柄随消息发回来;`DeviceFuture.done()` 先看消息到没到,再看 GPU event 发没发生——两步都过,才算"搬完"。

### ② producer event 同步(line 190)

提交搬运单前,副驾先 record 一个 event 发给服务器:"我这边的显存状态已冻结,你可以安全读写了"——防止服务器在 vLLM 还在写 KV 时动手抢。

### ③ 两个空钩子,一条路线选择(line 224-228)

`wait_for_layer_load` / `save_kv_layer` 竟然是 `pass`!这不是偷懒:vLLM 给了两条路——"逐层加载 / 逐层保存"(这两个钩子),或"整块搬运"(metadata + submit_transfers)。本实现选了后者:所有搬运信息在 `build_connector_meta` 里打包、一次提交,服务器端用流水线整块处理——LMCache 官方实现同样是整块路线。

## 5. 一张表总结:消息从哪来

| 消息 | 从哪个钩子发出 | 同步 or 异步 |
|---|---|---|
| LOOKUP | get_num_new_matched_tokens(首次) | 异步,发完就走 |
| QUERY_PREFETCH_STATUS | get_num_new_matched_tokens(轮询) | 同步,快问快答 |
| FREE_LOOKUP_LOCKS | update_state_after_alloc | 异步 |
| RETRIEVE | start_load_kv | 异步,DeviceFuture 跟踪 |
| STORE | wait_for_save | 异步,DeviceFuture 跟踪 |
| END_SESSION | request_finished | 异步 |
| REGISTER / UNREGISTER | register_kv_caches / shutdown | 同步,要确认结果 |

---

一句话:**vLLM 每走一步,副驾就问一句、记一笔、开一单**——问(LOOKUP/QUERY)、记(tracker 账本)、开单(RETRIEVE/STORE),就是这个文件全部的工作。
