# mini-llmcache

请帮我测试一下 mini-llmcache 是否正常工作，包括模型加载、推理、缓存等， 并提供测试结果，测试过程遇到Bug, 请自行定位并修复。

## 环境

Docker 镜像： quay.io/ascend/vllm-ascend:v0.23.0rc1-a3  
Docker 容器： vllm-ascend-env
权重路径： /home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B

## 测试 mini-llmcache

```bash
python -m mini_llmcache.server --port 45881 --l1-size-gb 8 \
    --l2-adapter '{"type": "fs", "base_path": "/tmp/mini-l2"}'
```

```bash
vllm serve Qwen/Qwen3-0.6B --enforce-eager \
    --max-model-len 4096 --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "MiniConnector",
                           "kv_connector_module_path": "mini_llmcache.integration.vllm_connector",
                           "kv_role": "kv_both",
                           "kv_connector_extra_config": {"mini.port": 45881}}'
```

```bash
curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d "{
    \"model\": \"Qwen/Qwen3-0.6B\", \"temperature\": 0, \"max_tokens\": 24,
    \"prompt\": \"$(python3 -c "print('A field guide to the birds of North America. ' * 80)")\"}"
```
