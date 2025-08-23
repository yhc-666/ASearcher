#!/bin/bash
# Test SGLang server directly to see the actual error

echo "Testing SGLang server startup directly..."
echo "========================================="

# Set environment
export CUDA_VISIBLE_DEVICES=0

# Run the exact command from the launcher
python3 -m sglang.launch_server \
    --host 10.164.106.66 \
    --port 13051 \
    --tokenizer-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B \
    --tokenizer-mode auto \
    --load-format auto \
    --trust-remote-code \
    --device cuda \
    --tp-size 1 \
    --base-gpu-id 0 \
    --nnodes 1 \
    --node-rank 0 \
    --model-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B \
    --dtype bfloat16 \
    --context-length 32768 \
    --mem-fraction-static 0.9 \
    --log-level info 2>&1 | head -100

echo ""
echo "========================================="
echo "If the server fails to start, the error will be shown above."