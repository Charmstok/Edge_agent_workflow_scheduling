#!/usr/bin/env bash

# Download the model weights, pull the vLLM image, and create the two local
# model containers. Weights are kept in the host Hugging Face cache.

set -euo pipefail

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/data/huggingface}"
CONTAINER_9B="${VLLM_9B_CONTAINER:-qwen35-9b}"
CONTAINER_27B="${VLLM_27B_CONTAINER:-qwen38-27b-fp8}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'error: required command not found: %s\n' "$1" >&2
    exit 1
  }
}

container_exists() {
  docker container inspect "$1" >/dev/null 2>&1
}

require_command docker
require_command hf
mkdir -p "$HF_CACHE_DIR"

printf 'Downloading Qwen3.5-9B into %s...\n' "$HF_CACHE_DIR"
HF_HOME="$HF_CACHE_DIR" hf download Qwen/Qwen3.5-9B

printf 'Downloading Qwen3.8-27B-FP8 into %s...\n' "$HF_CACHE_DIR"
HF_HOME="$HF_CACHE_DIR" hf download Qwen/Qwen3.8-27B-FP8

printf 'Pulling %s...\n' "$IMAGE"
docker pull "$IMAGE"

create_container() {
  local name="$1"
  shift
  if container_exists "$name"; then
    printf 'Container already exists, leaving it unchanged: %s\n' "$name"
    return 0
  fi

  docker create \
    --name "$name" \
    --gpus all \
    --ipc=host \
    --restart unless-stopped \
    --env VLLM_USE_RUST_FRONTEND=1 \
    --env HF_HUB_OFFLINE=1 \
    --env HF_HOME=/root/.cache/huggingface \
    --volume "${HF_CACHE_DIR}:/root/.cache/huggingface:ro" \
    "$@"
  printf 'Created container: %s\n' "$name"
}

create_container "$CONTAINER_9B" \
  --publish 8000:8000 \
  "$IMAGE" \
  --model Qwen/Qwen3.5-9B \
  --served-model-name Qwen3.5-9B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.34

create_container "$CONTAINER_27B" \
  --publish 8001:8000 \
  "$IMAGE" \
  --model Qwen/Qwen3.8-27B-FP8 \
  --served-model-name Qwen3.8-27B-FP8 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --max-model-len 32768 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.54

printf '\nContainers are ready. Start them with:\n'
printf '  ./scripts/start_vllm_containers.sh\n'
