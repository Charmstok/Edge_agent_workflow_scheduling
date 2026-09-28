#!/usr/bin/env bash

# Start containers created by create_vllm_containers.sh.

set -euo pipefail

CONTAINER_9B="${VLLM_9B_CONTAINER:-qwen35-9b}"
CONTAINER_27B="${VLLM_27B_CONTAINER:-qwen38-27b-fp8}"
STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-900}"

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    printf 'error: required command not found: %s\n' "$1" >&2
    exit 1
  }
}

start_container() {
  local name="$1"
  if ! docker container inspect "$name" >/dev/null 2>&1; then
    printf 'error: container does not exist: %s\n' "$name" >&2
    printf 'Run ./scripts/create_vllm_containers.sh first.\n' >&2
    exit 1
  fi
  if [[ "$(docker inspect --format '{{.State.Status}}' "$name")" == "running" ]]; then
    printf 'Already running: %s\n' "$name"
    return 0
  fi
  docker start "$name" >/dev/null
  printf 'Started %s\n' "$name"
}

wait_for_endpoint() {
  local name="$1"
  local endpoint="$2"
  local deadline=$((SECONDS + STARTUP_TIMEOUT))

  printf 'Waiting for %s at %s ...\n' "$name" "$endpoint"
  while (( SECONDS < deadline )); do
    if curl --noproxy '*' --fail --silent --max-time 5 "$endpoint" >/dev/null 2>&1; then
      printf 'Ready: %s\n' "$name"
      return 0
    fi
    sleep 5
  done

  printf 'error: %s did not become ready within %s seconds\n' \
    "$name" "$STARTUP_TIMEOUT" >&2
  printf 'Inspect its logs with: docker logs --tail 100 %s\n' "$name" >&2
  return 1
}

require_command docker
require_command curl
start_container "$CONTAINER_9B"
start_container "$CONTAINER_27B"

wait_for_endpoint "$CONTAINER_9B" "http://127.0.0.1:8000/v1/models"
wait_for_endpoint "$CONTAINER_27B" "http://127.0.0.1:8001/v1/models"

printf '\nEndpoints:\n'
printf '  Qwen3.5-9B:       http://127.0.0.1:8000/v1\n'
printf '  Qwen3.8-27B-FP8: http://127.0.0.1:8001/v1\n'
