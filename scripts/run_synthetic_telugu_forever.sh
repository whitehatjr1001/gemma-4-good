#!/usr/bin/env bash
set -u

MODEL="${MODEL:-qwen3:30b-a3b}"
OLLAMA_HOST="${OLLAMA_HOST:-http://192.168.10.151:11434}"
WORKERS="${WORKERS:-2}"
NUM_PREDICT="${NUM_PREDICT:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
RETRIES="${RETRIES:-2}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"

mkdir -p runs/synthetic_telugu

echo "synthetic Telugu supervisor starting"
echo "model=${MODEL}"
echo "ollama=${OLLAMA_HOST}"
echo "workers=${WORKERS}"
echo "num_predict=${NUM_PREDICT}"

while true; do
  echo "heartbeat $(date -u '+%Y-%m-%dT%H:%M:%SZ'): starting/resuming generation"
  OLLAMA_HOST="${OLLAMA_HOST}" uv run python scripts/generate_synthetic_telugu.py \
    --model "${MODEL}" \
    --temperature "${TEMPERATURE}" \
    --num-predict "${NUM_PREDICT}" \
    --workers "${WORKERS}" \
    --retries "${RETRIES}" \
    --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  status=$?
  echo "heartbeat $(date -u '+%Y-%m-%dT%H:%M:%SZ'): generation exited with status ${status}"
  if [ "${status}" -eq 0 ]; then
    echo "heartbeat $(date -u '+%Y-%m-%dT%H:%M:%SZ'): generation complete"
    exit 0
  fi
  echo "heartbeat $(date -u '+%Y-%m-%dT%H:%M:%SZ'): sleeping ${SLEEP_SECONDS}s before retry"
  sleep "${SLEEP_SECONDS}"
done
