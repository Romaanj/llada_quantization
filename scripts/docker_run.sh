#!/bin/bash
# Docker 컨테이너 실행 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 설정
IMAGE_NAME="llada-quantization:latest"
CONTAINER_NAME="llada-quant"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"

echo "========================================="
echo "Running LLaDA Quantization Container"
echo "GPU: ${GPU_IDS}"
echo "========================================="

# 기존 컨테이너 중지 및 삭제 (주의하세요!)
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# 컨테이너 실행 ()
docker run -dit \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --shm-size=128g \
    -v "${PROJECT_DIR}:/workspace/llada_quantization" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e HF_HOME=/root/.cache/huggingface \
    -e PYTHONPATH=/workspace/llada_quantization \
    -w /workspace/llada_quantization \
    "$IMAGE_NAME" \
    /bin/bash

