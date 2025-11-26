#!/bin/bash
# Docker 이미지 빌드 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "========================================="
echo "Building LLaDA Quantization Docker Image"
echo "========================================="

# 빌드 옵션
IMAGE_NAME="llada-quantization"
IMAGE_TAG="latest"

# Docker 빌드
docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

echo "========================================="
echo "Build completed: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "========================================="

