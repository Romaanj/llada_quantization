# A100 GPU 환경을 위한 LLaDA Quantization Dockerfile
# PyTorch 2.1.2 + CUDA 11.8 기반 (flash-attn 사전 빌드 wheel 사용)

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /workspace

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt /workspace/requirements.txt

# pip 업그레이드
RUN pip install --upgrade pip

# PyTorch는 이미 베이스 이미지에 설치되어 있으므로 제외하고 나머지 패키지 설치
# torch, torchvision, torchaudio 라인 제거 후 설치
RUN sed -e '/^torch==/d' -e '/^torchvision==/d' -e '/^torchaudio==/d' \
    /workspace/requirements.txt > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt && \
    rm /tmp/requirements_filtered.txt

# Flash Attention 2 설치 (A100 최적화) - PyTorch 2.1 + CUDA 11.8 호환 wheel
ARG FLASH_ATTN_WHL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir $FLASH_ATTN_WHL

# AutoGPTQ 설치 (CUDA 12.1 호환)
RUN pip install --no-cache-dir auto-gptq --no-build-isolation

# 프로젝트 코드 복사
COPY . /workspace/llada_quantization

# 작업 디렉토리 변경
WORKDIR /workspace/llada_quantization

# PYTHONPATH 설정
ENV PYTHONPATH="/workspace/llada_quantization:${PYTHONPATH}"

# 기본 명령어
CMD ["/bin/bash"]

