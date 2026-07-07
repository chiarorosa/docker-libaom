# Base oficial de desenvolvimento da NVIDIA com CUDA 12.6 e Ubuntu 24.04 LTS
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Bloqueia prompts interativos do apt-get durante o build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV CCACHE_DIR=/workspace/build/.ccache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    clang \
    cmake \
    ninja-build \
    git \
    nasm \
    yasm \
    perl \
    python3 \
    python3-pip \
    python3-venv \
    pkg-config \
    ca-certificates \
    curl \
    wget \
    time \
    file \
    ccache \
    sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Garante que as permissões nativas em ext4 sejam herdadas perfeitamente pelo volume nomeado.
RUN useradd -m -s /bin/bash researcher \
    && echo "researcher ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p /workspace/src /workspace/build /workspace/logs /workspace/results \
    && mkdir -p /workspace/build/.ccache \
    && chown -R researcher:researcher /workspace

WORKDIR /workspace

USER researcher

CMD ["/bin/bash"]