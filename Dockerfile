# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE} AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build build-essential ccache pkg-config \
    python3-dev python3-pip python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ----- OpenFHE GPU 코드 빌드 -----
ARG REPO_URL=https://github.com/leodec/openfhe-gpu-public.git
ARG REPO_REF=main
ARG CMAKE_BUILD_TYPE=Release
ARG CMAKE_CUDA_ARCHS="100;90"

WORKDIR /opt
RUN git clone --depth=1 --branch ${REPO_REF} ${REPO_URL} openfhe-gpu

# export-set 관련 오류 제거 (유지)
RUN perl -0777 -pe 's/^[ \t]*install\s*\(\s*EXPORT\b.*?\)\s*\n//msg; s/^[ \t]*export\s*\(\s*EXPORT\b.*?\)\s*\n//msg' \
    -i /opt/openfhe-gpu/CMakeLists.txt

# ⚠️ configure 시 정책/아키텍처 명시 + Ninja 사용
RUN cmake -S /opt/openfhe-gpu -B /opt/openfhe-gpu/build \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHS}" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DOPENFHE_ENABLE_CUDA=ON \
    -DOPENFHE_ENABLE_RMM=ON \
    -DOPENFHE_BUILD_EXAMPLES=ON \
    -DOPENFHE_BUILD_BENCHMARKS=OFF \
    -DOPENFHE_BUILD_UNITTESTS=OFF \
    && cmake --build /opt/openfhe-gpu/build -j"$(nproc)" \
    && echo "== Built binaries ==" && (ls -R /opt/openfhe-gpu/build/bin || true)

# 실행 시 .so 경로
ENV LD_LIBRARY_PATH="/opt/openfhe-gpu/build/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/openfhe-gpu/build/bin:${PATH}"

# PyTorch 확인/설치(베이스에 없을 때만)
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
RUN python - <<'PY'
try:
    import torch; print("Torch already present:", torch.__version__)
except Exception:
    import os, subprocess
    subprocess.check_call(["pip","install","--no-cache-dir","--index-url", os.environ.get("PYTORCH_INDEX_URL",""),
                           "torch","torchvision","torchaudio"])
PY

# 비루트 사용자
ARG USERNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USERNAME} && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}
USER ${USERNAME}
WORKDIR /workspace

# 진입 시 간단 확인
CMD bash -lc 'echo "== nvcc ==" && nvcc --version; \
    python - <<EOF\nimport torch\nprint(\"== PyTorch ==\")\nprint(\"torch:\", torch.__version__)\nprint(\"cuda:\", getattr(torch.version,\"cuda\",None))\nprint(\"available:\", torch.cuda.is_available())\nprint(\"device:\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"-\")\nEOF; bash'
