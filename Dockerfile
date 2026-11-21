# syntax=docker/dockerfile:1.7
#
# Wan2.2-S2V-14B: audio-driven performance from a single still.
#
# Base choices, and why:
#
# * CUDA 12.8 (not 13.x). The prebuilt flash-attention wheels that the model
#   needs are built for cu12 + torch 2.8, and 12.x containers start on the
#   R550/R570 drivers that most Vast hosts run. An R580+ driver runs this image
#   fine; an R525-era host will not.
# * Ubuntu 24.04 + the distro Python 3.12, in a venv. The flash-attn wheel is
#   cp312; pinning the interpreter is what makes `pip install <wheel>` exact.
# * A *runtime* CUDA image. Nothing compiles here: flash-attn is a prebuilt
#   wheel (the source build needs nvcc and ~1 h), so no devel toolchain is
#   shipped.
#
# Weights are NOT baked in: 45.6 GB, mounted at /models and downloaded once
# (scripts/download-models.sh). The upstream code is pinned by commit and
# fetched as a tarball, so no git is needed at build or run time.

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

# Upstream commit, not a branch: a release that renders must not change under
# the image. Bump this deliberately and re-run the tests.
ARG WAN_REPO=https://github.com/Wan-Video/Wan2.2
ARG WAN_REF=1ea34ff48f87168174e12956e200b1d908b1c5ff

# Prebuilt wheel: cu12, torch 2.8, cp312, pre-C++11 ABI (what PyTorch's own
# linux wheels use).
ARG FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2/flash_attn-2.8.3.post2%2Bcu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/.hf \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    VIRTUAL_ENV=/opt/wan/.venv \
    PATH=/opt/wan/.venv/bin:${PATH} \
    PYTHONPATH=/opt/Wan2.2

# ffmpeg: Wan writes H.264 with imageio and muxes with the ffmpeg binary.
# libsndfile1: librosa/soundfile for the audio encoder's decode path.
# libgl1/libglib2.0-0: OpenCV imports that come with the wan package.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
        libgl1 \
        libglib2.0-0 \
        python3 \
        python3-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/wan/.venv \
    && /opt/wan/.venv/bin/pip install --upgrade pip wheel

# torch first, from the cu128 index, in its own layer (~890 MB download). The
# cu128 tag matters: plain PyPI would give a CPU wheel and silence the GPU.
RUN pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.8.0 \
        torchvision==0.23.0 \
        torchaudio==2.8.0

# flash-attn is not optional here: wan/modules/s2v/model_s2v.py calls
# flash_attention() directly, and that asserts a flash-attn build exists. The
# SDPA fallback in wan/modules/attention.py is only used by other tasks.
RUN pip install --no-deps "${FLASH_ATTN_WHEEL}" \
    && pip install einops==0.8.1

COPY requirements-core.txt requirements-ml.txt requirements-dev.txt requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt -r /tmp/requirements-dev.txt

# Upstream code at a known commit, extracted without git. Extract to a staging
# directory first, then rename, so the archive's `Wan2.2-<sha>/` folder becomes
# exactly /opt/Wan2.2.
RUN mkdir -p /opt/wan-src \
    && curl -fsSL "${WAN_REPO}/archive/${WAN_REF}.tar.gz" -o /tmp/wan.tar.gz \
    && tar -xzf /tmp/wan.tar.gz -C /opt/wan-src \
    && mv /opt/wan-src/Wan2.2-* /opt/Wan2.2 \
    && rmdir /opt/wan-src \
    && rm -f /tmp/wan.tar.gz \
    && test -f /opt/Wan2.2/generate.py \
    && printf '%s\n' "${WAN_REF}" > /opt/Wan2.2/.upstream-commit \
    && test -f /opt/Wan2.2/wan/configs/wan_s2v_14B.py

WORKDIR /opt/wan-docker

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY examples ./examples
COPY docs ./docs
COPY README.md LICENSE ./

RUN chmod +x scripts/*.sh \
    && ln -sf /opt/wan-docker/scripts/entrypoint.sh /usr/local/bin/wan22-s2v \
    && mkdir -p /models /performances /output

VOLUME ["/models", "/output"]
EXPOSE 8000

ENTRYPOINT ["/opt/wan-docker/scripts/entrypoint.sh"]
CMD ["doctor"]
