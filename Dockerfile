# syntax=docker/dockerfile:1.7
# Runtime image only.  Model weights and voice references are deliberately
# mounted at runtime and are never copied into this image.
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="Aegis TTS" \
      org.opencontainers.image.description="Local Qwen3-TTS Base voice-cloning service" \
      org.opencontainers.image.vendor="Aegis"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/cache/huggingface/hub \
    TRANSFORMERS_CACHE=/cache/huggingface/transformers \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    QWEN_MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    QWEN_MODEL_DIR=/models/Qwen3-TTS-12Hz-1.7B-Base \
    QWEN_DEVICE=cuda:0 \
    QWEN_DTYPE=bfloat16 \
    QWEN_ATTENTION_IMPLEMENTATION=sdpa

WORKDIR /app

# FFmpeg/SoX are retained only for local audio inspection/conversion support.
# CUDA user-space libraries are provided by the pinned PyTorch cu121 wheels;
# the NVIDIA Container Toolkit supplies the host driver at runtime.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
        libsox-fmt-all \
        sox \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python3.12 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python3.12 -m pip check \
    && rm -f /tmp/requirements.txt

# Keep host-identity customization after dependency installation so changing
# the runtime UID/GID does not invalidate the expensive PyTorch dependency
# layer during ordinary rebuilds.
ARG AEGIS_UID=1002
ARG AEGIS_GID=1002
RUN groupadd --gid "${AEGIS_GID}" aegis \
    && useradd --uid "${AEGIS_UID}" --gid "${AEGIS_GID}" --create-home --home-dir /home/aegis --shell /usr/sbin/nologin aegis \
    && mkdir -p /models /cache/huggingface /samples /tmp \
    && chown -R aegis:aegis /app /models /cache /samples /tmp

COPY --chown=aegis:aegis app /app/app

USER aegis

EXPOSE 8001

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/sh", "-c", "exec python3.12 -m uvicorn app.main:app --host \"${AEGIS_TTS_HOST:-0.0.0.0}\" --port \"${AEGIS_TTS_PORT:-8001}\""]
