# syntax=docker/dockerfile:1.7
# vLLM itself stays pinned to the validated AegisVoice Q00 digest.
FROM vllm/vllm-omni@sha256:8b0cc5438eb27b34cdfd22011b735da6a94835a09e9c56ddf9e8cb300d679919

ARG AEGIS_SOURCE_COMMIT=unknown
LABEL org.opencontainers.image.revision=$AEGIS_SOURCE_COMMIT
ENV AEGIS_TTS_SOURCE_COMMIT=$AEGIS_SOURCE_COMMIT

WORKDIR /gateway
COPY gateway/requirements-runtime.txt /tmp/aegis-gateway-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/aegis-gateway-requirements.txt \
    && python3 -c "import numpy, pyworld, scipy" \
    && rm -f /tmp/aegis-gateway-requirements.txt

COPY gateway /gateway
RUN chmod 0555 /gateway/run-vllm-with-default-reference.sh
