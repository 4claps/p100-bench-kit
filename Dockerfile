# P100 bench kit: web app + pinned llama.cpp engines + Hermes agent battery.
#
#   docker build -t p100-bench-kit .
#   docker run -d --name bench --gpus all -p 8090:8090 \
#     -v /sys/firmware/dmi/tables:/host-dmi:ro \
#     -v /path/to/models:/models -v bench-data:/data p100-bench-kit
#
# Build args: CUDA_ARCHS (default 60;61;70 = P100, P40/P4, V100) for the upstream
# engine, KMIC_CUDA_ARCHS (default 60, P100 only), JOBS.

ARG CUDA_VERSION=12.9.1
ARG UV_VERSION=0.12.23

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---------------- engines ----------------
# NCCL 2.27 ships in the CUDA 12.9 base images (held packages; do not apt-install it)
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS engines
ARG CUDA_ARCHS="60;61;70"
ARG JOBS=""
ARG UPSTREAM_COMMIT=79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6
ARG KMIC_COMMIT=ae35056eba07a52dc85c91b035170013d5e46220
ENV CUDA_ARCHS=${CUDA_ARCHS} JOBS=${JOBS}
RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY docker/build-engine.sh /usr/local/bin/build-engine.sh
RUN build-engine.sh upstream https://github.com/ggml-org/llama.cpp "$UPSTREAM_COMMIT" \
      "llama.cpp upstream (baseline)" layer '{}' \
      '["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.0"]' \
      "Pinned upstream build; the fair baseline for cross-hardware comparisons."
# Kmic is built for sm_60 only, as its p100-docs/BUILD.md does: its P100 mmvq
# tuning is gated on __CUDA_ARCH_LIST__ == 600, and other arch lists fail to compile.
ARG KMIC_CUDA_ARCHS=60
RUN CUDA_ARCHS="$KMIC_CUDA_ARCHS" build-engine.sh kmic https://github.com/Kmic-68/llama.cpp "$KMIC_COMMIT" \
      "Kmic-68 llama.cpp (P100 tuned)" tensor \
      '{"GGML_CUDA_P2P": "1", "GGML_CUDA_GRAPHS_PRE_VOLTA": "3", "LLAMA_SPEC_SAMPLE_TEMP": "1.0", "LLAMA_SPEC_DRAFT_TOPK": "20"}' \
      '["--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0.0"]' \
      "Pascal-focused fork with tensor split and its own decode paths; env from its QUICKSTART."

# ---------------- python (app + hermes) ----------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04 AS python
ARG HERMES_COMMIT=4a3e5c4094d96c2e184dead99bccb4f4aabb2e1b
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN uv python install 3.12 && uv venv -p 3.12 /opt/benchkit-venv
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install -p /opt/benchkit-venv/bin/python -r /tmp/requirements.txt
RUN git init -q /opt/hermes-agent \
    && git -C /opt/hermes-agent remote add origin https://github.com/NousResearch/hermes-agent \
    && git -C /opt/hermes-agent fetch -q --depth 1 origin "$HERMES_COMMIT" \
    && git -C /opt/hermes-agent checkout -q FETCH_HEAD \
    && rm -rf /opt/hermes-agent/.git
# Hermes only supports editable installs; uv sync uses its own lock file, so
# every dependency version is pinned by the commit above.
RUN cd /opt/hermes-agent && UV_PROJECT_ENVIRONMENT=/opt/hermes-venv uv sync --frozen --no-dev -p 3.12

# ---------------- runtime ----------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
# python3 is what the agent tasks call; there is deliberately no `python`
# on PATH (err_python_env tests exactly that). dmidecode reads RAM type/speed.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 python3 dmidecode pciutils ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 bench
COPY --from=python /opt/python /opt/python
COPY --from=python /opt/benchkit-venv /opt/benchkit-venv
COPY --from=python /opt/hermes-venv /opt/hermes-venv
COPY --from=python /opt/hermes-agent /opt/hermes-agent
COPY --from=engines /opt/engines /opt/engines
COPY benchkit /opt/benchkit/benchkit
COPY models /opt/benchkit/models
COPY README.md LICENSE /opt/benchkit/
ENV BENCHKIT_DATA_DIR=/data BENCHKIT_MODELS_DIR=/models BENCHKIT_ENGINES_DIR=/opt/engines \
    BENCHKIT_HERMES_PYTHON=/opt/hermes-venv/bin/python BENCHKIT_HERMES_SOURCE=/opt/hermes-agent \
    BENCHKIT_MANIFEST=/opt/benchkit/models/manifest.yaml PYTHONUNBUFFERED=1
WORKDIR /opt/benchkit
VOLUME ["/data"]
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s CMD /opt/benchkit-venv/bin/python -c \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=4)"
CMD ["/opt/benchkit-venv/bin/python", "-m", "benchkit.main"]
