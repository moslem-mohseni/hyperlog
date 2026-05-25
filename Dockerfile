# HyLog production container.
#
# Two-stage build: the builder installs dependencies into a wheel cache,
# the runtime image copies the installed environment + source. CUDA 12.x
# runtime is the base because Phase-3+ training and inference both need
# bitsandbytes 4-bit kernels.

# ----- builder ----------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential git curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/hylog

COPY pyproject.toml README.md LICENSE CITATION.cff ./
COPY src/ ./src/

RUN python -m pip install --upgrade pip wheel setuptools \
    && python -m pip install \
        "torch==2.4.*" \
        --index-url https://download.pytorch.org/whl/cu121 \
    && python -m pip install \
        "transformers>=4.45" \
        "peft>=0.12" \
        "bitsandbytes>=0.43.0" \
        "accelerate>=0.34" \
        "datasets>=2.20" \
        "hydra-core>=1.3,<2.0" \
        "omegaconf>=2.3" \
        "mlflow>=2.16" \
        "scikit-learn>=1.5" \
        "numpy>=1.26" \
        "pandas>=2.2" \
        "scipy>=1.13" \
        "click>=8.1" \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.30" \
        "pydantic>=2.8" \
        "tqdm>=4.66" \
        "rich>=13.7" \
        "pyyaml>=6.0" \
    && python -m pip install -e . --no-deps

# ----- runtime ----------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HYLOG_HOME=/opt/hylog \
    PATH=/usr/local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv git ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/hylog

# Bring in the installed Python environment + the project source.
COPY --from=builder /usr/lib/python3 /usr/lib/python3
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /opt/hylog /opt/hylog

COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY splits/ ./splits/
COPY data/licenses.yaml data/checksums.txt ./data/

LABEL org.opencontainers.image.title="HyLog" \
      org.opencontainers.image.description="Hybrid SLM pipeline for cross-system log anomaly detection" \
      org.opencontainers.image.authors="Moslem Mohseni Khah" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/moslem-mohseni/hyperlog"

# Default entrypoint: print version + verification status. Real
# workloads override via `docker run hylog:latest hylog-loso ...`.
ENTRYPOINT ["python", "-m"]
CMD ["hylog.cli.train", "--dry-run"]
