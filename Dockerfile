# Official Docker Hub Python image
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/home/poseidon/.cache/fastembed

WORKDIR /app

# Install ripgrep for fast, structured code search (Milestone 5+)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, deterministic lockfile installs
RUN pip install --no-cache-dir uv

# Create a non-root user for clean container execution
RUN useradd --create-home --shell /bin/bash poseidon

# Copy dependency definitions first for Docker layer caching
COPY pyproject.toml uv.lock* README.md ./
COPY src/ ./src/

# Install package and dev tools into system environment
RUN uv pip install --system --no-cache ".[dev]"

# Prepare workspace mount point and model cache dir owned by non-root user
RUN mkdir -p /workspace /home/poseidon/.cache/fastembed \
    && chown -R poseidon:poseidon /app /workspace /home/poseidon/.cache

USER poseidon

ENTRYPOINT ["poseidon"]
CMD ["--help"]
