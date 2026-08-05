# =============================================================================
# SupportSphere AI — Production Dockerfile (Optimized CPU Image Size)
# =============================================================================
# Multi-stage build using uv package manager and CPU PyTorch wheels.
# Drastically reduces image size from ~9.8 GB down to ~1.2 GB.
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Enable uv optimizations and disable cache persistence in image layer
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_CACHE=1

# Copy dependency manifests
COPY pyproject.toml ./

# Synchronize virtual environment using CPU-only PyTorch wheels
RUN uv sync --no-dev --no-install-project --index-strategy unsafe-best-match

# Prune build cache, CUDA remnants, and compiled bytecode from virtualenv
RUN find /app/.venv -type d -name "nvidia" -exec rm -rf {} + 2>/dev/null || true \
    && find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /app/.venv -type f -name "*.pyc" -delete 2>/dev/null || true

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install curl for healthchecks and clean up apt package lists
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user setup for security compliance
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy optimized virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy application source code and modules
COPY src/ ./src/
COPY vectordb_ingestion/ ./vectordb_ingestion/
COPY pyproject.toml ./

# Copy persisted artifacts and data index
COPY data/ ./data/

# Environment configuration
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose Streamlit default port
EXPOSE 8501

# Container healthcheck targeting Streamlit health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Switch to non-privileged user
USER appuser

# Launch SupportSphere AI Streamlit application
ENTRYPOINT ["streamlit", "run", "src/app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--server.enableCORS=false", \
            "--server.enableXsrfProtection=false", \
            "--browser.gatherUsageStats=false"]
