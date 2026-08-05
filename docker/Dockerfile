# =============================================================================
# SupportSphere AI — Production Dockerfile
# =============================================================================
# Multi-stage build using uv package manager, locked dependency graphs,
# and CPU-optimized virtual environment caching.
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Pin exact uv binary version for deterministic, reproducible build environment
COPY --from=ghcr.io/astral-sh/uv:0.5.29 /uv /usr/local/bin/uv

WORKDIR /app

# Enable uv optimizations and disable cache persistence in image layer
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# Copy dependency manifests (pyproject.toml AND uv.lock) first to leverage Docker layer caching.
# If dependencies haven't changed, this layer is reused, skipping re-installation.
COPY pyproject.toml uv.lock ./

# Synchronize virtual environment strictly against uv.lock (--locked).
# Fails the build immediately if pyproject.toml and uv.lock are out of sync.
# Does NOT use --index-strategy unsafe-best-match for deterministic reproducibility.
RUN uv sync --locked --no-dev --no-install-project

# Prune compiled bytecode and cache from virtualenv
RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
    && find /app/.venv -type f -name "*.pyc" -delete 2>/dev/null || true

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install minimal runtime requirements (curl for healthchecks) and clean apt caches
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user setup for security compliance
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy pre-built, optimized virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy Streamlit configuration directory explicitly to bake web UI settings into image
COPY .streamlit/ ./.streamlit/

# Copy application source code and runtime assets
COPY src/ ./src/
COPY vectordb_ingestion/ ./vectordb_ingestion/
COPY pyproject.toml ./
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

RUN mkdir -p /app/logs /app/data && chown -R appuser:appgroup /app

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
