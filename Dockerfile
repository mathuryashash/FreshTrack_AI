# FreshTrack AI - Production Dockerfile
# Multi-stage build for smaller final image

# ===== BUILD STAGE =====
FROM python:3.11-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
# CPU torch first, so -r requirements.txt sees it satisfied and never pulls CUDA wheels
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.12.0 torchvision==0.27.0 \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ===== RUNTIME STAGE =====
FROM python:3.11-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Set working directory
WORKDIR /app

# Copy source code
COPY --chown=appuser:appgroup src/ ./src/

# The model is NOT baked into the image: checkpoints are untracked and
# excluded by .dockerignore. Mount the promoted model at runtime:
#   docker run -v "$PWD/models/checkpoints:/app/models/checkpoints:ro" ...
# (needs freshtrack_v2.ckpt + model_meta.json; /health reports "degraded" without them)
RUN mkdir -p /app/data /app/models/checkpoints && chown -R appuser:appgroup /app/data /app/models

# Switch to non-root user
USER appuser

# Environment variables (override at runtime)
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MODEL_CHECKPOINT=models/checkpoints/freshtrack_v2.ckpt
ENV MODEL_META=models/checkpoints/model_meta.json
# Filesystem path (not a URL). Persist it with: -v freshtrack-data:/app/data
ENV DATABASE_URL=/app/data/freshtrack.db
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]