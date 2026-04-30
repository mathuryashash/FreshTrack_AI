# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies with retry mechanism
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    || (sleep 10 && apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0) \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements and install (faster builds, no ML training deps)
COPY requirements-api.txt .
RUN pip install --no-cache-dir --retries=5 --timeout=30 -r requirements-api.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy source code
COPY src/ ./src/
COPY download_model.py .

# Create appuser and appgroup first, then create models and data directories with proper permissions
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser && \
    mkdir -p models/checkpoints data && chown -R appuser:appgroup models data

# Download model if not present, then start server
USER appuser

CMD python download_model.py && uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
