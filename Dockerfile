# Use official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy production requirements and install (faster builds, no ML training deps)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt \
    -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy source code
COPY src/ ./src/
COPY models/ ./models/
COPY download_model.py .

# Download model if not present, then start server
CMD python download_model.py && uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
