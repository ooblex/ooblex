# Multi-stage build for ML Worker service
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash ooblex

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/ooblex/.local

# Copy application code
COPY --chown=ooblex:ooblex . .

# Create model cache directory
RUN mkdir -p /models /cache && chown -R ooblex:ooblex /models /cache

# Set Python path
ENV PATH=/home/ooblex/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Switch to non-root user
USER ooblex

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import tensorflow as tf; assert torch.cuda.is_available()" || exit 1

# Run the application
CMD ["python", "ml_worker_parallel.py"]