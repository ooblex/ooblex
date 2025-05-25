FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install IPFS binary (optional, for running local node)
RUN wget https://dist.ipfs.io/go-ipfs/v0.23.0/go-ipfs_v0.23.0_linux-amd64.tar.gz && \
    tar -xvzf go-ipfs_v0.23.0_linux-amd64.tar.gz && \
    cd go-ipfs && \
    ./install.sh && \
    cd .. && \
    rm -rf go-ipfs go-ipfs_v0.23.0_linux-amd64.tar.gz

# Copy application code
COPY . .

# Create directories for data
RUN mkdir -p /data/ipfs /data/contracts

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV IPFS_PATH=/data/ipfs

# Expose ports
EXPOSE 8080 5001 4001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the blockchain service
CMD ["python", "-m", "blockchain_service"]