# Docker Configuration Guide

This document explains the various Docker configurations available in the Ooblex project.

## Docker Compose Files

The project provides multiple Docker Compose configurations for different use cases:

### 1. `docker-compose.yml` (Default/Production)
**Purpose**: Full production deployment with all services
**Use when**: Running the complete Ooblex platform in production

```bash
docker-compose up -d
```

**Services included**:
- API Gateway
- ML Worker (GPU-enabled)
- Decoder
- MJPEG Streaming
- RabbitMQ
- Redis
- PostgreSQL

---

### 2. `docker-compose.simple.yml`
**Purpose**: Simplified deployment for basic functionality
**Use when**: Testing basic features or running on resource-constrained environments

```bash
docker-compose -f docker-compose.simple.yml up -d
```

**Services included**:
- Core API
- Lightweight ML processing
- Essential infrastructure only

---

### 3. `docker-compose.dev.yml`
**Purpose**: Development environment with hot-reload and debugging
**Use when**: Actively developing and testing changes

```bash
docker-compose -f docker-compose.dev.yml up
```

**Features**:
- Volume mounts for code hot-reloading
- Debug ports exposed
- Development-friendly logging
- API in development mode

---

### 4. `docker-compose.webrtc.yml`
**Purpose**: WebRTC real-time audio/video streaming
**Use when**: Enabling P2P WebRTC connectivity

```bash
docker-compose -f docker-compose.webrtc.yml up -d
```

**Services included**:
- WebRTC gateway
- STUN/TURN server support
- Real-time media processing

---

### 5. `docker-compose.ninjasdk.yml`
**Purpose**: NinjaSDK integration for VDO.Ninja audio ingestion
**Use when**: Integrating with VDO.Ninja for WebRTC audio

```bash
docker-compose -f docker-compose.ninjasdk.yml up -d
```

**Services included**:
- NinjaSDK audio ingestion service
- WebRTC P2P audio streaming
- Audio processing pipeline

---

### 6. `docker-compose.observability.yml`
**Purpose**: Monitoring and observability stack
**Use when**: Setting up metrics, logging, and monitoring

```bash
docker-compose -f docker-compose.observability.yml up -d
```

**Services included**:
- Prometheus (metrics)
- Grafana (dashboards)
- Log aggregation
- Metrics exporters

---

## Dockerfile Variants

### ML Worker Dockerfiles

#### `services/ml-worker/Dockerfile` (GPU - Default)
- **Base**: CUDA-enabled image
- **Purpose**: GPU-accelerated ML inference
- **Hardware**: Requires NVIDIA GPU with CUDA support
- **Use**: Production ML workloads with GPU

#### `services/ml-worker/Dockerfile.cpu`
- **Base**: Python 3.11-slim
- **Purpose**: CPU-only ML inference
- **Hardware**: Works on any system
- **Use**: Development or non-GPU environments

#### `services/ml-worker/Dockerfile.simple`
- **Base**: Python 3.11-slim
- **Purpose**: Minimal ML worker with basic models
- **Use**: Testing or lightweight deployments

---

### Root-level Dockerfiles

#### `Dockerfile.simple`
- **Purpose**: Single-container simplified deployment
- **Use**: Quick testing or demo environments

#### `Dockerfile.whisper`
- **Purpose**: Whisper speech recognition service
- **Use**: Audio transcription workloads

#### `Dockerfile.llm`
- **Purpose**: Large Language Model inference
- **Use**: LLM-based processing

---

### API Dockerfiles

#### `services/api/Dockerfile` (Production)
- **Base**: Python 3.11-slim (multi-stage)
- **Purpose**: Optimized production API server
- **Features**: Minimal size, security-hardened

#### `services/api/Dockerfile.dev`
- **Base**: Python 3.11
- **Purpose**: Development API server
- **Features**: Full debugging tools, hot-reload

---

## Common Usage Patterns

### Quick Start (Recommended)
```bash
# Simple deployment for testing
docker-compose -f docker-compose.simple.yml up -d
```

### Full Production Deployment
```bash
# Complete system with all services
docker-compose up -d

# Add observability
docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d
```

### Development Workflow
```bash
# Development with hot-reload
docker-compose -f docker-compose.dev.yml up

# Rebuild after dependency changes
docker-compose -f docker-compose.dev.yml up --build
```

### WebRTC Streaming
```bash
# WebRTC gateway
docker-compose -f docker-compose.webrtc.yml up -d

# OR with NinjaSDK integration
docker-compose -f docker-compose.ninjasdk.yml up -d
```

---

## Environment Configuration

All Docker deployments support environment configuration via `.env` files:

```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env

# Environment variables are automatically loaded by docker-compose
```

---

## Hardware Requirements

### Minimum (Simple/Dev)
- CPU: 2 cores
- RAM: 4GB
- Storage: 10GB

### Recommended (Production)
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (for ML worker)

### WebRTC/Real-time
- CPU: 4+ cores
- RAM: 8GB+
- Network: Low-latency connection
- Bandwidth: 10+ Mbps

---

## Troubleshooting

### GPU Not Detected
```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If not working, install nvidia-docker2
sudo apt-get install nvidia-docker2
sudo systemctl restart docker
```

### Port Conflicts
```bash
# Check for port usage
sudo netstat -tulpn | grep :8000

# Modify ports in docker-compose.yml or use PORT environment variables
```

### Volume Permissions
```bash
# Fix permission issues
sudo chown -R $USER:$USER ./data
```

---

## Best Practices

1. **Use specific compose files** for your use case rather than modifying the default
2. **Always use `.env` files** instead of hardcoding secrets
3. **Enable observability** in production with `docker-compose.observability.yml`
4. **Use multi-stage builds** for smaller production images
5. **Regular updates**: Keep base images and dependencies updated
6. **Resource limits**: Set memory and CPU limits in production
7. **Health checks**: All services include health checks for reliability

---

## See Also

- [README.md](README.md) - General project documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [.env.example](.env.example) - Environment configuration template
