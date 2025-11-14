# Ooblex - Real-Time AI Video Processing

[![CI/CD](https://github.com/ooblex/ooblex/workflows/Ooblex%20CI%2FCD/badge.svg)](https://github.com/ooblex/ooblex/actions)
[![Security](https://img.shields.io/badge/security-patched-success.svg)](SECURITY_FIXES.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Process live video streams with AI models in real-time. 200-400ms latency via WebRTC.

**Pipeline:** Browser → WebRTC → AI Processing → Browser

---

## Architecture

![Ooblex Architecture](docs/images/diagram.png)

**Core Components:**
- **WebRTC** - Browser video I/O (low latency, NAT traversal)
- **Redis** - Frame queue (in-memory, fast)
- **RabbitMQ** - Task distribution to workers
- **ML Workers** - Run AI models (TensorFlow, PyTorch, ONNX, OpenCV)

**Scaling:** Add more workers = more throughput (horizontal scaling)

**Latency:** 200-400ms end-to-end

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/ooblex/ooblex.git
cd ooblex

# Start infrastructure + workers (no AI models needed)
docker compose -f docker-compose.simple.yml up
```

Open http://localhost:8800

### Option 2: Local Development

```bash
# Start services
docker compose up -d redis rabbitmq

# Run worker with OpenCV effects (no model downloads)
python3 code/brain_simple.py &

# Run demo/validation
python3 demo.py
```

---

## Built-In Effects (No Models Required)

Works immediately with `brain_simple.py` - no downloads needed:

| Effect | FPS (CPU) | Description |
|--------|-----------|-------------|
| Face Detection | ~50 | OpenCV Haar cascades |
| Pixelate Faces | ~40 | Privacy filter |
| Cartoon | ~30 | Bilateral filter + edges |
| Background Blur | ~60 | Gaussian blur |
| Edge Detection | ~80 | Canny edges |
| Grayscale | ~100 | B&W conversion |
| Mirror, Invert, Sepia, Denoise | 25-120 | Basic transforms |

All run on **CPU** - no GPU required.

---

## Add Your Own AI Models

See [models/README.md](models/README.md) for how to integrate:

**Supported Frameworks:**
- TensorFlow / TensorFlow Lite
- PyTorch / TorchScript
- ONNX Runtime (recommended)
- OpenVINO (Intel CPUs)
- TensorRT (NVIDIA GPUs)
- MediaPipe

**Note:** Original 2020 face swap models are no longer available (too large for GitHub). Use `brain_simple.py` for zero-friction demo or add your own models.

---

## Core Files

| File | Purpose | Lines |
|------|---------|-------|
| `code/api.py` | WebSocket gateway | 120 |
| `code/brain_simple.py` | OpenCV effects worker | 400 |
| `code/decoder.py` | Video decoder | ~200 |
| `code/mjpeg.py` | MJPEG streaming | ~150 |
| `code/webrtc.py` | WebRTC gateway | ~180 |

**Infrastructure:**
- Redis 7.x - Frame queue
- RabbitMQ 3.x - Task distribution
- Docker Compose - Deployment

---

## Testing

```bash
# Validation
python3 demo.py --quick

# Full test suite (220+ tests)
pytest tests/unit -v
pytest tests/integration -v  # requires Redis + RabbitMQ
pytest tests/e2e -v
pytest tests/benchmarks -v

# CI/CD runs on every push
# - Linting (flake8, black, isort)
# - Unit + integration tests
# - Docker builds
# - Security scanning
```

---

## Production Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guide:

- AWS, GCP, Azure deployment
- Kubernetes with auto-scaling
- SSL/TLS setup (Let's Encrypt)
- Monitoring (Prometheus + Grafana)
- Security hardening checklist

**Quick production start:**
```bash
cp .env.example .env
# Edit .env - change all passwords
sudo certbot certonly --standalone -d yourdomain.com
docker-compose up -d
```

---

## Security

**All critical vulnerabilities patched** (November 2024)

✅ Fixed 4 CVEs:
- CVE-2024-33663 (python-jose) - **CRITICAL** CVSS 9.3
- CVE-2024-33664 (python-jose) - **HIGH** CVSS 5.3
- CVE-2024-12797 (cryptography) - **HIGH**
- CVE-2025-53643 (aiohttp) - **LOW** CVSS 3.7

See [SECURITY_FIXES.md](SECURITY_FIXES.md) for details.

**Report vulnerabilities:** Use GitHub Security Advisories (not public issues)

---

## Performance

**Benchmarks** (RTX 3080, 16GB RAM):

| Effect | Resolution | Latency | FPS |
|--------|-----------|---------|-----|
| Face Detection | 640x480 | 180ms | 30 |
| Background Blur | 1280x720 | 120ms | 45 |
| Style Transfer | 640x480 | 350ms | 15 |

**Scaling:**
- 1 worker: ~15-30 FPS
- 5 workers: ~60-120 FPS
- 10 workers: ~150-250 FPS

---

## Contributing

```bash
git clone https://github.com/ooblex/ooblex.git
cd ooblex
pip install -r requirements.txt
pre-commit install

# Run tests
pytest -v

# Format code
black .
isort .
```

**Guidelines:**
- All PRs must pass CI/CD
- Add tests for new features
- No unnecessary dependencies
- See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## Documentation

- [QUICKSTART.md](QUICKSTART.md) - 5-minute setup
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment
- [SECURITY_FIXES.md](SECURITY_FIXES.md) - Security updates
- [CONTRIBUTING.md](CONTRIBUTING.md) - Developer guide
- [SUGGESTIONS.md](SUGGESTIONS.md) - Roadmap
- [CLEANUP_REPORT.md](CLEANUP_REPORT.md) - What was removed

**Website docs:** [docs/](docs/)
- [docs/api.md](docs/api.md) - API documentation
- [docs/models.md](docs/models.md) - ML model guide
- [docs/webrtc.md](docs/webrtc.md) - WebRTC details

---

## Configuration

```bash
# Core services
REDIS_URL=redis://localhost:6379
RABBITMQ_URL=amqp://guest:guest@localhost:5672

# WebRTC
WEBRTC_STUN_SERVERS=stun:stun.l.google.com:19302

# ML workers
MODEL_PATH=/models
CUDA_VISIBLE_DEVICES=0
ML_WORKER_REPLICAS=2

# API
API_PORT=8800
LOG_LEVEL=info
```

Copy `.env.example` to `.env` and edit as needed.

---

## Troubleshooting

**High latency (>500ms):**
- Add more workers: `docker compose up -d --scale ml-worker=5`
- Use GPU instead of CPU
- Reduce video resolution

**Workers not processing:**
```bash
docker compose logs rabbitmq
docker compose logs ml-worker
```

**WebRTC not connecting:**
- Check ports 8000-8010 UDP are open
- Verify STUN/TURN configuration

---

## License

Apache License 2.0 - see [LICENSE](LICENSE)

**Original Author:** Steve Seguin (2018-2020)

---

## Support

- [GitHub Issues](https://github.com/ooblex/ooblex/issues)
- [GitHub Discussions](https://github.com/ooblex/ooblex/discussions)
