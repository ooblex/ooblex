# Ooblex - Real-Time Bidirectional AI Video Processing

[![CI/CD Pipeline](https://github.com/ooblex/ooblex/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/ooblex/ooblex/actions)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](docker-compose.yml)

**Transform live video streams with AI in real-time. Bidirectional WebRTC pipeline with sub-400ms latency.**

Ooblex enables **true real-time AI video processing** - send your webcam feed to the cloud, apply AI transformations (face swap, style transfer, background removal, object detection), and receive the processed video back in your browser. All with latency low enough for **live conversations, remote control, and interactive applications**.

---

## 🎯 What Makes Ooblex Unique

###Real-Time Bidirectional Processing
- **Browser → Cloud → Browser**: WebRTC native, not traditional streaming protocols
- **Ultra-Low Latency**: 200-400ms end-to-end (fast enough for conversation)
- **Live Feedback**: See AI transformations applied to yourself in real-time
- **Interactive**: Change effects on-the-fly, adjust parameters dynamically

### Scalable AI Processing
- **Parallel Workers**: Add ML workers dynamically, they auto-balance load
- **Redis Queue**: Millions of frames buffered and processed efficiently
- **Distributed**: Workers can run anywhere (cloud, edge, local GPU servers)
- **Framework Agnostic**: TensorFlow, PyTorch, ONNX, OpenVINO, TensorRT

### Production-Ready Infrastructure
- **Docker-First**: One command to run entire stack
- **WebRTC Gateway**: Browser-native video input/output
- **Monitoring**: Prometheus metrics, health checks, structured logging
- **Tested**: 100+ automated tests, CI/CD pipeline, installation validation

---

## 🚀 Use Cases

### 1. **Live Video Effects for Streaming**
Apply AI-powered effects to your video before streaming to Twitch/YouTube:
- Real-time background replacement
- Face filters and makeup
- Style transfer (cartoon, oil painting)
- Object detection and tracking

### 2. **AI-Powered Video Calls**
Enhance video conferencing with real-time AI:
- Smart background blur
- Noise reduction and enhancement
- Automatic framing and lighting adjustment
- Real-time translation with lip-sync

### 3. **Edge AI Security Cameras**
Build smart security cameras with cloud AI:
- Remote cameras send video to cloud for AI processing
- Person/vehicle detection with low latency
- Smart alerts and object tracking
- Recording only when motion detected

### 4. **Interactive AI Experiences**
Create engaging real-time applications:
- Virtual try-on (clothes, makeup, accessories)
- Real-time deepfakes for entertainment
- Interactive art installations
- AR/VR preprocessing

### 5. **Remote Robot/Drone Control**
Control autonomous systems with AI-augmented vision:
- Low-latency video feedback with object detection
- Path planning with semantic segmentation
- Gesture control recognition
- Real-time decision making

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         YOUR BROWSER                            │
│  ┌──────────────┐              ┌──────────────┐                │
│  │   Webcam     │              │  AI Video    │                │
│  │   Input      │              │  Output      │                │
│  └──────┬───────┘              └──────▲───────┘                │
│         │                             │                         │
└─────────┼─────────────────────────────┼─────────────────────────┘
          │                             │
          │  WebRTC (H.264/VP8)         │  WebRTC (H.264/VP8)
          ▼                             │
┌─────────────────────────────────────────────────────────────────┐
│                      OOBLEX CLOUD                               │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  WebRTC Gateway (Janus/aiortc)                           │  │
│  │  • Receives browser WebRTC streams                       │  │
│  │  • Decodes H.264/VP8 to raw frames                       │  │
│  │  • Encodes processed frames back to WebRTC              │  │
│  └────────┬──────────────────────────────────────▲──────────┘  │
│           │                                       │             │
│           │ Frame Extraction                      │ Processed   │
│           ▼                                       │ Frames      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Redis Queue (Frame Buffer)                             │   │
│  │  • Stores raw frames as JPEG/PNG                        │   │
│  │  • Acts as distributed queue for workers                │   │
│  │  • TTL-based automatic cleanup                          │   │
│  └────────┬────────────────────────────────────────────────┘   │
│           │                                                     │
│           │ Tasks via RabbitMQ                                 │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ML Worker Pool (Parallel Processing)                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│   │
│  │  │ Worker 1 │  │ Worker 2 │  │ Worker 3 │  │ Worker N ││   │
│  │  │  GPU/CPU │  │  GPU/CPU │  │  GPU/CPU │  │  GPU/CPU ││   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘│   │
│  │  • Face swap    • Style xfer  • Obj detect  • Custom   │   │
│  │  • Background removal, filters, effects, etc.          │   │
│  └────────┬────────────────────────────────────────────────┘   │
│           │                                                     │
│           │ Results back to Redis                              │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MJPEG Streaming Server (Alternative Output)            │   │
│  │  • HTTP-based streaming for debugging                   │   │
│  │  • Lower latency than WebRTC for same-network          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  API Gateway (WebSocket + REST)                         │   │
│  │  • Client connection management                         │   │
│  │  • Task orchestration                                   │   │
│  │  • Health monitoring                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

**Why This Architecture?**

- **WebRTC**: Native browser support, NAT traversal, adaptive bitrate, low latency
- **Redis Queue**: Fast, distributed, handles millions of frames
- **RabbitMQ**: Reliable task distribution, worker acknowledgment, retry logic
- **Parallel Workers**: Horizontal scaling - add workers = more throughput
- **Stateless Workers**: Can crash and restart without affecting system

---

## 🎬 Quick Start

### ⚡ Zero-Friction Demo (No Downloads!)

**Run Ooblex in 30 seconds** with OpenCV effects (no AI models required):

```bash
git clone https://github.com/ooblex/ooblex.git
cd ooblex

# Start services
docker compose up -d redis rabbitmq

# Run simple worker (no models needed!)
python3 code/brain_simple.py &

# Or use Docker
./run-simple-demo.sh
```

**Available effects** (run instantly, no GPU needed):
- 👤 Face Detection - Detect faces with bounding boxes
- 🔒 Pixelate Faces - Privacy filter
- 🎨 Cartoon - Comic book style
- 🌫️ Background Blur - Blur effect
- 📐 Edge Detection - Canny edges
- ⚫ Grayscale, Sepia, Denoise, Mirror, Invert

**All effects run 30-100+ FPS on CPU!**

**Validate installation:**
```bash
# Run comprehensive demo and validation
python3 demo.py

# Quick validation only
python3 demo.py --quick

# Test specific effect
python3 demo.py --effect=FaceOn
```

---

### Option 1: Docker Compose (Full Stack)

**Quick Start with Simple Effects:**

```bash
# Start everything with brain_simple.py (no models needed)
docker compose -f docker-compose.simple.yml up

# Open browser
open http://localhost:8800

# Or just the infrastructure
docker compose up -d redis rabbitmq
python3 code/brain_simple.py
```

**Run full WebRTC demo:**

```bash
# Start everything (Redis, RabbitMQ, WebRTC gateway, ML workers)
docker compose up -d

# Open browser
open http://localhost:8800

# Allow webcam access, select effect, see real-time processing
```

**What you'll see:**
- Your webcam video on the left
- Processed video on the right (real-time!)
- ~200-400ms latency
- Multiple effects to choose from

**Scale workers:**
```bash
docker compose -f docker-compose.webrtc.yml up -d --scale ml-worker=5
```

---

### Option 2: Bare Metal (Ubuntu)

```bash
sudo chmod +x *.sh

# Install dependencies (takes 30-60 minutes)
sudo ./install_opencv.sh      # OpenCV 4.x
sudo ./install_nginx.sh        # NGINX with SSL
sudo ./install_redis.sh        # Redis 7.x
sudo ./install_rabbitmq.sh     # RabbitMQ 3.x
sudo ./install_janus.sh        # Janus WebRTC gateway (optional)

# Configure
nano code/config.py  # Update Redis/RabbitMQ URLs and domain

# Run simple demo (no models)
cd code
python3 api.py &
python3 brain_simple.py &    # ← Simple effects, no downloads
python3 decoder.py &
python3 mjpeg.py &
```

---

### ⚠️ Note: Original AI Models Not Included

The original 2020 TensorFlow face swap models are no longer available (too large for GitHub, links inactive).

**You have two options:**

1. **Use Simple Effects** (recommended for demo) - Works immediately with `brain_simple.py`
2. **Add Your Own Models** - See [models/README.md](models/README.md) for how to add TensorFlow, PyTorch, ONNX, or other models

The **simple effects demonstrate the full Ooblex pipeline** without requiring any model downloads!

---

## 🧪 Testing

We take testing seriously to prevent code bloat:

### Run All Tests

```bash
# Installation validation
pytest tests/test_installation.py -v

# Unit tests (100+ tests, no external services needed)
pytest tests/unit -v

# Integration tests (requires Redis + RabbitMQ)
docker compose up -d redis rabbitmq
pytest tests/integration -v

# End-to-end tests
pytest tests/e2e -v

# Performance benchmarks
pytest tests/benchmarks -v --benchmark-only

# All tests with coverage
pytest --cov=services --cov=code --cov-report=html
```

### CI/CD Pipeline

Every push triggers:
- ✅ Linting (flake8, black, isort)
- ✅ Unit tests (100+ tests)
- ✅ Integration tests
- ✅ Docker builds
- ✅ Security scanning

---

## 🚀 Production Deployment

Ready to deploy Ooblex to production?

**See the comprehensive deployment guide:** [DEPLOYMENT.md](DEPLOYMENT.md)

Covers:
- ☁️ Cloud deployments (AWS, GCP, Azure)
- 🐳 Kubernetes with auto-scaling
- 🔒 SSL/TLS setup with Let's Encrypt
- 📊 Monitoring with Prometheus + Grafana
- 🔐 Security hardening checklist
- ⚡ Performance tuning
- 🌍 Multi-region deployment

**Quick Production Start:**
```bash
# 1. Complete security checklist
cp .env.example .env
nano .env  # Change all passwords and secrets

# 2. Get SSL certificate
sudo certbot certonly --standalone -d yourdomain.com

# 3. Deploy with Docker Compose
docker-compose up -d

# 4. Setup monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for full instructions.

---

## 📦 Components

### Core Services

| Service | Purpose | Lines |
|---------|---------|-------|
| **api.py** | WebSocket gateway | 120 |
| **brain.py** | ML worker | 252 |
| **decoder.py** | Video decoder | ~200 |
| **mjpeg.py** | MJPEG streaming | ~150 |
| **webrtc.py** | WebRTC gateway | ~180 |

### Infrastructure

- **Redis 7.x**: Frame queue, session storage
- **RabbitMQ 3.x**: Task distribution
- **Docker Compose**: One-command deployment
- **NGINX**: Reverse proxy, SSL

---

## 🎨 Available Effects

### ✅ Built-In (No Models Required)

These work immediately with `brain_simple.py`:

| Effect | Performance | Description |
|--------|-------------|-------------|
| **Face Detection** | ~50 FPS | OpenCV Haar cascades |
| **Pixelate Faces** | ~40 FPS | Privacy filter |
| **Cartoon** | ~30 FPS | Bilateral filter + edges |
| **Background Blur** | ~60 FPS | Gaussian blur |
| **Edge Detection** | ~80 FPS | Canny edges |
| **Grayscale** | ~100 FPS | B&W conversion |
| **Sepia** | ~90 FPS | Vintage effect |
| **Denoise** | ~25 FPS | Non-local means |
| **Mirror** | ~120 FPS | Horizontal flip |
| **Invert** | ~100 FPS | Color negative |

**All run on CPU, no GPU needed!**

---

### 🚀 Add Your Own Models

Want real AI models? See [models/README.md](models/README.md) for how to add:

**Supported Frameworks:**
- TensorFlow / TensorFlow Lite
- PyTorch / TorchScript
- ONNX Runtime (recommended)
- OpenVINO (Intel CPUs)
- TensorRT (NVIDIA GPUs)
- MediaPipe

**Popular Models:**
- Face swap (InsightFace)
- Style transfer (Fast Neural Style)
- Object detection (YOLOv8)
- Background removal (MediaPipe)
- Pose estimation (MediaPipe)

See examples in `models/README.md`

---

## ⚙️ Configuration

```bash
# Core Services
REDIS_URL=redis://localhost:6379
RABBITMQ_URL=amqp://guest:guest@localhost:5672

# WebRTC
WEBRTC_STUN_SERVERS=stun:stun.l.google.com:19302

# ML Configuration
MODEL_PATH=/models
CUDA_VISIBLE_DEVICES=0
ML_WORKER_REPLICAS=2

# API
API_PORT=8800
LOG_LEVEL=info
```

---

## 📊 Performance

### Benchmarks (RTX 3080, 16GB RAM)

| Effect | Resolution | Latency | FPS |
|--------|-----------|---------|-----|
| Face Detection | 640x480 | 180ms | 30 |
| Style Transfer | 640x480 | 350ms | 15 |
| Background Blur | 1280x720 | 120ms | 45 |
| Object Detection | 640x480 | 200ms | 25 |

### Scaling

- **1 worker**: ~15-30 FPS
- **5 workers**: ~60-120 FPS
- **10 workers**: ~150-250 FPS

**Total latency: 200-400ms** (sub-second for real-time feel)

---

## 🤝 Contributing

We welcome contributions!

### Development Setup

```bash
git clone https://github.com/ooblex/ooblex.git
cd ooblex

pip install -r requirements.txt
pre-commit install

pytest -v
black .
isort .
```

### Guidelines

- ✅ All PRs must pass CI/CD
- ✅ Add tests for new features
- ✅ Benchmark performance
- ❌ No unnecessary dependencies
- ❌ No AI-generated features without validation

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup
- **[HOW_IT_WORKS.md](HOW_IT_WORKS.md)** - Architecture deep dive
- **[CLEANUP_REPORT.md](CLEANUP_REPORT.md)** - What we removed and why
- **[API.md](docs/api.md)** - API documentation
- **[MODELS.md](docs/models.md)** - ML model guide
- **[DEPLOYMENT.md](docs/deployment.md)** - Production deployment

---

## 🗺️ Roadmap

### v2.1 (Current)
- ✅ Core pipeline working
- ✅ Docker deployment
- ✅ Basic AI effects
- ✅ 100+ tests
- ✅ CI/CD pipeline

### v2.2 (Next)
- ⏳ WHIP/WHEP protocol
- ⏳ Model marketplace
- ⏳ Web UI for configuration
- ⏳ Kubernetes deployment

### v2.3 (Future)
- 📅 Mobile SDK
- 📅 Edge deployment (RPi, Jetson)
- 📅 Multi-stream support
- 📅 Cloud templates (AWS, GCP, Azure)

---

## 🐛 Troubleshooting

### Common Issues

**WebRTC not connecting:**
- Ensure ports 8000-8010 UDP are open
- Check STUN/TURN configuration

**High latency (>500ms):**
- Add more workers
- Use GPU instead of CPU
- Reduce video resolution

**Workers not processing:**
- Check RabbitMQ connection: `docker compose logs rabbitmq`
- Check worker logs: `docker compose logs ml-worker`

See [TROUBLESHOOTING.md](docs/troubleshooting.md) for more.

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- **Original Author**: Steve Seguin (2018-2020)
- **WebRTC**: Janus Gateway, aiortc
- **ML Frameworks**: TensorFlow, PyTorch, ONNX

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ooblex/ooblex/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ooblex/ooblex/discussions)

---

**Built with ❤️ for real-time AI video processing**
