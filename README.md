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

### Option 1: Docker Compose (Recommended)

**Run full WebRTC demo with AI effects:**

```bash
git clone https://github.com/ooblex/ooblex.git
cd ooblex

# Start everything (Redis, RabbitMQ, WebRTC gateway, ML workers)
./run-webrtc-demo.sh

# Open browser
open https://localhost/webrtc-demo.html

# Allow webcam access, select effect, see AI-processed video
```

**What you'll see:**
- Your webcam video on the left
- AI-processed video on the right (real-time)
- ~200-400ms latency
- Multiple effects to choose from

**Scale workers for more throughput:**
```bash
docker compose -f docker-compose.webrtc.yml up -d --scale ml-worker=5
```

### Option 2: Simple OpenCV Demo

```bash
./run-simple-demo.sh
# Processes test images through the AI pipeline
```

### Option 3: Bare Metal (Ubuntu)

```bash
sudo chmod +x *.sh

# Install dependencies (takes 30-60 minutes)
sudo ./install_opencv.sh      # OpenCV 4.x
sudo ./install_nginx.sh        # NGINX with SSL
sudo ./install_redis.sh        # Redis 7.x
sudo ./install_rabbitmq.sh     # RabbitMQ 3.x
sudo ./install_janus.sh        # Janus WebRTC gateway

# Configure
nano code/config.py  # Update Redis/RabbitMQ URLs and domain

# Run services
cd code
python3 api.py &
python3 brain.py &
python3 decoder.py &
python3 mjpeg.py &
python3 webrtc.py &
```

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

## 🎨 Available AI Effects

✅ **Face Detection** - MTCNN-based face detection

✅ **Style Transfer** - Apply artistic styles

✅ **Background Blur** - Depth-based blur

✅ **Edge Detection** - Canny, Sobel

✅ **Object Detection** - YOLO

✅ **Cartoon Filter** - Bilateral filtering

**Supported ML Frameworks:**
- TensorFlow / TensorFlow Lite
- PyTorch / TorchScript
- ONNX Runtime
- OpenVINO
- TensorRT
- MediaPipe

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
