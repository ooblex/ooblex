# Ooblex - Modern WebRTC AI Video Processing Platform

[![CI/CD Pipeline](https://github.com/ooblex/ooblex/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/ooblex/ooblex/actions)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-brightgreen.svg)](docker-compose.yml)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-brightgreen.svg)](k8s/)

Ooblex is a distributed, scalable platform for real-time AI video processing using WebRTC. It enables low-latency video transformations including face swapping, style transfer, object detection, and more.

## 🚀 Features

- **Real-time Video Processing**: WebRTC-based streaming with sub-second latency
- **AI Transformations**: Face swap, style transfer, background removal, object detection
- **Scalable Architecture**: Microservices design with horizontal scaling
- **GPU Acceleration**: Support for NVIDIA GPUs for ML workloads
- **Modern Stack**: Python 3.11+, PyTorch/TensorFlow 2.x, FastAPI, Redis, RabbitMQ
- **Cloud Native**: Docker, Kubernetes, and Helm chart support
- **Production Ready**: Health checks, monitoring, logging, and security hardening

## 📋 Prerequisites

- Docker 24+ and Docker Compose 2.x
- Python 3.11+ (for local development)
- NVIDIA GPU + CUDA drivers (optional, for GPU acceleration)
- 8GB+ RAM minimum (16GB+ recommended)
- SSL certificates (self-signed provided for development)

## 🔧 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ooblex/ooblex.git
cd ooblex
```

### 2. Initial Setup
```bash
# Run the setup script
./deploy.sh setup

# Or use Make
make setup
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

### 4. Start Services
```bash
# Using deployment script
./deploy.sh start

# Or using Make
make dev

# Or using Docker Compose directly
docker-compose up -d
```

### 5. Access the Application
- Web Interface: https://localhost
- API Gateway: https://localhost:8800
- WebRTC Gateway: wss://localhost:8100
- MJPEG Stream: http://localhost:8081
- RabbitMQ Management: http://localhost:15672 (admin/admin)
- Grafana Dashboard: http://localhost:3000 (admin/admin)

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Nginx     │────▶│  API Gateway │────▶│   Redis     │
│  (Ingress)  │     │   (FastAPI)  │     │   Cache     │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  RabbitMQ    │
                    │  Message Bus │
                    └──────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  ML Worker   │   │  ML Worker   │   │  ML Worker   │
│   (GPU)      │   │   (GPU)      │   │   (GPU)      │
└──────────────┘   └──────────────┘   └──────────────┘
```

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

### Building Images
```bash
# Build all services
make build

# Build specific service
docker-compose build api
```

### Running Tests
```bash
# Run unit tests
make test

# Run with coverage
make test-coverage

# Run benchmarks
make bench
```

## 🚢 Deployment

### Docker Compose (Development/Staging)
```bash
# Start all services
./deploy.sh start development

# Stop services
./deploy.sh stop

# View logs
./deploy.sh logs [service]

# Check status
./deploy.sh status
```

### Kubernetes (Production)
```bash
# Deploy to Kubernetes
./deploy.sh k8s

# Or using kubectl
kubectl apply -f k8s/

# Or using Helm
helm install ooblex ./charts/ooblex
```

### Systemd (Bare Metal)
```bash
# Copy service files
sudo cp launch_scripts/*.service /etc/systemd/system/

# Create user and directories
sudo useradd -r -s /bin/false ooblex
sudo mkdir -p /opt/ooblex
sudo chown -R ooblex:ooblex /opt/ooblex

# Enable and start services
sudo systemctl daemon-reload
sudo systemctl enable ooblex-api ooblex-brain ooblex-webrtc
sudo systemctl start ooblex-api ooblex-brain ooblex-webrtc
```

## 📊 Monitoring

### Prometheus Metrics
- API metrics: http://localhost:8800/metrics
- Custom ML metrics tracked
- Resource usage monitoring

### Grafana Dashboards
- Service health overview
- ML model performance
- WebRTC connection stats
- Resource utilization

### Logging
- Structured JSON logging
- Centralized log aggregation
- Log levels: DEBUG, INFO, WARN, ERROR

## 🔒 Security

- JWT-based authentication
- SSL/TLS encryption
- Rate limiting
- Input validation
- Secrets management
- Network policies (Kubernetes)
- Security scanning in CI/CD

## 📦 ML Models

Place your models in the `models/` directory:
```
models/
├── face_detection.onnx
├── face_swap.onnx
├── style_transfer.onnx
└── background_removal.onnx
```

Supported formats:
- ONNX (recommended)
- PyTorch (.pt, .pth)
- TensorFlow SavedModel

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Environment Variables

Key configuration options:
```bash
# Core Settings
NODE_ENV=development
LOG_LEVEL=info

# Services
REDIS_URL=redis://redis:6379
RABBITMQ_URL=amqp://admin:admin@rabbitmq:5672
DATABASE_URL=postgresql://ooblex:password@postgres:5432/ooblex

# ML Configuration  
MODEL_PATH=/models
CUDA_VISIBLE_DEVICES=0
ML_WORKER_REPLICAS=2

# WebRTC
WEBRTC_STUN_SERVERS=stun:stun.l.google.com:19302

# Features
ENABLE_FACE_SWAP=true
ENABLE_STYLE_TRANSFER=true
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Check NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi
   ```

2. **Services not starting**
   ```bash
   # Check logs
   ./deploy.sh logs [service-name]
   
   # Check service health
   docker-compose ps
   ```

3. **WebRTC connection issues**
   - Ensure SSL certificates are trusted
   - Check firewall rules for UDP ports 10000-10100
   - Verify STUN/TURN server configuration

## 📚 Documentation

- [API Documentation](docs/api.md)
- [WebRTC Integration](docs/webrtc.md)
- [ML Model Guide](docs/models.md)
- [Deployment Guide](docs/deployment.md)
- [Security Best Practices](docs/security.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original Ooblex concept and implementation
- [Janus WebRTC Gateway](https://janus.conf.meetecho.com/)
- [MediaPipe](https://mediapipe.dev/)
- [PyTorch](https://pytorch.org/) & [TensorFlow](https://tensorflow.org/)

---

Made with ❤️ by the Ooblex Team
