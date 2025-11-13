# Ooblex - Improvement Suggestions

**Date:** 2025-11-13
**Status:** Post-cleanup recommendations

This document outlines suggested improvements to make Ooblex production-ready, more performant, and easier to use.

---

## 🔥 Critical (Do First)

### 1. **Fix Security Vulnerabilities**

**Current Issue:** GitHub detected 13 vulnerabilities (3 critical, 2 high)

**Actions:**
```bash
# Audit dependencies
pip-audit

# Update critical packages
pip install --upgrade tensorflow torch pillow cryptography

# Check for known vulnerabilities
safety check
```

**Specific Packages to Update:**
- `tensorflow`: Likely has security issues in older versions
- `pillow`: Image processing library often has CVEs
- `cryptography`: SSL/TLS vulnerabilities
- `aiohttp`: Web server vulnerabilities

**Priority:** CRITICAL - Do this immediately

---

### 2. **Add Missing Model Files**

**Current Issue:** Original TensorFlow face swap models not included (too large for Git)

**Solutions:**
1. **Host models externally**:
   ```bash
   # Create model download script
   ./scripts/download_models.sh
   ```

2. **Use Git LFS** for large files:
   ```bash
   git lfs install
   git lfs track "*.h5" "*.pb" "*.onnx"
   ```

3. **Model Registry** (recommended):
   - Create S3/GCS bucket for models
   - Add model metadata (name, size, accuracy, speed)
   - Implement download-on-demand
   - Cache models locally

**Files Needed:**
- `models/encoder256.h5`
- `models/decoder256_A.h5`
- `models/decoder256_B.h5`
- `models/encoder256_ENCODER.h5`
- `models/decoder256_A_TAYLOR.h5`

**Priority:** HIGH - Needed for full functionality

---

### 3. **Automate SSL Certificate Setup**

**Current Issue:** Manual SSL certificate configuration required

**Solution:**
```bash
# Add Let's Encrypt automation
sudo apt-get install certbot python3-certbot-nginx

# Create setup script
#!/bin/bash
# scripts/setup_ssl.sh

read -p "Enter your domain: " DOMAIN
read -p "Enter your email: " EMAIL

sudo certbot --nginx \
  -d $DOMAIN \
  --non-interactive \
  --agree-tos \
  --email $EMAIL

# Auto-renewal
sudo systemctl enable certbot.timer
```

**Priority:** HIGH - Blocks WebRTC in production

---

## 🚀 Performance Improvements

### 4. **Implement Frame Skipping**

**Benefit:** Reduce latency and CPU usage

```python
# services/decoder/decoder.py

class FrameSkipper:
    def __init__(self, skip_ratio=2):
        self.skip_ratio = skip_ratio
        self.counter = 0

    def should_process(self):
        self.counter += 1
        return self.counter % self.skip_ratio == 0

# Process every 2nd frame (15 FPS from 30 FPS input)
skipper = FrameSkipper(skip_ratio=2)

if skipper.should_process():
    process_frame(frame)
```

**Expected Impact:** 2x throughput, ~100ms latency reduction

---

### 5. **Add GPU Auto-Detection**

**Current Issue:** Manual CUDA configuration required

```python
# services/ml-worker/gpu_detector.py

import torch
import tensorflow as tf

def detect_gpu():
    """Detect available GPUs and return best configuration"""
    result = {
        'has_gpu': False,
        'framework': None,
        'device_count': 0,
        'devices': []
    }

    # Check PyTorch
    if torch.cuda.is_available():
        result['has_gpu'] = True
        result['framework'] = 'pytorch'
        result['device_count'] = torch.cuda.device_count()
        result['devices'] = [torch.cuda.get_device_name(i)
                             for i in range(result['device_count'])]

    # Check TensorFlow
    elif len(tf.config.list_physical_devices('GPU')) > 0:
        result['has_gpu'] = True
        result['framework'] = 'tensorflow'
        result['device_count'] = len(tf.config.list_physical_devices('GPU'))

    return result

# Auto-configure based on detection
gpu_info = detect_gpu()
if gpu_info['has_gpu']:
    print(f"Using {gpu_info['framework']} with {gpu_info['device_count']} GPU(s)")
else:
    print("No GPU detected, using CPU")
```

---

### 6. **Implement Batch Processing**

**Benefit:** Better GPU utilization, higher throughput

```python
# services/ml-worker/batch_processor.py

class BatchProcessor:
    def __init__(self, batch_size=8, timeout=0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.batch = []

    async def add_frame(self, frame):
        self.batch.append(frame)

        if len(self.batch) >= self.batch_size:
            return await self.process_batch()

    async def process_batch(self):
        if not self.batch:
            return []

        # Stack frames into batch
        batch_array = np.stack(self.batch)

        # Process all frames at once (GPU efficient)
        results = model.predict(batch_array)

        self.batch = []
        return results
```

**Expected Impact:** 3-5x throughput on GPU

---

## 🏗️ Architecture Improvements

### 7. **Add WHIP/WHEP Protocol Support**

**Current Issue:** Using older WebRTC methods

**WHIP/WHEP Benefits:**
- Standardized WebRTC ingest/egress
- Better browser compatibility
- Simpler implementation
- Lower latency

```python
# services/webrtc/whip_server.py

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, Request

app = FastAPI()

@app.post("/whip")
async def whip_endpoint(request: Request):
    """WHIP (WebRTC-HTTP Ingestion Protocol) endpoint"""
    offer = await request.body()

    pc = RTCPeerConnection()

    # Handle incoming tracks
    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            # Process video track
            asyncio.create_task(process_video_track(track))

    # Create answer
    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer, type="offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return Response(content=pc.localDescription.sdp, media_type="application/sdp")
```

**Priority:** MEDIUM - Improves WebRTC compatibility

---

### 8. **Add Model Warmup**

**Current Issue:** First frame has high latency

```python
# services/ml-worker/ml_worker.py

class MLWorker:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.warmed_up = False

    def warmup(self):
        """Pre-run model with dummy data to initialize GPU"""
        if self.warmed_up:
            return

        # Create dummy input
        dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)

        # Run inference 3 times to warm up
        for _ in range(3):
            _ = self.model.predict(dummy_input)

        self.warmed_up = True
        print("Model warmed up")

# Warmup on startup
worker = MLWorker("model.h5")
worker.warmup()
```

**Expected Impact:** Eliminate first-frame latency spike

---

### 9. **Implement Health Checks**

**Current Issue:** No way to monitor worker health

```python
# services/shared/health.py

from dataclasses import dataclass
from datetime import datetime

@dataclass
class HealthStatus:
    service: str
    status: str  # healthy, degraded, unhealthy
    last_check: datetime
    metrics: dict

class HealthChecker:
    def __init__(self):
        self.checks = {}

    async def check_redis(self, redis_client):
        try:
            await redis_client.ping()
            return HealthStatus(
                service="redis",
                status="healthy",
                last_check=datetime.utcnow(),
                metrics={"latency_ms": 1.2}
            )
        except Exception as e:
            return HealthStatus(
                service="redis",
                status="unhealthy",
                last_check=datetime.utcnow(),
                metrics={"error": str(e)}
            )

    async def check_all(self):
        """Run all health checks"""
        return {
            "redis": await self.check_redis(),
            "rabbitmq": await self.check_rabbitmq(),
            "ml_workers": await self.check_workers(),
        }
```

**Add to API:**
```python
@app.get("/health/detailed")
async def detailed_health():
    checker = HealthChecker()
    return await checker.check_all()
```

---

## 🎨 User Experience

### 10. **Create Web Configuration UI**

**Current Issue:** Configuration requires editing files

**Solution:** Build simple web UI for configuration

```html
<!-- html/config.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Ooblex Configuration</title>
</head>
<body>
    <h1>Ooblex Configuration</h1>

    <form id="config-form">
        <h2>Redis</h2>
        <input type="text" name="redis_url" placeholder="redis://localhost:6379">

        <h2>RabbitMQ</h2>
        <input type="text" name="rabbitmq_url" placeholder="amqp://localhost:5672">

        <h2>ML Workers</h2>
        <input type="number" name="worker_count" value="2" min="1" max="10">

        <h2>Video Settings</h2>
        <select name="resolution">
            <option value="480p">480p (640x480)</option>
            <option value="720p">720p (1280x720)</option>
            <option value="1080p">1080p (1920x1080)</option>
        </select>

        <button type="submit">Save Configuration</button>
    </form>

    <script src="js/config.js"></script>
</body>
</html>
```

---

### 11. **Add Model Marketplace**

**Vision:** Let users upload/download/share models

```python
# services/api/model_marketplace.py

@app.post("/models/upload")
async def upload_model(
    name: str,
    file: UploadFile,
    description: str,
    framework: str,  # tensorflow, pytorch, onnx
    input_size: tuple,
    current_user: User = Depends(get_current_user)
):
    """Upload a model to the marketplace"""
    # Validate model file
    # Store in S3/GCS
    # Add to database
    # Return model ID

@app.get("/models")
async def list_models(
    framework: Optional[str] = None,
    sort_by: str = "downloads"
):
    """List available models"""
    # Query database
    # Return model list with metadata

@app.get("/models/{model_id}/download")
async def download_model(model_id: str):
    """Download a model"""
    # Stream model file from storage
```

---

## 📱 Mobile & Edge

### 12. **Create Mobile SDK**

**Platforms:** iOS, Android, React Native

```swift
// iOS SDK Example
// OoblexSDK.swift

class OoblexClient {
    let apiURL: URL
    let streamKey: String

    init(apiURL: URL, streamKey: String) {
        self.apiURL = apiURL
        self.streamKey = streamKey
    }

    func startProcessing(effect: String) async throws {
        // Connect WebRTC
        // Start sending camera frames
        // Receive processed frames
    }

    func stopProcessing() {
        // Clean up connection
    }
}

// Usage
let client = OoblexClient(
    apiURL: URL(string: "https://api.ooblex.com")!,
    streamKey: "user_stream_123"
)

await client.startProcessing(effect: "face_swap")
```

---

### 13. **Raspberry Pi / Jetson Nano Support**

**Goal:** Run ML workers on edge devices

```bash
# deploy/edge/install_jetson.sh

#!/bin/bash
# Install Ooblex on Jetson Nano

# Install JetPack dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    libopencv-dev \
    python3-opencv

# Install TensorRT (NVIDIA optimized inference)
pip3 install nvidia-pyindex
pip3 install nvidia-tensorrt

# Configure for edge deployment
export OOBLEX_MODE=edge
export ML_BACKEND=tensorrt
export FRAME_RESOLUTION=480p
export WORKER_COUNT=1

# Run worker only (no API/WebRTC on edge)
python3 services/ml-worker/ml_worker_edge.py
```

---

## 🔒 Security & Privacy

### 14. **Add End-to-End Encryption**

**Concern:** Video frames contain sensitive data

```python
# services/shared/encryption.py

from cryptography.fernet import Fernet

class FrameEncryption:
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)

    def encrypt_frame(self, frame_data: bytes) -> bytes:
        """Encrypt frame before storing in Redis"""
        return self.cipher.encrypt(frame_data)

    def decrypt_frame(self, encrypted_data: bytes) -> bytes:
        """Decrypt frame for processing"""
        return self.cipher.decrypt(encrypted_data)

# Usage
encryption = FrameEncryption(Fernet.generate_key())

# Encrypt before Redis
encrypted = encryption.encrypt_frame(encoded_frame)
redis_client.setex(frame_id, 10, encrypted)

# Decrypt in worker
encrypted_data = redis_client.get(frame_id)
frame_data = encryption.decrypt_frame(encrypted_data)
```

---

### 15. **Implement Rate Limiting**

**Protect against abuse**

```python
# services/api/rate_limiter.py

from fastapi import HTTPException
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    def __init__(self, max_requests=100, window_seconds=60):
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests = {}  # client_id -> list of timestamps

    async def check_rate_limit(self, client_id: str):
        now = datetime.utcnow()

        # Initialize if new client
        if client_id not in self.requests:
            self.requests[client_id] = []

        # Remove old requests outside window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window
        ]

        # Check if over limit
        if len(self.requests[client_id]) >= self.max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Add current request
        self.requests[client_id].append(now)

# Usage
rate_limiter = RateLimiter(max_requests=1000, window_seconds=60)

@app.post("/process")
async def process(request: ProcessRequest, client_id: str = Depends(get_client_id)):
    await rate_limiter.check_rate_limit(client_id)
    # ... process request
```

---

## 🧪 Testing & Quality

### 16. **Add Contract Tests**

**Ensure API compatibility**

```python
# tests/contract/test_api_contracts.py

def test_process_endpoint_contract():
    """Test /process endpoint follows contract"""
    response = client.post("/process", json={
        "stream_token": "test",
        "process_type": "face_swap",
        "parameters": {}
    })

    assert response.status_code == 200

    # Validate response schema
    data = response.json()
    assert "task_id" in data
    assert "status" in data
    assert "message" in data

    assert isinstance(data["task_id"], str)
    assert data["status"] in ["queued", "processing", "completed", "failed"]
```

---

### 17. **Add Load Tests**

**Test system under load**

```python
# tests/load/test_load.py

import asyncio
import aiohttp
from locust import HttpUser, task, between

class OoblexUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def process_frame(self):
        self.client.post("/process", json={
            "stream_token": f"user_{self.user_id}",
            "process_type": "face_detection",
            "parameters": {}
        })

    @task
    def check_health(self):
        self.client.get("/health")

# Run with: locust -f test_load.py --headless -u 100 -r 10 -t 5m
```

---

## 📊 Monitoring & Observability

### 18. **Add Distributed Tracing**

**Track requests through system**

```python
# services/shared/tracing.py (enhance existing)

from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing(service_name: str):
    """Setup distributed tracing with Jaeger"""
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )

    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    return tracer

# Usage
tracer = setup_tracing("ooblex-api")

@app.post("/process")
async def process(request: ProcessRequest):
    with tracer.start_as_current_span("process_request") as span:
        span.set_attribute("stream_token", request.stream_token)
        span.set_attribute("process_type", request.process_type)

        # ... process request
```

---

### 19. **Add Grafana Dashboards**

**Visualize metrics**

Create `monitoring/grafana/dashboards/ooblex.json`:
- Processing latency over time
- Frames per second by worker
- Queue depth (Redis)
- Worker CPU/GPU utilization
- Error rates
- WebRTC connection count

---

## 🚀 Deployment

### 20. **Create Cloud Deployment Templates**

**AWS CloudFormation:**
```yaml
# deploy/aws/cloudformation.yaml
AWSTemplateFormatVersion: '2010-09-09'
Resources:
  OoblexECS:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: ooblex-production

  OoblexTaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: ooblex
      ContainerDefinitions:
        - Name: api
          Image: ooblex/api:latest
        - Name: ml-worker
          Image: ooblex/ml-worker:latest
          ResourceRequirements:
            - Type: GPU
              Value: 1
```

**Kubernetes Helm Chart:**
```yaml
# deploy/kubernetes/helm/values.yaml
replicaCount:
  api: 2
  mlWorker: 5

image:
  repository: ooblex
  tag: latest

resources:
  mlWorker:
    limits:
      nvidia.com/gpu: 1
```

**Terraform:**
```hcl
# deploy/terraform/main.tf
provider "aws" {
  region = "us-west-2"
}

module "ooblex" {
  source = "./modules/ooblex"

  instance_type = "g4dn.xlarge"  # GPU instance
  worker_count = 5
  redis_instance_type = "cache.r6g.large"
}
```

---

## 📝 Documentation

### 21. **Create Video Tutorials**

**Topics:**
1. Quick Start (5 min)
2. Adding Custom Effects (10 min)
3. Production Deployment (15 min)
4. Performance Tuning (10 min)
5. Troubleshooting Common Issues (8 min)

---

### 22. **Interactive API Documentation**

**Use Swagger/OpenAPI** (already in FastAPI):
- Add more examples
- Add curl commands
- Add response examples
- Add error codes documentation

---

## Priority Summary

| Priority | Item | Impact | Effort |
|----------|------|--------|--------|
| **CRITICAL** | Fix Security Vulnerabilities | High | Low |
| **CRITICAL** | Add Model Files | High | Low |
| **HIGH** | SSL Automation | High | Medium |
| **HIGH** | Frame Skipping | High | Low |
| **HIGH** | GPU Auto-Detection | Medium | Low |
| **MEDIUM** | Batch Processing | High | Medium |
| **MEDIUM** | WHIP/WHEP Support | Medium | High |
| **MEDIUM** | Web Config UI | Medium | Medium |
| **LOW** | Model Marketplace | High | Very High |
| **LOW** | Mobile SDK | High | Very High |

---

## Next Steps

1. **Week 1**: Fix security vulnerabilities, add model download script
2. **Week 2**: Implement SSL automation, add frame skipping
3. **Week 3**: Add GPU detection, implement health checks
4. **Week 4**: Create web config UI, add comprehensive tests
5. **Month 2**: WHIP/WHEP support, batch processing
6. **Month 3+**: Mobile SDK, model marketplace

---

**Remember:** Focus on core functionality first. Every feature should serve the primary goal: **ultra-low-latency real-time AI video processing**.

Avoid feature creep. Avoid AI slop. Test everything.
