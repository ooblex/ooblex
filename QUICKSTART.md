# Ooblex Quick Start Guide

## 🚀 Getting Started in 2 Minutes

### Prerequisites
- Docker and Docker Compose
- Python 3.7+ (for the demo controller)
- 4GB RAM minimum

### Step 1: Clone and Setup
```bash
git clone https://github.com/ooblex/ooblex.git
cd ooblex
```

### Step 2: Start Core Services
```bash
# Start the minimal stack
docker-compose up -d redis rabbitmq mjpeg

# Verify services are running
docker-compose ps
```

### Step 3: Run the Demo
```bash
# Install Python dependencies
pip install redis aioredis pillow numpy

# Run with test pattern (no camera needed!)
python demo_controller.py

# You should see:
# ✓ Connected to Redis
# ✓ Generating test pattern for stream: demo
# ✓ Publishing frames...
# View stream at: http://localhost:8081/demo.mjpg
```

### Step 4: View the Stream
Open your browser to: **http://localhost:8081/demo.mjpg**

You'll see a moving test pattern being "processed" by the AI pipeline.

## 📹 Using Real Video

### From a Video File
```bash
# Process a video file
python demo_controller.py --input path/to/video.mp4 --effect face_detection
```

### From Webcam (Linux/Mac)
```bash
# Requires a webcam connected
python demo_controller.py --input webcam --effect style_transfer
```

### From RTSP Camera
```bash
# Connect to IP camera
python demo_controller.py --input rtsp://camera.local:554/stream --effect object_detection
```

## 🎨 Available Effects

The demo supports these effects (via MediaPipe):
- `face_detection` - Detect and mark faces
- `object_detection` - Identify objects
- `style_transfer` - Apply artistic style
- `test_overlay` - Simple overlay effect (default)

## 🔧 Troubleshooting

### No Stream Showing?
```bash
# Check MJPEG server logs
docker-compose logs -f mjpeg

# Verify Redis is working
docker exec -it ooblex_redis_1 redis-cli ping
# Should return: PONG
```

### Connection Refused?
```bash
# Ensure services are running
docker-compose ps

# Restart if needed
docker-compose restart
```

### Performance Issues?
- The demo runs on CPU by default
- For GPU acceleration, you'd need to implement CUDA support
- Reduce frame rate: `python demo_controller.py --fps 15`

## 📁 Understanding the Structure

```
ooblex/
├── docker-compose.yml      # Service definitions
├── demo_controller.py      # Test/demo script
├── services/
│   ├── decoder/           # Video decoding service
│   ├── ml-worker/         # AI processing (needs models)
│   └── mjpeg/            # HTTP streaming output
└── code/                  # Legacy Python 2 code (reference only)
```

## 🎯 Next Steps

1. **Add ML Models**: Place ONNX/TensorFlow models in `models/`
2. **Enable GPU**: Modify docker-compose.yml for GPU support
3. **Add WebRTC**: Implement the WebRTC gateway for browser input
4. **Scale Up**: Add more ML workers for parallel processing

## 💡 What's Actually Working?

- ✅ Video input (file/webcam/RTSP via demo script)
- ✅ Frame extraction and queuing
- ✅ Basic ML processing pipeline
- ✅ MJPEG output streaming
- ✅ Test pattern generation

## ⚠️ What's Not Implemented?

- ❌ WebRTC browser input (docs describe it but not built)
- ❌ Advanced ML models (uses MediaPipe fallbacks)
- ❌ Kubernetes deployment (configs are examples only)
- ❌ Authentication/security features
- ❌ The web UI (HTML exists but backend APIs missing)

## 🆘 Getting Help

- Check logs: `docker-compose logs [service-name]`
- Review architecture: [HOW_IT_WORKS.md](HOW_IT_WORKS.md)
- Legacy code reference: [code/MIGRATION.md](code/MIGRATION.md)

Remember: This is a demonstration/educational project showing how to build a video processing pipeline. For production use, you'll need to implement proper error handling, security, and scalability features.