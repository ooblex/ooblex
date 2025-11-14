# How Ooblex Works - Complete Overview

## 🎯 What Ooblex Does

Ooblex is a real-time AI video processing platform. It:
1. **Accepts** live video from any source (webcam, mobile, OBS, IP cameras)
2. **Processes** it with AI models (face swap, style transfer, etc.)
3. **Outputs** the transformed video with minimal latency

Think of it as "Instagram filters for live video streams" but much more powerful.

## 🏗️ Architecture Explained

### The Microservices Approach

Yes, I created a microservices architecture! Here's why and how it works:

```
┌─────────────────────────────────────────────────────────┐
│                    Load Balancer                         │
│                     (Nginx)                              │
└──────┬──────────────────────────────────────────┬───────┘
       │                                          │
       ▼                                          ▼
┌──────────────┐                          ┌──────────────┐
│  Web UI      │                          │ Video Input  │
│  (React)     │                          │  (WebRTC)    │
└──────┬───────┘                          └──────┬───────┘
       │                                          │
       ▼                                          ▼
┌─────────────────────────────────────────────────────────┐
│                   API Gateway                            │
│                  (Port 8800)                             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Message Queue                           │
│                  (RabbitMQ)                              │
└──────┬──────────┬──────────┬──────────┬────────────────┘
       │          │          │          │
       ▼          ▼          ▼          ▼
   ML Worker  ML Worker  ML Worker   ...more workers
   (GPU #1)   (GPU #2)   (CPU)       (auto-scaling)
```

### Why Microservices?

1. **Scalability**: Add more ML workers for more concurrent streams
2. **Flexibility**: Different models can run on different hardware
3. **Reliability**: If one worker crashes, others continue
4. **Efficiency**: GPU resources are shared optimally

## 📹 Video Flow - Step by Step

### 1. Video Input
User starts streaming from their device:
- **Browser**: Uses WebRTC getUserMedia API
- **OBS**: Connects via RTMP to `rtmp://server:1935/live`
- **Mobile**: Uses native SDKs (iOS/Android)
- **IP Camera**: Connects via RTSP

### 2. Ingestion
The appropriate gateway receives the stream:
- **WebRTC Gateway** (port 8100): Handles browser connections
- **RTMP Server** (port 1935): Handles OBS/streaming software
- **WHIP Endpoint** (port 8800): Standards-based WebRTC

### 3. Frame Extraction
```python
# Simplified version of what happens
async def process_stream(stream):
    while True:
        frame = await stream.get_frame()  # Extract single frame
        frame_id = generate_id()
        
        # Send to Redis queue
        await redis.lpush("frames_to_process", {
            "id": frame_id,
            "data": frame,
            "effect": "face_swap",
            "stream_id": stream.id
        })
```

### 4. AI Processing
ML Workers continuously poll for frames:
```python
# ML Worker process
async def ml_worker():
    while True:
        # Get frame from queue
        frame_data = await redis.brpop("frames_to_process")
        
        # Apply AI model
        if frame_data["effect"] == "face_swap":
            processed = await face_swap_model.process(frame_data["data"])
        elif frame_data["effect"] == "style_transfer":
            processed = await style_transfer_model.process(frame_data["data"])
        
        # Send back processed frame
        await redis.lpush(f"processed_frames:{frame_data['stream_id']}", processed)
```

### 5. Output Encoding
Different services handle different output formats:

- **WebRTC Output**: Encodes to VP8/H264 and sends back via DataChannel
- **MJPEG Server**: Creates HTTP stream of JPEG images
- **HLS Server**: Packages into segments for CDN delivery

### 6. Viewing the Result

Users can view the processed video through:
- **Web Browser**: `https://server/watch/stream-id`
- **MJPEG URL**: `http://server:8081/stream.mjpg`
- **HLS Playlist**: `http://server:8084/stream/playlist.m3u8`
- **VLC Player**: Any of the above URLs

## 🚀 Running a Live Demo

### Quick Test - Command Line
```bash
# 1. Start Ooblex
docker-compose up -d

# 2. Stream from webcam (Linux/Mac)
ffmpeg -f v4l2 -i /dev/video0 \
       -c:v libx264 -preset ultrafast \
       -f rtsp rtsp://localhost:8554/test

# 3. View processed stream
open http://localhost:8081/test.mjpg
```

### Full Demo - Web Interface
```bash
# 1. Open browser
open https://localhost/demo

# 2. Click "Use Webcam"
# 3. Select an AI effect
# 4. See the magic happen!
```

## 🎮 How Each Component Works

### API Gateway (`services/api/main.py`)
- FastAPI application
- Handles authentication
- Routes requests to appropriate services
- WebSocket support for real-time updates

### ML Workers (`services/ml-worker/`)
- Loads AI models on startup
- Processes frames from queue
- Supports GPU acceleration
- Auto-scales based on load

### WebRTC Gateway (`services/webrtc/`)
- Handles WebRTC signaling
- Manages peer connections
- STUN/TURN integration
- Supports WHIP/WHEP protocols

### MJPEG Server (`services/mjpeg/`)
- Simple HTTP streaming
- No client plugins needed
- Works everywhere
- Good for monitoring/preview

### Media Server (`services/media-server/`)
- SFU (Selective Forwarding Unit)
- MCU (Multipoint Control Unit)
- Handles many-to-many streaming
- Room management

## 📊 Performance Characteristics

| Component | Latency | Use Case |
|-----------|---------|----------|
| WebRTC | 50-150ms | Real-time communication |
| MJPEG | 100-300ms | Simple monitoring |
| HLS | 2-6 seconds | Large scale broadcast |
| Edge (WASM) | 10-50ms | Privacy-sensitive |

## 🔧 Configuration

### Processing Options
```yaml
# .env file
DEFAULT_EFFECT=face_swap
ML_WORKER_REPLICAS=3
ENABLE_GPU=true
MAX_RESOLUTION=1080p
```

### Scaling
```bash
# Add more ML workers
docker-compose up -d --scale ml-worker=5

# Add more decode workers  
docker-compose up -d --scale decoder=3
```

## 🎯 Common Use Cases

1. **Live Streaming with Effects**
   - Streamer uses OBS → Applies effects → Streams to Twitch

2. **Video Conferencing**
   - Join meeting → Apply background blur → Others see processed video

3. **Security Monitoring**
   - IP cameras → Object detection → Alert on specific objects

4. **Virtual Production**
   - Multiple cameras → Green screen removal → Composite scene

## 💡 Why This Architecture?

The microservices design allows:
- **Horizontal scaling**: Add workers as needed
- **Technology flexibility**: Use best tool for each job
- **Fault tolerance**: One crash doesn't kill everything
- **Easy updates**: Update one service without touching others
- **Resource optimization**: GPU-heavy tasks on GPU nodes

Compare to original monolithic design:
- Old: Single Python script, one stream at a time
- New: Distributed system, hundreds of concurrent streams

## 🐛 Troubleshooting

**No video showing?**
```bash
# Check all services are running
docker-compose ps

# Check logs
docker-compose logs -f webrtc
docker-compose logs -f ml-worker
```

**High latency?**
- Check GPU is being used: `nvidia-smi`
- Reduce resolution in settings
- Use fewer ML workers if CPU-bound

**Can't connect?**
- Check firewall allows ports 8100, 8081
- Verify SSL certificates are valid
- Try with localhost first

## 🎓 Understanding the Code

Start here to understand the system:
1. `docker-compose.yml` - See all services
2. `services/api/main.py` - API entry points
3. `services/ml-worker/ml_worker.py` - AI processing
4. `services/mjpeg/mjpeg_server.py` - Simple output

The beauty is each service is independent - you can understand one without understanding all!