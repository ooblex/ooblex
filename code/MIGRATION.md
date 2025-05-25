# Migration Guide: Legacy Code to Modern Ooblex

This directory contains the original Ooblex implementation from 2018. The codebase has been completely modernized with a new microservices architecture. This guide helps you migrate from the legacy code to the new system.

## ⚠️ Important Note

The Python files in this directory use Python 2.7 and outdated libraries. They are kept for reference but should NOT be used in production. Use the modern implementation in the `services/` directory instead.

## Migration Mapping

### Old Components → New Services

| Legacy File | Purpose | New Service | Location |
|------------|---------|-------------|----------|
| `api.py` | WebSocket API server | API Gateway | `services/api/main.py` |
| `brain.py` | ML processing coordinator | ML Worker | `services/ml-worker/ml_worker.py` |
| `webrtc.py` | WebRTC signaling | WebRTC Gateway | `services/webrtc/` |
| `decoder.py` | Video decoding | Media Server | `services/media-server/` |
| `mjpeg.py` | MJPEG streaming | Streaming Server | `services/streaming/` |
| `model.py` | TensorFlow 1.x models | ML Models | `ml_models/` |
| `detect_face.py` | Face detection | Edge Computing | `services/edge-compute/` |

### Key Differences

#### 1. Python Version
- **Old**: Python 2.7
- **New**: Python 3.11+

#### 2. Web Framework
- **Old**: SimpleWebSocketServer
- **New**: FastAPI with WebSocket support

#### 3. Message Queue
- **Old**: Direct AMQP with amqpstorm
- **New**: RabbitMQ with aio-pika (async)

#### 4. ML Framework
- **Old**: TensorFlow 1.x
- **New**: TensorFlow 2.x / PyTorch

#### 5. WebRTC
- **Old**: Custom signaling
- **New**: WHIP/WHEP standards + VDO.Ninja support

## Migration Steps

### 1. Update Dependencies

Replace old requirements:
```bash
# Old
SimpleWebSocketServer
amqpstorm
tensorflow==1.12

# New - see services/api/requirements.txt
fastapi==0.115.6
aio-pika==9.5.0
tensorflow==2.18.0
```

### 2. Update API Endpoints

Old WebSocket API:
```python
# Legacy api.py
class SimpleChat(WebSocket):
    def handleMessage(self):
        # Process message
```

New FastAPI WebSocket:
```python
# services/api/main.py
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    # Process messages
```

### 3. Update ML Processing

Old TensorFlow 1.x:
```python
# Legacy model.py
with tf.Session() as sess:
    sess.run(output)
```

New TensorFlow 2.x:
```python
# services/ml-worker/ml_worker.py
model = tf.keras.models.load_model('model.h5')
output = model.predict(input_data)
```

### 4. Update WebRTC

Old custom signaling:
```python
# Legacy webrtc.py
def handle_offer(offer):
    # Custom handling
```

New WHIP/WHEP:
```python
# services/webrtc/whip_whep_server.py
async def handle_whip_post(self, request):
    # Standard WHIP protocol
```

### 5. Use Docker

Instead of manual installation:
```bash
# Don't use old install scripts
# Use Docker Compose instead:
docker-compose up -d
```

## Running Legacy Code (Not Recommended)

If you absolutely must run the legacy code:

1. Use Python 2.7 virtual environment:
```bash
virtualenv -p python2.7 venv
source venv/bin/activate
pip install -r legacy_requirements.txt
```

2. Update hardcoded paths in files
3. Use legacy install scripts with caution

## Recommended Approach

1. **Use the modern stack**: Deploy with Docker Compose
2. **Import models**: Convert old models to new format
3. **Migrate data**: Use migration scripts if needed
4. **Test thoroughly**: The new system has different behavior

## Getting Help

- Modern docs: [GitHub README](https://github.com/ooblex/ooblex#readme)
- API docs: [docs/api.md](../docs/api.md)
- Community: [Discord](https://discord.gg/ooblex)

## Why Keep Legacy Code?

- Historical reference
- Understanding original architecture
- Extracting specific algorithms
- Model conversion reference

Remember: The modern Ooblex is production-ready with better performance, security, and features. Always use the new implementation for any real deployment.