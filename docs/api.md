# API Documentation

## Overview

The Ooblex API provides a comprehensive RESTful interface for managing WebRTC streams, AI processing pipelines, and real-time video analytics. Built on FastAPI, it offers high-performance endpoints with automatic documentation and validation.

## Getting Started

### Base URL
```
https://api.ooblex.com/v1
```

### Authentication
All API requests require authentication using API keys or JWT tokens:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.ooblex.com/v1/streams
```

### Quick Start
```python
import requests

# Create a new stream
response = requests.post(
    "https://api.ooblex.com/v1/streams",
    headers={"Authorization": "Bearer YOUR_API_KEY"},
    json={
        "name": "My Stream",
        "type": "webrtc",
        "resolution": "1920x1080",
        "ai_models": ["emotion", "face_detection"]
    }
)

stream_id = response.json()["id"]
```

## Detailed Usage Examples

### Stream Management

#### Create Stream
```http
POST /v1/streams
Content-Type: application/json

{
    "name": "Conference Room",
    "type": "webrtc",
    "protocol": "whip",
    "resolution": "1920x1080",
    "framerate": 30,
    "bitrate": 2500000,
    "ai_models": ["emotion", "face_detection", "object_tracking"],
    "edge_processing": true,
    "blockchain_verification": true
}
```

Response:
```json
{
    "id": "stream_abc123",
    "name": "Conference Room",
    "whip_url": "https://api.ooblex.com/whip/stream_abc123",
    "whep_url": "https://api.ooblex.com/whep/stream_abc123",
    "vdo_ninja_url": "https://vdo.ninja/?view=stream_abc123",
    "status": "active",
    "created_at": "2024-01-15T10:00:00Z"
}
```

#### List Streams
```http
GET /v1/streams?page=1&limit=10&status=active
```

#### Get Stream Details
```http
GET /v1/streams/{stream_id}
```

#### Update Stream
```http
PATCH /v1/streams/{stream_id}
Content-Type: application/json

{
    "ai_models": ["emotion", "face_detection", "pose_estimation"],
    "recording_enabled": true
}
```

#### Delete Stream
```http
DELETE /v1/streams/{stream_id}
```

### AI Processing

#### Get Available Models
```http
GET /v1/ai/models
```

Response:
```json
{
    "models": [
        {
            "id": "emotion",
            "name": "Emotion Detection",
            "version": "2.1.0",
            "edge_compatible": true,
            "performance": {
                "fps": 30,
                "latency_ms": 15,
                "accuracy": 0.94
            }
        },
        {
            "id": "face_detection",
            "name": "Face Detection",
            "version": "3.0.1",
            "edge_compatible": true,
            "performance": {
                "fps": 60,
                "latency_ms": 8,
                "accuracy": 0.98
            }
        }
    ]
}
```

#### Configure AI Pipeline
```http
POST /v1/streams/{stream_id}/ai/pipeline
Content-Type: application/json

{
    "models": [
        {
            "id": "face_detection",
            "enabled": true,
            "confidence_threshold": 0.8,
            "edge_processing": true
        },
        {
            "id": "emotion",
            "enabled": true,
            "confidence_threshold": 0.7,
            "sampling_rate": 5
        }
    ],
    "output_format": "json",
    "webhook_url": "https://your-app.com/webhooks/ai-results"
}
```

#### Get AI Results
```http
GET /v1/streams/{stream_id}/ai/results?start_time=2024-01-15T10:00:00Z&end_time=2024-01-15T11:00:00Z
```

### Analytics

#### Get Stream Analytics
```http
GET /v1/streams/{stream_id}/analytics?period=day&metrics=viewers,quality,ai_detections
```

Response:
```json
{
    "stream_id": "stream_abc123",
    "period": "day",
    "metrics": {
        "viewers": {
            "peak": 156,
            "average": 89,
            "total_unique": 234
        },
        "quality": {
            "average_bitrate": 2340000,
            "packet_loss": 0.02,
            "jitter_ms": 12
        },
        "ai_detections": {
            "faces_detected": 1234,
            "emotions": {
                "happy": 456,
                "neutral": 678,
                "surprised": 100
            }
        }
    }
}
```

### Recording

#### Start Recording
```http
POST /v1/streams/{stream_id}/recording/start
Content-Type: application/json

{
    "format": "mp4",
    "resolution": "1920x1080",
    "include_ai_overlay": true,
    "storage": {
        "type": "s3",
        "bucket": "my-recordings",
        "path": "/recordings/{date}/{stream_id}"
    }
}
```

#### Stop Recording
```http
POST /v1/streams/{stream_id}/recording/stop
```

#### Get Recordings
```http
GET /v1/streams/{stream_id}/recordings
```

### Webhook Management

#### Register Webhook
```http
POST /v1/webhooks
Content-Type: application/json

{
    "url": "https://your-app.com/webhooks/ooblex",
    "events": [
        "stream.started",
        "stream.stopped",
        "ai.face_detected",
        "ai.emotion_detected",
        "recording.completed"
    ],
    "secret": "your-webhook-secret"
}
```

## Configuration Options

### API Configuration
```python
# config.py
API_CONFIG = {
    "version": "v1",
    "title": "Ooblex API",
    "description": "Real-time video processing and AI analytics",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "cors_origins": ["https://app.ooblex.com"],
    "rate_limiting": {
        "requests_per_minute": 60,
        "burst": 100
    },
    "authentication": {
        "type": "bearer",
        "token_expiry": 3600,
        "refresh_enabled": True
    }
}
```

### Stream Defaults
```python
STREAM_DEFAULTS = {
    "resolution": "1280x720",
    "framerate": 30,
    "bitrate": 1500000,
    "codec": "h264",
    "audio_codec": "opus",
    "audio_bitrate": 128000,
    "max_duration": 14400,  # 4 hours
    "idle_timeout": 300     # 5 minutes
}
```

### AI Processing Options
```python
AI_CONFIG = {
    "max_concurrent_models": 5,
    "processing_threads": 4,
    "edge_device_priority": True,
    "model_cache_size": "2GB",
    "result_retention_days": 30,
    "batch_processing": {
        "enabled": True,
        "batch_size": 10,
        "timeout_ms": 100
    }
}
```

## Best Practices

### Performance Optimization

1. **Use Batch Endpoints**: When processing multiple operations, use batch endpoints to reduce API calls:
```http
POST /v1/streams/batch
Content-Type: application/json

{
    "operations": [
        {"action": "create", "data": {...}},
        {"action": "update", "id": "stream_123", "data": {...}}
    ]
}
```

2. **Enable Caching**: Use ETags and conditional requests:
```http
GET /v1/streams/{stream_id}
If-None-Match: "etag-12345"
```

3. **Implement Webhooks**: Use webhooks instead of polling for real-time updates:
```python
# Webhook handler
@app.post("/webhooks/ooblex")
async def handle_webhook(request: Request):
    signature = request.headers.get("X-Ooblex-Signature")
    if verify_signature(request.body, signature):
        event = await request.json()
        await process_event(event)
```

### Error Handling

Always implement robust error handling:

```python
import asyncio
from typing import Optional

async def create_stream_with_retry(data: dict, max_retries: int = 3) -> Optional[dict]:
    for attempt in range(max_retries):
        try:
            response = await api_client.post("/v1/streams", json=data)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                wait_time = int(response.headers.get("Retry-After", 60))
                await asyncio.sleep(wait_time)
            else:
                response.raise_for_status()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Security

1. **API Key Rotation**: Rotate API keys regularly:
```http
POST /v1/auth/rotate-key
Authorization: Bearer OLD_API_KEY
```

2. **IP Whitelisting**: Configure allowed IPs:
```http
PUT /v1/account/security/ip-whitelist
Content-Type: application/json

{
    "ips": ["192.168.1.0/24", "10.0.0.1"]
}
```

3. **Request Signing**: For sensitive operations, use request signing:
```python
import hmac
import hashlib

def sign_request(method: str, path: str, body: str, secret: str) -> str:
    message = f"{method}:{path}:{body}"
    signature = hmac.new(
        secret.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return signature
```

## Troubleshooting

### Common Issues

#### 401 Unauthorized
- Check API key validity
- Ensure proper Authorization header format
- Verify API key permissions

#### 429 Too Many Requests
- Implement exponential backoff
- Check Retry-After header
- Consider upgrading rate limits

#### 500 Internal Server Error
- Check service status at status.ooblex.com
- Review request payload for invalid data
- Contact support with request ID

### Debug Mode

Enable debug mode for detailed error information:
```http
GET /v1/streams?debug=true
X-Debug-Token: YOUR_DEBUG_TOKEN
```

### Health Check

Monitor API health:
```http
GET /v1/health
```

Response:
```json
{
    "status": "healthy",
    "version": "1.2.3",
    "services": {
        "api": "operational",
        "streaming": "operational",
        "ai_processing": "operational",
        "storage": "operational"
    },
    "latency_ms": {
        "database": 2,
        "cache": 1,
        "ai_engine": 15
    }
}
```

### Support

For additional support:
- Documentation: https://docs.ooblex.com
- API Status: https://status.ooblex.com
- Support Email: support@ooblex.com
- Community Forum: https://forum.ooblex.com