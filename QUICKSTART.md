# Ooblex Quick Start Guide

**Get Ooblex running in 5 minutes with zero friction.**

## Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- 4GB RAM minimum

**No GPU, no AI models, no downloads needed!**

---

## Quick Start (2 Minutes)

### Step 1: Clone and Start Services
```bash
git clone https://github.com/ooblex/ooblex.git
cd ooblex

# Start Redis + RabbitMQ
docker compose up -d redis rabbitmq
sleep 10  # Wait for services
```

### Step 2: Install Dependencies
```bash
pip install opencv-python redis amqpstorm numpy
```

### Step 3: Run Worker
```bash
python3 code/brain_simple.py
```

✅ **Done! Ooblex is running with 10 real-time effects!**

---

## What You Built

A real-time video processing pipeline:

```
Input → Redis → ML Worker → Redis → Output
```

**10 effects available** (30-100+ FPS on CPU):
- Face Detection, Pixelate Faces, Cartoon
- Background Blur, Edge Detection
- Grayscale, Sepia, Denoise, Mirror, Invert

---

## Test It

```python
import redis, cv2, numpy as np

r = redis.Redis()
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
_, encoded = cv2.imencode('.jpg', frame)
r.setex("test_001", 30, encoded.tobytes())
# Worker processes automatically!
```

---

## Next Steps

**Scale Workers:**
```bash
# Run multiple workers for parallel processing
python3 code/brain_simple.py &
python3 code/brain_simple.py &
python3 code/brain_simple.py &
```

**Add WebRTC:** See `run-webrtc-demo.sh`

**Add AI Models:** See `models/README.md`

**Deploy:** See `DEPLOYMENT.md`

---

## Troubleshooting

**Redis connection error:**
```bash
redis-cli ping  # Should return PONG
docker compose logs redis
```

**Worker not processing:**
```bash
# Check RabbitMQ: http://localhost:15672 (guest/guest)
# Look for "tf-task" queue
```

---

**Full docs:** [README.md](README.md) | **Issues:** [GitHub](https://github.com/ooblex/ooblex/issues)
