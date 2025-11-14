# NinjaSDK Integration - 5 Minute Quick Start

Get real-time voice-to-text working in 5 minutes!

## Step 1: Install Dependencies (1 minute)

```bash
cd ooblex

# Node.js dependencies
cd services/ninjasdk-audio-ingestion
npm install
cd ../..

# Python dependencies
pip install -r requirements.whisper.txt
pip install -r requirements.llm.txt
```

## Step 2: Start Infrastructure (1 minute)

```bash
# Start Redis and RabbitMQ
docker-compose -f docker-compose.ninjasdk.yml up -d redis rabbitmq

# Wait for services to be ready
sleep 10
```

## Step 3: Start Workers (30 seconds)

Open 3 terminals:

**Terminal 1 - Audio Ingestion:**
```bash
cd services/ninjasdk-audio-ingestion
node server.js
```

**Terminal 2 - Whisper STT:**
```bash
python code/whisper_worker.py
```

**Terminal 3 - LLM (Optional):**
```bash
python code/llm_worker.py
```

## Step 4: Open Demo (30 seconds)

```bash
# Start web server
docker-compose -f docker-compose.ninjasdk.yml up -d nginx

# Open in browser
open http://localhost:8800/ninjasdk/voice-to-text-demo.html
```

## Step 5: Test! (2 minutes)

1. Click "**Start Speaking**"
2. Allow microphone access
3. Say: "**Hello, this is a test of Ooblex voice to text**"
4. Watch the transcription appear in real-time!

---

## 🎉 Success!

You now have:
- ✅ P2P WebRTC audio streaming
- ✅ Real-time speech-to-text
- ✅ AI-powered responses (if LLM enabled)

## Next Steps

### Customize Your Room

```bash
# Edit room settings
export NINJASDK_ROOM=my-custom-room
export NINJASDK_PASSWORD=secret123

# Restart audio service
```

### Improve Performance

```bash
# Use smaller Whisper model (faster)
export WHISPER_MODEL=tiny
python code/whisper_worker.py

# Use GPU (requires CUDA)
export WHISPER_DEVICE=cuda
python code/whisper_worker.py
```

### Enable AI Responses

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama2

# Start LLM worker
export LLM_BACKEND=ollama
export LLM_MODEL=llama2
python code/llm_worker.py
```

## Docker Quick Start (Alternative)

Prefer Docker? Run everything with one command:

```bash
# Copy environment file
cp .env.ninjasdk.example .env.ninjasdk

# Start all services
docker-compose -f docker-compose.ninjasdk.yml up

# Open demo
open http://localhost:8800/ninjasdk/voice-to-text-demo.html
```

## Troubleshooting

### "Cannot connect to Redis"

```bash
# Check Redis is running
docker ps | grep redis

# Test connection
redis-cli ping
```

### "Whisper model not found"

```bash
# Model downloads on first run, be patient
# Or pre-download:
python -c "import whisper; whisper.load_model('base')"
```

### "No audio detected"

1. Check microphone permissions in browser
2. Try a different browser (Chrome/Firefox recommended)
3. Check audio service logs: `docker logs ooblex-ninjasdk-audio-1`

### "HTTPS required for microphone"

Some browsers require HTTPS. Quick fix:

```bash
# Use ngrok for testing
ngrok http 8800

# Open the https:// URL provided
```

## Configuration Files

- **Audio Service**: `services/ninjasdk-audio-ingestion/.env`
- **Workers**: `.env.ninjasdk`
- **Docker**: `docker-compose.ninjasdk.yml`

## What's Happening?

1. **Browser** → Streams audio via P2P WebRTC
2. **NinjaSDK Service** → Captures audio, converts to PCM16
3. **Redis** → Buffers audio chunks
4. **Whisper** → Transcribes speech to text
5. **RabbitMQ** → Distributes text results
6. **LLM** → Generates AI responses (optional)
7. **Browser** ← Receives transcriptions via data channel

## Performance Tips

| Action | Speed Gain | Quality Impact |
|--------|------------|----------------|
| Use GPU | 10-50x | None |
| Use `tiny` model | 32x | -20% |
| Use `base` model | 16x | -10% |
| Reduce chunk size | 2x latency | None |
| Skip LLM | 2-4s latency | N/A |

## Production Deployment

### With Docker Swarm

```bash
docker stack deploy -c docker-compose.ninjasdk.yml ooblex-voice
```

### With Kubernetes

```bash
kubectl apply -f k8s/ninjasdk/
```

### With PM2

```bash
pm2 start ecosystem.config.js
pm2 save
```

## Learn More

- [Full Documentation](README.md)
- [Architecture Details](ARCHITECTURE.md)
- [API Reference](API.md)
- [Examples](../../html/ninjasdk/)

## Get Help

- 📖 [Documentation](README.md)
- 💬 [Discord Community](#)
- 🐛 [Report Issues](https://github.com/ooblex/ooblex/issues)
- 📧 [Email Support](mailto:support@ooblex.com)

---

**Estimated Time**: 5 minutes
**Difficulty**: Beginner
**Prerequisites**: Docker, Node.js, Python

Happy coding! 🎙️✨
