# Ooblex WebRTC Video Processing Demo

This is a fully functional WebRTC video processing demo that shows real-time video transformation using parallel ML workers.

## Features

- ✅ Real WebRTC video streaming between browser and server
- ✅ Parallel frame processing across multiple ML workers
- ✅ Redis-based frame queuing for distributed processing
- ✅ Multiple video effects (style transfer, background blur, cartoon, edge detection)
- ✅ Real-time performance metrics and monitoring
- ✅ Docker Compose setup for easy deployment

## Architecture

```
Browser (WebRTC) <--> WebRTC Server <--> Redis Queue <--> ML Workers (1-3)
                           |                                    |
                           v                                    v
                      Prometheus <------------------------ Metrics
                           |
                           v
                       Grafana
```

## Quick Start

1. **Run the demo:**
   ```bash
   ./run-webrtc-demo.sh
   ```

2. **Open your browser:**
   - Navigate to: https://localhost/webrtc-demo.html
   - Accept the self-signed certificate warning

3. **Use the demo:**
   - Click "Start Camera" to enable your webcam
   - Click "Connect" to establish WebRTC connection
   - Select different effects from the transform options
   - Monitor real-time FPS and latency

## Components

### WebRTC Server (`services/webrtc/webrtc_server.py`)
- Handles WebRTC signaling and media routing
- Samples video frames and queues them for processing
- Manages peer connections and data channels
- Serves processed frames back to clients

### ML Workers (`services/ml-worker/ml_worker_parallel.py`)
- Process frames from Redis queue in parallel
- Support multiple transformation types
- Use ONNX models when available, fallback to OpenCV effects
- Export Prometheus metrics

### Web Interface (`html/webrtc-demo.html`)
- Captures webcam video using getUserMedia
- Establishes WebRTC connection with server
- Displays original and processed video side-by-side
- Provides real-time performance metrics

## Video Processing Effects

1. **Style Transfer** - Applies artistic style to video
2. **Background Blur** - Blurs background while keeping foreground sharp
3. **Cartoon Effect** - Converts video to cartoon-like appearance
4. **Edge Detection** - Shows edges/outlines in the video

## Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

Key metrics:
- Active WebRTC connections
- Frames processed per second
- Processing latency
- Queue sizes
- Worker utilization

## Configuration

### Environment Variables
- `REDIS_URL`: Redis connection URL
- `ML_WORKER_COUNT`: Number of parallel workers per container
- `LOG_LEVEL`: Logging level (INFO/DEBUG)

### Scaling
To add more workers, edit `docker-compose.webrtc.yml`:
```yaml
ml-worker-4:
  build:
    context: ./services/ml-worker
    dockerfile: Dockerfile.cpu
  # ... same configuration as other workers
```

## Development

### Running locally without Docker:

1. **Start Redis:**
   ```bash
   redis-server
   ```

2. **Start WebRTC server:**
   ```bash
   cd services/webrtc
   pip install -r requirements.txt
   python webrtc_server.py
   ```

3. **Start ML workers:**
   ```bash
   cd services/ml-worker
   pip install -r requirements-cpu.txt
   python ml_worker_parallel.py
   ```

4. **Serve the web interface:**
   ```bash
   cd html
   python -m http.server 8000
   ```

### Adding new effects:

1. Add processing method to `ModelProcessor` class in `ml_worker_parallel.py`
2. Add UI button in `webrtc-demo.html`
3. Update transform handling in both server and worker

## Troubleshooting

### "No camera access"
- Ensure your browser has camera permissions
- HTTPS is required for getUserMedia (self-signed cert is OK for localhost)

### "Connection failed"
- Check that all services are running: `docker-compose -f docker-compose.webrtc.yml ps`
- Verify Redis is accessible
- Check browser console for errors

### "Low FPS"
- Reduce `FRAME_SAMPLE_RATE` to process fewer frames
- Add more ML workers
- Check CPU usage on workers

### "Effects not working"
- Ensure ONNX models are in the `models/` directory
- Check ML worker logs for errors
- Verify Redis connectivity

## Performance Tuning

- **Frame sampling**: Adjust `FRAME_SAMPLE_RATE` in `webrtc_server.py`
- **Queue size**: Modify `MAX_QUEUE_SIZE` to control memory usage
- **Worker count**: Scale `ML_WORKER_COUNT` based on CPU cores
- **Model optimization**: Use quantized ONNX models for faster inference

## Security Notes

- This demo uses self-signed certificates (not for production)
- WebRTC connections are encrypted by default
- Consider implementing authentication for production use
- Validate and sanitize all inputs

## Next Steps

- Add GPU support for faster processing
- Implement model hot-swapping
- Add WebRTC simulcast for adaptive quality
- Create mobile app using the mobile SDKs
- Add collaborative features (multiple users)