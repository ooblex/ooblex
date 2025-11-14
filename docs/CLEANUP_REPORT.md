# Ooblex Cleanup Report - Removing AI Slop

**Date:** 2025-11-13
**Commit Reference:** Original working version at f974894d8ec0355009615cfab5f654f108862288 (2020)

## Core Functionality (What MUST be preserved)

### Original 2020 Architecture
```
Browser WebRTC → api.py (WebSocket) → RabbitMQ → brain.py (TensorFlow ML Worker)
                                                      ↓
                                              Redis Frame Queue
                                                      ↓
                                              decoder.py (Frame decode)
                                                      ↓
                                              mjpeg.py (MJPEG output) → Browser
```

### Core Services (from `/code/` directory)
1. **api.py** - Simple WebSocket server for client communication (120 lines)
2. **brain.py** - ML worker with TensorFlow face swap models (252 lines)
3. **decoder.py** - Video frame decoder from WebRTC
4. **mjpeg.py** - MJPEG streaming output server
5. **webrtc.py** - WebRTC gateway integration with Janus
6. **config.py** - Configuration for Redis/RabbitMQ/Domain
7. **tensorthread_shell.py** - Template for custom ML workers

### Supporting Files (KEEP)
- Installation scripts: `install_*.sh`
- HTML client: `html/index.html`, `html/js/ooblex.v0.js`
- Janus configs: `janus_confs/`
- Launch scripts: `launch_scripts/*.service`

## Services to REMOVE (AI Slop)

### 1. Blockchain Service (~1,085 lines) ❌
- `services/blockchain/blockchain_service.py`
- `services/blockchain/ipfs_client.py`
- **Reason:** Never in original. No blockchain in video processing platform.

### 2. Collaboration Service (~595 lines) ❌
- `services/collaboration/collaboration_server.py`
- `services/collaboration/models.py`
- **Reason:** Never in original. Not core to video ML processing.

### 3. Edge Compute Service ❌
- `services/edge-compute/edge_server.py`
- **Reason:** Never in original. Vague feature without implementation.

### 4. Media Server Service ❌
- `services/media-server/media_server.py`
- **Reason:** Duplicate of existing functionality.

### 5. Streaming Service ❌
- `services/streaming/streaming_server.py`
- `services/streaming/example_client.py`
- **Reason:** Duplicate of MJPEG functionality.

### 6. Over-Complicated API Rewrite ❌
- `services/api/main.py` - Replaced simple 120-line WebSocket with FastAPI + OAuth + JWT + Blockchain calls
- **Decision:** Simplify or revert to original pattern

## Services to KEEP and MODERNIZE

### From `/services/` directory:

1. **services/decoder/** ✅
   - Modern Python 3 rewrite of decoder.py
   - Keep but ensure alignment with original functionality

2. **services/mjpeg/** ✅
   - Modern rewrite of mjpeg.py
   - Keep but simplify

3. **services/ml-worker/** ✅
   - Modern rewrite of brain.py
   - Keep but ensure it follows original tensor thread pattern

4. **services/webrtc/** ✅
   - Modern WebRTC gateway
   - Keep but simplify (remove WHIP/WHEP if not working)

5. **services/shared/** ✅
   - Common utilities (tracing, etc.)
   - Keep if minimal

## Docker Compose Files

Keep:
- `docker-compose.simple.yml` ✅ - Minimal working setup
- `docker-compose.webrtc.yml` ✅ - Full WebRTC demo

Remove or simplify:
- `docker-compose.yml` - Remove blockchain/collaboration references
- `docker-compose.observability.yml` - Optional, can keep if lightweight

## Testing Requirements

### Installation Tests
- Validate install scripts still work on Ubuntu 20.04/22.04
- Check dependency installation
- Verify service startup

### Unit Tests
- Mock Redis/RabbitMQ connections
- Test frame processing without actual ML models
- Test WebSocket message handling
- Test frame encoding/decoding

### Integration Tests
- Test full pipeline with dummy frames
- Test service-to-service communication
- Test docker-compose startup

### GitHub Actions CI
- Run unit tests on push
- Validate Python dependencies
- Check Docker builds
- Run integration tests

## Success Criteria

1. ✅ Codebase reduced by ~2,000+ lines of unnecessary code
2. ✅ Core functionality preserved and working
3. ✅ Dependencies updated and secure
4. ✅ Automated tests prevent future slop
5. ✅ Documentation clearly describes what works
6. ✅ Docker setup works out of the box
7. ✅ Original simplicity restored with modern best practices
