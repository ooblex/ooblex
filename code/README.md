# Legacy Code Directory

⚠️ **WARNING: This directory contains the original Ooblex implementation from 2018 using Python 2.7 and outdated dependencies.**

## DO NOT USE FOR PRODUCTION

This code is kept for historical reference only. For the modern, production-ready implementation, use the services in the parent directory.

## Modern Alternatives

- **API Server**: Use `services/api/` (FastAPI-based)
- **ML Processing**: Use `services/ml-worker/` (TF 2.x/PyTorch)
- **WebRTC**: Use `services/webrtc/` (WHIP/WHEP support)
- **Streaming**: Use `services/streaming/` (HLS/DASH)

## Migration Guide

See [MIGRATION.md](MIGRATION.md) for detailed instructions on migrating from this legacy code to the modern stack.

## Quick Start with Modern Stack

```bash
# From the parent directory
docker-compose up -d
```

## Why Keep This Code?

1. Historical reference
2. Algorithm documentation
3. Model conversion reference
4. Understanding original architecture

For any new development, always use the modern microservices architecture.