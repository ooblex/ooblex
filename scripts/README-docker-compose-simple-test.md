# Docker Compose Simple Test

This directory contains test scripts and validation tools for `docker-compose.simple.yml`.

## Quick Start

To validate that `docker-compose.simple.yml` works correctly, run:

```bash
./scripts/test-docker-compose-simple.sh
```

This script will:
1. Validate the docker-compose configuration syntax
2. Build all required Docker images
3. Start all services (redis, rabbitmq, worker, api, mjpeg)
4. Wait for all services to become healthy
5. Verify connectivity to all services
6. Display sample logs from each service
7. Clean up all containers and volumes

## What's Being Tested

The test validates:
- **Redis** (port 6379): Frame queue storage
- **RabbitMQ** (port 5672): Task distribution
- **Worker**: ML worker instances running `brain_simple.py`
- **API** (port 8800): WebSocket + WebRTC API server
- **MJPEG** (port 8081): Streaming server

## Manual Testing

After starting the services with:
```bash
docker compose -f docker-compose.simple.yml up -d
```

You can manually test:

### API Service
```bash
curl http://localhost:8800/health
```

### MJPEG Stream
```bash
curl http://localhost:8081/stream
```

### Redis
```bash
redis-cli -h localhost -p 6379 ping
```

### RabbitMQ
```bash
# Check if RabbitMQ is accepting connections
nc -z localhost 5672 && echo "RabbitMQ is up"
```

### View Logs
```bash
# All services
docker compose -f docker-compose.simple.yml logs -f

# Specific service
docker compose -f docker-compose.simple.yml logs -f api
docker compose -f docker-compose.simple.yml logs -f worker
```

### Check Service Status
```bash
docker compose -f docker-compose.simple.yml ps
```

## Python Test Suite

For programmatic testing using pytest:

```bash
# Install test dependencies
pip install pytest requests redis pika

# Run the test suite
pytest tests/test_docker_compose_simple.py -v
```

The Python test suite includes:
- Config validation
- Image build verification
- Service startup validation
- Health checks for all services
- Connectivity tests
- Log verification

## GitHub Actions

The validation is also run automatically in GitHub Actions on every push to `main`, `develop`, and `claude/**` branches.

The workflow job `test-docker-compose-simple` performs:
- Config validation
- Image building
- Service health checks
- Connectivity verification
- Log inspection

## Troubleshooting

### Services not starting
```bash
# Check logs for errors
docker compose -f docker-compose.simple.yml logs

# Check specific service
docker compose -f docker-compose.simple.yml logs api
```

### Port conflicts
If ports 6379, 5672, 8800, or 8081 are already in use:
```bash
# Find what's using the port
sudo lsof -i :8800

# Stop the conflicting service or modify docker-compose.simple.yml
```

### Clean restart
```bash
# Stop and remove all containers and volumes
docker compose -f docker-compose.simple.yml down -v

# Rebuild and restart
docker compose -f docker-compose.simple.yml build
docker compose -f docker-compose.simple.yml up -d
```

### Resource issues
If containers are being OOM killed:
```bash
# Check Docker resource limits
docker stats

# Increase Docker Desktop memory allocation or add resource limits to docker-compose.simple.yml
```

## Cleanup

To stop all services and remove containers and volumes:
```bash
docker compose -f docker-compose.simple.yml down -v
```

To also remove built images:
```bash
docker compose -f docker-compose.simple.yml down -v --rmi all
```
