#!/bin/bash
# Test script for docker-compose.simple.yml
# Validates that all services start correctly and are healthy

set -e

COMPOSE_FILE="docker-compose.simple.yml"
TIMEOUT=120

echo "=== Testing docker-compose.simple.yml ==="

# Function to wait for a service to be healthy
wait_for_service() {
    local service=$1
    local max_attempts=$2
    local attempt=0

    echo "Waiting for $service to be healthy..."
    while [ $attempt -lt $max_attempts ]; do
        if docker compose -f $COMPOSE_FILE ps $service | grep -q "healthy\|running"; then
            echo "✓ $service is healthy"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "✗ $service failed to become healthy"
    return 1
}

# Function to check if a port is responding
wait_for_port() {
    local host=$1
    local port=$2
    local max_attempts=$3
    local attempt=0

    echo "Waiting for $host:$port to respond..."
    while [ $attempt -lt $max_attempts ]; do
        if nc -z $host $port 2>/dev/null; then
            echo "✓ $host:$port is responding"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "✗ $host:$port is not responding"
    return 1
}

# Cleanup function
cleanup() {
    echo ""
    echo "=== Cleaning up ==="
    docker compose -f $COMPOSE_FILE logs --tail=50
    docker compose -f $COMPOSE_FILE down -v
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Step 1: Validate config
echo ""
echo "Step 1: Validating docker-compose config..."
docker compose -f $COMPOSE_FILE config > /dev/null
echo "✓ Config is valid"

# Step 2: Build images
echo ""
echo "Step 2: Building images..."
docker compose -f $COMPOSE_FILE build
echo "✓ Images built successfully"

# Step 3: Start services
echo ""
echo "Step 3: Starting services..."
docker compose -f $COMPOSE_FILE up -d
echo "✓ Services started"

# Step 4: Wait for services to be healthy
echo ""
echo "Step 4: Checking service health..."

# Check Redis
wait_for_service "redis" 30 || exit 1
wait_for_port "localhost" 6379 30 || exit 1

# Check RabbitMQ
wait_for_service "rabbitmq" 30 || exit 1
wait_for_port "localhost" 5672 30 || exit 1

# Check API
wait_for_port "localhost" 8800 60 || exit 1

# Check MJPEG
wait_for_port "localhost" 8081 60 || exit 1

# Step 5: Check all services are running
echo ""
echo "Step 5: Verifying all services are running..."
docker compose -f $COMPOSE_FILE ps

# Count running services
EXPECTED_SERVICES=5  # redis, rabbitmq, worker (2 replicas count as 1), api, mjpeg
RUNNING_SERVICES=$(docker compose -f $COMPOSE_FILE ps --services --filter "status=running" | wc -l)

if [ $RUNNING_SERVICES -ge $EXPECTED_SERVICES ]; then
    echo "✓ All expected services are running ($RUNNING_SERVICES/$EXPECTED_SERVICES)"
else
    echo "✗ Not all services are running ($RUNNING_SERVICES/$EXPECTED_SERVICES)"
    exit 1
fi

# Step 6: Test Redis connectivity
echo ""
echo "Step 6: Testing Redis connectivity..."
if command -v redis-cli &> /dev/null; then
    redis-cli -h localhost -p 6379 ping
    echo "✓ Redis is responding to commands"
fi

# Step 7: Show service logs (last 10 lines each)
echo ""
echo "Step 7: Sample service logs..."
for service in redis rabbitmq worker api mjpeg; do
    echo ""
    echo "--- $service logs (last 10 lines) ---"
    docker compose -f $COMPOSE_FILE logs --tail=10 $service
done

echo ""
echo "=== ✓ All tests passed! ==="
echo ""
echo "To manually test the services:"
echo "  - API: http://localhost:8800"
echo "  - MJPEG Stream: http://localhost:8081/stream"
echo "  - Redis: localhost:6379"
echo "  - RabbitMQ: localhost:5672"
echo ""
