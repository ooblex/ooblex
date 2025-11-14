#!/bin/bash
# Start the complete WebRTC demo with parallel processing

set -e

echo "🚀 Starting Ooblex WebRTC Demo with Parallel Processing"
echo "======================================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it first."
    exit 1
fi

# Create required directories
echo "📁 Creating directories..."
mkdir -p ssl models logs

# Generate self-signed SSL certificate if not exists
if [ ! -f ssl/cert.pem ]; then
    echo "🔐 Generating self-signed SSL certificate..."
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
        -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Ooblex/CN=localhost"
fi

# Create simple model placeholders if not exist
if [ ! -f models/face_detection.onnx ]; then
    echo "📦 Creating model placeholders..."
    echo "placeholder" > models/face_detection.onnx
    echo "placeholder" > models/style_transfer.onnx
    echo "placeholder" > models/background_blur.onnx
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.webrtc.yml down 2>/dev/null || true

# Start services
echo "🐳 Starting services..."
docker-compose -f docker-compose.webrtc.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 5

# Check service health
echo "🏥 Checking service health..."
services=("redis:6379" "webrtc:8000")
for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    container_name="ooblex_${name}_1"
    
    if docker exec $container_name echo "OK" &>/dev/null; then
        echo "  ✅ $name is running"
    else
        echo "  ❌ $name failed to start"
        echo "     Check logs: docker logs $container_name"
    fi
done

# Count ML workers
worker_count=$(docker ps --filter "name=ml-worker" -q | wc -l)
echo "  ✅ $worker_count ML workers running"

# Display access information
echo ""
echo "✨ WebRTC Demo Ready!"
echo "===================="
echo ""
echo "🌐 Access the demo at:"
echo "   https://localhost/webrtc-demo.html"
echo ""
echo "📊 Monitor performance:"
echo "   Grafana: http://localhost:3000 (admin/admin)"
echo "   Redis Commander: http://localhost:8081"
echo ""
echo "🔍 View logs:"
echo "   All services: docker-compose -f docker-compose.webrtc.yml logs -f"
echo "   Workers only: docker-compose -f docker-compose.webrtc.yml logs -f ml-worker"
echo ""
echo "🎚️ Scale workers:"
echo "   More workers: docker-compose -f docker-compose.webrtc.yml up -d --scale ml-worker=5"
echo "   Fewer workers: docker-compose -f docker-compose.webrtc.yml up -d --scale ml-worker=2"
echo ""
echo "🧪 Test the workflow:"
echo "   python3 test_webrtc_workflow.py"
echo ""
echo "🛑 Stop everything:"
echo "   docker-compose -f docker-compose.webrtc.yml down"
echo ""
echo "⚠️  Note: Your browser may warn about the self-signed certificate."
echo "   This is normal for local development. Click 'Advanced' → 'Proceed'."
echo ""

# Optional: Run the test
read -p "Run workflow test now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running workflow test..."
    python3 test_webrtc_workflow.py
fi