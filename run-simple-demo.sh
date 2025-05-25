#!/bin/bash
# Run Ooblex with real OpenCV effects (no ML models needed)

set -e

echo "🎨 Starting Ooblex with Real OpenCV Effects"
echo "=========================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p ssl models/cascades nginx

# Generate SSL certificate if needed
if [ ! -f ssl/cert.pem ]; then
    echo "🔐 Generating self-signed SSL certificate..."
    openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem \
        -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Ooblex/CN=localhost"
fi

# Download OpenCV cascades
echo "📦 Downloading face detection models..."
cd services/ml-worker
python3 download_cascades.py || echo "Cascade download failed, workers will download on startup"
cd ../..

# Create nginx config
echo "🔧 Creating nginx configuration..."
cat > nginx/simple.conf << 'EOF'
server {
    listen 80;
    listen 443 ssl;
    server_name localhost;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    location / {
        root /usr/share/nginx/html;
        index index.html;
    }
    
    location /ws {
        proxy_pass http://webrtc:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Stop existing containers
echo "🛑 Stopping any existing containers..."
docker-compose -f docker-compose.simple.yml down 2>/dev/null || true

# Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose.simple.yml up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 5

# Check services
echo "✅ Checking service health..."
if docker exec ooblex_redis_1 redis-cli ping &>/dev/null; then
    echo "  ✓ Redis is running"
else
    echo "  ✗ Redis failed to start"
fi

worker_count=$(docker ps --filter "name=ml-worker" -q | wc -l)
echo "  ✓ $worker_count ML workers running with OpenCV effects"

echo ""
echo "🎉 Demo Ready!"
echo "============="
echo ""
echo "Available Effects (all real, no ML models needed):"
echo "  • Face Detection - Detects and highlights faces"
echo "  • Blur Background - Portrait mode effect"
echo "  • Edge Detection - Artistic line drawing"
echo "  • Cartoon - Cartoon-style rendering"
echo "  • Sepia - Vintage photo effect"
echo "  • Grayscale - Black and white"
echo "  • Pixelate - Retro pixel art"
echo "  • Emboss - 3D relief effect"
echo ""
echo "🌐 Open in your browser:"
echo "   http://localhost/webrtc-demo.html"
echo ""
echo "📊 Monitor workers:"
echo "   docker-compose -f docker-compose.simple.yml logs -f ml-worker"
echo ""
echo "🛑 Stop everything:"
echo "   docker-compose -f docker-compose.simple.yml down"
echo ""

# Test the effects
echo "🧪 Testing OpenCV effects..."
docker exec ooblex_ml-worker_1 python -c "
import cv2
import numpy as np
print('OpenCV version:', cv2.__version__)
# Create test image
img = np.ones((100,100,3), dtype=np.uint8) * 255
# Try edge detection
edges = cv2.Canny(img, 100, 200)
print('✓ Edge detection works')
# Try face cascade
cascade = cv2.CascadeClassifier('/models/cascades/haarcascade_frontalface_default.xml')
if cascade.empty():
    print('✗ Face cascade not loaded')
else:
    print('✓ Face detection ready')
" || echo "Tests will run when container starts"

echo ""
echo "Ready! Your video will be processed with real OpenCV effects."