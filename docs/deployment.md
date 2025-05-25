# Deployment Guide

## Overview

This guide covers deploying Ooblex in various environments, from single-server setups to distributed edge computing networks. We support containerized deployments, Kubernetes orchestration, edge device deployment, and blockchain-integrated architectures.

## Getting Started

### System Requirements

#### Minimum Requirements
- CPU: 4 cores (x86_64 or ARM64)
- RAM: 8GB
- Storage: 50GB SSD
- Network: 100 Mbps
- OS: Ubuntu 20.04+, Debian 11+, RHEL 8+

#### Recommended Requirements
- CPU: 8+ cores with AVX2 support
- RAM: 32GB
- Storage: 500GB NVMe SSD
- Network: 1 Gbps
- GPU: NVIDIA RTX 3060+ or Edge TPU
- OS: Ubuntu 22.04 LTS

### Quick Deployment

```bash
# Clone repository
git clone https://github.com/ooblex/ooblex.git
cd ooblex

# Run automated deployment
./deploy.sh --environment production --components all

# Or use Docker Compose
docker-compose up -d

# Verify deployment
curl http://localhost:8080/health
```

## Detailed Deployment Scenarios

### Single Server Deployment

For small to medium deployments on a single server:

```bash
#!/bin/bash
# Single server deployment script

# 1. Install dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.9 python3-pip \
    nginx redis-server \
    postgresql-13 \
    gstreamer1.0-tools \
    libopencv-dev

# 2. Install Python packages
pip3 install -r requirements.txt

# 3. Configure PostgreSQL
sudo -u postgres createdb ooblex
sudo -u postgres createuser ooblex_user -P
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ooblex TO ooblex_user;"

# 4. Configure Redis
sudo sed -i 's/supervised no/supervised systemd/' /etc/redis/redis.conf
sudo systemctl restart redis

# 5. Install Janus Gateway
./install_janus.sh

# 6. Configure Nginx
sudo cp nginx/ooblex.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/ooblex.conf /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# 7. Setup systemd services
sudo cp launch_scripts/*.service /etc/systemd/system/
sudo systemctl daemon-reload

# 8. Start services
for service in api brain decoder webrtc janus; do
    sudo systemctl enable ooblex-$service
    sudo systemctl start ooblex-$service
done

# 9. Configure firewall
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8188/tcp  # WebSocket
sudo ufw allow 10000:60000/udp  # RTP
```

### Docker Deployment

#### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Service
  api:
    build:
      context: .
      dockerfile: docker/api.Dockerfile
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgresql://ooblex:password@postgres:5432/ooblex
      - REDIS_URL=redis://redis:6379
      - JANUS_URL=ws://janus:8188
      - AI_BACKEND=brain:50051
    depends_on:
      - postgres
      - redis
      - brain
    volumes:
      - ./config:/app/config
      - media_storage:/app/media
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  # AI Brain Service
  brain:
    build:
      context: .
      dockerfile: docker/brain.Dockerfile
    ports:
      - "50051:50051"
    volumes:
      - ./models:/app/models
      - model_cache:/app/cache
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - TF_FORCE_GPU_ALLOW_GROWTH=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # WebRTC/Janus Gateway
  janus:
    build:
      context: .
      dockerfile: docker/janus.Dockerfile
    ports:
      - "8088:8088"  # HTTP
      - "8188:8188"  # WebSocket
      - "10000-10500:10000-10500/udp"  # RTP
    volumes:
      - ./janus_confs:/etc/janus
    environment:
      - JANUS_NAT_1_1_MAPPING=${PUBLIC_IP}
    restart: unless-stopped

  # Video Decoder Service
  decoder:
    build:
      context: .
      dockerfile: docker/decoder.Dockerfile
    volumes:
      - media_storage:/app/media
      - /dev/dri:/dev/dri  # Hardware acceleration
    environment:
      - DECODER_WORKERS=4
      - HARDWARE_ACCEL=vaapi
    restart: unless-stopped

  # PostgreSQL Database
  postgres:
    image: postgres:14-alpine
    environment:
      - POSTGRES_DB=ooblex
      - POSTGRES_USER=ooblex
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
      - static_files:/usr/share/nginx/html
    depends_on:
      - api
      - janus
    restart: unless-stopped

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  media_storage:
  model_cache:
  static_files:
  prometheus_data:
  grafana_data:
```

#### Custom Dockerfile Examples

```dockerfile
# docker/api.Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libpq-dev \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY code/ ./code/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 ooblex && chown -R ooblex:ooblex /app
USER ooblex

# Run API server
CMD ["uvicorn", "code.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

### Kubernetes Deployment

#### Helm Chart Configuration

```yaml
# helm/ooblex/values.yaml
global:
  imageRegistry: registry.ooblex.com
  imagePullSecrets:
    - name: ooblex-registry

api:
  replicaCount: 3
  image:
    repository: ooblex/api
    tag: "latest"
  service:
    type: LoadBalancer
    port: 80
  ingress:
    enabled: true
    className: nginx
    annotations:
      cert-manager.io/cluster-issuer: letsencrypt-prod
    hosts:
      - host: api.ooblex.com
        paths:
          - path: /
            pathType: Prefix
    tls:
      - secretName: api-tls
        hosts:
          - api.ooblex.com
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "2000m"
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70

brain:
  replicaCount: 2
  image:
    repository: ooblex/brain
    tag: "latest"
  nodeSelector:
    gpu: "true"
  resources:
    requests:
      memory: "4Gi"
      cpu: "2000m"
      nvidia.com/gpu: 1
    limits:
      memory: "8Gi"
      cpu: "4000m"
      nvidia.com/gpu: 1

janus:
  replicaCount: 5
  image:
    repository: ooblex/janus
    tag: "latest"
  service:
    type: LoadBalancer
    annotations:
      service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
  ports:
    http: 8088
    websocket: 8188
    rtpStart: 10000
    rtpEnd: 10500
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app: janus
          topologyKey: kubernetes.io/hostname

postgresql:
  enabled: true
  auth:
    postgresPassword: "changeme"
    database: "ooblex"
  primary:
    persistence:
      size: 100Gi
      storageClass: fast-ssd
  metrics:
    enabled: true

redis:
  enabled: true
  auth:
    enabled: true
    password: "changeme"
  master:
    persistence:
      size: 20Gi
  replica:
    replicaCount: 2
  metrics:
    enabled: true

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true
    adminPassword: "changeme"
```

#### Kubernetes Manifests

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ooblex-api
  namespace: ooblex
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ooblex-api
  template:
    metadata:
      labels:
        app: ooblex-api
    spec:
      containers:
      - name: api
        image: registry.ooblex.com/ooblex/api:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: ooblex-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: ooblex-config
              key: redis-url
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ooblex-api
  namespace: ooblex
spec:
  selector:
    app: ooblex-api
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ooblex-api-hpa
  namespace: ooblex
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ooblex-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Edge Deployment

#### Edge Device Setup

```bash
#!/bin/bash
# Edge device deployment script (Raspberry Pi, Jetson, etc.)

# 1. Detect device type
DEVICE_TYPE=$(cat /proc/device-tree/model 2>/dev/null || echo "unknown")

# 2. Install edge-specific dependencies
if [[ $DEVICE_TYPE == *"Raspberry Pi"* ]]; then
    # Raspberry Pi setup
    sudo apt-get install -y libraspberrypi-dev
    pip3 install picamera2
elif [[ $DEVICE_TYPE == *"Jetson"* ]]; then
    # NVIDIA Jetson setup
    sudo apt-get install -y nvidia-jetpack
    pip3 install jetson-stats
fi

# 3. Install Edge TPU runtime (if available)
if lsusb | grep -q "Google Inc."; then
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
        sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update
    sudo apt-get install -y libedgetpu1-std
fi

# 4. Configure edge agent
cat > /etc/ooblex/edge-config.yaml <<EOF
edge:
  device_id: $(hostname)-$(date +%s)
  capabilities:
    - face_detection
    - emotion_recognition
    - object_detection
  hardware:
    accelerator: $(detect_accelerator)
    memory_limit: 2GB
  cloud:
    sync_url: https://cloud.ooblex.com/edge/sync
    sync_interval: 300
    offline_mode: true
  models:
    storage_path: /var/lib/ooblex/models
    max_cache_size: 1GB
EOF

# 5. Setup edge services
sudo systemctl enable ooblex-edge-agent
sudo systemctl start ooblex-edge-agent
```

#### Edge Orchestration

```python
# edge_orchestrator.py
import asyncio
from typing import List, Dict
import yaml

class EdgeOrchestrator:
    def __init__(self, config_path: str):
        self.config = yaml.safe_load(open(config_path))
        self.edge_nodes = {}
        self.model_registry = {}
    
    async def register_edge_node(self, node_info: Dict):
        """Register a new edge node"""
        node_id = node_info['device_id']
        self.edge_nodes[node_id] = {
            'info': node_info,
            'status': 'online',
            'last_seen': asyncio.get_event_loop().time(),
            'workload': 0,
            'models': set()
        }
        
        # Deploy appropriate models
        await self.deploy_models_to_node(node_id)
    
    async def deploy_models_to_node(self, node_id: str):
        """Deploy models based on node capabilities"""
        node = self.edge_nodes[node_id]
        capabilities = node['info']['capabilities']
        
        for capability in capabilities:
            if capability in self.model_registry:
                model = self.model_registry[capability]
                await self.deploy_model(node_id, model)
    
    async def deploy_model(self, node_id: str, model: Dict):
        """Deploy a model to an edge node"""
        # Optimize model for edge device
        optimized_model = await self.optimize_for_edge(
            model, 
            self.edge_nodes[node_id]['info']['hardware']
        )
        
        # Transfer model to edge
        await self.transfer_model(node_id, optimized_model)
        
        # Update node status
        self.edge_nodes[node_id]['models'].add(model['name'])
    
    def select_best_node(self, task: str) -> str:
        """Select best edge node for a task"""
        eligible_nodes = [
            node_id for node_id, node in self.edge_nodes.items()
            if task in node['info']['capabilities'] 
            and node['status'] == 'online'
        ]
        
        if not eligible_nodes:
            raise Exception(f"No edge nodes available for task: {task}")
        
        # Select node with lowest workload
        return min(eligible_nodes, key=lambda n: self.edge_nodes[n]['workload'])

# Usage
orchestrator = EdgeOrchestrator('/etc/ooblex/edge-orchestrator.yaml')
await orchestrator.start()
```

### High Availability Deployment

#### Multi-Region Setup

```yaml
# terraform/multi-region.tf
provider "aws" {
  alias  = "us-east-1"
  region = "us-east-1"
}

provider "aws" {
  alias  = "eu-west-1"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "ap-southeast-1"
  region = "ap-southeast-1"
}

# Global Route53 hosted zone
resource "aws_route53_zone" "main" {
  name = "ooblex.com"
}

# Regional deployments
module "us_east_deployment" {
  source = "./modules/regional-deployment"
  providers = {
    aws = aws.us-east-1
  }
  
  region = "us-east-1"
  vpc_cidr = "10.1.0.0/16"
  cluster_name = "ooblex-us-east"
  node_count = 5
  instance_type = "c5.2xlarge"
}

module "eu_west_deployment" {
  source = "./modules/regional-deployment"
  providers = {
    aws = aws.eu-west-1
  }
  
  region = "eu-west-1"
  vpc_cidr = "10.2.0.0/16"
  cluster_name = "ooblex-eu-west"
  node_count = 3
  instance_type = "c5.2xlarge"
}

# Global load balancing
resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.ooblex.com"
  type    = "A"
  
  set_identifier = "us-east-1"
  geolocation_routing_policy {
    continent = "NA"
  }
  
  alias {
    name                   = module.us_east_deployment.alb_dns_name
    zone_id                = module.us_east_deployment.alb_zone_id
    evaluate_target_health = true
  }
}

# Cross-region replication
resource "aws_s3_bucket" "media_storage" {
  bucket = "ooblex-media-storage"
  
  replication_configuration {
    role = aws_iam_role.replication.arn
    
    rules {
      id     = "replicate-all"
      status = "Enabled"
      
      destination {
        bucket        = aws_s3_bucket.media_storage_replica.arn
        storage_class = "GLACIER_IR"
      }
    }
  }
}
```

## Configuration Options

### Environment Configuration

```bash
# .env.production
# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
API_WORKERS=4
API_CORS_ORIGINS=https://app.ooblex.com,https://dashboard.ooblex.com

# Database Configuration
DATABASE_URL=postgresql://ooblex:password@postgres:5432/ooblex
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=40

# Redis Configuration
REDIS_URL=redis://redis:6379/0
REDIS_POOL_SIZE=10

# WebRTC Configuration
JANUS_HTTP_URL=http://janus:8088/janus
JANUS_WS_URL=ws://janus:8188
TURN_SERVER=turn:turn.ooblex.com:3478
TURN_USERNAME=ooblex
TURN_PASSWORD=secure-password

# AI Configuration
AI_BACKEND_URL=grpc://brain:50051
AI_MODEL_PATH=/models
AI_CACHE_SIZE=4GB
AI_DEVICE=gpu

# Storage Configuration
MEDIA_STORAGE_PATH=/var/lib/ooblex/media
MEDIA_STORAGE_TYPE=s3
S3_BUCKET=ooblex-media
S3_REGION=us-east-1
S3_ACCESS_KEY=AKIAXXXXXXXXXXXXXXXX
S3_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Security Configuration
JWT_SECRET_KEY=your-256-bit-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=60
ENCRYPTION_KEY=your-32-byte-encryption-key

# Monitoring Configuration
PROMETHEUS_ENABLED=true
PROMETHEUS_PORT=9090
GRAFANA_ENABLED=true
GRAFANA_PORT=3000

# Feature Flags
ENABLE_EDGE_COMPUTING=true
ENABLE_BLOCKCHAIN=true
ENABLE_WHIP_WHEP=true
ENABLE_VDO_NINJA=true
```

### Nginx Configuration

```nginx
# nginx/ooblex.conf
upstream api_backend {
    least_conn;
    server api1:8080 weight=1 max_fails=3 fail_timeout=30s;
    server api2:8080 weight=1 max_fails=3 fail_timeout=30s;
    server api3:8080 weight=1 max_fails=3 fail_timeout=30s;
}

upstream websocket_backend {
    ip_hash;
    server janus1:8188;
    server janus2:8188;
    server janus3:8188;
}

server {
    listen 80;
    server_name ooblex.com www.ooblex.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ooblex.com www.ooblex.com;
    
    ssl_certificate /etc/nginx/ssl/ooblex.crt;
    ssl_certificate_key /etc/nginx/ssl/ooblex.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # API endpoints
    location /api/ {
        proxy_pass http://api_backend/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    # WebSocket endpoints
    location /ws/ {
        proxy_pass http://websocket_backend/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
    
    # WHIP endpoint
    location /whip/ {
        proxy_pass http://api_backend/whip/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # CORS for WHIP
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods "POST, GET, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Content-Type, Authorization" always;
        
        if ($request_method = OPTIONS) {
            return 204;
        }
    }
    
    # WHEP endpoint
    location /whep/ {
        proxy_pass http://api_backend/whep/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # CORS for WHEP
        add_header Access-Control-Allow-Origin * always;
        add_header Access-Control-Allow-Methods "POST, GET, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Content-Type, Authorization" always;
        
        if ($request_method = OPTIONS) {
            return 204;
        }
    }
    
    # Static files
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
        
        # Cache static assets
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
    
    # Health checks
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

## Best Practices

### Security Hardening

```bash
#!/bin/bash
# Security hardening script

# 1. System updates
sudo apt-get update && sudo apt-get upgrade -y

# 2. Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 10000:60000/udp
sudo ufw --force enable

# 3. Fail2ban configuration
sudo apt-get install -y fail2ban
cat > /etc/fail2ban/jail.local <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[sshd]
enabled = true

[nginx-http-auth]
enabled = true

[nginx-noscript]
enabled = true
EOF
sudo systemctl restart fail2ban

# 4. SSL/TLS configuration
sudo certbot certonly --nginx -d ooblex.com -d www.ooblex.com

# 5. Secure kernel parameters
cat >> /etc/sysctl.conf <<EOF
# IP Spoofing protection
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Ignore Directed pings
net.ipv4.icmp_ignore_bogus_error_responses = 1

# Enable TCP/IP SYN cookies
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Increase system file descriptor limit
fs.file-max = 65535
EOF
sudo sysctl -p

# 6. Application security
# Enable secret scanning
pip install detect-secrets
detect-secrets scan --baseline .secrets.baseline

# 7. Database security
sudo -u postgres psql <<EOF
ALTER USER ooblex_user WITH ENCRYPTED PASSWORD 'strong_password';
REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO ooblex_user;
EOF
```

### Performance Tuning

```bash
#!/bin/bash
# Performance tuning script

# 1. CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 2. Network optimization
cat >> /etc/sysctl.conf <<EOF
# Network performance tuning
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr
net.core.default_qdisc = fq
EOF
sudo sysctl -p

# 3. Redis optimization
cat >> /etc/redis/redis.conf <<EOF
maxmemory 4gb
maxmemory-policy allkeys-lru
save ""
EOF

# 4. PostgreSQL tuning
cat >> /etc/postgresql/14/main/postgresql.conf <<EOF
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 10MB
min_wal_size = 2GB
max_wal_size = 8GB
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4
EOF

# 5. Application optimization
export PYTHONOPTIMIZE=2
export OMP_NUM_THREADS=4
export TF_ENABLE_ONEDNN_OPTS=1
```

### Monitoring Setup

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ooblex-api'
    static_configs:
      - targets: ['api1:9090', 'api2:9090', 'api3:9090']
    
  - job_name: 'ooblex-brain'
    static_configs:
      - targets: ['brain1:9090', 'brain2:9090']
    
  - job_name: 'janus-gateway'
    static_configs:
      - targets: ['janus1:7088', 'janus2:7088', 'janus3:7088']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node1:9100', 'node2:9100', 'node3:9100']
    
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres:9187']
    
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts/*.yml'
```

### Backup and Recovery

```bash
#!/bin/bash
# Backup script

BACKUP_DIR="/backup/ooblex/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 1. Database backup
pg_dump -h localhost -U ooblex_user ooblex | gzip > $BACKUP_DIR/postgres.sql.gz

# 2. Redis backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb $BACKUP_DIR/

# 3. Media files backup
rsync -av /var/lib/ooblex/media/ $BACKUP_DIR/media/

# 4. Configuration backup
tar -czf $BACKUP_DIR/config.tar.gz /etc/ooblex/ /etc/nginx/sites-available/ooblex.conf

# 5. Model backup
tar -czf $BACKUP_DIR/models.tar.gz /var/lib/ooblex/models/

# 6. Upload to S3
aws s3 sync $BACKUP_DIR s3://ooblex-backups/$(date +%Y%m%d)/

# 7. Cleanup old backups
find /backup/ooblex/ -type d -mtime +30 -exec rm -rf {} \;
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
```bash
# Check service status
sudo systemctl status ooblex-api

# Check logs
sudo journalctl -u ooblex-api -f

# Verify dependencies
sudo systemctl status postgresql redis nginx

# Check port availability
sudo netstat -tlnp | grep -E '8080|8188|5432|6379'
```

2. **WebRTC Connection Issues**
```bash
# Test STUN/TURN
turnutils_stunclient -p 3478 turn.ooblex.com

# Check firewall rules
sudo ufw status verbose

# Verify NAT configuration
curl -s https://api.ipify.org

# Test WebSocket connection
wscat -c wss://ooblex.com/ws/
```

3. **Performance Issues**
```bash
# Check system resources
htop
iostat -x 1
vmstat 1

# Database performance
sudo -u postgres psql -c "SELECT * FROM pg_stat_activity;"

# Redis performance
redis-cli --latency

# Network performance
iperf3 -s  # On server
iperf3 -c server_ip  # On client
```

### Disaster Recovery

```bash
#!/bin/bash
# Disaster recovery script

# 1. Stop all services
for service in api brain decoder webrtc janus; do
    sudo systemctl stop ooblex-$service
done

# 2. Restore database
gunzip < /backup/postgres.sql.gz | psql -h localhost -U ooblex_user ooblex

# 3. Restore Redis
sudo systemctl stop redis
cp /backup/dump.rdb /var/lib/redis/
sudo chown redis:redis /var/lib/redis/dump.rdb
sudo systemctl start redis

# 4. Restore media files
rsync -av /backup/media/ /var/lib/ooblex/media/

# 5. Restore configuration
tar -xzf /backup/config.tar.gz -C /

# 6. Start services
for service in api brain decoder webrtc janus; do
    sudo systemctl start ooblex-$service
done

# 7. Verify recovery
curl http://localhost:8080/health
```

### Support Resources

- Documentation: https://docs.ooblex.com
- Community Forum: https://forum.ooblex.com
- GitHub Issues: https://github.com/ooblex/ooblex/issues
- Enterprise Support: support@ooblex.com