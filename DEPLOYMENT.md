# Ooblex Production Deployment Guide

This guide covers deploying Ooblex to production environments with best practices for security, scalability, and reliability.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Security Checklist](#security-checklist)
- [Deployment Options](#deployment-options)
  - [AWS Deployment](#aws-deployment)
  - [Google Cloud Platform](#google-cloud-platform)
  - [Azure Deployment](#azure-deployment)
  - [Kubernetes](#kubernetes-deployment)
  - [Bare Metal](#bare-metal-deployment)
- [SSL/TLS Configuration](#ssltls-configuration)
- [Monitoring and Logging](#monitoring-and-logging)
- [Scaling Strategies](#scaling-strategies)
- [Backup and Recovery](#backup-and-recovery)
- [Performance Tuning](#performance-tuning)

---

## Prerequisites

Before deploying to production:

1. **Domain Name** - You need a domain with DNS control
2. **SSL Certificates** - Required for WebRTC (uses Let's Encrypt or cloud provider)
3. **Cloud Account** - AWS, GCP, Azure, or bare metal server
4. **Resource Requirements:**
   - Minimum: 2 vCPUs, 4GB RAM, 20GB storage
   - Recommended: 4 vCPUs, 8GB RAM, 50GB storage
   - With GPU: Tesla T4 or better for ML models

---

## Security Checklist

⚠️ **CRITICAL**: Complete these before going live:

### 1. Environment Variables

```bash
# Generate secure secrets
openssl rand -hex 32  # For JWT_SECRET
openssl rand -hex 32  # For API keys

# Never commit .env to git!
cp .env.example .env
nano .env
```

Edit `.env`:
```bash
# Change ALL default passwords
RABBITMQ_USER=your_secure_username
RABBITMQ_PASS=your_secure_password_here

# Strong JWT secret
JWT_SECRET=your_64_char_random_hex_here

# Enable security features
ENABLE_RATE_LIMITING=true
ENABLE_METRICS=true
DEBUG=false
TESTING=false
```

### 2. Firewall Rules

Only expose necessary ports:
```bash
# Allow HTTPS (443)
ufw allow 443/tcp

# Allow SSH (change default port!)
ufw allow 2222/tcp

# Deny all other incoming
ufw default deny incoming
ufw enable
```

### 3. Update Dependencies

```bash
# Check for vulnerabilities
pip install safety
safety check --json

# Update dependencies
pip install --upgrade -r requirements.txt
```

### 4. SSL/TLS Only

- **NEVER** run production without HTTPS
- WebRTC **requires** HTTPS for getUserMedia()
- Use Let's Encrypt (free) or cloud provider certificates

---

## Deployment Options

### AWS Deployment

#### Option A: EC2 with Docker Compose

**1. Launch EC2 Instance**

```bash
# Choose instance type
# - t3.medium (2 vCPU, 4GB RAM) - minimal
# - c5.xlarge (4 vCPU, 8GB RAM) - recommended
# - g4dn.xlarge (4 vCPU, 16GB RAM, T4 GPU) - with AI models

# Ubuntu 22.04 LTS AMI
# Security group: Allow 443, 80, 22 (from your IP only)
```

**2. Setup Script**

```bash
#!/bin/bash
# AWS EC2 setup script

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone https://github.com/ooblex/ooblex.git
cd ooblex

# Configure environment
cp .env.example .env
nano .env  # Edit configuration

# Get SSL certificates
sudo apt install certbot python3-certbot-nginx -y
sudo certbot certonly --standalone -d yourdomain.com

# Start services
docker-compose up -d

# Setup automatic renewal
sudo crontab -e
# Add: 0 0 * * * certbot renew --quiet --deploy-hook "docker-compose restart nginx"
```

**3. Configure Load Balancer (for high traffic)**

```bash
# Create Application Load Balancer
aws elbv2 create-load-balancer \
  --name ooblex-lb \
  --subnets subnet-xxx subnet-yyy \
  --security-groups sg-xxx

# Create target group
aws elbv2 create-target-group \
  --name ooblex-targets \
  --protocol HTTPS \
  --port 443 \
  --vpc-id vpc-xxx \
  --health-check-path /health
```

#### Option B: ECS (Fargate)

See `deploy/aws/ecs-task-definition.json` for configuration.

```bash
# Build and push images
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin xxx.dkr.ecr.us-east-1.amazonaws.com
docker build -t ooblex-api -f Dockerfile.simple .
docker tag ooblex-api:latest xxx.dkr.ecr.us-east-1.amazonaws.com/ooblex-api:latest
docker push xxx.dkr.ecr.us-east-1.amazonaws.com/ooblex-api:latest

# Deploy to ECS
aws ecs create-service \
  --cluster ooblex-cluster \
  --service-name ooblex-api \
  --task-definition ooblex-api:1 \
  --desired-count 2 \
  --launch-type FARGATE
```

---

### Google Cloud Platform

#### Option A: Compute Engine with Docker

```bash
# Create VM instance
gcloud compute instances create ooblex-prod \
  --zone=us-central1-a \
  --machine-type=n1-standard-2 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --tags=http-server,https-server

# SSH into instance
gcloud compute ssh ooblex-prod --zone=us-central1-a

# Run setup script (same as AWS above)
```

#### Option B: Cloud Run (serverless)

```bash
# Build container
gcloud builds submit --tag gcr.io/your-project/ooblex-api

# Deploy to Cloud Run
gcloud run deploy ooblex-api \
  --image gcr.io/your-project/ooblex-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 1 \
  --max-instances 10 \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars REDIS_URL=redis://your-redis:6379
```

**Note:** Cloud Run works for stateless services. For full Ooblex stack, use Compute Engine or GKE.

#### Option C: Google Kubernetes Engine

See [Kubernetes Deployment](#kubernetes-deployment) section below.

---

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name ooblex-rg --location eastus

# Create container registry
az acr create --resource-group ooblex-rg --name ooblexregistry --sku Basic

# Build and push image
az acr build --registry ooblexregistry --image ooblex-api:v1 -f Dockerfile.simple .

# Deploy container
az container create \
  --resource-group ooblex-rg \
  --name ooblex-api \
  --image ooblexregistry.azurecr.io/ooblex-api:v1 \
  --cpu 2 \
  --memory 4 \
  --dns-name-label ooblex-prod \
  --ports 443 8800 \
  --environment-variables \
    REDIS_URL=redis://your-redis:6379 \
    RABBITMQ_URL=amqp://user:pass@your-rabbitmq:5672
```

#### Azure Kubernetes Service (AKS)

```bash
# Create AKS cluster
az aks create \
  --resource-group ooblex-rg \
  --name ooblex-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group ooblex-rg --name ooblex-cluster

# Deploy (see Kubernetes section)
kubectl apply -f deploy/kubernetes/
```

---

### Kubernetes Deployment

**Directory structure:**
```
deploy/kubernetes/
├── namespace.yaml
├── redis.yaml
├── rabbitmq.yaml
├── api-deployment.yaml
├── worker-deployment.yaml
├── ingress.yaml
└── hpa.yaml
```

**1. Create namespace:**

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ooblex
```

**2. Deploy Redis:**

```yaml
# redis.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: ooblex
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        persistentVolumeClaim:
          claimName: redis-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: ooblex
spec:
  ports:
  - port: 6379
  selector:
    app: redis
```

**3. Deploy ML Workers:**

```yaml
# worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ooblex-worker
  namespace: ooblex
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ooblex-worker
  template:
    metadata:
      labels:
        app: ooblex-worker
    spec:
      containers:
      - name: worker
        image: your-registry/ooblex-worker:latest
        env:
        - name: REDIS_URL
          value: redis://redis:6379
        - name: RABBITMQ_URL
          valueFrom:
            secretKeyRef:
              name: ooblex-secrets
              key: rabbitmq-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

**4. Horizontal Pod Autoscaler:**

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ooblex-worker-hpa
  namespace: ooblex
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ooblex-worker
  minReplicas: 2
  maxReplicas: 20
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

**5. Deploy:**

```bash
kubectl apply -f deploy/kubernetes/

# Check status
kubectl get pods -n ooblex
kubectl get services -n ooblex

# Scale manually
kubectl scale deployment ooblex-worker --replicas=10 -n ooblex
```

---

### Bare Metal Deployment

For dedicated servers or on-premises:

```bash
# 1. System Requirements
# - Ubuntu 22.04 LTS or RHEL 9
# - 8GB+ RAM
# - 4+ CPU cores
# - 100GB SSD

# 2. Install dependencies
sudo apt update
sudo apt install -y \
  python3.11 \
  python3-pip \
  redis-server \
  rabbitmq-server \
  nginx \
  certbot \
  python3-certbot-nginx

# 3. Clone and setup
git clone https://github.com/ooblex/ooblex.git /opt/ooblex
cd /opt/ooblex

# 4. Virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Configure systemd services
sudo cp deploy/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ooblex-api ooblex-worker ooblex-mjpeg
sudo systemctl start ooblex-api ooblex-worker ooblex-mjpeg

# 6. Setup NGINX reverse proxy
sudo cp deploy/nginx/ooblex.conf /etc/nginx/sites-available/
sudo ln -s /etc/nginx/sites-available/ooblex.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# 7. Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# 8. Setup automatic restarts
sudo crontab -e
# Add: @reboot cd /opt/ooblex && ./deploy/start-services.sh
```

---

## SSL/TLS Configuration

### Let's Encrypt (Recommended for most deployments)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate (standalone mode)
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Or with nginx plugin
sudo certbot --nginx -d yourdomain.com

# Certificates are stored in:
# /etc/letsencrypt/live/yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/yourdomain.com/privkey.pem

# Auto-renewal (runs twice daily)
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer
```

### NGINX SSL Configuration

```nginx
# /etc/nginx/sites-available/ooblex.conf
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL certificates
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # SSL security settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;

    # WebSocket proxy
    location /ws {
        proxy_pass http://localhost:8800;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }

    # API proxy
    location /api {
        proxy_pass http://localhost:8800;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Static files
    location / {
        root /opt/ooblex/html;
        try_files $uri $uri/ =404;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

---

## Monitoring and Logging

### Prometheus + Grafana

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=your_secure_password
      - GF_SERVER_ROOT_URL=https://metrics.yourdomain.com
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    command:
      - '--path.rootfs=/host'
    volumes:
      - '/:/host:ro,rslave'

volumes:
  prometheus_data:
  grafana_data:
```

### Application Logging

Use structured logging:

```python
import structlog

logger = structlog.get_logger()

# Log important events
logger.info("frame_processed", 
    stream_key=stream_key,
    effect=effect_name,
    latency_ms=latency,
    frame_size_kb=frame_size
)
```

Ship logs to cloud:

```bash
# CloudWatch (AWS)
pip install watchtower
aws logs create-log-group --log-group-name /ooblex/production

# Cloud Logging (GCP)
pip install google-cloud-logging

# Azure Monitor (Azure)
pip install opencensus-ext-azure
```

---

## Scaling Strategies

### Horizontal Scaling (Add More Workers)

```bash
# Docker Compose
docker-compose up -d --scale ml-worker=10

# Kubernetes
kubectl scale deployment ooblex-worker --replicas=20 -n ooblex

# Auto-scaling based on queue depth
# Monitor RabbitMQ queue size and scale workers dynamically
```

### Vertical Scaling (Bigger Instances)

- Start with t3.medium (2 vCPU, 4GB RAM)
- Scale to c5.2xlarge (8 vCPU, 16GB RAM) for more throughput
- Use GPU instances (g4dn.xlarge) for AI models

### Geographic Distribution

```bash
# Deploy to multiple regions
# - US East (us-east-1)
# - EU West (eu-west-1)
# - Asia Pacific (ap-southeast-1)

# Use Route53 / Cloud DNS for geo-routing
# Shared Redis/RabbitMQ or region-specific
```

---

## Backup and Recovery

### Redis Backups

```bash
# Enable AOF persistence
redis-cli CONFIG SET appendonly yes

# Manual backup
redis-cli BGSAVE

# Automated backups to S3
0 2 * * * /opt/ooblex/scripts/backup-redis-to-s3.sh
```

### RabbitMQ Backups

```bash
# Export definitions
rabbitmqctl export_definitions /backup/rabbitmq-defs.json

# Backup to S3
aws s3 cp /backup/rabbitmq-defs.json s3://your-backup-bucket/
```

### Database Backups (if using API persistence)

```bash
# PostgreSQL backup
pg_dump ooblex > /backup/ooblex-$(date +%Y%m%d).sql

# Automated backups
0 1 * * * /opt/ooblex/scripts/backup-db.sh
```

---

## Performance Tuning

### Redis Optimization

```bash
# /etc/redis/redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
save ""  # Disable RDB if using AOF
appendonly yes
appendfsync everysec
```

### RabbitMQ Optimization

```bash
# /etc/rabbitmq/rabbitmq.conf
vm_memory_high_watermark.relative = 0.6
disk_free_limit.absolute = 10GB
channel_max = 2048
```

### Kernel Tuning (Linux)

```bash
# /etc/sysctl.conf
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 8192
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_fin_timeout = 30
vm.swappiness = 10

# Apply
sudo sysctl -p
```

### NGINX Worker Processes

```nginx
# /etc/nginx/nginx.conf
worker_processes auto;
worker_rlimit_nofile 65535;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}
```

---

## Troubleshooting

### High Latency

1. Check Redis latency: `redis-cli --latency`
2. Check network latency: `ping your-server`
3. Monitor CPU usage: `htop`
4. Check worker queue depth: `rabbitmqctl list_queues`

### Out of Memory

```bash
# Check memory usage
free -h
docker stats

# Increase Redis maxmemory
redis-cli CONFIG SET maxmemory 8gb

# Reduce frame retention
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

### Connection Issues

```bash
# Check service health
systemctl status redis rabbitmq-server nginx

# Check ports
netstat -tulpn | grep LISTEN

# Check logs
journalctl -u ooblex-api -f
tail -f /var/log/nginx/error.log
```

---

## Support

For production support:
- GitHub Issues: https://github.com/ooblex/ooblex/issues
- Documentation: https://github.com/ooblex/ooblex
- Security Issues: security@ooblex.io (please do not open public issues)

---

**Next Steps:**
1. Complete [Security Checklist](#security-checklist)
2. Choose [Deployment Option](#deployment-options)
3. Configure [SSL/TLS](#ssltls-configuration)
4. Setup [Monitoring](#monitoring-and-logging)
5. Test with demo.py and real traffic
6. Monitor for 24-48 hours before going fully live
