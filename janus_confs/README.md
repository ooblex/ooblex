# Janus WebRTC Gateway Configuration

This directory contains configuration files for Janus WebRTC Gateway, which handles WebRTC connections in Ooblex.

## 🚀 Modern Deployment

For production deployments, Janus is configured automatically via Docker:

```bash
# Janus runs as part of the stack
docker-compose up -d janus
```

## Manual Configuration (Development Only)

If you're running Janus manually:

### 1. Copy Configuration Files

```bash
# Copy to Janus config directory
cd ~/ooblex/janus_confs
sudo cp *.cfg /opt/janus/etc/janus/
```

### 2. Update RabbitMQ Settings

Edit RabbitMQ connection settings in:
- `janus.eventhandler.rabbitmqevh.cfg`
- `janus.transport.rabbitmq.cfg`

Update to match your RabbitMQ server:
```ini
host = rabbitmq         # or your RabbitMQ host
port = 5672
username = guest        # change in production
password = guest        # change in production
vhost = /
```

### 3. Configure SSL Certificates

Update SSL certificate paths in the following files:

```bash
# Files that need SSL updates:
janus.cfg                    # Lines 67-68
janus.transport.http.cfg     # Lines 64-65
janus.transport.websockets.cfg # Lines 37-38
```

#### Using Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt install certbot

# Generate certificate
sudo certbot certonly --standalone -d your-domain.com

# Update configs to use the certificate
sudo find /opt/janus/etc/janus -type f -exec sed -i \
  's|/etc/letsencrypt/live/api.ooblex.com|/etc/letsencrypt/live/your-domain.com|g' {} \;
```

#### Using Self-Signed Certificates (Development)

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Move to appropriate location
sudo mkdir -p /opt/janus/certs
sudo mv cert.pem key.pem /opt/janus/certs/

# Update configs
sudo find /opt/janus/etc/janus -type f -exec sed -i \
  's|cert_pem = .*|cert_pem = /opt/janus/certs/cert.pem|g' {} \;
sudo find /opt/janus/etc/janus -type f -exec sed -i \
  's|cert_key = .*|cert_key = /opt/janus/certs/key.pem|g' {} \;
```

### 4. Plugin Configuration

#### VideoRoom Plugin (`janus.plugin.videoroom.cfg`)
- Configured for multi-party video conferencing
- Default room: 1234
- Supports up to 6 publishers per room
- VP8/VP9/H.264 codec support

#### Streaming Plugin (`janus.plugin.streaming.cfg`)
- Configured for one-to-many streaming
- RTP/RTSP input support
- Multiple mountpoints for different streams

### 5. Start Janus

```bash
# Start with logging
sudo /opt/janus/bin/janus -o

# Or use systemd service
sudo systemctl start janus
sudo journalctl -u janus -f
```

## Docker Configuration

When using Docker, environment variables configure Janus:

```yaml
# docker-compose.yml
janus:
  environment:
    - RABBITMQ_HOST=rabbitmq
    - RABBITMQ_PORT=5672
    - ADMIN_KEY=${JANUS_ADMIN_KEY}
    - STUN_SERVER=stun:stun.l.google.com:19302
    - TURN_SERVER=turn:your-turn-server.com:3478
    - TURN_USER=username
    - TURN_PASS=password
```

## Configuration Files

| File | Purpose |
|------|---------|
| `janus.cfg` | Main configuration |
| `janus.plugin.videoroom.cfg` | Multi-party video rooms |
| `janus.plugin.streaming.cfg` | One-to-many streaming |
| `janus.transport.http.cfg` | REST API configuration |
| `janus.transport.websockets.cfg` | WebSocket transport |
| `janus.transport.rabbitmq.cfg` | RabbitMQ integration |
| `janus.eventhandler.rabbitmqevh.cfg` | Event handling via RabbitMQ |

## Modern Features

The configuration supports:
- WHIP/WHEP protocols (via custom plugin)
- VDO.Ninja integration
- Scalable room management
- Recording capabilities
- Event-driven architecture via RabbitMQ

## Troubleshooting

1. **SSL Errors**: Ensure certificate paths are correct and files are readable
2. **RabbitMQ Connection**: Verify RabbitMQ is running and credentials are correct
3. **Port Conflicts**: Default ports are 8088 (HTTP), 8089 (HTTPS), 8188 (WS), 8989 (Admin)

For production deployments, always use the Docker-based setup for easier management and better security.