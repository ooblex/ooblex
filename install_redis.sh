#!/bin/bash
set -e

# Redis 7.x Installation Script
# Supports Ubuntu 20.04+ and container environments

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Configuration
REDIS_VERSION="7.2.3"
REDIS_PORT="6379"
REDIS_PASSWORD=""  # Set to empty for no password
REDIS_DATA_DIR="/var/lib/redis"
REDIS_LOG_DIR="/var/log/redis"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    print_error "Cannot detect OS version"
    exit 1
fi

print_status "Installing Redis $REDIS_VERSION on $OS $VER"

# Install dependencies
print_status "Installing dependencies..."
apt-get update -y
apt-get install -y \
    build-essential \
    tcl \
    wget \
    make \
    gcc \
    libc6-dev \
    libssl-dev \
    pkg-config

# Create redis user and group
print_status "Creating redis user and group..."
if ! id -u redis >/dev/null 2>&1; then
    groupadd -r redis
    useradd -r -g redis -s /bin/false -d /var/lib/redis -c "Redis Database Server" redis
fi

# Download and extract Redis
print_status "Downloading Redis $REDIS_VERSION..."
cd /tmp
wget -q http://download.redis.io/releases/redis-$REDIS_VERSION.tar.gz
tar xzf redis-$REDIS_VERSION.tar.gz
cd redis-$REDIS_VERSION

# Build Redis
print_status "Building Redis..."
make -j$(nproc)
make BUILD_TLS=yes -j$(nproc)

# Run tests (optional, can be skipped for faster installation)
print_status "Running Redis tests (this may take a while)..."
make test || print_warning "Some tests failed, but continuing installation"

# Install Redis
print_status "Installing Redis..."
make install PREFIX=/usr/local/bin

# Create directories
print_status "Creating Redis directories..."
mkdir -p $REDIS_DATA_DIR
mkdir -p $REDIS_LOG_DIR
mkdir -p /etc/redis
mkdir -p /var/run/redis

# Set permissions
chown redis:redis $REDIS_DATA_DIR
chown redis:redis $REDIS_LOG_DIR
chown redis:redis /var/run/redis

# Generate random password if not set
if [ -z "$REDIS_PASSWORD" ]; then
    REDIS_PASSWORD=$(openssl rand -base64 32)
    print_status "Generated Redis password: $REDIS_PASSWORD"
    print_warning "Please save this password securely!"
fi

# Create Redis configuration file
print_status "Creating Redis configuration..."
cat > /etc/redis/redis.conf << EOF
# Redis 7.x Configuration File

# Network
bind 127.0.0.1 ::1
protected-mode yes
port $REDIS_PORT
tcp-backlog 511
timeout 0
tcp-keepalive 300

# General
daemonize no
supervised systemd
pidfile /var/run/redis/redis-server.pid
loglevel notice
logfile $REDIS_LOG_DIR/redis-server.log
databases 16
always-show-logo no

# Snapshotting
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir $REDIS_DATA_DIR

# Replication
replica-serve-stale-data yes
replica-read-only yes
repl-diskless-sync yes
repl-diskless-sync-delay 5
repl-diskless-sync-max-replicas 0
repl-diskless-load disabled
repl-ping-replica-period 10
repl-timeout 60
repl-disable-tcp-nodelay no
repl-backlog-size 1mb
repl-backlog-ttl 3600

# Security
requirepass $REDIS_PASSWORD
# ACL configuration
aclfile /etc/redis/users.acl

# Clients
maxclients 10000

# Memory Management
maxmemory 0
maxmemory-policy noeviction
maxmemory-samples 5

# Lazy Freeing
lazyfree-lazy-eviction no
lazyfree-lazy-expire no
lazyfree-lazy-server-del no
replica-lazy-flush no
lazyfree-lazy-user-del no
lazyfree-lazy-user-flush no

# Persistence
appendonly yes
appendfilename "appendonly.aof"
appenddirname "appendonlydir"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes
aof-timestamp-enabled no

# Modules
# loadmodule /path/to/module.so

# Slow Log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Event notification
notify-keyspace-events ""

# Advanced config
hash-max-listpack-entries 512
hash-max-listpack-value 64
list-max-listpack-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-listpack-entries 128
zset-max-listpack-value 64
hll-sparse-max-bytes 3000
stream-node-max-bytes 4096
stream-node-max-entries 100
activerehashing yes
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
hz 10
dynamic-hz yes
aof-rewrite-incremental-fsync yes
rdb-save-incremental-fsync yes

# CPU affinity
# server-cpulist 0-3
# bio-cpulist 4,5
# aof-rewrite-cpulist 6,7
# bgsave-cpulist 8,9

# Disable some dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command KEYS ""
rename-command CONFIG ""
EOF

# Create ACL file for user management
print_status "Creating ACL configuration..."
cat > /etc/redis/users.acl << EOF
# Redis ACL Users Configuration
# Format: user <username> [... rules ...]

# Default user with full access (using the password from redis.conf)
user default on ~* &* +@all

# Example read-only user (uncomment and modify as needed)
# user readonly on ~* &* +@read -@dangerous
EOF

chown redis:redis /etc/redis/redis.conf
chown redis:redis /etc/redis/users.acl
chmod 640 /etc/redis/redis.conf
chmod 640 /etc/redis/users.acl

# Create systemd service file
print_status "Creating systemd service..."
cat > /etc/systemd/system/redis.service << EOF
[Unit]
Description=Redis In-Memory Data Store
After=network.target

[Service]
Type=notify
ExecStart=/usr/local/bin/redis-server /etc/redis/redis.conf
ExecStop=/usr/local/bin/redis-cli shutdown
TimeoutStopSec=0
Restart=always
User=redis
Group=redis
RuntimeDirectory=redis
RuntimeDirectoryMode=0755

# Hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$REDIS_DATA_DIR $REDIS_LOG_DIR /var/run/redis
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true
RestrictRealtime=true
RestrictNamespaces=true
RestrictSUIDSGID=true
PrivateDevices=true
ProtectHostname=true
ProtectClock=true
ProtectKernelLogs=true
RestrictAddressFamilies=AF_INET AF_INET6 AF_UNIX
SystemCallFilter=@system-service
SystemCallErrorNumber=EPERM

[Install]
WantedBy=multi-user.target
EOF

# Create Redis CLI wrapper with authentication
print_status "Creating Redis CLI wrapper..."
cat > /usr/local/bin/redis-cli-auth << EOF
#!/bin/bash
# Redis CLI with automatic authentication
exec /usr/local/bin/redis-cli -a "$REDIS_PASSWORD" "\$@"
EOF
chmod +x /usr/local/bin/redis-cli-auth

# Enable and start Redis
print_status "Enabling and starting Redis service..."
systemctl daemon-reload
systemctl enable redis
systemctl start redis

# Wait for Redis to start
sleep 2

# Install Python Redis client
print_status "Installing Python Redis client..."
apt-get install -y python3-pip
pip3 install redis redis-py-cluster

# Create benchmark script
print_status "Creating benchmark script..."
cat > /usr/local/bin/redis-benchmark-test << 'EOF'
#!/bin/bash
# Redis benchmark test

echo "Running Redis benchmark..."
PASS=$(grep "^requirepass" /etc/redis/redis.conf | awk '{print $2}')
/usr/local/bin/redis-benchmark -a "$PASS" -q -n 10000 -c 50 -P 12
EOF
chmod +x /usr/local/bin/redis-benchmark-test

# Create monitoring script
print_status "Creating monitoring script..."
cat > /usr/local/bin/redis-monitor << 'EOF'
#!/bin/bash
# Redis monitoring script

PASS=$(grep "^requirepass" /etc/redis/redis.conf | awk '{print $2}')
watch -n 1 "/usr/local/bin/redis-cli -a '$PASS' INFO | grep -E 'used_memory_human|connected_clients|instantaneous_ops_per_sec|keyspace_'"
EOF
chmod +x /usr/local/bin/redis-monitor

# Verify installation
print_status "Verifying Redis installation..."

# Check if Redis is running
if systemctl is-active --quiet redis; then
    print_status "Redis service is running"
else
    print_error "Redis service is not running"
    exit 1
fi

# Test Redis connection
if /usr/local/bin/redis-cli -a "$REDIS_PASSWORD" ping | grep -q PONG; then
    print_status "Redis is responding to commands"
else
    print_error "Redis is not responding"
    exit 1
fi

# Get Redis info
REDIS_INFO=$(/usr/local/bin/redis-cli -a "$REDIS_PASSWORD" INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')
print_status "Redis $REDIS_INFO installed successfully!"

# Display helpful information
print_status "Installation complete!"
echo ""
echo "Redis Configuration:"
echo "  - Version: $REDIS_INFO"
echo "  - Port: $REDIS_PORT"
echo "  - Data directory: $REDIS_DATA_DIR"
echo "  - Log directory: $REDIS_LOG_DIR"
echo "  - Config file: /etc/redis/redis.conf"
echo "  - Password: $REDIS_PASSWORD"
echo ""
echo "Service Management:"
echo "  - Start: systemctl start redis"
echo "  - Stop: systemctl stop redis"
echo "  - Restart: systemctl restart redis"
echo "  - Status: systemctl status redis"
echo ""
echo "Redis CLI:"
echo "  - With auth: redis-cli-auth"
echo "  - Manual: redis-cli -a '$REDIS_PASSWORD'"
echo ""
echo "Monitoring and Testing:"
echo "  - Monitor: redis-monitor"
echo "  - Benchmark: redis-benchmark-test"
echo "  - Logs: journalctl -u redis -f"
echo ""
echo "Python Usage:"
echo "  import redis"
echo "  r = redis.Redis(host='localhost', port=$REDIS_PORT, password='$REDIS_PASSWORD', decode_responses=True)"
echo ""
echo "Security Notes:"
echo "  - Password protected: YES"
echo "  - Dangerous commands disabled: FLUSHDB, FLUSHALL, KEYS, CONFIG"
echo "  - Bound to: localhost only"
echo "  - Protected mode: ON"

# Clean up
print_status "Cleaning up..."
rm -rf /tmp/redis-$REDIS_VERSION*

print_status "Redis $REDIS_VERSION installation completed successfully!"