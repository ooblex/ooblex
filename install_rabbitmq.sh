#!/bin/bash
set -e

# RabbitMQ 3.12.x Installation Script
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
RABBITMQ_VERSION="3.12.10"
ERLANG_VERSION="25.3"

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

print_status "Installing RabbitMQ $RABBITMQ_VERSION on $OS $VER"

# Install prerequisites
print_status "Installing prerequisites..."
apt-get update -y
apt-get install -y \
    curl \
    gnupg \
    apt-transport-https \
    software-properties-common \
    lsb-release

# Add RabbitMQ and Erlang repositories
print_status "Adding RabbitMQ and Erlang repositories..."

# TeamRabbitMQ's main signing key
curl -1sLf "https://keys.openpgp.org/vks/v1/by-fingerprint/0A9AF2115F4687BD29803A206B73A36E6026DFCA" | apt-key add -

# Cloudsmith: modern Erlang repository
curl -1sLf https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-erlang/gpg.E495BB49CC4BBE5B.key | apt-key add -

# Cloudsmith: RabbitMQ repository
curl -1sLf https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/gpg.9F4587F226208342.key | apt-key add -

# Add apt repositories
CODENAME=$(lsb_release -sc)

# Modern Erlang
echo "deb https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-erlang/deb/ubuntu $CODENAME main" > /etc/apt/sources.list.d/rabbitmq-erlang.list
echo "deb-src https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-erlang/deb/ubuntu $CODENAME main" >> /etc/apt/sources.list.d/rabbitmq-erlang.list

# RabbitMQ
echo "deb https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/deb/ubuntu $CODENAME main" > /etc/apt/sources.list.d/rabbitmq.list
echo "deb-src https://dl.cloudsmith.io/public/rabbitmq/rabbitmq-server/deb/ubuntu $CODENAME main" >> /etc/apt/sources.list.d/rabbitmq.list

# Update package index
print_status "Updating package index..."
apt-get update -y

# Install Erlang
print_status "Installing Erlang..."
apt-get install -y erlang-base \
    erlang-asn1 \
    erlang-crypto \
    erlang-eldap \
    erlang-ftp \
    erlang-inets \
    erlang-mnesia \
    erlang-os-mon \
    erlang-parsetools \
    erlang-public-key \
    erlang-runtime-tools \
    erlang-snmp \
    erlang-ssl \
    erlang-syntax-tools \
    erlang-tftp \
    erlang-tools \
    erlang-xmerl

# Install RabbitMQ server
print_status "Installing RabbitMQ server..."
apt-get install -y rabbitmq-server

# Enable RabbitMQ service
print_status "Enabling RabbitMQ service..."
systemctl enable rabbitmq-server

# Start RabbitMQ service
print_status "Starting RabbitMQ service..."
systemctl start rabbitmq-server

# Wait for RabbitMQ to fully start
print_status "Waiting for RabbitMQ to start..."
sleep 5

# Enable management plugin
print_status "Enabling RabbitMQ management plugin..."
rabbitmq-plugins enable rabbitmq_management

# Enable other useful plugins
print_status "Enabling additional RabbitMQ plugins..."
rabbitmq-plugins enable rabbitmq_management_agent
rabbitmq-plugins enable rabbitmq_prometheus
rabbitmq-plugins enable rabbitmq_shovel
rabbitmq-plugins enable rabbitmq_shovel_management

# Create admin user (optional)
print_status "Creating admin user..."
ADMIN_USER="admin"
ADMIN_PASS="admin123"

# Check if user already exists
if rabbitmqctl list_users | grep -q "^$ADMIN_USER"; then
    print_warning "Admin user already exists, skipping creation"
else
    rabbitmqctl add_user $ADMIN_USER $ADMIN_PASS
    rabbitmqctl set_user_tags $ADMIN_USER administrator
    rabbitmqctl set_permissions -p / $ADMIN_USER ".*" ".*" ".*"
    print_status "Admin user created: $ADMIN_USER (password: $ADMIN_PASS)"
    print_warning "Please change the default password!"
fi

# Configure RabbitMQ for production use
print_status "Configuring RabbitMQ for production..."
cat > /etc/rabbitmq/rabbitmq.conf << EOF
# RabbitMQ Configuration
# Network
listeners.tcp.default = 5672
management.tcp.port = 15672

# Memory and disk alarms
vm_memory_high_watermark.relative = 0.6
disk_free_limit.absolute = 5GB

# Message TTL
# message_ttl = 3600000

# Connection limits
# channel_max = 128
# connection_max = infinity

# Logging
log.file.level = info
log.console = true
log.console.level = info

# Clustering
# cluster_formation.peer_discovery_backend = rabbit_peer_discovery_classic_config
# cluster_formation.classic_config.nodes.1 = rabbit@hostname1
# cluster_formation.classic_config.nodes.2 = rabbit@hostname2
EOF

# Set up environment variables
print_status "Setting up environment variables..."
cat > /etc/rabbitmq/rabbitmq-env.conf << EOF
# RabbitMQ Environment Variables
NODENAME=rabbit@localhost
NODE_IP_ADDRESS=0.0.0.0
NODE_PORT=5672

# Erlang VM settings
RABBITMQ_SERVER_ERL_ARGS="+K true +A128 +P 1048576"

# Memory settings (optional)
# RABBITMQ_VM_MEMORY_HIGH_WATERMARK=0.6
EOF

# Create systemd override for container compatibility
print_status "Creating systemd override for container compatibility..."
mkdir -p /etc/systemd/system/rabbitmq-server.service.d
cat > /etc/systemd/system/rabbitmq-server.service.d/override.conf << EOF
[Service]
# Container-friendly settings
PrivateDevices=false
PrivateTmp=false
ProtectSystem=false
ProtectHome=false
NoNewPrivileges=false
EOF

# Reload systemd and restart RabbitMQ
systemctl daemon-reload
systemctl restart rabbitmq-server

# Create monitoring script
print_status "Creating monitoring script..."
cat > /usr/local/bin/rabbitmq-health-check << 'EOF'
#!/bin/bash
# RabbitMQ health check script

STATUS=$(rabbitmqctl status 2>&1)
if echo "$STATUS" | grep -q "Status of node rabbit@"; then
    echo "RabbitMQ is healthy"
    exit 0
else
    echo "RabbitMQ is not healthy"
    exit 1
fi
EOF
chmod +x /usr/local/bin/rabbitmq-health-check

# Verify installation
print_status "Verifying RabbitMQ installation..."

# Check if RabbitMQ is running
if systemctl is-active --quiet rabbitmq-server; then
    print_status "RabbitMQ service is running"
else
    print_error "RabbitMQ service is not running"
    exit 1
fi

# Check RabbitMQ status
if rabbitmqctl status > /dev/null 2>&1; then
    print_status "RabbitMQ is responding to commands"
else
    print_error "RabbitMQ is not responding"
    exit 1
fi

# Check management plugin
if curl -s -o /dev/null -w "%{http_code}" http://localhost:15672 | grep -q "200\|301"; then
    print_status "RabbitMQ Management UI is accessible"
else
    print_warning "RabbitMQ Management UI may not be accessible"
fi

# Display version info
print_status "Installed versions:"
echo "  - RabbitMQ: $(rabbitmqctl version)"
echo "  - Erlang: $(erl -eval 'erlang:display(erlang:system_info(otp_release)), halt().' -noshell)"

# Display helpful information
print_status "Installation complete!"
echo ""
echo "RabbitMQ Configuration:"
echo "  - AMQP Port: 5672"
echo "  - Management UI: http://localhost:15672"
echo "  - Default User: guest/guest (localhost only)"
echo "  - Admin User: $ADMIN_USER/$ADMIN_PASS"
echo "  - Config File: /etc/rabbitmq/rabbitmq.conf"
echo "  - Environment: /etc/rabbitmq/rabbitmq-env.conf"
echo ""
echo "Service Management:"
echo "  - Start: systemctl start rabbitmq-server"
echo "  - Stop: systemctl stop rabbitmq-server"
echo "  - Restart: systemctl restart rabbitmq-server"
echo "  - Status: systemctl status rabbitmq-server"
echo ""
echo "Useful Commands:"
echo "  - List users: rabbitmqctl list_users"
echo "  - List queues: rabbitmqctl list_queues"
echo "  - List exchanges: rabbitmqctl list_exchanges"
echo "  - Health check: /usr/local/bin/rabbitmq-health-check"
echo ""
echo "Clustering:"
echo "  - Join cluster: rabbitmqctl join_cluster rabbit@hostname"
echo "  - Cluster status: rabbitmqctl cluster_status"

print_status "RabbitMQ $RABBITMQ_VERSION installation completed successfully!"