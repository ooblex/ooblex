#!/bin/bash
set -e

# NGINX Installation Script with WebRTC modules
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

print_status "Installing NGINX with WebRTC modules on $OS $VER"

# Update package list
print_status "Updating package list..."
apt-get update -y

# Install dependencies
print_status "Installing dependencies..."
apt-get install -y \
    build-essential \
    libpcre3 \
    libpcre3-dev \
    zlib1g \
    zlib1g-dev \
    libssl-dev \
    libgd-dev \
    libxml2 \
    libxml2-dev \
    uuid-dev \
    wget \
    git

# Create nginx user if it doesn't exist
if ! id -u nginx >/dev/null 2>&1; then
    print_status "Creating nginx user..."
    useradd -r -s /bin/false nginx
fi

# Download and compile NGINX with rtmp module
NGINX_VERSION="1.24.0"
NGINX_RTMP_VERSION="1.2.2"

print_status "Downloading NGINX $NGINX_VERSION..."
cd /tmp
wget -q http://nginx.org/download/nginx-${NGINX_VERSION}.tar.gz
tar -xzf nginx-${NGINX_VERSION}.tar.gz

print_status "Downloading nginx-rtmp-module..."
git clone --depth 1 -b v${NGINX_RTMP_VERSION} https://github.com/arut/nginx-rtmp-module.git

print_status "Configuring NGINX build..."
cd nginx-${NGINX_VERSION}
./configure \
    --prefix=/etc/nginx \
    --sbin-path=/usr/sbin/nginx \
    --modules-path=/usr/lib/nginx/modules \
    --conf-path=/etc/nginx/nginx.conf \
    --error-log-path=/var/log/nginx/error.log \
    --http-log-path=/var/log/nginx/access.log \
    --pid-path=/var/run/nginx.pid \
    --lock-path=/var/run/nginx.lock \
    --http-client-body-temp-path=/var/cache/nginx/client_temp \
    --http-proxy-temp-path=/var/cache/nginx/proxy_temp \
    --http-fastcgi-temp-path=/var/cache/nginx/fastcgi_temp \
    --http-uwsgi-temp-path=/var/cache/nginx/uwsgi_temp \
    --http-scgi-temp-path=/var/cache/nginx/scgi_temp \
    --user=nginx \
    --group=nginx \
    --with-compat \
    --with-file-aio \
    --with-threads \
    --with-http_addition_module \
    --with-http_auth_request_module \
    --with-http_dav_module \
    --with-http_flv_module \
    --with-http_gunzip_module \
    --with-http_gzip_static_module \
    --with-http_mp4_module \
    --with-http_random_index_module \
    --with-http_realip_module \
    --with-http_secure_link_module \
    --with-http_slice_module \
    --with-http_ssl_module \
    --with-http_stub_status_module \
    --with-http_sub_module \
    --with-http_v2_module \
    --with-mail \
    --with-mail_ssl_module \
    --with-stream \
    --with-stream_realip_module \
    --with-stream_ssl_module \
    --with-stream_ssl_preread_module \
    --add-module=../nginx-rtmp-module

print_status "Compiling NGINX..."
make -j$(nproc)
make install

# Create necessary directories
print_status "Creating directories..."
mkdir -p /var/cache/nginx/{client_temp,proxy_temp,fastcgi_temp,uwsgi_temp,scgi_temp}
mkdir -p /var/log/nginx
mkdir -p /etc/nginx/conf.d

# Create systemd service file
print_status "Creating systemd service..."
cat > /etc/systemd/system/nginx.service << 'EOF'
[Unit]
Description=The NGINX HTTP and reverse proxy server
After=syslog.target network-online.target remote-fs.target nss-lookup.target
Wants=network-online.target

[Service]
Type=forking
PIDFile=/var/run/nginx.pid
ExecStartPre=/usr/sbin/nginx -t
ExecStart=/usr/sbin/nginx
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
PrivateTmp=true

[Install]
WantedBy=multi-user.target
EOF

# Create basic nginx.conf with WebRTC-friendly settings
print_status "Creating NGINX configuration..."
cat > /etc/nginx/nginx.conf << 'EOF'
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;

    # Performance optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/json application/javascript;

    # WebSocket support
    map $http_upgrade $connection_upgrade {
        default upgrade;
        '' close;
    }

    # Include additional configurations
    include /etc/nginx/conf.d/*.conf;
}

# RTMP configuration for WebRTC
rtmp {
    server {
        listen 1935;
        chunk_size 4096;

        application live {
            live on;
            record off;
        }
    }
}
EOF

# Create a default server configuration
cat > /etc/nginx/conf.d/default.conf << 'EOF'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;
    root /var/www/html;
    index index.html index.htm;

    location / {
        try_files $uri $uri/ =404;
    }

    # WebSocket proxy example
    location /ws {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Create web root directory
mkdir -p /var/www/html

# Enable and start NGINX
print_status "Enabling and starting NGINX service..."
systemctl daemon-reload
systemctl enable nginx
systemctl start nginx

# Install Certbot for SSL certificates
print_status "Installing Certbot for SSL certificates..."
if [ "$VER" == "20.04" ] || [ "$VER" == "22.04" ]; then
    apt-get install -y snapd
    snap install core
    snap refresh core
    snap install --classic certbot
    ln -sf /snap/bin/certbot /usr/bin/certbot
else
    apt-get install -y certbot python3-certbot-nginx
fi

# Verify installation
print_status "Verifying NGINX installation..."
if nginx -v 2>&1 | grep -q "nginx version"; then
    NGINX_VER=$(nginx -v 2>&1 | awk -F'/' '{print $2}')
    print_status "NGINX $NGINX_VER installed successfully!"
else
    print_error "NGINX installation verification failed"
    exit 1
fi

# Check if NGINX is running
if systemctl is-active --quiet nginx; then
    print_status "NGINX is running"
else
    print_error "NGINX is not running"
    exit 1
fi

# Display helpful information
print_status "Installation complete!"
echo ""
echo "NGINX Configuration:"
echo "  - Config directory: /etc/nginx"
echo "  - Log directory: /var/log/nginx"
echo "  - Web root: /var/www/html"
echo "  - RTMP port: 1935"
echo ""
echo "SSL Certificate Setup:"
echo "  - For standalone certificate: certbot certonly --standalone -d yourdomain.com"
echo "  - For NGINX integration: certbot --nginx -d yourdomain.com"
echo ""
echo "Service Management:"
echo "  - Start: systemctl start nginx"
echo "  - Stop: systemctl stop nginx"
echo "  - Restart: systemctl restart nginx"
echo "  - Status: systemctl status nginx"

# Cleanup
print_status "Cleaning up temporary files..."
rm -rf /tmp/nginx-${NGINX_VERSION}*
rm -rf /tmp/nginx-rtmp-module

print_status "NGINX installation with WebRTC support completed successfully!"