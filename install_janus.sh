#!/bin/bash
# Janus WebRTC Gateway Installation Script for Ooblex
# Updated for Janus 1.2.x with modern dependencies

set -e

echo "Installing Janus WebRTC Gateway for Ooblex..."

# Update system
sudo apt-get update

# Install base dependencies
sudo apt-get install -y \
    libmicrohttpd-dev \
    libjansson-dev \
    libssl-dev \
    libsofia-sip-ua-dev \
    libglib2.0-dev \
    libopus-dev \
    libogg-dev \
    libcurl4-openssl-dev \
    liblua5.3-dev \
    libconfig-dev \
    pkg-config \
    gengetopt \
    libtool \
    automake \
    build-essential \
    cmake \
    git \
    gtk-doc-tools \
    libgnutls28-dev

# Install libnice (latest version for better WebRTC support)
echo "Installing libnice..."
cd /tmp
git clone https://gitlab.freedesktop.org/libnice/libnice
cd libnice
git checkout 0.1.21  # Use stable version
meson builddir
ninja -C builddir
sudo ninja -C builddir install
sudo ldconfig

# Install libsrtp 2.5
echo "Installing libsrtp..."
cd /tmp
wget https://github.com/cisco/libsrtp/archive/v2.5.0.tar.gz
tar -xzf v2.5.0.tar.gz
cd libsrtp-2.5.0
./configure --prefix=/usr --enable-openssl
make shared_library
sudo make install
sudo ldconfig

# Install usrsctp for data channels
echo "Installing usrsctp..."
cd /tmp
git clone https://github.com/sctplab/usrsctp
cd usrsctp
git checkout 0.9.5.0
./bootstrap
./configure --prefix=/usr --disable-programs --disable-inet --disable-inet6
make
sudo make install
sudo ldconfig

# Install libwebsockets 4.3
echo "Installing libwebsockets..."
cd /tmp
git clone https://github.com/warmcat/libwebsockets
cd libwebsockets
git checkout v4.3-stable
mkdir build
cd build
cmake .. -DLWS_WITH_STATIC=OFF -DLWS_WITHOUT_TESTAPPS=ON -DCMAKE_INSTALL_PREFIX=/usr
make
sudo make install
sudo ldconfig

# Install RabbitMQ-C
echo "Installing RabbitMQ-C..."
cd /tmp
git clone https://github.com/alanxz/rabbitmq-c
cd rabbitmq-c
git checkout v0.13.0
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr
make
sudo make install
sudo ldconfig

# Install Janus
echo "Installing Janus Gateway..."
cd /tmp
git clone https://github.com/meetecho/janus-gateway.git
cd janus-gateway
git checkout v1.2.1  # Use stable version
./autogen.sh
./configure \
    --prefix=/opt/janus \
    --enable-websockets \
    --enable-post-processing \
    --enable-json-logger \
    --enable-data-channels \
    --enable-rabbitmq \
    --enable-rest \
    --enable-unix-sockets \
    --enable-all-js-modules

make
sudo make install
sudo make configs

# Create directories and set permissions
sudo mkdir -p /opt/janus/etc/janus
sudo mkdir -p /opt/janus/share/janus/recordings
sudo mkdir -p /var/log/janus

# Create systemd service file
sudo tee /etc/systemd/system/janus.service > /dev/null <<EOF
[Unit]
Description=Janus WebRTC Gateway
After=network.target

[Service]
Type=simple
ExecStart=/opt/janus/bin/janus
Restart=always
RestartSec=5
User=janus
Group=janus
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

# Create janus user
sudo useradd -r -s /bin/false janus || true
sudo chown -R janus:janus /opt/janus
sudo chown -R janus:janus /var/log/janus

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable janus

echo "Janus WebRTC Gateway installation completed!"
echo "Configuration files are in: /opt/janus/etc/janus/"
echo "Start with: sudo systemctl start janus"
echo "View logs: sudo journalctl -u janus -f"