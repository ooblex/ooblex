#!/bin/bash
# GStreamer Installation Script for Ooblex
# Updated for modern GStreamer 1.24.x and container-friendly installation

set -e

# GStreamer version
VERSION=${GSTREAMER_VERSION:-1.24.0}

echo "Installing GStreamer $VERSION for Ooblex..."

# Update package lists
sudo apt-get update

# Install build dependencies and runtime libraries
sudo apt-get install -y \
    build-essential \
    meson \
    ninja-build \
    pkg-config \
    python3 \
    python3-pip \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-good1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-nice \
    gstreamer1.0-vaapi \
    gstreamer1.0-rtsp \
    libsrtp2-dev \
    libnice-dev \
    libwebrtc-audio-processing-dev \
    libopus-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    libde265-dev \
    libva-dev \
    libva-drm2 \
    libva-x11-2 \
    vainfo

# Install Python bindings
pip3 install --user pygobject

# For WebRTC support
sudo apt-get install -y \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-nice \
    libnice10 \
    libnice-dev

# For hardware acceleration (optional)
if lspci | grep -i nvidia > /dev/null; then
    echo "NVIDIA GPU detected, installing NVENC support..."
    sudo apt-get install -y gstreamer1.0-plugins-bad
fi

# Verify installation
echo "Verifying GStreamer installation..."
gst-inspect-1.0 --version

# Test WebRTC elements
echo "Checking WebRTC support..."
gst-inspect-1.0 webrtcbin || echo "Warning: webrtcbin not found"
gst-inspect-1.0 rtpbin || echo "Warning: rtpbin not found"

# Increase network buffer sizes for better streaming performance
echo "Optimizing network settings..."
sudo sysctl -w net.core.rmem_max=33554432
sudo sysctl -w net.core.wmem_max=33554432
sudo sysctl -w net.core.rmem_default=8388608
sudo sysctl -w net.core.wmem_default=8388608

# Make settings persistent
echo "net.core.rmem_max=33554432" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_max=33554432" | sudo tee -a /etc/sysctl.conf
echo "net.core.rmem_default=8388608" | sudo tee -a /etc/sysctl.conf
echo "net.core.wmem_default=8388608" | sudo tee -a /etc/sysctl.conf

echo "GStreamer installation completed successfully!"
echo "You can now use GStreamer with Ooblex for WebRTC streaming."