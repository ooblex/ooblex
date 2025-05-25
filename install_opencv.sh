#!/bin/bash
set -e

# OpenCV 4.x Installation Script with Python 3 Support
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
OPENCV_VERSION="4.8.1"
OPENCV_CONTRIB_VERSION="4.8.1"
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
BUILD_DIR="/tmp/opencv_build"

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

print_status "Installing OpenCV $OPENCV_VERSION on $OS $VER"

# Update system packages
print_status "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install build dependencies
print_status "Installing build tools and dependencies..."
apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    wget \
    unzip \
    yasm \
    checkinstall

# Install image and video libraries
print_status "Installing image and video libraries..."
apt-get install -y \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libv4l-dev \
    v4l-utils \
    libxvidcore-dev \
    libx264-dev \
    libx265-dev \
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libvorbis-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev

# Install GUI libraries (optional, can be skipped for headless)
print_status "Installing GUI libraries..."
apt-get install -y \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module

# Install optimization libraries
print_status "Installing optimization libraries..."
apt-get install -y \
    libatlas-base-dev \
    liblapacke-dev \
    gfortran \
    libeigen3-dev

# Install Python development packages
print_status "Installing Python development packages..."
apt-get install -y \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-matplotlib

# Update pip and install Python packages
print_status "Updating pip and installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy matplotlib scikit-image scikit-learn

# Install additional useful libraries
print_status "Installing additional libraries..."
apt-get install -y \
    libtbb-dev \
    libdc1394-22-dev \
    libopenblas-dev \
    liblapack-dev \
    libjasper-dev \
    libwebp-dev \
    libopenexr-dev \
    libgdal-dev \
    libgphoto2-dev \
    libhdf5-dev \
    libprotobuf-dev \
    protobuf-compiler

# Create build directory
print_status "Creating build directory..."
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Download OpenCV
print_status "Downloading OpenCV $OPENCV_VERSION..."
wget -q -O opencv.zip https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip
wget -q -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV_CONTRIB_VERSION.zip

# Extract archives
print_status "Extracting archives..."
unzip -q opencv.zip
unzip -q opencv_contrib.zip
rm opencv.zip opencv_contrib.zip

# Create build directory
cd opencv-$OPENCV_VERSION
mkdir -p build && cd build

# Configure build with CMake
print_status "Configuring OpenCV build..."
cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-$OPENCV_CONTRIB_VERSION/modules \
    -D WITH_CUDA=OFF \
    -D WITH_CUDNN=OFF \
    -D WITH_CUBLAS=OFF \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_OPENCL=ON \
    -D WITH_IPP=ON \
    -D WITH_LAPACK=ON \
    -D WITH_EIGEN=ON \
    -D WITH_OPENMP=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GSTREAMER=ON \
    -D BUILD_TIFF=ON \
    -D BUILD_PNG=ON \
    -D BUILD_JPEG=ON \
    -D BUILD_JASPER=ON \
    -D BUILD_WEBP=ON \
    -D BUILD_OPENEXR=ON \
    -D BUILD_EXAMPLES=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D BUILD_opencv_java=OFF \
    -D BUILD_NEW_PYTHON_SUPPORT=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_PC_FILE_NAME=opencv4.pc \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python$PYTHON_VERSION \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    -D INSTALL_C_EXAMPLES=ON \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    ..

# Build OpenCV
print_status "Building OpenCV (this may take a while)..."
make -j$(nproc)

# Install OpenCV
print_status "Installing OpenCV..."
make install
ldconfig

# Create pkg-config file if not exists
if [ ! -f /usr/local/lib/pkgconfig/opencv4.pc ]; then
    print_status "Creating pkg-config file..."
    mkdir -p /usr/local/lib/pkgconfig
    cat > /usr/local/lib/pkgconfig/opencv4.pc << EOF
prefix=/usr/local
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include/opencv4

Name: OpenCV
Description: Open Source Computer Vision Library
Version: $OPENCV_VERSION
Libs: -L\${libdir} -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_dnn -lopencv_ml -lopencv_flann -lopencv_photo -lopencv_stitching -lopencv_gapi
Cflags: -I\${includedir}
EOF
fi

# Update library cache
print_status "Updating library cache..."
echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf
ldconfig

# Create Python symlink if needed
PYTHON_CV2_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])")
if [ -f "/usr/local/lib/python$PYTHON_VERSION/site-packages/cv2/python-$PYTHON_VERSION/cv2.cpython-*.so" ]; then
    print_status "Creating Python cv2 symlink..."
    ln -sf /usr/local/lib/python$PYTHON_VERSION/site-packages/cv2/python-$PYTHON_VERSION/cv2.cpython-*.so \
        $PYTHON_CV2_PATH/cv2.so
fi

# Verify installation
print_status "Verifying OpenCV installation..."

# Check C++ installation
if pkg-config --exists opencv4; then
    CV_VERSION=$(pkg-config --modversion opencv4)
    print_status "OpenCV $CV_VERSION C++ libraries installed successfully!"
else
    print_error "OpenCV C++ installation verification failed"
    exit 1
fi

# Check Python installation
python3 -c "import cv2; print(f'OpenCV Python: {cv2.__version__}')" 2>/dev/null
if [ $? -eq 0 ]; then
    print_status "OpenCV Python bindings installed successfully!"
else
    print_warning "OpenCV Python bindings may not be properly installed"
fi

# Create test program
print_status "Creating test program..."
cat > /tmp/test_opencv.cpp << 'EOF'
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    cv::Mat img = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::putText(img, "OpenCV", cv::Point(10, 50), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
EOF

# Compile and run test
print_status "Testing OpenCV installation..."
g++ /tmp/test_opencv.cpp -o /tmp/test_opencv `pkg-config --cflags --libs opencv4`
if /tmp/test_opencv; then
    print_status "OpenCV test passed!"
else
    print_error "OpenCV test failed"
fi

# Clean up
print_status "Cleaning up..."
rm -rf $BUILD_DIR
rm -f /tmp/test_opencv*

# Display information
print_status "Installation complete!"
echo ""
echo "OpenCV Configuration:"
echo "  - Version: $OPENCV_VERSION"
echo "  - Install prefix: /usr/local"
echo "  - Python version: $PYTHON_VERSION"
echo "  - pkg-config file: /usr/local/lib/pkgconfig/opencv4.pc"
echo ""
echo "Usage Examples:"
echo "  - C++: g++ program.cpp \`pkg-config --cflags --libs opencv4\`"
echo "  - Python: import cv2"
echo ""
echo "Test your installation:"
echo "  - C++: pkg-config --modversion opencv4"
echo "  - Python: python3 -c 'import cv2; print(cv2.__version__)'"

print_status "OpenCV $OPENCV_VERSION installation completed successfully!"