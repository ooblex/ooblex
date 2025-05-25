#!/bin/bash
set -e

# TensorFlow 2.x Installation Script with GPU Support
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
PYTHON_VERSION="3"
TF_VERSION="2.15.0"  # Latest stable version
CUDA_VERSION="12.2"
CUDNN_VERSION="8.9"

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

print_status "Installing TensorFlow $TF_VERSION on $OS $VER"

# Detect GPU
HAS_GPU=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_GPU=true
        print_status "NVIDIA GPU detected"
    fi
fi

# Update system
print_status "Updating system packages..."
apt-get update -y
apt-get upgrade -y

# Install Python and development tools
print_status "Installing Python and development tools..."
apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-venv \
    build-essential \
    pkg-config \
    git \
    wget \
    curl \
    unzip \
    software-properties-common

# Install system dependencies
print_status "Installing system dependencies..."
apt-get install -y \
    libhdf5-dev \
    libc-ares-dev \
    libeigen3-dev \
    gcc \
    gfortran \
    g++ \
    libgfortran5 \
    libatlas-base-dev \
    libopenblas-dev \
    liblapack-dev \
    cython3 \
    libfreetype6-dev \
    libpng-dev \
    pkg-config

# Update pip
print_status "Updating pip..."
python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel

# Install Python scientific packages
print_status "Installing Python scientific packages..."
python${PYTHON_VERSION} -m pip install --upgrade \
    numpy \
    scipy \
    pandas \
    matplotlib \
    scikit-learn \
    pillow \
    h5py

# GPU Support Installation
if [ "$HAS_GPU" = true ]; then
    print_status "Installing NVIDIA GPU support..."
    
    # Add NVIDIA package repositories
    print_status "Adding NVIDIA repositories..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo $VER | tr -d '.')/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    apt-get update
    
    # Install CUDA Toolkit
    print_status "Installing CUDA Toolkit $CUDA_VERSION..."
    apt-get install -y cuda-toolkit-${CUDA_VERSION//./-}
    
    # Install cuDNN
    print_status "Installing cuDNN $CUDNN_VERSION..."
    apt-get install -y libcudnn8=${CUDNN_VERSION}.*-1+cuda${CUDA_VERSION}
    apt-get install -y libcudnn8-dev=${CUDNN_VERSION}.*-1+cuda${CUDA_VERSION}
    
    # Install TensorRT (optional)
    print_status "Installing TensorRT..."
    apt-get install -y libnvinfer8 libnvinfer-dev libnvinfer-plugin8
    
    # Set up environment variables
    print_status "Setting up CUDA environment variables..."
    cat >> /etc/profile.d/cuda.sh << EOF
export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-${CUDA_VERSION}
EOF
    
    # Source the environment
    source /etc/profile.d/cuda.sh
    
    # Install TensorFlow with GPU support
    print_status "Installing TensorFlow with GPU support..."
    python${PYTHON_VERSION} -m pip install tensorflow[and-cuda]==$TF_VERSION
    
else
    # Install CPU-only TensorFlow
    print_status "Installing TensorFlow (CPU only)..."
    python${PYTHON_VERSION} -m pip install tensorflow-cpu==$TF_VERSION
fi

# Install additional TensorFlow packages
print_status "Installing additional TensorFlow packages..."
python${PYTHON_VERSION} -m pip install \
    tensorflow-hub \
    tensorflow-datasets \
    tensorboard \
    tf-slim \
    tensorflow-probability \
    tensorflow-addons \
    tensorflow-text

# Install Keras (now integrated with TensorFlow)
print_status "Installing Keras and related packages..."
python${PYTHON_VERSION} -m pip install \
    keras \
    keras-preprocessing \
    keras-applications \
    keras-tuner

# Install additional ML/DL tools
print_status "Installing additional ML/DL tools..."
python${PYTHON_VERSION} -m pip install \
    jupyter \
    jupyterlab \
    ipython \
    notebook \
    opencv-python \
    albumentations \
    tqdm \
    seaborn \
    plotly \
    tensorboard-plugin-profile

# Create test script
print_status "Creating TensorFlow test script..."
cat > /tmp/test_tensorflow.py << 'EOF'
#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import sys

print("TensorFlow Test Script")
print("=" * 50)

# Version information
print(f"TensorFlow version: {tf.__version__}")
print(f"Python version: {sys.version}")

# Check GPU availability
print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")
if tf.config.list_physical_devices('GPU'):
    print("GPU Details:")
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"  - {gpu}")

# Test basic operations
print("\nRunning basic operations test...")
try:
    # Create tensors
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    
    # Perform operations
    c = tf.matmul(a, b)
    print(f"Matrix multiplication result:\n{c.numpy()}")
    
    # Test neural network layers
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Generate dummy data
    x = np.random.random((100, 5))
    y = np.random.randint(2, size=(100, 1))
    
    # Compile and train
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x, y, epochs=2, verbose=0)
    
    print(f"\nNeural network test completed successfully!")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    
except Exception as e:
    print(f"Error during test: {e}")
    sys.exit(1)

print("\nAll tests passed!")
EOF

chmod +x /tmp/test_tensorflow.py

# Create Jupyter service file
print_status "Creating Jupyter service file..."
cat > /etc/systemd/system/jupyter.service << EOF
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/usr/local/bin/jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create TensorBoard service file
print_status "Creating TensorBoard service file..."
cat > /etc/systemd/system/tensorboard.service << EOF
[Unit]
Description=TensorBoard Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/usr/local/bin/tensorboard --logdir=/var/log/tensorboard --bind_all
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Create log directory for TensorBoard
mkdir -p /var/log/tensorboard

# Run test
print_status "Running TensorFlow test..."
python${PYTHON_VERSION} /tmp/test_tensorflow.py

# Verify installation
print_status "Verifying TensorFlow installation..."
TF_VERSION_INSTALLED=$(python${PYTHON_VERSION} -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    print_status "TensorFlow $TF_VERSION_INSTALLED installed successfully!"
else
    print_error "TensorFlow installation verification failed"
    exit 1
fi

# Display helpful information
print_status "Installation complete!"
echo ""
echo "TensorFlow Configuration:"
echo "  - Version: $TF_VERSION_INSTALLED"
echo "  - Python: python${PYTHON_VERSION}"
if [ "$HAS_GPU" = true ]; then
    echo "  - GPU Support: Enabled"
    echo "  - CUDA Version: $CUDA_VERSION"
    echo "  - cuDNN Version: $CUDNN_VERSION"
else
    echo "  - GPU Support: Disabled (CPU only)"
fi
echo ""
echo "Installed Packages:"
echo "  - TensorFlow Core"
echo "  - TensorBoard"
echo "  - Keras"
echo "  - Jupyter Notebook/Lab"
echo "  - NumPy, SciPy, Pandas, Matplotlib"
echo "  - OpenCV, Scikit-learn"
echo ""
echo "Services:"
echo "  - Jupyter: systemctl start jupyter"
echo "  - TensorBoard: systemctl start tensorboard"
echo ""
echo "Usage Examples:"
echo "  - Python: python${PYTHON_VERSION}"
echo "  - IPython: ipython${PYTHON_VERSION}"
echo "  - Jupyter: jupyter notebook"
echo "  - TensorBoard: tensorboard --logdir=./logs"
echo ""
echo "Test Installation:"
echo "  python${PYTHON_VERSION} -c 'import tensorflow as tf; print(tf.__version__)'"
echo "  python${PYTHON_VERSION} -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo ""
echo "Environment Setup (GPU only):"
echo "  source /etc/profile.d/cuda.sh"

# Cleanup
print_status "Cleaning up..."
rm -f /tmp/test_tensorflow.py

print_status "TensorFlow $TF_VERSION installation completed successfully!"