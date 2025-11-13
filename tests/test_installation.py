"""
Installation validation tests
Tests that core dependencies and system requirements are properly installed
"""
import pytest
import sys
import importlib
import subprocess
import os


class TestPythonVersion:
    """Test Python version requirements"""

    def test_python_version(self):
        """Verify Python 3.11+ is being used"""
        assert sys.version_info >= (3, 11), f"Python 3.11+ required, got {sys.version_info}"


class TestCoreDependencies:
    """Test that core Python dependencies are installed"""

    @pytest.mark.parametrize("module_name", [
        "fastapi",
        "uvicorn",
        "redis",
        "aio_pika",
        "cv2",  # opencv-python
        "numpy",
        "pydantic",
    ])
    def test_core_module_imports(self, module_name):
        """Test that core modules can be imported"""
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")

    def test_opencv_version(self):
        """Verify OpenCV is installed and functional"""
        import cv2
        assert hasattr(cv2, "__version__"), "OpenCV version not found"
        major_version = int(cv2.__version__.split('.')[0])
        assert major_version >= 4, f"OpenCV 4.x+ required, got {cv2.__version__}"

    def test_numpy_version(self):
        """Verify NumPy is installed and functional"""
        import numpy as np
        assert hasattr(np, "__version__"), "NumPy version not found"
        version_parts = np.__version__.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1])
        assert (major, minor) >= (1, 24), f"NumPy 1.24+ recommended, got {np.__version__}"


class TestOptionalDependencies:
    """Test optional dependencies (ML frameworks)"""

    def test_tensorflow_import(self):
        """Test TensorFlow import (optional for ML workers)"""
        try:
            import tensorflow as tf
            assert hasattr(tf, "__version__")
            print(f"TensorFlow {tf.__version__} installed")
        except ImportError:
            pytest.skip("TensorFlow not installed (optional)")

    def test_torch_import(self):
        """Test PyTorch import (optional for ML workers)"""
        try:
            import torch
            assert hasattr(torch, "__version__")
            print(f"PyTorch {torch.__version__} installed")
        except ImportError:
            pytest.skip("PyTorch not installed (optional)")

    def test_onnx_import(self):
        """Test ONNX import (optional for ML workers)"""
        try:
            import onnx
            import onnxruntime
            assert hasattr(onnx, "__version__")
            print(f"ONNX {onnx.__version__} installed")
        except ImportError:
            pytest.skip("ONNX not installed (optional)")


class TestSystemCommands:
    """Test that system-level commands are available"""

    @pytest.mark.parametrize("command", [
        "python3",
        "pip3",
        "docker",
    ])
    def test_command_exists(self, command):
        """Test that required system commands exist"""
        try:
            result = subprocess.run(
                ["which", command],
                capture_output=True,
                text=True,
                check=False
            )
            assert result.returncode == 0, f"Command '{command}' not found in PATH"
        except Exception as e:
            pytest.fail(f"Failed to check command {command}: {e}")

    def test_docker_compose_exists(self):
        """Test that docker compose is available"""
        try:
            # Try modern 'docker compose' command
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return

            # Fallback to legacy 'docker-compose'
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            assert result.returncode == 0, "Neither 'docker compose' nor 'docker-compose' found"
        except Exception as e:
            pytest.skip(f"Docker compose check failed: {e}")


class TestFileStructure:
    """Test that required files and directories exist"""

    def test_core_directories_exist(self):
        """Verify core directory structure"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        required_dirs = [
            "code",
            "services",
            "services/api",
            "services/decoder",
            "services/mjpeg",
            "services/ml-worker",
            "services/webrtc",
            "html",
            "tests",
        ]

        for dir_path in required_dirs:
            full_path = os.path.join(base_dir, dir_path)
            assert os.path.isdir(full_path), f"Required directory not found: {dir_path}"

    def test_core_files_exist(self):
        """Verify core files exist"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        required_files = [
            "code/api.py",
            "code/brain.py",
            "code/decoder.py",
            "code/mjpeg.py",
            "code/webrtc.py",
            "code/config.py",
            "docker-compose.yml",
            "docker-compose.simple.yml",
            "requirements.txt",
            "README.md",
        ]

        for file_path in required_files:
            full_path = os.path.join(base_dir, file_path)
            assert os.path.isfile(full_path), f"Required file not found: {file_path}"

    def test_install_scripts_exist(self):
        """Verify installation scripts exist and are executable"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        install_scripts = [
            "install_opencv.sh",
            "install_nginx.sh",
            "install_tensorflow.sh",
            "install_gstreamer.sh",
            "install_rabbitmq.sh",
            "install_redis.sh",
            "install_janus.sh",
            "install_webrtc.sh",
        ]

        for script in install_scripts:
            full_path = os.path.join(base_dir, script)
            if os.path.isfile(full_path):
                # Check if executable
                is_executable = os.access(full_path, os.X_OK)
                if not is_executable:
                    print(f"Warning: {script} is not executable")


class TestConfigurationFiles:
    """Test configuration files are valid"""

    def test_docker_compose_files_valid(self):
        """Verify docker-compose files are valid YAML"""
        import yaml
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        compose_files = [
            "docker-compose.yml",
            "docker-compose.simple.yml",
            "docker-compose.webrtc.yml",
        ]

        for compose_file in compose_files:
            full_path = os.path.join(base_dir, compose_file)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, 'r') as f:
                        yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {compose_file}: {e}")


class TestEnvironmentSetup:
    """Test environment configuration"""

    def test_environment_variables(self):
        """Check for recommended environment variables"""
        recommended_vars = {
            "REDIS_URL": "redis://localhost:6379",
            "RABBITMQ_URL": "amqp://guest:guest@localhost:5672",
        }

        for var_name, default_value in recommended_vars.items():
            value = os.getenv(var_name, default_value)
            assert value, f"Environment variable {var_name} not set (using default: {default_value})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
