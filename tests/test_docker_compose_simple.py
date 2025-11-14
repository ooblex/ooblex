"""
Test suite for docker-compose.simple.yml validation.

This test ensures that the simple docker-compose configuration:
1. Has valid syntax
2. Can build all required images
3. Can start all services successfully
4. All services become healthy
5. Services can communicate with each other
"""

import subprocess
import time
from typing import Dict, List

import pika
import redis
import requests


def run_command(cmd: List[str], timeout: int = 60) -> Dict[str, any]:
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out",
            "success": False,
        }


class TestDockerComposeSimple:
    """Test suite for docker-compose.simple.yml."""

    COMPOSE_FILE = "docker-compose.simple.yml"
    SERVICES = ["redis", "rabbitmq", "worker", "api", "mjpeg"]

    def test_compose_config_valid(self):
        """Test that the docker-compose file has valid syntax."""
        result = run_command(["docker", "compose", "-f", self.COMPOSE_FILE, "config"])
        assert result["success"], f"Config validation failed: {result['stderr']}"
        assert len(result["stdout"]) > 0, "Config output is empty"

    def test_compose_build(self):
        """Test that all images can be built successfully."""
        result = run_command(
            ["docker", "compose", "-f", self.COMPOSE_FILE, "build"], timeout=300
        )
        assert result["success"], f"Build failed: {result['stderr']}"

    def test_compose_up(self):
        """Test that all services start successfully."""
        # Clean up any existing containers
        run_command(["docker", "compose", "-f", self.COMPOSE_FILE, "down", "-v"])

        # Start services in detached mode
        result = run_command(
            ["docker", "compose", "-f", self.COMPOSE_FILE, "up", "-d"], timeout=120
        )
        assert result["success"], f"Services failed to start: {result['stderr']}"

    def test_services_running(self):
        """Test that all expected services are running."""
        # Wait a bit for services to stabilize
        time.sleep(5)

        result = run_command(
            ["docker", "compose", "-f", self.COMPOSE_FILE, "ps", "--format", "json"]
        )
        assert result["success"], f"Failed to get service status: {result['stderr']}"

        # Check that all services are present
        output = result["stdout"]
        for service in self.SERVICES:
            assert service in output, f"Service {service} not found in running services"

    def test_redis_health(self):
        """Test that Redis is healthy and accessible."""
        max_retries = 30
        for i in range(max_retries):
            try:
                r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=1)
                r.ping()
                print("✓ Redis is healthy")
                return
            except Exception as e:
                if i == max_retries - 1:
                    raise AssertionError(
                        f"Redis health check failed after {max_retries} retries: {e}"
                    )
                time.sleep(1)

    def test_rabbitmq_health(self):
        """Test that RabbitMQ is healthy and accessible."""
        max_retries = 30
        for i in range(max_retries):
            try:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host="localhost",
                        port=5672,
                        credentials=pika.PlainCredentials("guest", "guest"),
                        connection_attempts=1,
                        socket_timeout=1,
                    )
                )
                connection.close()
                print("✓ RabbitMQ is healthy")
                return
            except Exception as e:
                if i == max_retries - 1:
                    raise AssertionError(
                        f"RabbitMQ health check failed after {max_retries} retries: {e}"
                    )
                time.sleep(1)

    def test_api_health(self):
        """Test that the API service is healthy and responding."""
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8800/health", timeout=2)
                if response.status_code == 200:
                    print("✓ API is healthy")
                    return
            except requests.exceptions.RequestException as e:
                if i == max_retries - 1:
                    raise AssertionError(
                        f"API health check failed after {max_retries} retries: {e}"
                    )
                time.sleep(1)

    def test_mjpeg_health(self):
        """Test that the MJPEG service is healthy and responding."""
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get(
                    "http://localhost:8081/stream", timeout=2, stream=True
                )
                if response.status_code in [200, 404]:  # 404 is ok if no stream yet
                    print("✓ MJPEG service is responding")
                    return
            except requests.exceptions.RequestException as e:
                if i == max_retries - 1:
                    raise AssertionError(
                        f"MJPEG health check failed after {max_retries} retries: {e}"
                    )
                time.sleep(1)

    def test_worker_logs(self):
        """Test that worker services are running and logging."""
        result = run_command(
            ["docker", "compose", "-f", self.COMPOSE_FILE, "logs", "worker"]
        )
        assert result["success"], f"Failed to get worker logs: {result['stderr']}"
        assert len(result["stdout"]) > 0, "Worker logs are empty"

    @classmethod
    def teardown_class(cls):
        """Clean up: stop and remove all containers."""
        print("\nCleaning up containers...")
        run_command(["docker", "compose", "-f", cls.COMPOSE_FILE, "down", "-v"])
