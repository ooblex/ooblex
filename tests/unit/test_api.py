"""
Unit tests for API service
"""
import pytest
from datetime import datetime

# Import from the path set up in conftest
from main import (
    ProcessRequest,
    ProcessResponse,
    HealthResponse,
    ConnectionManager,
)
from config import settings


class TestModels:
    """Test Pydantic models"""

    def test_process_request_valid(self):
        """Test ProcessRequest with valid data"""
        request = ProcessRequest(
            stream_token="token123",
            process_type="face_swap",
            model_name="model_v1",
            parameters={"threshold": 0.5}
        )

        assert request.stream_token == "token123"
        assert request.process_type == "face_swap"
        assert request.model_name == "model_v1"
        assert request.parameters == {"threshold": 0.5}

    def test_process_request_valid_types(self):
        """Test ProcessRequest with all valid process types"""
        valid_types = [
            "face_swap", "style_transfer", "object_detection",
            "background_removal", "trump", "taylor"
        ]

        for process_type in valid_types:
            request = ProcessRequest(
                stream_token="token123",
                process_type=process_type
            )
            assert request.process_type == process_type

    def test_process_request_invalid_type(self):
        """Test ProcessRequest with invalid process type"""
        with pytest.raises(ValueError):
            ProcessRequest(
                stream_token="token123",
                process_type="invalid_type"
            )

    def test_process_request_defaults(self):
        """Test ProcessRequest default values"""
        request = ProcessRequest(
            stream_token="token123",
            process_type="face_swap"
        )

        assert request.model_name is None
        assert request.parameters == {}

    def test_process_response(self):
        """Test ProcessResponse model"""
        response = ProcessResponse(
            task_id="task_123",
            status="queued",
            message="Task queued for processing"
        )

        assert response.task_id == "task_123"
        assert response.status == "queued"
        assert response.message == "Task queued for processing"

    def test_health_response(self):
        """Test HealthResponse model"""
        response = HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            services={"redis": "healthy", "rabbitmq": "healthy"}
        )

        assert response.status == "healthy"
        assert isinstance(response.timestamp, datetime)
        assert response.services["redis"] == "healthy"


class TestSettings:
    """Test configuration settings"""

    def test_default_settings(self):
        """Test default settings values"""
        assert settings.app_name == "Ooblex API"
        assert settings.api_port == 8800
        assert settings.jwt_algorithm == "HS256"

    def test_redis_url_default(self):
        """Test Redis URL default"""
        assert "redis://" in settings.redis_url

    def test_rabbitmq_url_default(self):
        """Test RabbitMQ URL default"""
        assert "amqp://" in settings.rabbitmq_url


class TestConnectionManager:
    """Test WebSocket connection manager"""

    def test_connection_manager_init(self):
        """Test ConnectionManager initialization"""
        manager = ConnectionManager()

        assert isinstance(manager.active_connections, dict)
        assert isinstance(manager.stream_tokens, dict)
        assert len(manager.active_connections) == 0
        assert len(manager.stream_tokens) == 0

    def test_disconnect_nonexistent_client(self):
        """Test disconnecting a client that doesn't exist"""
        manager = ConnectionManager()

        # Should not raise error
        manager.disconnect("nonexistent_client")
        assert len(manager.active_connections) == 0

    def test_stream_token_mapping(self):
        """Test stream token to client mapping"""
        manager = ConnectionManager()

        # Manually set up the mapping
        manager.stream_tokens["stream_123"] = "client_abc"

        assert manager.stream_tokens.get("stream_123") == "client_abc"
        assert manager.stream_tokens.get("nonexistent") is None
