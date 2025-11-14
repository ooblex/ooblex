"""
Unit tests for API service
"""

from datetime import datetime, timedelta

import pytest
from api.config import settings
from api.main import (
    ProcessRequest,
    TokenData,
    User,
    create_access_token,
    get_password_hash,
    verify_password,
)
from jose import jwt


class TestAuthentication:
    """Test authentication functions"""

    def test_password_hashing(self):
        """Test password hashing and verification"""
        password = "testpassword123"
        hashed = get_password_hash(password)

        assert hashed != password
        assert verify_password(password, hashed)
        assert not verify_password("wrongpassword", hashed)

    def test_create_access_token(self):
        """Test JWT token creation"""
        data = {"sub": "testuser"}
        token = create_access_token(data)

        # Decode token
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )

        assert payload["sub"] == "testuser"
        assert "exp" in payload

    def test_create_access_token_with_expiry(self):
        """Test JWT token with custom expiry"""
        data = {"sub": "testuser"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)

        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        exp_time = datetime.fromtimestamp(payload["exp"])

        # Check expiry is approximately 30 minutes from now
        assert (exp_time - datetime.utcnow()).total_seconds() > 1700  # ~28 minutes
        assert (exp_time - datetime.utcnow()).total_seconds() < 1900  # ~31 minutes


class TestModels:
    """Test Pydantic models"""

    def test_user_model(self):
        """Test User model"""
        user = User(
            username="testuser", email="test@example.com", full_name="Test User"
        )

        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.disabled is None

    def test_process_request_validation(self):
        """Test ProcessRequest validation"""
        # Valid request
        request = ProcessRequest(
            stream_token="token123",
            process_type="face_swap",
            model_name="model_v1",
            parameters={"threshold": 0.5},
        )

        assert request.stream_token == "token123"
        assert request.process_type == "face_swap"

        # Invalid process type should raise error
        with pytest.raises(ValueError):
            ProcessRequest(stream_token="token123", process_type="invalid_type")

    def test_token_data_optional(self):
        """Test TokenData with optional username"""
        token_data = TokenData()
        assert token_data.username is None

        token_data = TokenData(username="testuser")
        assert token_data.username == "testuser"


@pytest.mark.asyncio
class TestAPIEndpoints:
    """Test API endpoints"""

    async def test_root_endpoint(self, api_client):
        """Test root endpoint"""
        response = await api_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "2.0.0"

    async def test_health_check(self, api_client):
        """Test health check endpoint"""
        response = await api_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "services" in data
        assert data["status"] in ["healthy", "degraded"]

    async def test_login_success(self, api_client, mocker):
        """Test successful login"""
        # Mock get_user and authenticate_user
        mock_user = mocker.patch("api.main.authenticate_user")
        mock_user.return_value = User(
            username="demo",
            email="demo@example.com",
            full_name="Demo User",
            disabled=False,
        )

        response = await api_client.post(
            "/token", data={"username": "demo", "password": "demo123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_login_failure(self, api_client, mocker):
        """Test failed login"""
        mock_user = mocker.patch("api.main.authenticate_user")
        mock_user.return_value = False

        response = await api_client.post(
            "/token", data={"username": "wrong", "password": "wrong"}
        )

        assert response.status_code == 401
        assert response.json()["detail"] == "Incorrect username or password"

    async def test_get_current_user(self, api_client, auth_headers, mocker):
        """Test get current user endpoint"""
        # Mock get_user
        mock_get_user = mocker.patch("api.main.get_user")
        mock_get_user.return_value = User(
            username="testuser",
            email="test@example.com",
            full_name="Test User",
            disabled=False,
        )

        response = await api_client.get("/me", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"

    async def test_process_request(self, api_client, auth_headers, mocker):
        """Test process request endpoint"""
        # Mock dependencies
        mock_get_user = mocker.patch("api.main.get_user")
        mock_get_user.return_value = User(username="testuser", disabled=False)

        request_data = {
            "stream_token": "test_token",
            "process_type": "face_swap",
            "model_name": "test_model",
            "parameters": {"threshold": 0.5},
        }

        response = await api_client.post(
            "/process", json=request_data, headers=auth_headers
        )

        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["message"] == "Task queued for processing"

    async def test_get_process_status(
        self, api_client, auth_headers, redis_client, mocker
    ):
        """Test get process status endpoint"""
        # Mock dependencies
        mock_get_user = mocker.patch("api.main.get_user")
        mock_get_user.return_value = User(username="testuser", disabled=False)

        # Add task to Redis
        task_id = "test_task_123"
        await redis_client.setex(
            f"task:{task_id}",
            3600,
            '{"status": "processing", "message": "Task in progress"}',
        )

        response = await api_client.get(f"/process/{task_id}", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["status"] == "processing"

    async def test_metrics_endpoint(self, api_client):
        """Test metrics endpoint"""
        response = await api_client.get("/metrics")

        assert response.status_code == 200
        # Prometheus metrics are plain text
        assert "api_requests_total" in response.text


@pytest.mark.asyncio
class TestWebSocket:
    """Test WebSocket functionality"""

    async def test_websocket_connection(self, api_client):
        """Test WebSocket connection"""
        async with api_client.websocket_connect("/ws/test_client") as websocket:
            # Send ping
            await websocket.send_json({"type": "ping"})

            # Receive pong
            data = await websocket.receive_json()
            assert data["type"] == "pong"
            assert "timestamp" in data

    async def test_websocket_subscribe(self, api_client):
        """Test WebSocket subscription"""
        async with api_client.websocket_connect("/ws/test_client") as websocket:
            # Subscribe to stream
            await websocket.send_json(
                {"type": "subscribe", "stream_token": "test_stream"}
            )

            # Receive confirmation
            data = await websocket.receive_json()
            assert data["type"] == "subscribed"
            assert data["stream_token"] == "test_stream"

    async def test_websocket_invalid_json(self, api_client):
        """Test WebSocket with invalid JSON"""
        async with api_client.websocket_connect("/ws/test_client") as websocket:
            # Send invalid JSON
            await websocket.send_text("invalid json")

            # Receive error
            data = await websocket.receive_json()
            assert data["type"] == "error"
            assert data["message"] == "Invalid JSON"
