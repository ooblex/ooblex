"""
Integration tests for API service
"""

import asyncio
import json
from datetime import datetime

import pytest
from api.main import app


@pytest.mark.asyncio
class TestAPIIntegration:
    """Integration tests for API endpoints with real dependencies"""

    async def test_full_authentication_flow(self, api_client):
        """Test complete authentication flow"""
        # 1. Failed login with wrong credentials
        response = await api_client.post(
            "/token", data={"username": "wrong", "password": "wrong"}
        )
        assert response.status_code == 401

        # 2. Successful login (using demo user)
        response = await api_client.post(
            "/token", data={"username": "demo", "password": "demo123"}
        )
        assert response.status_code == 200
        token_data = response.json()
        assert "access_token" in token_data

        # 3. Use token to access protected endpoint
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        response = await api_client.get("/me", headers=headers)
        assert response.status_code == 200
        user_data = response.json()
        assert user_data["username"] == "demo"

        # 4. Try to access protected endpoint without token
        response = await api_client.get("/me")
        assert response.status_code == 401

    async def test_process_workflow(
        self, api_client, redis_client, rabbitmq_connection
    ):
        """Test complete processing workflow"""
        # Login first
        response = await api_client.post(
            "/token", data={"username": "demo", "password": "demo123"}
        )
        token_data = response.json()
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}

        # Create processing request
        process_data = {
            "stream_token": "integration_test_stream",
            "process_type": "face_swap",
            "model_name": "test_model",
        }

        response = await api_client.post("/process", json=process_data, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert "task_id" in result
        task_id = result["task_id"]

        # Check task status
        response = await api_client.get(f"/process/{task_id}", headers=headers)
        assert response.status_code == 200
        status_data = response.json()
        assert status_data["status"] == "queued"

        # Verify task was stored in Redis
        task_info = await redis_client.get(f"task:{task_id}")
        assert task_info is not None
        task_data = json.loads(task_info)
        assert task_data["status"] == "queued"

        # Verify message was sent to RabbitMQ
        channel = await rabbitmq_connection.channel()
        queue = await channel.declare_queue("tasks", durable=True)

        # Should have at least one message
        message = await queue.get()
        if message:
            async with message.process():
                body = json.loads(message.body.decode())
                assert body["task_id"] == task_id
                assert body["process_type"] == "face_swap"

    async def test_websocket_pubsub(self, api_client, redis_client):
        """Test WebSocket pub/sub functionality"""
        client1_id = "integration_test_client1"
        client2_id = "integration_test_client2"
        stream_token = "integration_test_stream"

        # Connect two clients
        async with api_client.websocket_connect(f"/ws/{client1_id}") as ws1:
            async with api_client.websocket_connect(f"/ws/{client2_id}") as ws2:
                # Client 1 subscribes to stream
                await ws1.send_json({"type": "subscribe", "stream_token": stream_token})

                # Wait for subscription confirmation
                response = await ws1.receive_json()
                assert response["type"] == "subscribed"

                # Verify subscription in Redis
                subscribers = await redis_client.smembers(f"subscribers:{stream_token}")
                assert client1_id in subscribers

                # Test ping/pong
                await ws2.send_json({"type": "ping"})
                pong = await ws2.receive_json()
                assert pong["type"] == "pong"

    async def test_health_check_with_dependencies(
        self, api_client, redis_client, rabbitmq_connection
    ):
        """Test health check reflects actual service status"""
        response = await api_client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["services"]["redis"] == "healthy"
        assert health_data["services"]["rabbitmq"] == "healthy"

        # Verify timestamp is recent
        timestamp = datetime.fromisoformat(
            health_data["timestamp"].replace("Z", "+00:00")
        )
        assert (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds() < 5

    async def test_concurrent_requests(self, api_client):
        """Test API handles concurrent requests properly"""
        # Login
        response = await api_client.post(
            "/token", data={"username": "demo", "password": "demo123"}
        )
        token_data = response.json()
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}

        # Make 10 concurrent requests
        async def make_request(i):
            process_data = {
                "stream_token": f"concurrent_stream_{i}",
                "process_type": "style_transfer",
                "parameters": {"style": f"style_{i}"},
            }
            response = await api_client.post(
                "/process", json=process_data, headers=headers
            )
            return response

        # Execute requests concurrently
        responses = await asyncio.gather(*[make_request(i) for i in range(10)])

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have unique task IDs
        task_ids = [r.json()["task_id"] for r in responses]
        assert len(set(task_ids)) == 10

    async def test_rate_limiting(self, api_client):
        """Test rate limiting functionality"""
        # This would require rate limiting middleware to be implemented
        # For now, just verify endpoint responds
        response = await api_client.get("/")
        assert response.status_code == 200

        # Make many requests quickly
        responses = []
        for _ in range(100):
            response = await api_client.get("/")
            responses.append(response.status_code)

        # All should succeed (no rate limiting implemented yet)
        assert all(status == 200 for status in responses)

    async def test_metrics_collection(self, api_client):
        """Test metrics are collected properly"""
        # Make some requests to generate metrics
        await api_client.get("/")
        await api_client.get("/health")

        # Login to generate auth metrics
        await api_client.post(
            "/token", data={"username": "demo", "password": "demo123"}
        )

        # Get metrics
        response = await api_client.get("/metrics")
        assert response.status_code == 200

        metrics_text = response.text

        # Verify expected metrics are present
        assert "api_requests_total" in metrics_text
        assert "api_request_duration_seconds" in metrics_text
        assert "websocket_connections_total" in metrics_text

        # Verify metrics have values
        assert 'api_requests_total{method="GET"' in metrics_text
        assert 'api_requests_total{method="POST"' in metrics_text
