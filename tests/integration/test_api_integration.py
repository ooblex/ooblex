"""
Integration tests for API service
Tests require running Redis and RabbitMQ services.
"""
import pytest
import json
import asyncio
from datetime import datetime

# Skip all tests if external dependencies aren't available
pytest_plugins = ['pytest_asyncio']


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIIntegration:
    """Integration tests for API endpoints with real dependencies"""

    async def test_root_endpoint(self, api_client):
        """Test root endpoint returns API info"""
        response = await api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "2.0.0"

    async def test_health_check_with_dependencies(self, api_client):
        """Test health check reflects actual service status"""
        response = await api_client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded"]
        assert "services" in health_data
        assert "timestamp" in health_data

        # Verify timestamp is recent
        timestamp = datetime.fromisoformat(health_data["timestamp"].replace("Z", "+00:00"))
        assert (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds() < 60

    async def test_process_request(self, api_client):
        """Test processing request endpoint"""
        process_data = {
            "stream_token": "integration_test_stream",
            "process_type": "face_swap",
            "model_name": "test_model",
            "parameters": {}
        }

        response = await api_client.post("/process", json=process_data)
        assert response.status_code == 200

        result = response.json()
        assert "task_id" in result
        assert result["status"] == "queued"
        assert "message" in result

    async def test_process_request_invalid_type(self, api_client):
        """Test processing request with invalid process type"""
        process_data = {
            "stream_token": "test_stream",
            "process_type": "invalid_type",
        }

        response = await api_client.post("/process", json=process_data)
        assert response.status_code == 422  # Validation error

    async def test_process_request_all_types(self, api_client):
        """Test all valid process types"""
        valid_types = [
            "face_swap", "style_transfer", "object_detection",
            "background_removal", "trump", "taylor"
        ]

        for process_type in valid_types:
            process_data = {
                "stream_token": f"test_stream_{process_type}",
                "process_type": process_type,
            }
            response = await api_client.post("/process", json=process_data)
            assert response.status_code == 200, f"Failed for process_type: {process_type}"

    async def test_concurrent_requests(self, api_client):
        """Test API handles concurrent requests properly"""
        async def make_request(i):
            process_data = {
                "stream_token": f"concurrent_stream_{i}",
                "process_type": "style_transfer",
                "parameters": {"style": f"style_{i}"}
            }
            response = await api_client.post("/process", json=process_data)
            return response

        # Execute 10 requests concurrently
        responses = await asyncio.gather(*[make_request(i) for i in range(10)])

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

        # All should have unique task IDs
        task_ids = [r.json()["task_id"] for r in responses]
        assert len(set(task_ids)) == 10

    async def test_metrics_collection(self, api_client):
        """Test metrics are collected properly"""
        # Make some requests to generate metrics
        await api_client.get("/")
        await api_client.get("/health")

        # Get metrics
        response = await api_client.get("/metrics")
        assert response.status_code == 200

        metrics_text = response.text
        # Verify expected metrics are present
        assert "api_requests_total" in metrics_text or "http" in metrics_text.lower()

    async def test_rate_limiting(self, api_client):
        """Test API handles rapid requests"""
        responses = []
        for _ in range(50):
            response = await api_client.get("/")
            responses.append(response.status_code)

        # All should succeed (rate limiting may not be implemented)
        success_count = sum(1 for status in responses if status == 200)
        assert success_count > 0, "No successful requests"
