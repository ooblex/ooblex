"""
Unit tests for SimpleMLWorker
Tests the simple ML worker functionality with mocked dependencies.
"""
import pytest
import json
import numpy as np
import cv2
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import sys
import os

# Add the ml-worker path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'ml-worker'))

from ml_worker_simple import SimpleMLWorker


@pytest.fixture
def worker():
    """Create SimpleMLWorker instance"""
    return SimpleMLWorker()


@pytest.fixture
def mock_redis_client():
    """Create mock Redis client"""
    mock = AsyncMock()
    mock.get = AsyncMock()
    mock.setex = AsyncMock()
    return mock


@pytest.fixture
def sample_frame():
    """Create a sample encoded frame"""
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, encoded = cv2.imencode('.jpg', image)
    return encoded.tobytes()


@pytest.fixture
def task_message():
    """Create a sample task message"""
    return {
        "task_id": "test_task_123",
        "stream_token": "test_stream",
        "process_type": "grayscale",
        "parameters": {}
    }


class MockAsyncContextManager:
    """Mock async context manager for message.process()"""
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


def create_mock_message(task_data):
    """Helper to create a properly mocked message"""
    mock_message = MagicMock()
    mock_message.body = json.dumps(task_data).encode()
    mock_message.process = MagicMock(return_value=MockAsyncContextManager())
    return mock_message


class TestSimpleMLWorkerInit:
    """Test SimpleMLWorker initialization"""

    def test_init_sets_default_values(self, worker):
        """Test that worker initializes with correct default values"""
        assert worker.redis_client is None
        assert worker.rabbitmq_connection is None
        assert worker.rabbitmq_channel is None
        assert worker.processor is not None
        assert worker.running is True

    def test_init_creates_processor(self, worker):
        """Test that worker creates OpenCV processor"""
        from ml_worker_opencv import OpenCVProcessor
        assert isinstance(worker.processor, OpenCVProcessor)


class TestSimpleMLWorkerSetup:
    """Test SimpleMLWorker setup"""

    @pytest.mark.asyncio
    async def test_setup_initializes_redis(self, worker):
        """Test that setup initializes Redis connection"""
        with patch('ml_worker_simple.redis.from_url', new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = AsyncMock()
            with patch('ml_worker_simple.connect_robust', new_callable=AsyncMock) as mock_rabbitmq:
                mock_connection = AsyncMock()
                mock_channel = AsyncMock()
                mock_channel.set_qos = AsyncMock()
                mock_channel.declare_queue = AsyncMock()
                mock_connection.channel = AsyncMock(return_value=mock_channel)
                mock_rabbitmq.return_value = mock_connection

                await worker.setup()

                assert worker.redis_client is not None
                mock_redis.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_initializes_rabbitmq(self, worker):
        """Test that setup initializes RabbitMQ connection"""
        with patch('ml_worker_simple.redis.from_url', new_callable=AsyncMock) as mock_redis:
            mock_redis.return_value = AsyncMock()
            with patch('ml_worker_simple.connect_robust', new_callable=AsyncMock) as mock_rabbitmq:
                mock_connection = AsyncMock()
                mock_channel = AsyncMock()
                mock_channel.set_qos = AsyncMock()
                mock_channel.declare_queue = AsyncMock()
                mock_connection.channel = AsyncMock(return_value=mock_channel)
                mock_rabbitmq.return_value = mock_connection

                await worker.setup()

                assert worker.rabbitmq_connection is not None
                assert worker.rabbitmq_channel is not None
                mock_rabbitmq.assert_called_once()


class TestSimpleMLWorkerTaskProcessing:
    """Test task processing functionality"""

    @pytest.mark.asyncio
    async def test_process_task_grayscale(self, worker, mock_redis_client, sample_frame, task_message):
        """Test processing a grayscale task"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = sample_frame

        mock_message = create_mock_message(task_message)

        await worker.process_task(mock_message)

        # Verify Redis was called to get frame
        mock_redis_client.get.assert_called_with(f"frame:{task_message['stream_token']}:latest")

    @pytest.mark.asyncio
    async def test_process_task_updates_status(self, worker, mock_redis_client, sample_frame):
        """Test that task processing updates status in Redis"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = sample_frame

        task_data = {
            "task_id": "test_task_456",
            "stream_token": "test_stream",
            "process_type": "sepia",
            "parameters": {}
        }

        mock_message = create_mock_message(task_data)

        await worker.process_task(mock_message)

        # Verify status was updated (setex was called)
        assert mock_redis_client.setex.called

    @pytest.mark.asyncio
    async def test_process_task_handles_missing_frame(self, worker, mock_redis_client):
        """Test that task handles missing frame data"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = None  # No frame data

        task_data = {
            "task_id": "test_task_789",
            "stream_token": "test_stream",
            "process_type": "blur",
            "parameters": {}
        }

        mock_message = create_mock_message(task_data)

        await worker.process_task(mock_message)

        # Should still complete without raising (error logged)
        assert True


class TestSimpleMLWorkerProcessTypes:
    """Test different process types"""

    @pytest.mark.asyncio
    async def test_process_type_face_detection(self, worker, mock_redis_client, sample_frame):
        """Test face detection process type"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = sample_frame

        task_data = {
            "task_id": "test_task",
            "stream_token": "test_stream",
            "process_type": "face_detection",
            "parameters": {}
        }

        mock_message = create_mock_message(task_data)

        await worker.process_task(mock_message)

        # Verify result was stored
        assert mock_redis_client.setex.called

    @pytest.mark.asyncio
    async def test_process_type_edge_detection(self, worker, mock_redis_client, sample_frame):
        """Test edge detection process type with parameters"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = sample_frame

        task_data = {
            "task_id": "test_task",
            "stream_token": "test_stream",
            "process_type": "edge_detection",
            "parameters": {
                "low_threshold": 100,
                "high_threshold": 200
            }
        }

        mock_message = create_mock_message(task_data)

        await worker.process_task(mock_message)

        assert mock_redis_client.setex.called

    @pytest.mark.asyncio
    async def test_process_type_unknown(self, worker, mock_redis_client, sample_frame):
        """Test handling of unknown process type"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = sample_frame

        task_data = {
            "task_id": "test_task",
            "stream_token": "test_stream",
            "process_type": "unknown_effect",
            "parameters": {}
        }

        mock_message = create_mock_message(task_data)

        # Should not raise error, just return original image
        await worker.process_task(mock_message)

        assert mock_redis_client.setex.called


class TestSimpleMLWorkerCleanup:
    """Test cleanup functionality"""

    @pytest.mark.asyncio
    async def test_cleanup_closes_connections(self, worker):
        """Test that cleanup closes all connections"""
        mock_redis = AsyncMock()
        mock_rabbitmq = AsyncMock()

        worker.redis_client = mock_redis
        worker.rabbitmq_connection = mock_rabbitmq

        await worker.cleanup()

        mock_redis.close.assert_called_once()
        mock_rabbitmq.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_handles_none_connections(self, worker):
        """Test that cleanup handles None connections gracefully"""
        worker.redis_client = None
        worker.rabbitmq_connection = None

        # Should not raise
        await worker.cleanup()


class TestSimpleMLWorkerValidProcessTypes:
    """Test all valid process types"""

    @pytest.mark.parametrize("process_type", [
        "face_detection",
        "background_blur",
        "edge_detection",
        "sepia",
        "grayscale",
        "cartoon",
        "vintage",
        "blur",
        "sharpen",
        "pixelate",
        "emboss",
        "none"
    ])
    @pytest.mark.asyncio
    async def test_all_valid_process_types(self, worker, mock_redis_client, sample_frame, process_type):
        """Test that all valid process types are handled"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = sample_frame

        task_data = {
            "task_id": f"test_task_{process_type}",
            "stream_token": "test_stream",
            "process_type": process_type,
            "parameters": {}
        }

        mock_message = create_mock_message(task_data)

        # Should not raise error
        await worker.process_task(mock_message)


class TestSimpleMLWorkerParameters:
    """Test parameter handling"""

    @pytest.mark.asyncio
    async def test_blur_with_kernel_size(self, worker, mock_redis_client, sample_frame):
        """Test blur with custom kernel size parameter"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = sample_frame

        task_data = {
            "task_id": "test_task",
            "stream_token": "test_stream",
            "process_type": "blur",
            "parameters": {"kernel_size": 25}
        }

        mock_message = create_mock_message(task_data)

        await worker.process_task(mock_message)

        assert mock_redis_client.setex.called

    @pytest.mark.asyncio
    async def test_pixelate_with_pixel_size(self, worker, mock_redis_client, sample_frame):
        """Test pixelate with custom pixel size parameter"""
        worker.redis_client = mock_redis_client
        mock_redis_client.get.return_value = sample_frame

        task_data = {
            "task_id": "test_task",
            "stream_token": "test_stream",
            "process_type": "pixelate",
            "parameters": {"pixel_size": 15}
        }

        mock_message = create_mock_message(task_data)

        await worker.process_task(mock_message)

        assert mock_redis_client.setex.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
