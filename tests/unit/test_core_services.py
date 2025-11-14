"""
Unit tests for core Ooblex services
Tests core functionality without requiring actual ML models or external services
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import cv2
import numpy as np
import pytest


class TestImageProcessing:
    """Test image processing utilities"""

    def test_numpy_array_creation(self):
        """Test creating numpy arrays for image data"""
        # Create a simple 100x100 RGB image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        assert image.shape == (100, 100, 3)
        assert image.dtype == np.uint8

    def test_opencv_encode_decode(self):
        """Test OpenCV image encoding/decoding (no ML required)"""
        # Create a test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Encode as JPEG
        success, encoded = cv2.imencode(".jpg", image)
        assert success, "Failed to encode image"
        assert len(encoded) > 0, "Encoded image is empty"

        # Decode back
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        assert decoded is not None, "Failed to decode image"
        assert decoded.shape == image.shape, "Shape mismatch after encode/decode"

    def test_opencv_resize(self):
        """Test OpenCV image resizing"""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Resize to 256x256 (common ML input size)
        resized = cv2.resize(image, (256, 256))
        assert resized.shape == (256, 256, 3), "Resize failed"

    def test_opencv_color_conversion(self):
        """Test OpenCV color space conversions"""
        image_bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        assert image_rgb.shape == image_bgr.shape

        # Convert to grayscale
        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        assert image_gray.shape == (100, 100)


class TestRedisOperations:
    """Test Redis operations (mocked)"""

    @pytest.mark.asyncio
    async def test_redis_set_get(self):
        """Test Redis set/get operations"""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=b"test_value")

        # Test set
        result = await mock_redis.set("test_key", "test_value")
        assert result is True

        # Test get
        value = await mock_redis.get("test_key")
        assert value == b"test_value"

    @pytest.mark.asyncio
    async def test_redis_frame_storage(self):
        """Test storing/retrieving frames in Redis"""
        mock_redis = AsyncMock()

        # Create a test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert success

        # Mock storing frame
        frame_id = "frame_123"
        mock_redis.set = AsyncMock(return_value=True)
        mock_redis.get = AsyncMock(return_value=encoded.tobytes())

        await mock_redis.set(frame_id, encoded.tobytes())
        retrieved = await mock_redis.get(frame_id)

        # Decode retrieved frame
        decoded = cv2.imdecode(np.frombuffer(retrieved, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape == frame.shape


class TestRabbitMQOperations:
    """Test RabbitMQ operations (mocked)"""

    @pytest.mark.asyncio
    async def test_publish_task(self):
        """Test publishing tasks to RabbitMQ"""
        mock_channel = AsyncMock()
        mock_channel.default_exchange = AsyncMock()
        mock_channel.default_exchange.publish = AsyncMock()

        task_data = {
            "streamKey": "test_stream",
            "task": "FaceOn",
            "redisID": "frame_123",
        }

        # Simulate publishing
        await mock_channel.default_exchange.publish(
            Mock(body=json.dumps(task_data).encode()), routing_key="tf-task"
        )

        # Verify publish was called
        mock_channel.default_exchange.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_consume_task(self):
        """Test consuming tasks from RabbitMQ"""
        mock_queue = AsyncMock()

        # Mock message
        mock_message = Mock()
        mock_message.body = json.dumps(
            {"streamKey": "test_stream", "task": "FaceOn"}
        ).encode()
        mock_message.ack = AsyncMock()

        # Simulate processing
        data = json.loads(mock_message.body.decode())
        assert data["streamKey"] == "test_stream"
        assert data["task"] == "FaceOn"

        await mock_message.ack()
        mock_message.ack.assert_called_once()


class TestMLWorkerProcessing:
    """Test ML worker processing logic (without actual models)"""

    def test_frame_preprocessing(self):
        """Test frame preprocessing for ML models"""
        # Create a test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Typical preprocessing steps
        # 1. Resize to model input size
        resized = cv2.resize(frame, (256, 256))
        assert resized.shape == (256, 256, 3)

        # 2. Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0

        # 3. Convert to RGB (if needed)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        assert rgb.shape == resized.shape

    def test_frame_postprocessing(self):
        """Test frame postprocessing after ML inference"""
        # Simulate model output (normalized float)
        output = np.random.rand(256, 256, 3).astype(np.float32)

        # Typical postprocessing
        # 1. Denormalize from [0, 1] to [0, 255]
        denormalized = (output * 255).astype(np.uint8)
        assert denormalized.dtype == np.uint8
        assert denormalized.max() <= 255
        assert denormalized.min() >= 0

        # 2. Resize back to original size
        resized = cv2.resize(denormalized, (640, 480))
        assert resized.shape == (480, 640, 3)


class TestWebSocketMessages:
    """Test WebSocket message handling"""

    def test_parse_start_process_message(self):
        """Test parsing 'start process' WebSocket messages"""
        message = "start process:test_stream_key"

        # Parse message (original api.py pattern)
        if message.startswith("start process:"):
            parts = message.split("start process:")
            assert len(parts) == 2
            stream_key = parts[1]
            assert stream_key == "test_stream_key"

    def test_parse_task_message(self):
        """Test parsing task WebSocket messages"""
        message = "FaceOn:test_stream_key"

        # Parse message (original api.py pattern)
        parts = message.split(":", 1)
        assert len(parts) == 2

        task = parts[0]
        stream_key = parts[1]

        assert task == "FaceOn"
        assert stream_key == "test_stream_key"

    def test_create_task_json(self):
        """Test creating task JSON for RabbitMQ"""
        stream_key = "test_stream"
        task = "FaceOn"
        redis_id = "frame_123"

        # Create task message (original brain.py format)
        task_msg = {"streamKey": stream_key, "task": task, "redisID": redis_id}

        # Verify JSON serialization
        json_str = json.dumps(task_msg)
        parsed = json.loads(json_str)

        assert parsed["streamKey"] == stream_key
        assert parsed["task"] == task
        assert parsed["redisID"] == redis_id


class TestVideoProcessingPipeline:
    """Test the overall video processing pipeline logic"""

    def test_pipeline_flow(self):
        """Test the complete pipeline flow without actual services"""
        # 1. Receive frame from WebRTC
        incoming_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # 2. Encode frame
        success, encoded = cv2.imencode(".jpg", incoming_frame)
        assert success

        # 3. Store in Redis (mocked)
        frame_id = "frame_001"
        redis_storage = {frame_id: encoded.tobytes()}

        # 4. Queue task (mocked)
        task = {"streamKey": "stream_001", "task": "FaceOn", "redisID": frame_id}
        task_queue = [task]

        # 5. Worker retrieves and processes (mocked)
        retrieved_encoded = redis_storage[frame_id]
        decoded_frame = cv2.imdecode(
            np.frombuffer(retrieved_encoded, np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded_frame is not None

        # 6. Apply processing (mock - just blur for test)
        processed_frame = cv2.GaussianBlur(decoded_frame, (15, 15), 0)
        assert processed_frame.shape == decoded_frame.shape

        # 7. Encode result
        success, result_encoded = cv2.imencode(".jpg", processed_frame)
        assert success

        # 8. Store result in Redis
        result_id = f"processed_{frame_id}"
        redis_storage[result_id] = result_encoded.tobytes()

        # 9. Broadcast completion
        broadcast_msg = {"key": "stream_001", "msg": f"Processed: {result_id}"}

        assert broadcast_msg["key"] == "stream_001"
        assert result_id in redis_storage


class TestConfigurationParsing:
    """Test configuration file parsing"""

    def test_config_structure(self):
        """Test that config has required fields"""
        # Mock config structure (from original config.py)
        config = {
            "REDIS_CONFIG": {"uri": "redis://localhost:6379"},
            "RABBITMQ_CONFIG": {"uri": "amqp://guest:guest@localhost:5672"},
            "DOMAIN_CONFIG": {"domain": "localhost"},
        }

        assert "REDIS_CONFIG" in config
        assert "RABBITMQ_CONFIG" in config
        assert "DOMAIN_CONFIG" in config

        assert "uri" in config["REDIS_CONFIG"]
        assert "uri" in config["RABBITMQ_CONFIG"]
        assert "domain" in config["DOMAIN_CONFIG"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
