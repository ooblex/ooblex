"""
End-to-end pipeline tests
Tests the complete video processing pipeline from input to output
"""

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest
import redis


class TestVideoPipelineE2E:
    """Test complete video processing pipeline"""

    @pytest.fixture
    def redis_client(self):
        """Create Redis client for testing"""
        try:
            client = redis.Redis(host="localhost", port=6379, db=0)
            client.ping()
            return client
        except (redis.ConnectionError, redis.exceptions.ConnectionError):
            pytest.skip("Redis not available")

    @pytest.fixture
    def test_frame(self):
        """Generate a test video frame"""
        # Create 640x480 RGB frame with gradient pattern
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add gradient
        for i in range(480):
            frame[i, :, 0] = int(255 * i / 480)  # Red gradient

        for j in range(640):
            frame[:, j, 1] = int(255 * j / 640)  # Green gradient

        # Add blue channel
        frame[:, :, 2] = 128

        return frame

    def test_frame_encoding_decoding(self, test_frame):
        """Test frame can be encoded and decoded"""
        # Encode as JPEG
        success, encoded = cv2.imencode(
            ".jpg", test_frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        assert success, "Failed to encode frame"
        assert len(encoded) > 0, "Encoded frame is empty"

        # Decode
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        assert decoded is not None, "Failed to decode frame"
        assert decoded.shape == test_frame.shape, "Shape mismatch"

        # Check similarity (JPEG is lossy)
        diff = cv2.absdiff(test_frame, decoded)
        mean_diff = np.mean(diff)
        assert mean_diff < 10, f"Too much difference after encode/decode: {mean_diff}"

    def test_frame_storage_in_redis(self, redis_client, test_frame):
        """Test storing and retrieving frames from Redis"""
        # Encode frame
        success, encoded = cv2.imencode(".jpg", test_frame)
        assert success

        # Store in Redis
        frame_id = "test_frame_001"
        redis_client.setex(frame_id, 60, encoded.tobytes())  # 60 second TTL

        # Retrieve from Redis
        retrieved = redis_client.get(frame_id)
        assert retrieved is not None, "Frame not found in Redis"

        # Decode retrieved frame
        decoded = cv2.imdecode(np.frombuffer(retrieved, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None, "Failed to decode retrieved frame"
        assert decoded.shape == test_frame.shape

    def test_multiple_frames_pipeline(self, redis_client, test_frame):
        """Test processing multiple frames through pipeline"""
        frame_ids = []

        # Store multiple frames
        for i in range(10):
            # Create slightly different frame
            frame = test_frame.copy()
            frame[:, :, 0] = (frame[:, :, 0] + i * 10) % 256

            # Encode and store
            success, encoded = cv2.imencode(".jpg", frame)
            assert success

            frame_id = f"test_frame_{i:03d}"
            redis_client.setex(frame_id, 60, encoded.tobytes())
            frame_ids.append(frame_id)

        # Verify all frames stored
        for frame_id in frame_ids:
            assert redis_client.exists(frame_id), f"Frame {frame_id} not found"

        # Retrieve and verify frames
        for frame_id in frame_ids:
            retrieved = redis_client.get(frame_id)
            assert retrieved is not None

            decoded = cv2.imdecode(np.frombuffer(retrieved, np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None
            assert decoded.shape == test_frame.shape

        # Cleanup
        for frame_id in frame_ids:
            redis_client.delete(frame_id)

    def test_frame_processing_simulation(self, test_frame):
        """Simulate processing a frame through ML worker"""
        # Preprocessing
        resized = cv2.resize(test_frame, (256, 256))
        assert resized.shape == (256, 256, 3)

        # Simulate ML processing (apply Gaussian blur as mock AI effect)
        processed = cv2.GaussianBlur(resized, (15, 15), 0)
        assert processed.shape == resized.shape

        # Postprocessing - resize back
        output = cv2.resize(processed, (640, 480))
        assert output.shape == test_frame.shape

        # Verify output is different from input
        diff = cv2.absdiff(test_frame, output)
        mean_diff = np.mean(diff)
        assert mean_diff > 0, "Processing had no effect"

    def test_pipeline_throughput(self, redis_client, test_frame):
        """Test pipeline throughput with timing"""
        import time

        num_frames = 30  # Simulate 1 second at 30 FPS

        start_time = time.time()

        for i in range(num_frames):
            # Encode
            success, encoded = cv2.imencode(".jpg", test_frame)
            assert success

            # Store in Redis
            frame_id = f"throughput_test_{i:03d}"
            redis_client.setex(frame_id, 10, encoded.tobytes())

            # Retrieve
            retrieved = redis_client.get(frame_id)
            assert retrieved is not None

            # Decode
            decoded = cv2.imdecode(np.frombuffer(retrieved, np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None

            # Cleanup immediately
            redis_client.delete(frame_id)

        elapsed = time.time() - start_time
        fps = num_frames / elapsed

        print(
            f"\nPipeline throughput: {fps:.2f} FPS ({elapsed:.2f}s for {num_frames} frames)"
        )

        # Should be able to handle at least 10 FPS on slow hardware
        assert fps > 10, f"Pipeline too slow: {fps:.2f} FPS"


class TestTaskOrchestration:
    """Test task creation and queueing"""

    def test_create_task_message(self):
        """Test creating task message for RabbitMQ"""
        stream_key = "test_stream"
        task_type = "FaceOn"
        redis_id = "frame_123"

        # Create task message (original format)
        task_msg = {"streamKey": stream_key, "task": task_type, "redisID": redis_id}

        # Verify JSON serialization
        json_str = json.dumps(task_msg)
        assert len(json_str) > 0

        # Verify can be decoded
        decoded = json.loads(json_str)
        assert decoded["streamKey"] == stream_key
        assert decoded["task"] == task_type
        assert decoded["redisID"] == redis_id

    def test_multiple_task_types(self):
        """Test different task types"""
        task_types = [
            "FaceOn",
            "TrumpOn",
            "StyleTransfer",
            "BackgroundBlur",
            "ObjectDetection",
        ]

        for task_type in task_types:
            task_msg = {"streamKey": "test", "task": task_type, "redisID": "frame_001"}

            json_str = json.dumps(task_msg)
            decoded = json.loads(json_str)

            assert decoded["task"] == task_type


class TestDataIntegrity:
    """Test data integrity through pipeline"""

    def test_frame_hash_preservation(self):
        """Test frame content preserved through encode/decode"""
        # Create deterministic frame
        np.random.seed(42)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Calculate hash
        original_hash = hashlib.md5(frame.tobytes()).hexdigest()

        # Encode as PNG (lossless)
        success, encoded = cv2.imencode(".png", frame)
        assert success

        # Decode
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        assert decoded is not None

        # Calculate hash of decoded
        decoded_hash = hashlib.md5(decoded.tobytes()).hexdigest()

        # Hashes should match (PNG is lossless)
        assert original_hash == decoded_hash, "Frame content changed"

    def test_frame_dimensions_preserved(self):
        """Test frame dimensions preserved through pipeline"""
        test_sizes = [
            (480, 640),
            (720, 1280),
            (1080, 1920),
            (256, 256),
        ]

        for height, width in test_sizes:
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

            # Encode/decode
            success, encoded = cv2.imencode(".jpg", frame)
            assert success

            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            assert decoded is not None
            assert decoded.shape == (
                height,
                width,
                3,
            ), f"Shape mismatch for {width}x{height}"


class TestErrorHandling:
    """Test error handling in pipeline"""

    def test_corrupted_frame_data(self):
        """Test handling of corrupted frame data"""
        # Create invalid encoded data
        corrupted_data = (
            b"\xff\xd8\xff\xe0\x00\x10\x4a\x46\x49\x46"  # Invalid JPEG header
        )

        # Try to decode
        decoded = cv2.imdecode(
            np.frombuffer(corrupted_data, np.uint8), cv2.IMREAD_COLOR
        )

        # Should return None or empty
        assert (
            decoded is None or decoded.size == 0
        ), "Should fail to decode corrupted data"

    def test_invalid_frame_dimensions(self):
        """Test handling of invalid dimensions"""
        # Try to create frame with invalid dimensions
        with pytest.raises((ValueError, cv2.error)):
            frame = np.zeros((-100, 640, 3), dtype=np.uint8)

    @pytest.mark.redis
    def test_redis_key_not_found(self):
        """Test handling of missing Redis keys"""
        try:
            client = redis.Redis(host="localhost", port=6379, db=0)
            client.ping()
        except redis.ConnectionError:
            pytest.skip("Redis not available")

        # Try to get non-existent key
        result = client.get("non_existent_key_12345")
        assert result is None, "Should return None for missing key"


class TestPerformanceMetrics:
    """Test performance metrics collection"""

    def test_latency_measurement(self):
        """Test measuring processing latency"""
        import time

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Measure encoding latency
        start = time.time()
        success, encoded = cv2.imencode(".jpg", frame)
        encode_latency = time.time() - start

        assert success
        print(f"\nEncode latency: {encode_latency*1000:.2f}ms")

        # Measure decoding latency
        start = time.time()
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decode_latency = time.time() - start

        assert decoded is not None
        print(f"Decode latency: {decode_latency*1000:.2f}ms")

        # Total should be under 100ms for this size
        total_latency = encode_latency + decode_latency
        assert total_latency < 0.1, f"Too slow: {total_latency*1000:.2f}ms"

    def test_memory_usage(self):
        """Test memory usage of frames"""
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        # Calculate raw size
        raw_size = frame.nbytes
        print(f"\nRaw frame size: {raw_size / 1024 / 1024:.2f} MB")

        # Encode
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert success

        encoded_size = len(encoded)
        print(f"Encoded size: {encoded_size / 1024:.2f} KB")

        # Compression ratio
        compression_ratio = raw_size / encoded_size
        print(f"Compression ratio: {compression_ratio:.2f}x")

        # Should achieve at least 10x compression
        assert compression_ratio > 10, "Poor compression ratio"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
