"""
Performance benchmarks for Ooblex
Tests throughput, latency, and resource usage
"""
import pytest
import numpy as np
import cv2
import time
import redis
import json
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestEncodingBenchmarks:
    """Benchmark video encoding performance"""

    @pytest.fixture
    def frame_480p(self):
        """Generate 480p test frame"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def frame_720p(self):
        """Generate 720p test frame"""
        return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    @pytest.fixture
    def frame_1080p(self):
        """Generate 1080p test frame"""
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    @pytest.mark.benchmark
    def test_jpeg_encoding_480p(self, frame_480p, benchmark):
        """Benchmark JPEG encoding for 480p frames"""
        def encode():
            return cv2.imencode('.jpg', frame_480p, [cv2.IMWRITE_JPEG_QUALITY, 85])

        result = benchmark(encode)
        assert result[0], "Encoding failed"
        print(f"\n480p JPEG encoding: {benchmark.stats.stats.mean*1000:.2f}ms average")

    @pytest.mark.benchmark
    def test_jpeg_encoding_720p(self, frame_720p, benchmark):
        """Benchmark JPEG encoding for 720p frames"""
        def encode():
            return cv2.imencode('.jpg', frame_720p, [cv2.IMWRITE_JPEG_QUALITY, 85])

        result = benchmark(encode)
        assert result[0], "Encoding failed"
        print(f"\n720p JPEG encoding: {benchmark.stats.stats.mean*1000:.2f}ms average")

    @pytest.mark.benchmark
    def test_jpeg_encoding_1080p(self, frame_1080p, benchmark):
        """Benchmark JPEG encoding for 1080p frames"""
        def encode():
            return cv2.imencode('.jpg', frame_1080p, [cv2.IMWRITE_JPEG_QUALITY, 85])

        result = benchmark(encode)
        assert result[0], "Encoding failed"
        print(f"\n1080p JPEG encoding: {benchmark.stats.stats.mean*1000:.2f}ms average")

    @pytest.mark.benchmark
    def test_png_encoding_480p(self, frame_480p, benchmark):
        """Benchmark PNG encoding for 480p frames"""
        def encode():
            return cv2.imencode('.png', frame_480p)

        result = benchmark(encode)
        assert result[0], "Encoding failed"
        print(f"\n480p PNG encoding: {benchmark.stats.stats.mean*1000:.2f}ms average")


class TestDecodingBenchmarks:
    """Benchmark video decoding performance"""

    @pytest.fixture
    def encoded_480p(self):
        """Pre-encoded 480p frame"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        success, encoded = cv2.imencode('.jpg', frame)
        return encoded

    @pytest.fixture
    def encoded_720p(self):
        """Pre-encoded 720p frame"""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        success, encoded = cv2.imencode('.jpg', frame)
        return encoded

    @pytest.mark.benchmark
    def test_jpeg_decoding_480p(self, encoded_480p, benchmark):
        """Benchmark JPEG decoding for 480p frames"""
        def decode():
            return cv2.imdecode(encoded_480p, cv2.IMREAD_COLOR)

        result = benchmark(decode)
        assert result is not None, "Decoding failed"
        print(f"\n480p JPEG decoding: {benchmark.stats.stats.mean*1000:.2f}ms average")

    @pytest.mark.benchmark
    def test_jpeg_decoding_720p(self, encoded_720p, benchmark):
        """Benchmark JPEG decoding for 720p frames"""
        def decode():
            return cv2.imdecode(encoded_720p, cv2.IMREAD_COLOR)

        result = benchmark(decode)
        assert result is not None, "Decoding failed"
        print(f"\n720p JPEG decoding: {benchmark.stats.stats.mean*1000:.2f}ms average")


class TestImageProcessingBenchmarks:
    """Benchmark common image processing operations"""

    @pytest.fixture
    def frame_640x480(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.mark.benchmark
    def test_resize_480p_to_256(self, frame_640x480, benchmark):
        """Benchmark resizing 480p to 256x256"""
        def resize():
            return cv2.resize(frame_640x480, (256, 256))

        result = benchmark(resize)
        assert result.shape == (256, 256, 3)
        print(f"\nResize 480p→256x256: {benchmark.stats.stats.mean*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_gaussian_blur(self, frame_640x480, benchmark):
        """Benchmark Gaussian blur"""
        def blur():
            return cv2.GaussianBlur(frame_640x480, (15, 15), 0)

        result = benchmark(blur)
        assert result.shape == frame_640x480.shape
        print(f"\nGaussian blur: {benchmark.stats.stats.mean*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_color_conversion(self, frame_640x480, benchmark):
        """Benchmark BGR to RGB conversion"""
        def convert():
            return cv2.cvtColor(frame_640x480, cv2.COLOR_BGR2RGB)

        result = benchmark(convert)
        assert result.shape == frame_640x480.shape
        print(f"\nColor conversion: {benchmark.stats.stats.mean*1000:.2f}ms")

    @pytest.mark.benchmark
    def test_canny_edge_detection(self, frame_640x480, benchmark):
        """Benchmark Canny edge detection"""
        gray = cv2.cvtColor(frame_640x480, cv2.COLOR_BGR2GRAY)

        def edge_detect():
            return cv2.Canny(gray, 100, 200)

        result = benchmark(edge_detect)
        assert result is not None
        print(f"\nCanny edge detection: {benchmark.stats.stats.mean*1000:.2f}ms")


class TestRedisBenchmarks:
    """Benchmark Redis operations"""

    @pytest.fixture
    def redis_client(self):
        """Create Redis client"""
        try:
            client = redis.Redis(host='localhost', port=6379, db=0)
            client.ping()
            return client
        except redis.ConnectionError:
            pytest.skip("Redis not available")

    @pytest.fixture
    def encoded_frame(self):
        """Pre-encoded frame"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        success, encoded = cv2.imencode('.jpg', frame)
        return encoded.tobytes()

    @pytest.mark.benchmark
    @pytest.mark.redis
    def test_redis_set(self, redis_client, encoded_frame, benchmark):
        """Benchmark Redis SET operation"""
        counter = [0]

        def redis_set():
            key = f"benchmark_frame_{counter[0]}"
            counter[0] += 1
            return redis_client.setex(key, 10, encoded_frame)

        result = benchmark(redis_set)
        print(f"\nRedis SET: {benchmark.stats.stats.mean*1000:.2f}ms")

    @pytest.mark.benchmark
    @pytest.mark.redis
    def test_redis_get(self, redis_client, encoded_frame, benchmark):
        """Benchmark Redis GET operation"""
        # Store a frame first
        redis_client.setex("benchmark_frame_get", 60, encoded_frame)

        def redis_get():
            return redis_client.get("benchmark_frame_get")

        result = benchmark(redis_get)
        assert result is not None
        print(f"\nRedis GET: {benchmark.stats.stats.mean*1000:.2f}ms")


class TestThroughputBenchmarks:
    """Benchmark overall system throughput"""

    def test_encoding_throughput_30fps(self):
        """Test if system can encode 30 frames per second"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        num_frames = 30

        start = time.time()
        for _ in range(num_frames):
            success, encoded = cv2.imencode('.jpg', frame)
            assert success

        elapsed = time.time() - start
        fps = num_frames / elapsed

        print(f"\nEncoding throughput: {fps:.2f} FPS")
        assert fps >= 30, f"Cannot maintain 30 FPS encoding: {fps:.2f} FPS"

    def test_decoding_throughput_30fps(self):
        """Test if system can decode 30 frames per second"""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        success, encoded = cv2.imencode('.jpg', frame)
        assert success

        num_frames = 30

        start = time.time()
        for _ in range(num_frames):
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            assert decoded is not None

        elapsed = time.time() - start
        fps = num_frames / elapsed

        print(f"\nDecoding throughput: {fps:.2f} FPS")
        assert fps >= 30, f"Cannot maintain 30 FPS decoding: {fps:.2f} FPS"

    @pytest.mark.redis
    def test_redis_throughput(self):
        """Test Redis read/write throughput"""
        try:
            client = redis.Redis(host='localhost', port=6379, db=0)
            client.ping()
        except redis.ConnectionError:
            pytest.skip("Redis not available")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        success, encoded = cv2.imencode('.jpg', frame)
        encoded_bytes = encoded.tobytes()

        num_operations = 100

        start = time.time()
        for i in range(num_operations):
            key = f"throughput_test_{i}"
            client.setex(key, 10, encoded_bytes)
            retrieved = client.get(key)
            assert retrieved is not None
            client.delete(key)

        elapsed = time.time() - start
        ops_per_sec = num_operations / elapsed

        print(f"\nRedis throughput: {ops_per_sec:.2f} ops/sec")
        assert ops_per_sec >= 50, f"Redis too slow: {ops_per_sec:.2f} ops/sec"


class TestParallelProcessing:
    """Test parallel processing performance"""

    def test_parallel_encoding(self):
        """Test encoding multiple frames in parallel"""
        num_frames = 10
        frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                  for _ in range(num_frames)]

        def encode_frame(frame):
            return cv2.imencode('.jpg', frame)

        # Sequential processing
        start = time.time()
        for frame in frames:
            success, encoded = encode_frame(frame)
            assert success
        sequential_time = time.time() - start

        # Parallel processing
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(encode_frame, frame) for frame in frames]
            results = [f.result() for f in as_completed(futures)]
        parallel_time = time.time() - start

        speedup = sequential_time / parallel_time

        print(f"\nSequential: {sequential_time:.3f}s")
        print(f"Parallel (4 workers): {parallel_time:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Should see some speedup (not necessarily 4x due to GIL)
        assert speedup > 1.0, f"No speedup from parallelization: {speedup:.2f}x"


class TestMemoryBenchmarks:
    """Benchmark memory usage"""

    def test_frame_memory_footprint(self):
        """Test memory footprint of frames"""
        resolutions = [
            ((480, 640), "480p"),
            ((720, 1280), "720p"),
            ((1080, 1920), "1080p"),
            ((2160, 3840), "4K"),
        ]

        print("\nMemory footprint:")
        for (h, w), name in resolutions:
            # Raw frame
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            raw_size = frame.nbytes

            # Encoded frame
            success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            encoded_size = len(encoded)

            compression = raw_size / encoded_size

            print(f"  {name}: Raw={raw_size/1024/1024:.2f}MB, "
                  f"Encoded={encoded_size/1024:.2f}KB, "
                  f"Compression={compression:.1f}x")

    def test_batch_memory_usage(self):
        """Test memory usage of frame batches"""
        batch_sizes = [1, 10, 30, 60]

        print("\nBatch memory usage:")
        for batch_size in batch_sizes:
            frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                      for _ in range(batch_size)]

            total_size = sum(f.nbytes for f in frames)

            print(f"  Batch of {batch_size}: {total_size/1024/1024:.2f}MB")


class TestLatencyBreakdown:
    """Detailed latency breakdown"""

    @pytest.mark.redis
    def test_end_to_end_latency(self):
        """Measure end-to-end latency breakdown"""
        try:
            client = redis.Redis(host='localhost', port=6379, db=0)
            client.ping()
        except redis.ConnectionError:
            pytest.skip("Redis not available")

        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        timings = {}

        # Encoding
        start = time.time()
        success, encoded = cv2.imencode('.jpg', frame)
        timings['encoding'] = time.time() - start
        assert success

        # Redis write
        start = time.time()
        client.setex("latency_test", 10, encoded.tobytes())
        timings['redis_write'] = time.time() - start

        # Redis read
        start = time.time()
        retrieved = client.get("latency_test")
        timings['redis_read'] = time.time() - start
        assert retrieved is not None

        # Decoding
        start = time.time()
        decoded = cv2.imdecode(np.frombuffer(retrieved, np.uint8), cv2.IMREAD_COLOR)
        timings['decoding'] = time.time() - start
        assert decoded is not None

        # Processing (mock with resize)
        start = time.time()
        processed = cv2.resize(decoded, (256, 256))
        processed = cv2.resize(processed, (640, 480))
        timings['processing'] = time.time() - start

        # Total
        timings['total'] = sum(timings.values())

        print("\nLatency breakdown (480p):")
        for operation, latency in timings.items():
            percentage = (latency / timings['total']) * 100
            print(f"  {operation:15s}: {latency*1000:6.2f}ms ({percentage:5.1f}%)")

        # Total should be under 100ms for this pipeline
        assert timings['total'] < 0.1, f"Total latency too high: {timings['total']*1000:.2f}ms"

        # Cleanup
        client.delete("latency_test")


if __name__ == "__main__":
    # Run with: pytest test_performance.py -v -s --benchmark-only
    pytest.main([__file__, "-v", "-s"])
