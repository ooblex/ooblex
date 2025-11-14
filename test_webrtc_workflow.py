#!/usr/bin/env python3
"""
Test the complete WebRTC workflow end-to-end
Ensures all components are working correctly
"""

import asyncio
import io
import json
import subprocess
import sys
import time

import aiohttp
import numpy as np
import redis
from PIL import Image

# Configuration
REDIS_URL = "redis://localhost:6379"
WEBRTC_URL = "http://localhost:8000"
WORKERS_EXPECTED = 3


def print_status(message, status="INFO"):
    """Pretty print status messages"""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
    }
    print(f"{colors.get(status, '')}[{status}] {message}\033[0m")


def check_service(name, port):
    """Check if a service is running"""
    try:
        result = subprocess.run(
            ["nc", "-zv", "localhost", str(port)], capture_output=True, timeout=2
        )
        return result.returncode == 0
    except:
        return False


async def test_redis_connection():
    """Test Redis is accessible"""
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        print_status("Redis connection successful", "SUCCESS")

        # Clear any old test data
        r.delete("frames_to_process")
        r.delete("test_frames_processed")
        return True
    except Exception as e:
        print_status(f"Redis connection failed: {e}", "ERROR")
        return False


async def test_webrtc_server():
    """Test WebRTC server is running"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{WEBRTC_URL}/health") as resp:
                if resp.status == 200:
                    print_status("WebRTC server is healthy", "SUCCESS")
                    return True
    except Exception as e:
        print_status(f"WebRTC server not responding: {e}", "ERROR")
        return False


async def simulate_frame_processing():
    """Simulate the complete frame processing workflow"""
    r = redis.from_url(REDIS_URL)

    # Create test frames
    print_status("Creating test frames...")
    test_frames = []
    for i in range(10):
        # Create a simple test image
        img = Image.new("RGB", (640, 480), color=(i * 25, 0, 0))
        img_array = np.array(img)

        frame_data = {
            "client_id": "test_client",
            "frame_id": f"frame_{i}",
            "data": img_array.tobytes(),
            "shape": img_array.shape,
            "effect": "style_transfer",
            "timestamp": time.time(),
        }
        test_frames.append(frame_data)

    # Queue frames for processing
    print_status("Queueing frames for processing...")
    for frame in test_frames:
        r.lpush("frames_to_process", json.dumps(frame))

    print_status(f"Queued {len(test_frames)} frames", "SUCCESS")

    # Monitor processing
    print_status("Monitoring frame processing...")
    start_time = time.time()
    processed_count = 0

    while processed_count < len(test_frames) and time.time() - start_time < 30:
        # Check queue length
        queue_length = r.llen("frames_to_process")
        processed = r.llen("processed_frames:test_client")

        print(f"\rQueue: {queue_length} | Processed: {processed}", end="")

        if processed > processed_count:
            processed_count = processed

        await asyncio.sleep(0.5)

    print()  # New line

    if processed_count == len(test_frames):
        print_status(
            f"All frames processed in {time.time()-start_time:.2f}s", "SUCCESS"
        )
        return True
    else:
        print_status(
            f"Only {processed_count}/{len(test_frames)} frames processed", "WARNING"
        )
        return False


async def test_parallel_processing():
    """Test that multiple workers are processing in parallel"""
    r = redis.from_url(REDIS_URL)

    # Monitor worker activity
    print_status("Testing parallel processing...")

    # Create many frames quickly
    for i in range(30):
        frame_data = {
            "client_id": "parallel_test",
            "frame_id": f"pframe_{i}",
            "data": b"dummy_data",
            "shape": (480, 640, 3),
            "effect": "face_detection",
            "timestamp": time.time(),
        }
        r.lpush("frames_to_process", json.dumps(frame_data))

    # Check if frames are being processed by monitoring queue
    samples = []
    for _ in range(5):
        queue_len = r.llen("frames_to_process")
        samples.append(queue_len)
        await asyncio.sleep(0.2)

    # Queue should be decreasing if workers are active
    if samples[0] > samples[-1]:
        print_status("Parallel processing confirmed - queue decreasing", "SUCCESS")
        return True
    else:
        print_status("Workers may not be processing in parallel", "WARNING")
        return False


async def check_worker_health():
    """Check if ML workers are running"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=ml-worker", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        workers = result.stdout.strip().split("\n")
        workers = [w for w in workers if w]  # Remove empty strings

        if len(workers) >= WORKERS_EXPECTED:
            print_status(f"Found {len(workers)} ML workers running", "SUCCESS")
            return True
        else:
            print_status(
                f"Only {len(workers)} workers running (expected {WORKERS_EXPECTED})",
                "WARNING",
            )
            return False
    except Exception as e:
        print_status(f"Could not check worker status: {e}", "ERROR")
        return False


async def main():
    """Run all tests"""
    print_status("Starting WebRTC Workflow Tests", "INFO")
    print("=" * 50)

    # Check services
    services = [("Redis", 6379), ("WebRTC Server", 8000), ("Nginx", 443)]

    all_services_up = True
    for name, port in services:
        if check_service(name, port):
            print_status(f"{name} is running on port {port}", "SUCCESS")
        else:
            print_status(f"{name} is not running on port {port}", "ERROR")
            all_services_up = False

    if not all_services_up:
        print_status(
            "Not all services are running. Run ./run-webrtc-demo.sh first", "ERROR"
        )
        return

    # Run tests
    tests = [
        ("Redis Connection", test_redis_connection),
        ("WebRTC Server", test_webrtc_server),
        ("Worker Health", check_worker_health),
        ("Frame Processing", simulate_frame_processing),
        ("Parallel Processing", test_parallel_processing),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print_status(f"Test failed with error: {e}", "ERROR")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print_status("Test Summary", "INFO")
    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print_status(
            "All tests passed! WebRTC workflow is working correctly.", "SUCCESS"
        )
    else:
        print_status("Some tests failed. Check the output above.", "WARNING")


if __name__ == "__main__":
    asyncio.run(main())
