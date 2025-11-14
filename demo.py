#!/usr/bin/env python3
"""
Ooblex Demo Script
==================
Demonstrates the complete Ooblex pipeline with brain_simple.py effects.
This script validates that all components work together correctly.

Usage:
    python3 demo.py                    # Run all effects demo
    python3 demo.py --effect=FaceOn    # Test specific effect
    python3 demo.py --quick            # Quick validation only
"""

import asyncio
import cv2
import json
import numpy as np
import os
import sys
import time
from typing import Optional

# Check dependencies
try:
    import redis
    import pika
    import websockets
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("\nInstall with: pip install -r requirements.txt")
    sys.exit(1)


class OoblexDemo:
    """Ooblex pipeline demonstration"""
    
    def __init__(self, redis_url="redis://localhost:6379", 
                 rabbitmq_url="amqp://guest:guest@localhost:5672"):
        self.redis_url = redis_url
        self.rabbitmq_url = rabbitmq_url
        self.redis_client = None
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        
    def check_dependencies(self):
        """Check all required dependencies are available"""
        print("🔍 Checking dependencies...")
        
        deps = {
            "cv2": "OpenCV (pip install opencv-python)",
            "redis": "Redis Python client (pip install redis)",
            "pika": "RabbitMQ client (pip install pika)",
            "numpy": "NumPy (pip install numpy)",
            "websockets": "WebSockets (pip install websockets)"
        }
        
        missing = []
        for module, description in deps.items():
            try:
                __import__(module)
                print(f"  ✅ {module}")
            except ImportError:
                print(f"  ❌ {module} - {description}")
                missing.append(module)
        
        if missing:
            print(f"\n❌ Missing dependencies: {', '.join(missing)}")
            print("Install with: pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies installed\n")
        return True
    
    def check_services(self):
        """Check Redis and RabbitMQ are running"""
        print("🔍 Checking services...")
        
        # Check Redis
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            print("  ✅ Redis (localhost:6379)")
        except Exception as e:
            print(f"  ❌ Redis - {e}")
            print("     Start with: docker run -d -p 6379:6379 redis:7-alpine")
            return False
        
        # Check RabbitMQ
        try:
            params = pika.URLParameters(self.rabbitmq_url)
            params.socket_timeout = 2
            self.rabbitmq_connection = pika.BlockingConnection(params)
            self.rabbitmq_channel = self.rabbitmq_connection.channel()
            print("  ✅ RabbitMQ (localhost:5672)")
        except Exception as e:
            print(f"  ❌ RabbitMQ - {e}")
            print("     Start with: docker run -d -p 5672:5672 rabbitmq:3.12-alpine")
            return False
        
        print("✅ All services running\n")
        return True
    
    def create_test_frame(self, width=640, height=480):
        """Create a test video frame with recognizable patterns"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add colorful gradient background
        for y in range(height):
            color_value = int(255 * (y / height))
            frame[y, :] = [color_value, 128, 255 - color_value]
        
        # Add shapes for visual interest
        cv2.rectangle(frame, (100, 100), (540, 380), (0, 255, 0), 3)
        cv2.circle(frame, (320, 240), 80, (255, 0, 0), -1)
        cv2.circle(frame, (320, 240), 60, (255, 255, 255), -1)
        
        # Add text
        cv2.putText(frame, "OOBLEX DEMO", (180, 250), 
                   cv2.FONT_HERSHEY_BOLD, 1.5, (0, 0, 0), 3)
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (260, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def test_redis_pipeline(self):
        """Test Redis frame storage and retrieval"""
        print("🔍 Testing Redis pipeline...")
        
        # Create test frame
        frame = self.create_test_frame()
        
        # Encode as JPEG
        encode_start = time.time()
        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        encode_time = (time.time() - encode_start) * 1000
        
        if not success:
            print("  ❌ Failed to encode frame")
            return False
        
        frame_data = encoded.tobytes()
        frame_size = len(frame_data) / 1024  # KB
        
        # Store in Redis
        redis_key = "demo:test_frame"
        store_start = time.time()
        self.redis_client.setex(redis_key, 30, frame_data)
        store_time = (time.time() - store_start) * 1000
        
        # Retrieve from Redis
        retrieve_start = time.time()
        retrieved = self.redis_client.get(redis_key)
        retrieve_time = (time.time() - retrieve_start) * 1000
        
        # Decode
        decode_start = time.time()
        decoded_frame = cv2.imdecode(
            np.frombuffer(retrieved, dtype=np.uint8), 
            cv2.IMREAD_COLOR
        )
        decode_time = (time.time() - decode_start) * 1000
        
        total_time = encode_time + store_time + retrieve_time + decode_time
        
        print(f"  ✅ Frame size: {frame_size:.1f} KB")
        print(f"  ✅ Encode: {encode_time:.2f}ms")
        print(f"  ✅ Redis store: {store_time:.2f}ms")
        print(f"  ✅ Redis retrieve: {retrieve_time:.2f}ms")
        print(f"  ✅ Decode: {decode_time:.2f}ms")
        print(f"  ✅ Total latency: {total_time:.2f}ms")
        
        if total_time > 400:
            print(f"  ⚠️  Latency above 400ms target ({total_time:.2f}ms)")
        else:
            print(f"  🎉 Sub-400ms latency achieved!")
        
        print()
        return True
    
    def test_rabbitmq_pipeline(self):
        """Test RabbitMQ task queue"""
        print("🔍 Testing RabbitMQ pipeline...")
        
        queue_name = "demo_tasks"
        
        # Declare queue
        self.rabbitmq_channel.queue_declare(queue=queue_name, durable=False)
        
        # Send test message
        test_message = {
            "streamKey": "demo_stream",
            "task": "FaceOn",
            "redisID": "demo:test_frame",
            "timestamp": time.time()
        }
        
        publish_start = time.time()
        self.rabbitmq_channel.basic_publish(
            exchange='',
            routing_key=queue_name,
            body=json.dumps(test_message)
        )
        publish_time = (time.time() - publish_start) * 1000
        
        # Receive message
        receive_start = time.time()
        method_frame, header_frame, body = self.rabbitmq_channel.basic_get(
            queue=queue_name,
            auto_ack=True
        )
        receive_time = (time.time() - receive_start) * 1000
        
        if method_frame:
            received_message = json.loads(body)
            print(f"  ✅ Message published: {publish_time:.2f}ms")
            print(f"  ✅ Message received: {receive_time:.2f}ms")
            print(f"  ✅ Task: {received_message['task']}")
            print(f"  ✅ Stream: {received_message['streamKey']}")
        else:
            print("  ❌ No message received")
            return False
        
        # Cleanup
        self.rabbitmq_channel.queue_delete(queue=queue_name)
        
        print()
        return True
    
    def test_effect_simulation(self, effect="FaceOn"):
        """Simulate applying an effect (without running brain_simple.py)"""
        print(f"🔍 Testing {effect} effect simulation...")
        
        # Create test frame
        frame = self.create_test_frame()
        
        # Simulate different effects with OpenCV
        process_start = time.time()
        
        if effect == "FaceOn":
            # Simulate face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            result = frame
            
        elif effect == "CartoonOn":
            # Cartoon effect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, 
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            result = cv2.bitwise_and(color, color, mask=edges)
            
        elif effect == "EdgeOn":
            # Edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        else:
            # Passthrough
            result = frame
        
        process_time = (time.time() - process_start) * 1000
        
        print(f"  ✅ Effect: {effect}")
        print(f"  ✅ Processing time: {process_time:.2f}ms")
        print(f"  ✅ Estimated FPS: {1000/process_time:.1f}")
        
        print()
        return True
    
    def test_throughput(self, num_frames=30):
        """Test pipeline throughput"""
        print(f"🔍 Testing throughput ({num_frames} frames)...")
        
        frame = self.create_test_frame()
        
        start_time = time.time()
        
        for i in range(num_frames):
            # Encode
            success, encoded = cv2.imencode('.jpg', frame)
            if not success:
                print(f"  ❌ Failed to encode frame {i}")
                return False
            
            # Store in Redis
            redis_key = f"demo:throughput_{i:03d}"
            self.redis_client.setex(redis_key, 10, encoded.tobytes())
        
        elapsed = time.time() - start_time
        fps = num_frames / elapsed
        avg_latency = (elapsed / num_frames) * 1000
        
        print(f"  ✅ Processed {num_frames} frames")
        print(f"  ✅ Total time: {elapsed:.2f}s")
        print(f"  ✅ Throughput: {fps:.1f} FPS")
        print(f"  ✅ Avg latency per frame: {avg_latency:.2f}ms")
        
        if fps < 10:
            print(f"  ⚠️  Throughput below 10 FPS target")
        else:
            print(f"  🎉 Good throughput achieved!")
        
        # Cleanup
        for i in range(num_frames):
            self.redis_client.delete(f"demo:throughput_{i:03d}")
        
        print()
        return True
    
    def cleanup(self):
        """Cleanup connections"""
        if self.redis_client:
            self.redis_client.close()
        if self.rabbitmq_connection:
            self.rabbitmq_connection.close()
    
    def run_full_demo(self):
        """Run complete demo"""
        print("="*60)
        print("🚀 OOBLEX PIPELINE DEMO")
        print("="*60)
        print()
        
        if not self.check_dependencies():
            return False
        
        if not self.check_services():
            print("\n💡 TIP: Start services with:")
            print("   docker-compose up -d redis rabbitmq")
            return False
        
        try:
            # Run tests
            if not self.test_redis_pipeline():
                return False
            
            if not self.test_rabbitmq_pipeline():
                return False
            
            # Test different effects
            effects = ["FaceOn", "CartoonOn", "EdgeOn"]
            for effect in effects:
                if not self.test_effect_simulation(effect):
                    return False
            
            if not self.test_throughput():
                return False
            
            print("="*60)
            print("🎉 ALL TESTS PASSED!")
            print("="*60)
            print()
            print("Next steps:")
            print("  1. Run the actual worker: python3 code/brain_simple.py")
            print("  2. Open the web interface: http://localhost:8800")
            print("  3. Start streaming and try effects!")
            print()
            
            return True
            
        finally:
            self.cleanup()
    
    def run_quick_test(self):
        """Run quick validation only"""
        print("🚀 OOBLEX QUICK VALIDATION\n")
        
        if not self.check_dependencies():
            return False
        
        if not self.check_services():
            return False
        
        try:
            if not self.test_redis_pipeline():
                return False
            
            print("✅ Quick validation passed!")
            print("Run full demo with: python3 demo.py")
            print()
            return True
            
        finally:
            self.cleanup()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ooblex Demo Script")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick validation only")
    parser.add_argument("--effect", type=str, 
                       help="Test specific effect (FaceOn, CartoonOn, etc.)")
    parser.add_argument("--redis", default="redis://localhost:6379",
                       help="Redis URL")
    parser.add_argument("--rabbitmq", default="amqp://guest:guest@localhost:5672",
                       help="RabbitMQ URL")
    
    args = parser.parse_args()
    
    demo = OoblexDemo(redis_url=args.redis, rabbitmq_url=args.rabbitmq)
    
    try:
        if args.quick:
            success = demo.run_quick_test()
        elif args.effect:
            if not demo.check_dependencies() or not demo.check_services():
                sys.exit(1)
            success = demo.test_effect_simulation(args.effect)
        else:
            success = demo.run_full_demo()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        demo.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        demo.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
