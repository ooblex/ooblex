#!/usr/bin/env python3
"""
Simple Demo Controller for Ooblex
Coordinates the video processing pipeline for demonstration
"""

import asyncio
import json
import logging
import aioredis
import aio_pika
from datetime import datetime
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoController:
    """Simple controller to demonstrate the video pipeline"""
    
    def __init__(self):
        self.redis_url = "redis://localhost:6379"
        self.rabbitmq_url = "amqp://admin:admin@localhost:5672"
        self.redis = None
        self.rabbitmq_conn = None
        self.channel = None
        
    async def connect(self):
        """Connect to Redis and RabbitMQ"""
        self.redis = await aioredis.from_url(self.redis_url)
        self.rabbitmq_conn = await aio_pika.connect_robust(self.rabbitmq_url)
        self.channel = await self.rabbitmq_conn.channel()
        logger.info("Connected to Redis and RabbitMQ")
        
    async def start_demo_stream(self, stream_id="demo", source="0"):
        """Start a demo video stream"""
        # Send decode request
        decode_queue = await self.channel.declare_queue('video_decode', durable=True)
        
        message = aio_pika.Message(
            body=json.dumps({
                'stream_id': stream_id,
                'source': source,  # 0 for webcam, or URL
                'action': 'start',
                'fps': 30,
                'width': 640,
                'height': 480,
                'target_fps': 15  # Lower FPS for demo
            }).encode()
        )
        
        await self.channel.default_exchange.publish(
            message,
            routing_key='video_decode'
        )
        
        logger.info(f"Started demo stream: {stream_id}")
        
    async def process_frames(self, stream_id="demo", effect="style_transfer"):
        """Process frames with ML effects"""
        # Subscribe to frame events
        frame_exchange = await self.channel.declare_exchange('frames', 'topic', durable=True)
        queue = await self.channel.declare_queue('', exclusive=True)
        await queue.bind(frame_exchange, routing_key=f"frame.{stream_id}")
        
        tasks_queue = await self.channel.declare_queue('tasks', durable=True)
        
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    data = json.loads(message.body.decode())
                    frame_number = data['frame_number']
                    
                    # Create ML processing task
                    task_id = f"{stream_id}-{frame_number}-{effect}"
                    task_message = aio_pika.Message(
                        body=json.dumps({
                            'task_id': task_id,
                            'stream_token': stream_id,
                            'process_type': effect,
                            'parameters': {
                                'style': 'vangogh' if effect == 'style_transfer' else None
                            }
                        }).encode()
                    )
                    
                    await self.channel.default_exchange.publish(
                        task_message,
                        routing_key='tasks'
                    )
                    
                    logger.info(f"Sent frame {frame_number} for {effect} processing")
                    
    async def monitor_output(self, stream_id="demo"):
        """Monitor processed frames"""
        while True:
            # Check for processed frames
            keys = [
                f"frame:{stream_id}:style_transfer",
                f"frame:{stream_id}:face_swap",
                f"frame:{stream_id}:background_removal",
                f"frame:{stream_id}:object_detection"
            ]
            
            for key in keys:
                frame_data = await self.redis.get(key)
                if frame_data:
                    # Publish to MJPEG stream
                    await self.redis.publish(f"processed_frames:{stream_id}", frame_data)
                    logger.info(f"Published processed frame to MJPEG stream: {key}")
                    
            await asyncio.sleep(0.1)  # Check every 100ms
            
    async def create_test_pattern(self):
        """Create a test pattern for demonstration"""
        # Create a simple test pattern
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some colors and text
        cv2.rectangle(frame, (0, 0), (width//2, height//2), (255, 0, 0), -1)
        cv2.rectangle(frame, (width//2, 0), (width, height//2), (0, 255, 0), -1)
        cv2.rectangle(frame, (0, height//2), (width//2, height), (0, 0, 255), -1)
        cv2.rectangle(frame, (width//2, height//2), (width, height), (255, 255, 0), -1)
        
        # Add text
        cv2.putText(frame, "Ooblex Demo", (width//2 - 100, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, datetime.now().strftime("%H:%M:%S"), (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        
    async def run_demo(self):
        """Run the demo pipeline"""
        await self.connect()
        
        # For demo, create test frames if no camera available
        stream_id = "demo"
        
        logger.info("Starting demo pipeline...")
        
        # Option 1: Use test pattern
        logger.info("Generating test pattern frames...")
        for i in range(100):  # Generate 100 frames
            frame_data = await self.create_test_pattern()
            
            # Store frame in Redis
            await self.redis.setex(f"frame:{stream_id}:latest", 5, frame_data)
            await self.redis.setex(f"frame:{stream_id}:{i}", 5, frame_data)
            
            # Publish to MJPEG
            await self.redis.publish(f"processed_frames:{stream_id}", frame_data)
            
            logger.info(f"Generated frame {i}")
            await asyncio.sleep(0.033)  # ~30 FPS
            
        # Option 2: Start real camera/video decode
        # await self.start_demo_stream(stream_id, "0")  # Use webcam
        # await self.process_frames(stream_id, "style_transfer")
        # await self.monitor_output(stream_id)
        
    async def cleanup(self):
        """Cleanup connections"""
        if self.redis:
            await self.redis.close()
        if self.rabbitmq_conn:
            await self.rabbitmq_conn.close()

async def main():
    """Main entry point"""
    controller = DemoController()
    
    try:
        await controller.run_demo()
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    finally:
        await controller.cleanup()

if __name__ == "__main__":
    print("=== Ooblex Demo Controller ===")
    print("This will generate test pattern frames and publish them")
    print("Make sure Redis and RabbitMQ are running!")
    print()
    print("To view the stream:")
    print("1. Run: docker-compose up redis rabbitmq mjpeg")
    print("2. Open: http://localhost:8081/demo.mjpg")
    print()
    print("Press Ctrl+C to stop")
    print()
    
    asyncio.run(main())