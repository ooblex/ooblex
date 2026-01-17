"""
Simple ML Worker with only OpenCV dependencies
For testing and development without heavy ML frameworks
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

import numpy as np
import cv2
import redis.asyncio as redis
from aio_pika import connect_robust, IncomingMessage

from config import settings
from logger import setup_logger
from ml_worker_opencv import OpenCVProcessor

# Setup logging
logger = setup_logger(__name__)


class SimpleMLWorker:
    """Simplified ML Worker using only OpenCV"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        self.processor = OpenCVProcessor()
        self.running = True
        
    async def setup(self):
        """Initialize connections"""
        # Redis connection
        self.redis_client = await redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=False
        )
        
        # RabbitMQ connection
        self.rabbitmq_connection = await connect_robust(settings.rabbitmq_url)
        self.rabbitmq_channel = await self.rabbitmq_connection.channel()
        await self.rabbitmq_channel.set_qos(prefetch_count=1)
        
        # Declare queue
        self.queue = await self.rabbitmq_channel.declare_queue("tasks", durable=True)
        
        logger.info("Simple ML Worker initialized successfully")
        
    async def process_task(self, message: IncomingMessage):
        """Process a single task from the queue"""
        async with message.process():
            start_time = time.time()
            
            try:
                # Parse task data
                task_data = json.loads(message.body.decode())
                task_id = task_data["task_id"]
                stream_token = task_data["stream_token"]
                process_type = task_data["process_type"]
                parameters = task_data.get("parameters", {})
                
                logger.info(f"Processing task {task_id} - Type: {process_type}")
                
                # Update task status
                await self.redis_client.setex(
                    f"task:{task_id}",
                    3600,
                    json.dumps({"status": "processing", "started_at": datetime.utcnow().isoformat()})
                )
                
                # Get input frame from Redis
                frame_key = f"frame:{stream_token}:latest"
                frame_data = await self.redis_client.get(frame_key)
                
                if not frame_data:
                    raise ValueError(f"No frame data found for stream {stream_token}")
                    
                # Decode image
                nparr = np.frombuffer(frame_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Process based on type
                result = image  # Default to original
                metadata = {}
                
                if process_type == "face_detection":
                    result, face_data = await self.processor.detect_faces(image)
                    metadata = {"faces": face_data}
                elif process_type == "background_blur":
                    result = await self.processor.blur_background(image)
                elif process_type == "edge_detection":
                    result = await self.processor.edge_detection(
                        image,
                        low_threshold=parameters.get("low_threshold", 50),
                        high_threshold=parameters.get("high_threshold", 150)
                    )
                elif process_type == "sepia":
                    result = await self.processor.apply_sepia(image)
                elif process_type == "grayscale":
                    result = await self.processor.convert_grayscale(image)
                elif process_type == "cartoon":
                    result = await self.processor.cartoon_effect(image)
                elif process_type == "vintage":
                    result = await self.processor.apply_vintage(image)
                elif process_type == "blur":
                    result = await self.processor.apply_blur(
                        image,
                        kernel_size=parameters.get("kernel_size", 15)
                    )
                elif process_type == "sharpen":
                    result = await self.processor.apply_sharpen(image)
                elif process_type == "pixelate":
                    result = await self.processor.pixelate(
                        image,
                        pixel_size=parameters.get("pixel_size", 10)
                    )
                elif process_type == "emboss":
                    result = await self.processor.apply_emboss(image)
                elif process_type == "none":
                    result = image
                else:
                    logger.warning(f"Unknown process type: {process_type}, returning original")
                    
                # Store metadata if any
                if metadata:
                    await self.redis_client.setex(
                        f"metadata:{stream_token}:{process_type}",
                        60,
                        json.dumps(metadata)
                    )
                    
                # Encode result
                _, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    
                # Store processed frame
                result_key = f"frame:{stream_token}:{process_type}"
                await self.redis_client.setex(result_key, 60, encoded.tobytes())
                
                # Update task status
                processing_duration = time.time() - start_time
                await self.redis_client.setex(
                    f"task:{task_id}",
                    3600,
                    json.dumps({
                        "status": "completed",
                        "completed_at": datetime.utcnow().isoformat(),
                        "duration": processing_duration,
                        "result_key": result_key
                    })
                )
                
                logger.info(f"Task {task_id} completed in {processing_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Task processing failed: {e}", exc_info=True)
                
                # Update task status with error
                if 'task_id' in locals():
                    await self.redis_client.setex(
                        f"task:{task_id}",
                        3600,
                        json.dumps({
                            "status": "failed",
                            "error": str(e),
                            "failed_at": datetime.utcnow().isoformat()
                        })
                    )
                
    async def run(self):
        """Main worker loop"""
        await self.setup()
        
        # Start consuming tasks
        logger.info("Simple ML Worker started, waiting for tasks...")
        await self.queue.consume(self.process_task)
        
        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Shutting down Simple ML Worker...")
            self.running = False
            
    async def cleanup(self):
        """Cleanup resources"""
        if self.rabbitmq_connection:
            await self.rabbitmq_connection.close()
        if self.redis_client:
            await self.redis_client.close()


async def main():
    """Main entry point"""
    worker = SimpleMLWorker()
    
    try:
        await worker.run()
    finally:
        await worker.cleanup()


if __name__ == "__main__":
    asyncio.run(main())