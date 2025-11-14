#!/usr/bin/env python3
"""
Video Frame Decoder Service for Ooblex
Decodes video streams into individual frames for processing
"""

import asyncio
import io
import json
import logging
import os
import traceback
from datetime import datetime
from typing import Any, Dict, Optional

import aio_pika
import aioredis
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoDecoder:
    """Handles video stream decoding"""

    def __init__(self, redis_url: str, rabbitmq_url: str):
        self.redis_url = redis_url
        self.rabbitmq_url = rabbitmq_url
        self.redis: Optional[aioredis.Redis] = None
        self.rabbitmq_connection: Optional[aio_pika.Connection] = None
        self.channel: Optional[aio_pika.Channel] = None
        self.active_streams: Dict[str, cv2.VideoCapture] = {}

    async def connect(self):
        """Connect to Redis and RabbitMQ"""
        try:
            # Connect to Redis
            self.redis = await aioredis.from_url(self.redis_url)
            logger.info("Connected to Redis")

            # Connect to RabbitMQ
            self.rabbitmq_connection = await aio_pika.connect_robust(self.rabbitmq_url)
            self.channel = await self.rabbitmq_connection.channel()

            # Declare queues
            self.decode_queue = await self.channel.declare_queue(
                "video_decode", durable=True
            )
            self.frame_exchange = await self.channel.declare_exchange(
                "frames", "topic", durable=True
            )

            logger.info("Connected to RabbitMQ")

        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    async def process_decode_requests(self):
        """Process video decode requests from queue"""
        async with self.decode_queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    try:
                        data = json.loads(message.body.decode())
                        await self.handle_decode_request(data)
                    except Exception as e:
                        logger.error(f"Error processing decode request: {e}")
                        logger.error(traceback.format_exc())

    async def handle_decode_request(self, data: Dict[str, Any]):
        """Handle a single decode request"""
        stream_id = data.get("stream_id")
        source = data.get("source")
        action = data.get("action", "start")

        if not stream_id or not source:
            logger.error("Missing stream_id or source in decode request")
            return

        if action == "start":
            await self.start_decoding(stream_id, source, data)
        elif action == "stop":
            await self.stop_decoding(stream_id)

    async def start_decoding(self, stream_id: str, source: str, config: Dict[str, Any]):
        """Start decoding a video stream"""
        if stream_id in self.active_streams:
            logger.warning(f"Stream {stream_id} already active")
            return

        try:
            # Create video capture
            if source.startswith(("rtsp://", "rtmp://", "http://", "https://")):
                cap = cv2.VideoCapture(source)
            elif source.isdigit():
                cap = cv2.VideoCapture(int(source))
            else:
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                logger.error(f"Failed to open video source: {source}")
                return

            # Set capture properties
            fps = config.get("fps", 30)
            width = config.get("width", 640)
            height = config.get("height", 480)

            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            self.active_streams[stream_id] = cap
            logger.info(f"Started decoding stream {stream_id} from {source}")

            # Start frame extraction
            asyncio.create_task(self.extract_frames(stream_id, cap, config))

        except Exception as e:
            logger.error(f"Error starting decoder for {stream_id}: {e}")

    async def stop_decoding(self, stream_id: str):
        """Stop decoding a video stream"""
        if stream_id in self.active_streams:
            cap = self.active_streams[stream_id]
            cap.release()
            del self.active_streams[stream_id]
            logger.info(f"Stopped decoding stream {stream_id}")

    async def extract_frames(
        self, stream_id: str, cap: cv2.VideoCapture, config: Dict[str, Any]
    ):
        """Extract frames from video stream"""
        frame_count = 0
        target_fps = config.get("target_fps", 30)
        frame_interval = 1.0 / target_fps
        last_frame_time = 0

        try:
            while stream_id in self.active_streams:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from stream {stream_id}")
                    await asyncio.sleep(0.1)
                    continue

                current_time = asyncio.get_event_loop().time()

                # Rate limiting
                if current_time - last_frame_time < frame_interval:
                    continue

                last_frame_time = current_time
                frame_count += 1

                # Prepare frame data
                frame_data = {
                    "stream_id": stream_id,
                    "frame_number": frame_count,
                    "timestamp": datetime.utcnow().isoformat(),
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "channels": frame.shape[2] if len(frame.shape) > 2 else 1,
                }

                # Store frame in Redis (with TTL)
                frame_key = f"frame:{stream_id}:{frame_count}"
                await self.redis.setex(frame_key, 5, frame.tobytes())

                # Store frame metadata
                metadata_key = f"frame_meta:{stream_id}:{frame_count}"
                await self.redis.setex(metadata_key, 5, json.dumps(frame_data))

                # Publish frame event
                await self.publish_frame_event(stream_id, frame_count, frame_data)

                # Store latest frame for snapshot access
                await self.redis.set(f"latest_frame:{stream_id}", frame.tobytes())
                await self.redis.set(
                    f"latest_frame_meta:{stream_id}", json.dumps(frame_data)
                )

                # Update stream stats
                await self.update_stream_stats(stream_id, frame_count)

        except Exception as e:
            logger.error(f"Error in frame extraction for {stream_id}: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.stop_decoding(stream_id)

    async def publish_frame_event(
        self, stream_id: str, frame_number: int, metadata: Dict[str, Any]
    ):
        """Publish frame event to RabbitMQ"""
        try:
            message = aio_pika.Message(
                body=json.dumps(
                    {
                        "stream_id": stream_id,
                        "frame_number": frame_number,
                        "metadata": metadata,
                    }
                ).encode(),
                delivery_mode=aio_pika.DeliveryMode.NOT_PERSISTENT,
            )

            await self.frame_exchange.publish(message, routing_key=f"frame.{stream_id}")

        except Exception as e:
            logger.error(f"Error publishing frame event: {e}")

    async def update_stream_stats(self, stream_id: str, frame_count: int):
        """Update stream statistics in Redis"""
        try:
            stats_key = f"stream_stats:{stream_id}"
            await self.redis.hset(
                stats_key,
                mapping={
                    "frame_count": str(frame_count),
                    "last_update": datetime.utcnow().isoformat(),
                    "status": "active",
                },
            )
            await self.redis.expire(stats_key, 300)  # 5 minute TTL

        except Exception as e:
            logger.error(f"Error updating stream stats: {e}")

    async def cleanup_inactive_streams(self):
        """Periodically cleanup inactive streams"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Get all stream stats
                keys = await self.redis.keys("stream_stats:*")

                for key in keys:
                    stats = await self.redis.hgetall(key)
                    if stats:
                        last_update = stats.get(b"last_update")
                        if last_update:
                            last_update_time = datetime.fromisoformat(
                                last_update.decode()
                            )
                            if (datetime.utcnow() - last_update_time).seconds > 300:
                                # Stream inactive for 5 minutes
                                stream_id = key.decode().split(":", 1)[1]
                                await self.stop_decoding(stream_id)

            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    async def run(self):
        """Main run loop"""
        await self.connect()

        # Start background tasks
        asyncio.create_task(self.cleanup_inactive_streams())

        # Process decode requests
        await self.process_decode_requests()


async def main():
    """Main entry point"""
    decoder = VideoDecoder(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://admin:admin@localhost:5672"),
    )

    try:
        await decoder.run()
    except KeyboardInterrupt:
        logger.info("Shutting down decoder service...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
