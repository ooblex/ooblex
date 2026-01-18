#!/usr/bin/env python3
"""
Async Multi-Process ML Worker with MediaPipe Effects

This worker provides low-latency frame processing using:
- Redis pub/sub for instant frame notifications
- Async I/O for non-blocking operations
- Multiple worker processes for parallel processing
- Effect chaining for composing multiple effects

Architecture:
    Frame Producer (decoder/webrtc)
           |
           v
    Redis (frame storage + pub/sub notification)
           |
           v
    ML Worker Pool (N processes)
           |
           v
    Redis (result storage + pub/sub notification)
           |
           v
    Stream Server (mjpeg/webrtc)

Usage:
    python ml_worker_mediapipe.py
    ML_WORKER_COUNT=4 python ml_worker_mediapipe.py

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    RABBITMQ_URL: RabbitMQ connection URL (default: amqp://guest:guest@localhost:5672)
    ML_WORKER_COUNT: Number of worker processes (default: CPU count)
    MODEL_CACHE_DIR: Directory for model downloads (default: /tmp/ooblex_models)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
import cv2
import redis.asyncio as aioredis
import aio_pika
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Add code directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../code'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class WorkerConfig:
    redis_url: str = os.getenv('REDIS_URL', 'redis://localhost:6379')
    rabbitmq_url: str = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672')
    worker_count: int = int(os.getenv('ML_WORKER_COUNT', mp.cpu_count()))
    model_cache_dir: str = os.getenv('MODEL_CACHE_DIR', '/tmp/ooblex_models')
    metrics_port: int = int(os.getenv('METRICS_PORT', '9092'))

    # Performance tuning
    frame_ttl: int = 30  # seconds
    result_ttl: int = 30  # seconds
    max_queue_size: int = 100
    batch_size: int = 1  # Process frames one at a time for lowest latency

    # Pub/sub channels
    frame_channel: str = "ooblex:frames:incoming"
    result_channel: str = "ooblex:frames:processed"


config = WorkerConfig()

# =============================================================================
# PROMETHEUS METRICS
# =============================================================================

frames_processed = Counter(
    'ml_frames_processed_total',
    'Total frames processed',
    ['worker_id', 'effect']
)

processing_time = Histogram(
    'ml_frame_processing_seconds',
    'Frame processing time in seconds',
    ['effect'],
    buckets=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
)

chain_processing_time = Histogram(
    'ml_chain_processing_seconds',
    'Effect chain processing time in seconds',
    ['chain_length'],
    buckets=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)

queue_size = Gauge('ml_queue_size', 'Number of frames waiting to be processed')
active_workers = Gauge('ml_active_workers', 'Number of active worker processes')
effect_usage = Counter('ml_effect_usage_total', 'Effect usage count', ['effect'])

# =============================================================================
# FRAME PROCESSOR (runs in worker processes)
# =============================================================================

# Global processor instance per process
_processor = None


def get_processor():
    """Get or create the MediaPipe processor for this process"""
    global _processor
    if _processor is None:
        from brain_mediapipe import MediaPipeEffects
        _processor = MediaPipeEffects()
        logger.info(f"Initialized MediaPipeEffects in process {os.getpid()}")
    return _processor


def process_frame_in_worker(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single frame in a worker process

    Args:
        task: Dictionary containing:
            - frame_data: Base64 or bytes of JPEG image
            - effect: Effect name or pipe-separated chain
            - stream_key: Stream identifier
            - frame_id: Unique frame ID
            - timestamp: Frame capture timestamp

    Returns:
        Dictionary with processed frame data and metadata
    """
    try:
        start_time = time.time()

        # Get processor
        processor = get_processor()

        # Decode frame
        if isinstance(task.get('frame_data'), str):
            import base64
            frame_bytes = base64.b64decode(task['frame_data'])
        else:
            frame_bytes = task['frame_data']

        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.warning(f"Could not decode frame: {task.get('frame_id')}")
            return None

        # Apply effect(s)
        effect_spec = task.get('effect', 'face_mesh')

        # Import and apply effect
        from brain_mediapipe import apply_effect
        processed = apply_effect(frame, effect_spec)

        # Encode result
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, encoded = cv2.imencode('.jpg', processed, encode_params)

        if not success:
            logger.warning("Could not encode processed frame")
            return None

        processing_time_ms = (time.time() - start_time) * 1000

        return {
            'frame_id': task.get('frame_id'),
            'stream_key': task.get('stream_key'),
            'effect': effect_spec,
            'frame_data': encoded.tobytes(),
            'processing_time_ms': processing_time_ms,
            'timestamp': time.time(),
            'original_timestamp': task.get('timestamp'),
            'worker_pid': os.getpid()
        }

    except Exception as e:
        logger.error(f"Frame processing error: {e}", exc_info=True)
        return None


# =============================================================================
# ASYNC WORKER POOL
# =============================================================================

class AsyncMLWorkerPool:
    """
    Manages async frame processing with multiple worker processes

    Features:
    - Redis pub/sub for low-latency notifications
    - Process pool for parallel CPU-bound work
    - Automatic load balancing
    - Graceful shutdown
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.rabbitmq_connection: Optional[aio_pika.Connection] = None
        self.rabbitmq_channel: Optional[aio_pika.Channel] = None
        self.executor: Optional[ProcessPoolExecutor] = None
        self.running = False
        self.worker_id = f"ml-worker-{os.getpid()}"
        self._shutdown_event = asyncio.Event()

    async def connect(self):
        """Initialize connections to Redis and RabbitMQ"""
        logger.info("Connecting to services...")

        # Connect to Redis
        self.redis = await aioredis.from_url(
            self.config.redis_url,
            decode_responses=False,
            max_connections=20
        )
        logger.info(f"Connected to Redis: {self.config.redis_url}")

        # Connect to RabbitMQ
        for attempt in range(5):
            try:
                self.rabbitmq_connection = await aio_pika.connect_robust(
                    self.config.rabbitmq_url
                )
                self.rabbitmq_channel = await self.rabbitmq_connection.channel()
                await self.rabbitmq_channel.set_qos(prefetch_count=1)
                logger.info(f"Connected to RabbitMQ: {self.config.rabbitmq_url}")
                break
            except Exception as e:
                logger.warning(f"RabbitMQ connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2)
        else:
            logger.warning("Could not connect to RabbitMQ, continuing with Redis only")

        # Initialize process pool
        self.executor = ProcessPoolExecutor(
            max_workers=self.config.worker_count,
            initializer=self._worker_init
        )
        active_workers.set(self.config.worker_count)
        logger.info(f"Initialized {self.config.worker_count} worker processes")

    @staticmethod
    def _worker_init():
        """Initialize each worker process"""
        # Pre-load MediaPipe models
        get_processor()

    async def process_from_redis(self, redis_id: str, task_data: Dict) -> Optional[str]:
        """
        Process a frame from Redis storage

        Args:
            redis_id: Redis key containing the frame data
            task_data: Task metadata (effect, stream_key, etc.)

        Returns:
            Redis key of the processed result
        """
        try:
            # Get frame from Redis
            frame_data = await self.redis.get(redis_id)
            if frame_data is None:
                logger.warning(f"Frame not found in Redis: {redis_id}")
                return None

            # Build task
            task = {
                'frame_data': frame_data,
                'frame_id': redis_id,
                'effect': task_data.get('effect', task_data.get('task', 'face_mesh')),
                'stream_key': task_data.get('streamKey', task_data.get('stream_key', '')),
                'timestamp': task_data.get('timestamp', time.time())
            }

            # Process in worker
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                process_frame_in_worker,
                task
            )

            if result is None:
                return None

            # Store result in Redis
            result_id = f"processed_{redis_id}"
            await self.redis.setex(
                result_id,
                self.config.result_ttl,
                result['frame_data']
            )

            # Update metrics
            effect = task['effect']
            processing_time.labels(effect=effect).observe(
                result['processing_time_ms'] / 1000
            )
            frames_processed.labels(
                worker_id=self.worker_id,
                effect=effect
            ).inc()

            # Track effect usage
            if '|' in effect or '+' in effect:
                effects = effect.replace('+', '|').split('|')
                chain_processing_time.labels(chain_length=str(len(effects))).observe(
                    result['processing_time_ms'] / 1000
                )
                for e in effects:
                    effect_usage.labels(effect=e.strip()).inc()
            else:
                effect_usage.labels(effect=effect).inc()

            # Publish notification
            await self.redis.publish(
                self.config.result_channel,
                json.dumps({
                    'result_id': result_id,
                    'stream_key': result['stream_key'],
                    'effect': effect,
                    'processing_time_ms': result['processing_time_ms'],
                    'timestamp': result['timestamp']
                })
            )

            logger.debug(
                f"Processed {redis_id} with {effect} "
                f"in {result['processing_time_ms']:.1f}ms"
            )

            return result_id

        except Exception as e:
            logger.error(f"Error processing frame from Redis: {e}", exc_info=True)
            return None

    async def handle_rabbitmq_task(self, message: aio_pika.IncomingMessage):
        """Handle incoming task from RabbitMQ"""
        async with message.process():
            try:
                task_data = json.loads(message.body)
                redis_id = task_data.get('redisID', task_data.get('redis_id', ''))

                if not redis_id:
                    logger.warning("Task missing redis_id")
                    return

                result_id = await self.process_from_redis(redis_id, task_data)

                # Send completion message via RabbitMQ broadcast
                if self.rabbitmq_channel and result_id:
                    stream_key = task_data.get('streamKey', '')
                    effect = task_data.get('effect', task_data.get('task', ''))

                    response = json.dumps({
                        'msg': f"Processed with {effect}",
                        'key': stream_key,
                        'result_id': result_id
                    })

                    await self.rabbitmq_channel.default_exchange.publish(
                        aio_pika.Message(body=response.encode()),
                        routing_key='broadcast-all'
                    )

            except Exception as e:
                logger.error(f"Error handling RabbitMQ task: {e}", exc_info=True)

    async def handle_redis_pubsub(self):
        """Listen for frame notifications via Redis pub/sub"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self.config.frame_channel)
        logger.info(f"Subscribed to Redis channel: {self.config.frame_channel}")

        async for message in pubsub.listen():
            if not self.running:
                break

            if message['type'] == 'message':
                try:
                    task_data = json.loads(message['data'])
                    redis_id = task_data.get('redis_id', task_data.get('redisID', ''))

                    if redis_id:
                        await self.process_from_redis(redis_id, task_data)

                except Exception as e:
                    logger.error(f"Error handling pub/sub message: {e}")

        await pubsub.unsubscribe(self.config.frame_channel)

    async def consume_rabbitmq_queue(self):
        """Consume tasks from RabbitMQ queue"""
        if not self.rabbitmq_channel:
            logger.info("RabbitMQ not connected, skipping queue consumption")
            return

        # Declare queue
        queue = await self.rabbitmq_channel.declare_queue(
            "tf-task",
            durable=False,
            arguments={'x-message-ttl': 10000}
        )

        logger.info("Consuming from RabbitMQ queue: tf-task")

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                if not self.running:
                    break
                await self.handle_rabbitmq_task(message)

    async def run(self):
        """Main run loop"""
        self.running = True

        # Connect to services
        await self.connect()

        # Start metrics server
        start_http_server(self.config.metrics_port)
        logger.info(f"Metrics server started on port {self.config.metrics_port}")

        logger.info("=" * 60)
        logger.info("MediaPipe ML Worker Pool Started")
        logger.info(f"  Workers: {self.config.worker_count}")
        logger.info(f"  Redis: {self.config.redis_url}")
        logger.info(f"  Metrics: http://localhost:{self.config.metrics_port}")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Supported Effects (can be chained with | or +):")
        logger.info("  Background: background_blur, background_remove, background_replace")
        logger.info("  Face Mesh:  face_mesh, face_contour")
        logger.info("  Face FX:    big_eyes, face_slim, face_distort")
        logger.info("  Style:      anime_style, beauty_filter, virtual_makeup")
        logger.info("  Classic:    cartoon, edge_detection, sepia, grayscale")
        logger.info("")
        logger.info("Example chains:")
        logger.info("  background_blur|big_eyes|beauty_filter")
        logger.info("  background_remove+anime_style")
        logger.info("=" * 60)

        # Run both Redis pub/sub and RabbitMQ consumers
        tasks = [
            asyncio.create_task(self.handle_redis_pubsub()),
        ]

        if self.rabbitmq_channel:
            tasks.append(asyncio.create_task(self.consume_rabbitmq_queue()))

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Cancel tasks
        for task in tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down ML Worker Pool...")
        self.running = False
        self._shutdown_event.set()

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            active_workers.set(0)

        # Close connections
        if self.redis:
            await self.redis.close()

        if self.rabbitmq_connection:
            await self.rabbitmq_connection.close()

        logger.info("Shutdown complete")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point"""
    worker_pool = AsyncMLWorkerPool(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(worker_pool.shutdown())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await worker_pool.run()
    except Exception as e:
        logger.error(f"Worker pool error: {e}", exc_info=True)
    finally:
        await worker_pool.shutdown()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║         Ooblex MediaPipe ML Worker                        ║
    ║         Real-time AI Video Effects                        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    asyncio.run(main())
