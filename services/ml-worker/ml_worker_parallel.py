#!/usr/bin/env python3
"""
Parallel ML Worker for Ooblex
Processes video frames from Redis queue with multiple workers
"""
import asyncio
import base64
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np
import onnxruntime as ort
import redis.asyncio as redis
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
frames_processed = Counter(
    "ml_frames_processed_total", "Total frames processed", ["worker_id", "process_type"]
)
processing_time = Histogram(
    "ml_frame_processing_seconds", "Frame processing time", ["process_type"]
)
queue_size = Gauge("ml_queue_size", "Number of frames in queue")
active_workers = Gauge("ml_active_workers", "Number of active workers")

# Configuration
REDIS_URL = "redis://localhost:6379"
FRAME_QUEUE_KEY = "frame_queue"
RESULT_QUEUE_KEY = "result_queue:{}"
MODEL_PATH = "/mnt/c/Users/steve/Code/claude/ooblex/models"
WORKER_COUNT = mp.cpu_count()


class ModelProcessor:
    """Handles model loading and processing"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load ONNX models"""
        model_files = {
            "style_transfer": "style_transfer.onnx",
            "background_blur": "background_blur.onnx",
            "face_swap": "face_swap.onnx",
        }

        for name, filename in model_files.items():
            try:
                path = os.path.join(self.model_path, filename)
                if os.path.exists(path):
                    # Use CPU provider for better parallel performance
                    self.models[name] = ort.InferenceSession(
                        path, providers=["CPUExecutionProvider"]
                    )
                    logger.info(f"Loaded model: {name}")
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")

    def process_frame(self, frame: np.ndarray, process_type: str) -> np.ndarray:
        """Process a single frame"""
        if process_type == "background_blur":
            return self._blur_background(frame)
        elif process_type == "style_transfer":
            return self._apply_style_transfer(frame)
        elif process_type == "face_swap":
            return self._apply_face_swap(frame)
        else:
            # Default: apply simple edge detection
            return self._apply_edge_detection(frame)

    def _blur_background(self, frame: np.ndarray) -> np.ndarray:
        """Apply background blur effect"""
        try:
            if "background_blur" in self.models:
                # Use ONNX model
                model = self.models["background_blur"]
                input_name = model.get_inputs()[0].name

                # Preprocess
                input_size = (256, 256)
                resized = cv2.resize(frame, input_size)
                input_data = resized.astype(np.float32) / 255.0
                input_data = np.transpose(input_data, (2, 0, 1))
                input_data = np.expand_dims(input_data, axis=0)

                # Run inference
                output = model.run(None, {input_name: input_data})[0]
                mask = output[0, 0]

                # Apply blur
                blurred = cv2.GaussianBlur(frame, (21, 21), 0)
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_3d = np.stack([mask_resized] * 3, axis=2)

                result = frame * mask_3d + blurred * (1 - mask_3d)
                return result.astype(np.uint8)
            else:
                # Fallback: simple blur
                return cv2.GaussianBlur(frame, (15, 15), 0)
        except Exception as e:
            logger.error(f"Background blur failed: {e}")
            return frame

    def _apply_style_transfer(self, frame: np.ndarray) -> np.ndarray:
        """Apply style transfer effect"""
        try:
            if "style_transfer" in self.models:
                model = self.models["style_transfer"]
                input_name = model.get_inputs()[0].name

                # Preprocess
                input_size = (256, 256)
                original_size = frame.shape[:2]
                resized = cv2.resize(frame, input_size)
                input_data = resized.astype(np.float32) / 255.0
                input_data = np.transpose(input_data, (2, 0, 1))
                input_data = np.expand_dims(input_data, axis=0)

                # Run inference
                output = model.run(None, {input_name: input_data})[0]
                styled = output[0]
                styled = np.transpose(styled, (1, 2, 0))
                styled = np.clip(styled * 255, 0, 255).astype(np.uint8)

                # Resize back
                result = cv2.resize(styled, (original_size[1], original_size[0]))
                return result
            else:
                # Fallback: color shift effect
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 0] = (hsv[:, :, 0] + 30) % 180
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            logger.error(f"Style transfer failed: {e}")
            return frame

    def _apply_face_swap(self, frame: np.ndarray) -> np.ndarray:
        """Apply face swap effect (simplified)"""
        try:
            # For demo: apply cartoon effect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
            )
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            return cartoon
        except Exception as e:
            logger.error(f"Face swap failed: {e}")
            return frame

    def _apply_edge_detection(self, frame: np.ndarray) -> np.ndarray:
        """Apply edge detection effect"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return result


def process_frame_worker(
    frame_data: Dict[str, Any], model_path: str
) -> Optional[Dict[str, Any]]:
    """Worker function to process a single frame"""
    try:
        # Initialize processor (will be cached per process)
        if not hasattr(process_frame_worker, "processor"):
            process_frame_worker.processor = ModelProcessor(model_path)

        processor = process_frame_worker.processor

        # Decode frame
        frame_bytes = base64.b64decode(frame_data["frame_data"])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process frame
        start_time = time.time()
        processed = processor.process_frame(frame, frame_data["process_type"])
        processing_duration = time.time() - start_time

        # Encode result
        _, buffer = cv2.imencode(".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return {
            "frame_id": frame_data["frame_id"],
            "peer_id": frame_data["peer_id"],
            "frame_data": base64.b64encode(buffer).decode("utf-8"),
            "processing_time": processing_duration,
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.error(f"Frame processing failed: {e}")
        return None


class MLWorkerPool:
    """Manages a pool of ML workers"""

    def __init__(self, worker_count: int = WORKER_COUNT):
        self.worker_count = worker_count
        self.redis_client: Optional[redis.Redis] = None
        self.executor = ProcessPoolExecutor(max_workers=worker_count)
        self.running = True
        self.worker_id = f"worker_{os.getpid()}"

    async def setup(self):
        """Initialize connections"""
        self.redis_client = await redis.from_url(REDIS_URL, decode_responses=False)
        logger.info(f"ML Worker Pool initialized with {self.worker_count} workers")

    async def process_frames(self):
        """Main processing loop"""
        active_workers.set(self.worker_count)

        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                frame_json = await self.redis_client.brpop(FRAME_QUEUE_KEY, timeout=1)

                if frame_json:
                    _, frame_data_raw = frame_json
                    frame_data = json.loads(frame_data_raw)

                    # Update queue size metric
                    current_size = await self.redis_client.llen(FRAME_QUEUE_KEY)
                    queue_size.set(current_size)

                    # Process frame in worker process
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, process_frame_worker, frame_data, MODEL_PATH
                    )

                    if result:
                        # Push result to peer-specific queue
                        result_key = RESULT_QUEUE_KEY.format(result["peer_id"])
                        await self.redis_client.lpush(result_key, json.dumps(result))

                        # Set expiry on result queue
                        await self.redis_client.expire(result_key, 60)

                        # Update metrics
                        frames_processed.labels(
                            worker_id=self.worker_id,
                            process_type=frame_data["process_type"],
                        ).inc()
                        processing_time.labels(
                            process_type=frame_data["process_type"]
                        ).observe(result["processing_time"])

            except Exception as e:
                if self.running:
                    logger.error(f"Error in processing loop: {e}")
                    await asyncio.sleep(0.1)

    async def run(self):
        """Run the worker pool"""
        await self.setup()

        # Start metrics server
        start_http_server(9091)
        logger.info("Metrics server started on port 9091")

        # Run processing
        try:
            await self.process_frames()
        except KeyboardInterrupt:
            logger.info("Shutting down ML Worker Pool")
            self.running = False

    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        if self.redis_client:
            await self.redis_client.close()
        active_workers.set(0)


async def main():
    """Main entry point"""
    # Parse worker count from environment or use default
    worker_count = int(os.environ.get("ML_WORKER_COUNT", WORKER_COUNT))

    worker_pool = MLWorkerPool(worker_count)

    try:
        await worker_pool.run()
    finally:
        await worker_pool.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
