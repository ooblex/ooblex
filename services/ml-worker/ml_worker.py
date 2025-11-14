"""
Modern ML Worker Service for Ooblex
Handles AI/ML processing tasks with GPU acceleration
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import redis.asyncio as redis
import tensorflow as tf
import torch
from aio_pika import IncomingMessage, connect_robust
from config import settings
from logger import setup_logger
from PIL import Image
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from tenacity import retry, stop_after_attempt, wait_exponential

# Setup logging
logger = setup_logger(__name__)

# Metrics
task_counter = Counter(
    "ml_tasks_processed_total", "Total ML tasks processed", ["task_type", "status"]
)
processing_time = Histogram(
    "ml_task_processing_seconds", "ML task processing time", ["task_type"]
)
model_load_time = Histogram(
    "ml_model_load_seconds", "Model loading time", ["model_name"]
)
gpu_memory_usage = Gauge("gpu_memory_usage_bytes", "GPU memory usage")
active_tasks = Gauge("ml_active_tasks", "Currently processing tasks")

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_selfie_segmentation = mp.solutions.selfie_segmentation


class ModelManager:
    """Manages ML model loading and caching"""

    def __init__(self, cache_size: int = 5):
        self.models: Dict[str, Any] = {}
        self.cache_size = cache_size
        self.model_usage: Dict[str, float] = {}

    async def load_model(self, model_name: str, model_type: str = "onnx") -> Any:
        """Load a model with caching"""
        if model_name in self.models:
            self.model_usage[model_name] = time.time()
            return self.models[model_name]

        # Evict least recently used model if cache is full
        if len(self.models) >= self.cache_size:
            lru_model = min(self.model_usage, key=self.model_usage.get)
            del self.models[lru_model]
            del self.model_usage[lru_model]
            logger.info(f"Evicted model {lru_model} from cache")

        # Load model
        start_time = time.time()
        model_path = os.path.join(settings.MODEL_PATH, f"{model_name}.{model_type}")

        try:
            if model_type == "onnx":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                model = ort.InferenceSession(model_path, providers=providers)
            elif model_type == "torch":
                model = torch.jit.load(model_path)
                model.cuda()
                model.eval()
            elif model_type == "tf":
                model = tf.saved_model.load(model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.models[model_name] = model
            self.model_usage[model_name] = time.time()

            load_time = time.time() - start_time
            model_load_time.labels(model_name=model_name).observe(load_time)
            logger.info(f"Loaded model {model_name} in {load_time:.2f}s")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise


class MLProcessor:
    """Handles ML processing tasks"""

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )

    async def process_face_swap(
        self, image: np.ndarray, model_name: str = "face_swap"
    ) -> np.ndarray:
        """Process face swap transformation"""
        try:
            # Detect face
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)

            if not results.detections:
                logger.warning("No face detected for face swap")
                return image

            # Get face bounding box
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            face_w = int(bbox.width * w)
            face_h = int(bbox.height * h)

            # Extract and resize face
            face = image[y : y + face_h, x : x + face_w]
            face_resized = cv2.resize(face, (256, 256))

            # Load model and process
            model = await self.model_manager.load_model(model_name)

            if isinstance(model, ort.InferenceSession):
                # ONNX model
                input_name = model.get_inputs()[0].name
                face_tensor = face_resized.astype(np.float32) / 255.0
                face_tensor = np.transpose(face_tensor, (2, 0, 1))
                face_tensor = np.expand_dims(face_tensor, axis=0)

                output = model.run(None, {input_name: face_tensor})[0]
                processed_face = output[0]
                processed_face = np.transpose(processed_face, (1, 2, 0))
                processed_face = (processed_face * 255).astype(np.uint8)
            else:
                # PyTorch or TensorFlow model
                # TODO: Implement model-specific processing
                processed_face = face_resized

            # Resize back and replace in original image
            processed_face = cv2.resize(processed_face, (face_w, face_h))
            result = image.copy()
            result[y : y + face_h, x : x + face_w] = processed_face

            return result

        except Exception as e:
            logger.error(f"Face swap processing failed: {e}")
            return image

    async def process_style_transfer(
        self, image: np.ndarray, model_name: str = "style_transfer"
    ) -> np.ndarray:
        """Process style transfer transformation"""
        try:
            # Resize for model input
            input_size = (512, 512)
            original_size = image.shape[:2]
            resized = cv2.resize(image, input_size)

            # Load model and process
            model = await self.model_manager.load_model(model_name)

            if isinstance(model, ort.InferenceSession):
                # ONNX model processing
                input_name = model.get_inputs()[0].name
                input_tensor = resized.astype(np.float32) / 255.0
                input_tensor = np.transpose(input_tensor, (2, 0, 1))
                input_tensor = np.expand_dims(input_tensor, axis=0)

                output = model.run(None, {input_name: input_tensor})[0]
                styled = output[0]
                styled = np.transpose(styled, (1, 2, 0))
                styled = (styled * 255).astype(np.uint8)
            else:
                # Fallback - just return original
                styled = resized

            # Resize back to original size
            result = cv2.resize(styled, (original_size[1], original_size[0]))
            return result

        except Exception as e:
            logger.error(f"Style transfer processing failed: {e}")
            return image

    async def process_background_removal(self, image: np.ndarray) -> np.ndarray:
        """Remove background using MediaPipe"""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.selfie_segmentation.process(rgb_image)

            # Create mask
            mask = results.segmentation_mask
            condition = mask > 0.5

            # Create transparent background
            h, w = image.shape[:2]
            result = np.zeros((h, w, 4), dtype=np.uint8)
            result[:, :, :3] = image
            result[:, :, 3] = condition.astype(np.uint8) * 255

            return result

        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            # Return original image with alpha channel
            h, w = image.shape[:2]
            result = np.zeros((h, w, 4), dtype=np.uint8)
            result[:, :, :3] = image
            result[:, :, 3] = 255
            return result

    async def process_object_detection(
        self, image: np.ndarray, model_name: str = "yolov8"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Detect objects in image"""
        try:
            # For now, using MediaPipe face detection as example
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_image)

            detections = []
            annotated_image = image.copy()

            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    box_w = int(bbox.width * w)
                    box_h = int(bbox.height * h)

                    # Draw bounding box
                    cv2.rectangle(
                        annotated_image, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2
                    )

                    # Add to detections list
                    detections.append(
                        {
                            "class": "face",
                            "confidence": (
                                detection.score[0] if detection.score else 0.0
                            ),
                            "bbox": [x, y, box_w, box_h],
                        }
                    )

            return annotated_image, {"detections": detections}

        except Exception as e:
            logger.error(f"Object detection failed: {e}")
            return image, {"detections": []}


class MLWorker:
    """Main ML Worker class"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        self.model_manager = ModelManager(cache_size=settings.MODEL_CACHE_SIZE)
        self.processor = MLProcessor(self.model_manager)
        self.running = True

    async def setup(self):
        """Initialize connections"""
        # Redis connection
        self.redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,  # We need binary data for images
        )

        # RabbitMQ connection
        self.rabbitmq_connection = await connect_robust(settings.RABBITMQ_URL)
        self.rabbitmq_channel = await self.rabbitmq_connection.channel()
        await self.rabbitmq_channel.set_qos(prefetch_count=1)

        # Declare queue
        self.queue = await self.rabbitmq_channel.declare_queue("tasks", durable=True)

        logger.info("ML Worker initialized successfully")

    async def process_task(self, message: IncomingMessage):
        """Process a single task from the queue"""
        async with message.process():
            active_tasks.inc()
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
                    json.dumps(
                        {
                            "status": "processing",
                            "started_at": datetime.utcnow().isoformat(),
                        }
                    ),
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
                if process_type == "face_swap":
                    result = await self.processor.process_face_swap(
                        image, model_name=parameters.get("model", "face_swap")
                    )
                elif process_type == "style_transfer":
                    result = await self.processor.process_style_transfer(
                        image, model_name=parameters.get("style", "style_transfer")
                    )
                elif process_type == "background_removal":
                    result = await self.processor.process_background_removal(image)
                elif process_type == "object_detection":
                    result, detections = await self.processor.process_object_detection(
                        image
                    )
                    # Store detection results
                    await self.redis_client.setex(
                        f"detections:{stream_token}:latest", 60, json.dumps(detections)
                    )
                else:
                    raise ValueError(f"Unknown process type: {process_type}")

                # Encode result
                if result.shape[2] == 4:  # RGBA
                    _, encoded = cv2.imencode(".png", result)
                else:  # RGB
                    _, encoded = cv2.imencode(
                        ".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )

                # Store processed frame
                result_key = f"frame:{stream_token}:{process_type}"
                await self.redis_client.setex(result_key, 60, encoded.tobytes())

                # Update task status
                processing_duration = time.time() - start_time
                await self.redis_client.setex(
                    f"task:{task_id}",
                    3600,
                    json.dumps(
                        {
                            "status": "completed",
                            "completed_at": datetime.utcnow().isoformat(),
                            "duration": processing_duration,
                            "result_key": result_key,
                        }
                    ),
                )

                # Update metrics
                task_counter.labels(task_type=process_type, status="success").inc()
                processing_time.labels(task_type=process_type).observe(
                    processing_duration
                )

                logger.info(f"Task {task_id} completed in {processing_duration:.2f}s")

            except Exception as e:
                logger.error(f"Task processing failed: {e}")
                task_counter.labels(
                    task_type=task_data.get("process_type", "unknown"), status="error"
                ).inc()

                # Update task status with error
                if "task_id" in locals():
                    await self.redis_client.setex(
                        f"task:{task_id}",
                        3600,
                        json.dumps(
                            {
                                "status": "failed",
                                "error": str(e),
                                "failed_at": datetime.utcnow().isoformat(),
                            }
                        ),
                    )

            finally:
                active_tasks.dec()

    async def update_gpu_metrics(self):
        """Update GPU metrics periodically"""
        while self.running:
            try:
                if torch.cuda.is_available():
                    # Get GPU memory usage
                    allocated = torch.cuda.memory_allocated()
                    gpu_memory_usage.set(allocated)

            except Exception as e:
                logger.error(f"Failed to update GPU metrics: {e}")

            await asyncio.sleep(10)

    async def run(self):
        """Main worker loop"""
        await self.setup()

        # Start metrics server
        start_http_server(9090)
        logger.info("Metrics server started on port 9090")

        # Start GPU metrics updater
        asyncio.create_task(self.update_gpu_metrics())

        # Start consuming tasks
        logger.info("ML Worker started, waiting for tasks...")
        await self.queue.consume(self.process_task)

        try:
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Shutting down ML Worker...")
            self.running = False

    async def cleanup(self):
        """Cleanup resources"""
        if self.rabbitmq_connection:
            await self.rabbitmq_connection.close()
        if self.redis_client:
            await self.redis_client.close()


async def main():
    """Main entry point"""
    worker = MLWorker()

    try:
        await worker.run()
    finally:
        await worker.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
