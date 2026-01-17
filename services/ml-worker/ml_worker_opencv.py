"""
OpenCV-based ML Worker Service for Ooblex
Real-time video effects using OpenCV
"""
import asyncio
import json
import logging
import os
import time
import urllib.request
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

import numpy as np
import cv2
import redis.asyncio as redis
from aio_pika import connect_robust, IncomingMessage
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from config import settings
from logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Metrics
task_counter = Counter('ml_tasks_processed_total', 'Total ML tasks processed', ['task_type', 'status'])
processing_time = Histogram('ml_task_processing_seconds', 'ML task processing time', ['task_type'])
active_tasks = Gauge('ml_active_tasks', 'Currently processing tasks')


class OpenCVProcessor:
    """Handles OpenCV-based video effects"""
    
    def __init__(self):
        # Download face detection cascade if not exists
        self.cascade_dir = os.path.join(os.path.dirname(__file__), 'cascades')
        os.makedirs(self.cascade_dir, exist_ok=True)
        
        # Face detection cascade
        self.face_cascade_path = os.path.join(self.cascade_dir, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(self.face_cascade_path):
            self._download_cascade()
            
        self.face_cascade = cv2.CascadeClassifier(self.face_cascade_path)
        
        # Eye detection cascade
        self.eye_cascade_path = os.path.join(self.cascade_dir, 'haarcascade_eye.xml')
        if not os.path.exists(self.eye_cascade_path):
            self._download_eye_cascade()
            
        self.eye_cascade = cv2.CascadeClassifier(self.eye_cascade_path)
        
    def _download_cascade(self):
        """Download Haar cascade for face detection"""
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        logger.info("Downloading face detection cascade...")
        urllib.request.urlretrieve(url, self.face_cascade_path)
        logger.info("Face cascade downloaded successfully")
        
    def _download_eye_cascade(self):
        """Download Haar cascade for eye detection"""
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
        logger.info("Downloading eye detection cascade...")
        urllib.request.urlretrieve(url, self.eye_cascade_path)
        logger.info("Eye cascade downloaded successfully")
        
    async def detect_faces(self, image: np.ndarray) -> Tuple[np.ndarray, list]:
        """Detect faces and draw bounding boxes"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        result = image.copy()
        face_data = []
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Detect eyes within face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = result[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            face_data.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "eyes": [[int(ex), int(ey), int(ew), int(eh)] for ex, ey, ew, eh in eyes]
            })
            
        return result, face_data
        
    async def blur_background(self, image: np.ndarray) -> np.ndarray:
        """Blur background while keeping faces sharp"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            # No faces detected, blur entire image slightly
            return cv2.GaussianBlur(image, (21, 21), 0)
            
        # Create mask for faces
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for (x, y, w, h) in faces:
            # Expand face region slightly
            expand_factor = 0.3
            x_expand = int(w * expand_factor / 2)
            y_expand = int(h * expand_factor / 2)
            
            x1 = max(0, x - x_expand)
            y1 = max(0, y - y_expand)
            x2 = min(image.shape[1], x + w + x_expand)
            y2 = min(image.shape[0], y + h + y_expand)
            
            # Create elliptical mask for more natural look
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            
        # Blur the mask for smooth transition
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (31, 31), 0)
        
        # Combine using mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (image * mask_3channel + blurred * (1 - mask_3channel)).astype(np.uint8)
        
        return result
        
    async def edge_detection(self, image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """Apply Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Convert back to BGR
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Optional: overlay edges on original image
        # result = cv2.addWeighted(image, 0.7, result, 0.3, 0)
        
        return result
        
    async def apply_sepia(self, image: np.ndarray) -> np.ndarray:
        """Apply sepia tone effect"""
        # Sepia transformation matrix
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        result = cv2.transform(image, kernel)
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
        
    async def convert_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert to grayscale"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert back to 3 channels for consistency
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return result
        
    async def cartoon_effect(self, image: np.ndarray) -> np.ndarray:
        """Apply cartoon effect"""
        # Convert to gray scale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply median blur to reduce noise
        gray = cv2.medianBlur(gray, 5)
        
        # Detect edges
        edges = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 10)
        
        # Convert back to color
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Apply bilateral filter to smooth the image
        smooth = cv2.bilateralFilter(image, 15, 80, 80)
        
        # Combine edges and smooth image
        result = cv2.bitwise_and(smooth, edges)
        
        return result
        
    async def apply_vintage(self, image: np.ndarray) -> np.ndarray:
        """Apply vintage effect"""
        # Add noise
        noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        
        # Apply vignette effect
        rows, cols = image.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        
        # Apply mask to each channel
        result = np.zeros_like(noisy, dtype=np.float32)
        for i in range(3):
            result[:,:,i] = noisy[:,:,i] * mask
            
        # Adjust color channels for vintage look
        result[:,:,0] = result[:,:,0] * 0.7  # Reduce blue
        result[:,:,1] = result[:,:,1] * 0.9  # Slightly reduce green
        result[:,:,2] = np.minimum(result[:,:,2] * 1.1, 255)  # Boost red
        
        return result.astype(np.uint8)
        
    async def apply_blur(self, image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Apply Gaussian blur"""
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
    async def apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter"""
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        
        result = cv2.filter2D(image, -1, kernel)
        return result
        
    async def pixelate(self, image: np.ndarray, pixel_size: int = 10) -> np.ndarray:
        """Pixelate effect"""
        # Downscale
        height, width = image.shape[:2]
        temp = cv2.resize(image, (width//pixel_size, height//pixel_size), 
                         interpolation=cv2.INTER_LINEAR)
        
        # Upscale back
        result = cv2.resize(temp, (width, height), 
                           interpolation=cv2.INTER_NEAREST)
        
        return result
        
    async def apply_emboss(self, image: np.ndarray) -> np.ndarray:
        """Apply emboss effect"""
        kernel = np.array([[-2,-1,0],
                          [-1, 1,1],
                          [ 0, 1,2]])
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        embossed = cv2.filter2D(gray, -1, kernel)
        embossed = cv2.cvtColor(embossed, cv2.COLOR_GRAY2BGR)
        
        return embossed


class MLWorker:
    """Main ML Worker class with OpenCV effects"""
    
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
            decode_responses=False  # We need binary data for images
        )
        
        # RabbitMQ connection
        self.rabbitmq_connection = await connect_robust(settings.rabbitmq_url)
        self.rabbitmq_channel = await self.rabbitmq_connection.channel()
        await self.rabbitmq_channel.set_qos(prefetch_count=1)
        
        # Declare queue
        self.queue = await self.rabbitmq_channel.declare_queue("tasks", durable=True)
        
        logger.info("ML Worker (OpenCV) initialized successfully")
        
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
                if process_type == "face_detection":
                    result, face_data = await self.processor.detect_faces(image)
                    # Store detection results
                    await self.redis_client.setex(
                        f"detections:{stream_token}:latest",
                        60,
                        json.dumps({"faces": face_data})
                    )
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
                    # Pass-through
                    result = image
                else:
                    # Unknown effect - log warning and return original
                    logger.warning(f"Unknown process type: {process_type}, returning original image")
                    result = image
                    
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
                
                # Update metrics
                task_counter.labels(task_type=process_type, status="success").inc()
                processing_time.labels(task_type=process_type).observe(processing_duration)
                
                logger.info(f"Task {task_id} completed in {processing_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Task processing failed: {e}", exc_info=True)
                task_counter.labels(task_type=task_data.get("process_type", "unknown"), status="error").inc()
                
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
                    
            finally:
                active_tasks.dec()
                
    async def run(self):
        """Main worker loop"""
        await self.setup()
        
        # Start metrics server
        start_http_server(9090)
        logger.info("Metrics server started on port 9090")
        
        # Start consuming tasks
        logger.info("ML Worker (OpenCV) started, waiting for tasks...")
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