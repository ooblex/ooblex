"""
gRPC server implementation for ML Worker service
"""

import asyncio
import logging
import uuid
from concurrent import futures
from typing import AsyncIterator

import cv2
import grpc
import numpy as np

# Import generated protobuf files
import ooblex_pb2
import ooblex_pb2_grpc
import torch
from config import settings
from google.protobuf.any_pb2 import Any
from google.protobuf.empty_pb2 import Empty
from google.protobuf.timestamp_pb2 import Timestamp
from grpc_reflection.v1alpha import reflection
from logger import setup_logger
from ml_worker import MLProcessor, ModelManager

logger = setup_logger(__name__)


class MLWorkerServicer(ooblex_pb2_grpc.MLWorkerServiceServicer):
    """gRPC ML Worker service implementation"""

    def __init__(self):
        self.model_manager = ModelManager(cache_size=settings.MODEL_CACHE_SIZE)
        self.processor = MLProcessor(self.model_manager)
        self.worker_id = str(uuid.uuid4())
        self.start_time = Timestamp()
        self.start_time.GetCurrentTime()
        self.active_tasks = 0

    async def LoadModel(self, request, context):
        """Load a model"""
        try:
            model = await self.model_manager.load_model(
                request.model_name, request.model_type or "onnx"
            )

            # Create model info
            model_info = ooblex_pb2.ModelInfo(
                name=request.model_name,
                version="1.0",
                type=request.model_type,
                metadata={"path": request.model_path},
            )

            now = Timestamp()
            now.GetCurrentTime()
            model_info.loaded_at.CopyFrom(now)

            return ooblex_pb2.LoadModelResponse(model=model_info, success=True)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return ooblex_pb2.LoadModelResponse(success=False, error_message=str(e))

    async def UnloadModel(self, request, context):
        """Unload a model"""
        if request.model_name in self.model_manager.models:
            del self.model_manager.models[request.model_name]
            del self.model_manager.model_usage[request.model_name]
            logger.info(f"Unloaded model: {request.model_name}")

        return Empty()

    async def ListModels(self, request, context):
        """List loaded models"""
        models = []

        for name, model in self.model_manager.models.items():
            model_info = ooblex_pb2.ModelInfo(
                name=name, version="1.0", type="unknown"  # TODO: Track model types
            )

            # Set loaded time from usage tracking
            timestamp = Timestamp()
            timestamp.FromSeconds(int(self.model_manager.model_usage.get(name, 0)))
            model_info.loaded_at.CopyFrom(timestamp)

            models.append(model_info)

        return ooblex_pb2.ListModelsResponse(models=models)

    async def ProcessFrame(self, request, context):
        """Process a single frame"""
        self.active_tasks += 1
        start_time = asyncio.get_event_loop().time()

        try:
            # Decode frame
            nparr = np.frombuffer(request.frame.data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

            # Process based on type
            result = None
            metadata = {}

            if request.process_type == ooblex_pb2.PROCESS_TYPE_FACE_SWAP:
                result = await self.processor.process_face_swap(
                    image, model_name=request.model_name or "face_swap"
                )

            elif request.process_type == ooblex_pb2.PROCESS_TYPE_STYLE_TRANSFER:
                result = await self.processor.process_style_transfer(
                    image, model_name=request.model_name or "style_transfer"
                )

            elif request.process_type == ooblex_pb2.PROCESS_TYPE_BACKGROUND_REMOVAL:
                result = await self.processor.process_background_removal(image)

            elif request.process_type == ooblex_pb2.PROCESS_TYPE_OBJECT_DETECTION:
                result, detections = await self.processor.process_object_detection(
                    image
                )

                # Add detections to metadata
                detections_any = Any()
                # TODO: Properly serialize detections
                metadata["detections"] = detections_any

            else:
                raise ValueError(f"Unsupported process type: {request.process_type}")

            # Encode result
            if result.shape[2] == 4:  # RGBA
                _, encoded = cv2.imencode(".png", result)
            else:  # RGB
                _, encoded = cv2.imencode(
                    ".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 90]
                )

            # Create response frame
            processed_frame = ooblex_pb2.Frame(
                data=encoded.tobytes(),
                format="jpeg" if result.shape[2] == 3 else "png",
                width=result.shape[1],
                height=result.shape[0],
            )

            # Set timestamp
            now = Timestamp()
            now.GetCurrentTime()
            processed_frame.timestamp.CopyFrom(now)

            # Calculate processing time
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return ooblex_pb2.ProcessFrameResponse(
                processed_frame=processed_frame,
                metadata=metadata,
                request_id=request.request_id,
                processing_time_ms=processing_time,
                success=True,
            )

        except Exception as e:
            logger.error(f"Frame processing failed: {e}")

            return ooblex_pb2.ProcessFrameResponse(
                request_id=request.request_id, success=False, error_message=str(e)
            )

        finally:
            self.active_tasks -= 1

    async def ProcessFrameStream(
        self, request_iterator: AsyncIterator[ooblex_pb2.ProcessFrameRequest], context
    ) -> AsyncIterator[ooblex_pb2.ProcessFrameResponse]:
        """Process stream of frames"""
        async for request in request_iterator:
            response = await self.ProcessFrame(request, context)
            yield response

    async def GetWorkerStatus(self, request, context):
        """Get worker status"""
        # GPU metrics
        gpu_memory_used = 0
        gpu_memory_total = 0
        gpu_utilization = 0

        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_total = (
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )
            # Note: GPU utilization requires nvidia-ml-py

        return ooblex_pb2.WorkerStatus(
            worker_id=self.worker_id,
            healthy=True,
            loaded_models=len(self.model_manager.models),
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization_percent=gpu_utilization,
            active_tasks=self.active_tasks,
            started_at=self.start_time,
        )


async def serve_grpc(port: int = 50052):
    """Start gRPC server"""
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),  # 100MB
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_pings_without_data", 0),
        ],
    )

    # Add servicer
    servicer = MLWorkerServicer()
    ooblex_pb2_grpc.add_MLWorkerServiceServicer_to_server(servicer, server)

    # Enable reflection
    SERVICE_NAMES = (
        ooblex_pb2.DESCRIPTOR.services_by_name["MLWorkerService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    # Start server
    server.add_insecure_port(f"[::]:{port}")
    await server.start()

    logger.info(f"ML Worker gRPC server started on port {port}")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(5)


# Performance monitoring interceptor
class MetricsInterceptor(grpc.aio.ServerInterceptor):
    """Metrics collection interceptor"""

    def __init__(self):
        from prometheus_client import Counter, Histogram

        self.request_count = Counter(
            "grpc_requests_total", "Total gRPC requests", ["method", "status"]
        )

        self.request_duration = Histogram(
            "grpc_request_duration_seconds", "gRPC request duration", ["method"]
        )

    async def intercept_service(self, continuation, handler_call_details):
        start_time = asyncio.get_event_loop().time()
        method = handler_call_details.method

        try:
            response = await continuation(handler_call_details)
            self.request_count.labels(method=method, status="success").inc()
            return response

        except Exception as e:
            self.request_count.labels(method=method, status="error").inc()
            raise

        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self.request_duration.labels(method=method).observe(duration)
