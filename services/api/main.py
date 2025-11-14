"""
Ooblex API Gateway - Simplified Modern Implementation
Based on original api.py (120 lines) - keeps core WebSocket + RabbitMQ functionality
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import redis.asyncio as redis
from aio_pika import DeliveryMode, Message, connect_robust
from config import settings
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from logger import setup_logger
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# Setup logging
logger = setup_logger(__name__)

# Metrics
request_count = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
request_duration = Histogram(
    "api_request_duration_seconds", "API request duration", ["method", "endpoint"]
)
websocket_connections = Counter(
    "websocket_connections_total", "Total WebSocket connections"
)
active_connections = Counter(
    "websocket_active_connections", "Active WebSocket connections"
)

# Global connections
redis_client: Optional[redis.Redis] = None
rabbitmq_connection = None
rabbitmq_channel = None


# Models
class ProcessRequest(BaseModel):
    stream_token: str
    process_type: str = Field(
        ...,
        pattern="^(face_swap|style_transfer|object_detection|background_removal|trump|taylor)$",
    )
    model_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}


class ProcessResponse(BaseModel):
    task_id: str
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global redis_client, rabbitmq_connection, rabbitmq_channel

    logger.info("Starting Ooblex API Gateway")

    # Initialize Redis
    try:
        redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,  # Keep as bytes for image data
        )
        await redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise

    # Initialize RabbitMQ
    try:
        rabbitmq_connection = await connect_robust(settings.RABBITMQ_URL)
        rabbitmq_channel = await rabbitmq_connection.channel()

        # Declare queues (matching original setup)
        await rabbitmq_channel.declare_queue("gst-launcher", durable=True)
        await rabbitmq_channel.declare_queue("tf-task", durable=True)
        await rabbitmq_channel.declare_queue("broadcast-all", durable=True)

        logger.info("Connected to RabbitMQ")
    except Exception as e:
        logger.error(f"Failed to connect to RabbitMQ: {e}")
        raise

    yield

    # Cleanup
    logger.info("Shutting down Ooblex API Gateway")
    if redis_client:
        await redis_client.close()
    if rabbitmq_connection:
        await rabbitmq_connection.close()


# Initialize FastAPI app
app = FastAPI(
    title="Ooblex API",
    description="Real-time AI video processing platform",
    version="2.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Simplified for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Ooblex API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}

    # Check Redis
    try:
        await redis_client.ping()
        services["redis"] = "healthy"
    except Exception:
        services["redis"] = "unhealthy"

    # Check RabbitMQ
    try:
        if rabbitmq_channel and not rabbitmq_channel.is_closed:
            services["rabbitmq"] = "healthy"
        else:
            services["rabbitmq"] = "unhealthy"
    except Exception:
        services["rabbitmq"] = "unhealthy"

    overall_status = (
        "healthy" if all(s == "healthy" for s in services.values()) else "degraded"
    )

    return HealthResponse(
        status=overall_status, timestamp=datetime.utcnow(), services=services
    )


@app.post("/process", response_model=ProcessResponse)
async def start_process(request: ProcessRequest):
    """Start a new processing task - matches original API pattern"""
    # Generate task ID
    task_id = f"{request.stream_token}_{request.process_type}_{int(datetime.utcnow().timestamp())}"

    # Create task message (matching original brain.py format)
    task_data = {
        "streamKey": request.stream_token,
        "task": request.process_type,
        "redisID": request.parameters.get("redis_id", ""),
        "timestamp": datetime.utcnow().isoformat(),
    }

    # Send to RabbitMQ (tf-task queue like original)
    try:
        message = Message(
            body=json.dumps(task_data).encode(),
            delivery_mode=DeliveryMode.PERSISTENT,
            content_type="application/json",
        )
        await rabbitmq_channel.default_exchange.publish(message, routing_key="tf-task")

        logger.info(f"Task {task_id} queued for processing")

        return ProcessResponse(
            task_id=task_id, status="queued", message="Task queued for processing"
        )
    except Exception as e:
        logger.error(f"Failed to queue task: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue task")


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


# WebSocket manager (simplified from original)
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.stream_tokens: Dict[str, str] = {}  # Map stream_token to client_id

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        active_connections.inc()
        websocket_connections.inc()
        logger.info(
            f"Client {client_id} connected. Total: {len(self.active_connections)}"
        )

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            active_connections.dec()
            # Clean up stream token mapping
            for token, cid in list(self.stream_tokens.items()):
                if cid == client_id:
                    del self.stream_tokens[token]
            logger.info(
                f"Client {client_id} disconnected. Remaining: {len(self.active_connections)}"
            )

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Failed to send message to {client_id}: {e}")
                self.disconnect(client_id)

    async def send_to_stream(self, message: str, stream_token: str):
        """Send message to client subscribed to stream_token"""
        client_id = self.stream_tokens.get(stream_token)
        if client_id:
            await self.send_personal_message(message, client_id)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint - matches original api.py pattern

    Expected messages:
    - "start process:{streamKey}" - Start video processing
    - "{task}:{streamKey}" - Queue specific ML task
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info(f"Received from {client_id}: {data}")

            # Handle "start process" command (original pattern)
            if data.startswith("start process:"):
                parts = data.split("start process:")
                if len(parts) != 2:
                    await manager.send_personal_message(
                        "No streamKey provided", client_id
                    )
                    continue

                stream_key = parts[1]
                manager.stream_tokens[stream_key] = client_id

                await manager.send_personal_message(
                    "Starting Video Processing", client_id
                )

                # Send to gst-launcher queue (original pattern)
                try:
                    message = Message(
                        body=stream_key.encode(), delivery_mode=DeliveryMode.PERSISTENT
                    )
                    await rabbitmq_channel.default_exchange.publish(
                        message, routing_key="gst-launcher"
                    )
                    await manager.send_personal_message("...", client_id)
                    await manager.send_personal_message("!", client_id)
                except Exception as e:
                    logger.error(f"Failed to publish to gst-launcher: {e}")
                    await manager.send_personal_message(f"Error: {e}", client_id)

            # Handle task commands (original pattern: "task:streamKey")
            elif ":" in data:
                parts = data.split(":", 1)
                if len(parts) != 2:
                    await manager.send_personal_message(
                        "Invalid format. Use task:streamKey", client_id
                    )
                    continue

                task = parts[0]
                stream_key = parts[1]

                # Create task message (original format)
                try:
                    task_msg = {"streamKey": stream_key, "task": task}
                    message = Message(
                        body=json.dumps(task_msg).encode(),
                        delivery_mode=DeliveryMode.PERSISTENT,
                        content_type="application/json",
                    )
                    await rabbitmq_channel.default_exchange.publish(
                        message,
                        routing_key=stream_key,  # Original routes to streamKey queue
                    )
                    logger.info(f"Task {task} queued for stream {stream_key}")
                except Exception as e:
                    logger.error(f"Failed to queue task: {e}")
                    await manager.send_personal_message(f"Error: {e}", client_id)
            else:
                # Echo or handle other messages
                await manager.send_personal_message(
                    f"Unknown command: {data}", client_id
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")


# Background task to listen to broadcast-all queue and send to clients
async def rabbitmq_consumer():
    """Listen to broadcast-all queue and send messages to WebSocket clients"""
    global rabbitmq_channel

    if not rabbitmq_channel:
        logger.error("RabbitMQ channel not initialized")
        return

    # Declare and consume from broadcast-all queue
    queue = await rabbitmq_channel.declare_queue("broadcast-all", durable=True)

    async with queue.iterator() as queue_iter:
        async for message in queue_iter:
            async with message.process():
                try:
                    data = json.loads(message.body.decode())
                    stream_key = data.get("key", "")
                    msg = data.get("msg", "")

                    # Send to appropriate client
                    if stream_key:
                        await manager.send_to_stream(msg, stream_key)
                        logger.debug(f"Sent message to stream {stream_key}")
                except Exception as e:
                    logger.error(f"Error processing broadcast message: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    # Start RabbitMQ consumer in background
    asyncio.create_task(rabbitmq_consumer())
    logger.info("Started RabbitMQ consumer task")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8800, ssl_keyfile=None, ssl_certfile=None)
