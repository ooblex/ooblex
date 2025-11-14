#!/usr/bin/env python3
"""
MJPEG Streaming Server for Ooblex
Provides simple HTTP-based video streaming with processed frames
"""

import asyncio
import io
import logging
from datetime import datetime
from typing import Dict, Optional, Set

import aioredis
import cv2
import numpy as np
from aiohttp import web
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MJPEGStream:
    """Manages a single MJPEG stream"""

    def __init__(self, stream_id: str, quality: int = 85):
        self.stream_id = stream_id
        self.quality = quality
        self.clients: Set[web.StreamResponse] = set()
        self.frame_count = 0
        self.last_frame: Optional[bytes] = None
        self.fps = 30

    async def add_client(self, response: web.StreamResponse):
        """Add a client to this stream"""
        self.clients.add(response)
        logger.info(
            f"Client added to stream {self.stream_id}. Total clients: {len(self.clients)}"
        )

    async def remove_client(self, response: web.StreamResponse):
        """Remove a client from this stream"""
        self.clients.discard(response)
        logger.info(
            f"Client removed from stream {self.stream_id}. Total clients: {len(self.clients)}"
        )

    async def send_frame(self, frame: bytes):
        """Send frame to all connected clients"""
        self.last_frame = frame
        self.frame_count += 1

        boundary = b"--frame\r\n"
        headers = b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(frame)

        disconnected = []
        for client in self.clients:
            try:
                await client.write(boundary + headers + frame + b"\r\n")
            except Exception as e:
                logger.error(f"Error sending frame to client: {e}")
                disconnected.append(client)

        # Remove disconnected clients
        for client in disconnected:
            await self.remove_client(client)


class MJPEGServer:
    """MJPEG HTTP Streaming Server"""

    def __init__(self, redis_url: str = "redis://localhost:6379", port: int = 8081):
        self.redis_url = redis_url
        self.port = port
        self.streams: Dict[str, MJPEGStream] = {}
        self.redis: Optional[aioredis.Redis] = None
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get("/{stream_id}.mjpg", self.handle_stream)
        self.app.router.add_get("/{stream_id}.jpg", self.handle_snapshot)
        self.app.router.add_get("/status", self.handle_status)
        self.app.router.add_get("/", self.handle_index)

    async def start(self):
        """Start the MJPEG server"""
        # Connect to Redis
        self.redis = await aioredis.from_url(self.redis_url)

        # Start frame processor
        asyncio.create_task(self.process_frames())

        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.port)
        await site.start()

        logger.info(f"MJPEG Server started on port {self.port}")

    async def process_frames(self):
        """Process frames from Redis"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("processed_frames:*")

        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    # Extract stream ID from channel name
                    channel = message["channel"].decode("utf-8")
                    stream_id = channel.split(":", 1)[1]

                    # Get or create stream
                    if stream_id not in self.streams:
                        self.streams[stream_id] = MJPEGStream(stream_id)

                    stream = self.streams[stream_id]

                    # Only process if clients are connected
                    if stream.clients:
                        # Get frame data
                        frame_data = message["data"]

                        # Convert to JPEG if needed
                        if not frame_data.startswith(b"\xff\xd8"):
                            # Assume it's raw frame data, convert to JPEG
                            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
                            frame_array = frame_array.reshape(
                                (480, 640, 3)
                            )  # Adjust dimensions
                            _, jpeg_data = cv2.imencode(
                                ".jpg",
                                frame_array,
                                [cv2.IMWRITE_JPEG_QUALITY, stream.quality],
                            )
                            frame_data = jpeg_data.tobytes()

                        # Send to clients
                        await stream.send_frame(frame_data)

                except Exception as e:
                    logger.error(f"Error processing frame: {e}")

    async def handle_stream(self, request: web.Request) -> web.StreamResponse:
        """Handle MJPEG stream request"""
        stream_id = request.match_info["stream_id"]

        # Get or create stream
        if stream_id not in self.streams:
            self.streams[stream_id] = MJPEGStream(stream_id)

        stream = self.streams[stream_id]

        # Create response
        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Pragma": "no-cache",
            },
        )

        await response.prepare(request)
        await stream.add_client(response)

        try:
            # Send last frame immediately if available
            if stream.last_frame:
                boundary = b"--frame\r\n"
                headers = (
                    b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                    % len(stream.last_frame)
                )
                await response.write(boundary + headers + stream.last_frame + b"\r\n")

            # Keep connection alive
            while True:
                await asyncio.sleep(1)
                if response.task.done():
                    break

        except Exception as e:
            logger.error(f"Stream error: {e}")
        finally:
            await stream.remove_client(response)

        return response

    async def handle_snapshot(self, request: web.Request) -> web.Response:
        """Handle single frame snapshot request"""
        stream_id = request.match_info["stream_id"]

        if stream_id in self.streams and self.streams[stream_id].last_frame:
            return web.Response(
                body=self.streams[stream_id].last_frame,
                content_type="image/jpeg",
                headers={"Cache-Control": "no-cache"},
            )
        else:
            # Return placeholder image
            placeholder = self._generate_placeholder(stream_id)
            return web.Response(
                body=placeholder,
                content_type="image/jpeg",
                headers={"Cache-Control": "no-cache"},
            )

    async def handle_status(self, request: web.Request) -> web.Response:
        """Handle status request"""
        status = {
            "server": "MJPEG Server",
            "version": "2.0",
            "uptime": datetime.now().isoformat(),
            "streams": {},
        }

        for stream_id, stream in self.streams.items():
            status["streams"][stream_id] = {
                "clients": len(stream.clients),
                "frames": stream.frame_count,
                "fps": stream.fps,
            }

        return web.json_response(status)

    async def handle_index(self, request: web.Request) -> web.Response:
        """Handle index page"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Ooblex MJPEG Streams</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .stream { margin: 20px 0; border: 1px solid #ccc; padding: 10px; }
                img { max-width: 640px; height: auto; }
                .info { margin: 10px 0; color: #666; }
            </style>
        </head>
        <body>
            <h1>Ooblex MJPEG Streaming Server</h1>
            <p>Available streams:</p>
            <div class="stream">
                <h3>Default Stream</h3>
                <img src="/stream.mjpg" alt="Default Stream">
                <div class="info">URL: http://localhost:8081/stream.mjpg</div>
            </div>
            <div class="stream">
                <h3>How to Use</h3>
                <p>Stream to any path, e.g.:</p>
                <ul>
                    <li>/webcam.mjpg</li>
                    <li>/camera1.mjpg</li>
                    <li>/processed.mjpg</li>
                </ul>
                <p>Get snapshot: /stream.jpg</p>
                <p>Server status: <a href="/status">/status</a></p>
            </div>
        </body>
        </html>
        """
        return web.Response(text=html, content_type="text/html")

    def _generate_placeholder(self, text: str) -> bytes:
        """Generate a placeholder image"""
        # Create a simple placeholder image
        img = Image.new("RGB", (640, 480), color=(50, 50, 50))

        # Convert to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()


async def main():
    """Main entry point"""
    import os

    server = MJPEGServer(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        port=int(os.getenv("MJPEG_PORT", "8081")),
    )

    await server.start()

    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down MJPEG server...")


if __name__ == "__main__":
    asyncio.run(main())
