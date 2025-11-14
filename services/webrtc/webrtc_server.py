#!/usr/bin/env python3
"""
Real WebRTC Server for Ooblex
Handles WebRTC signaling, video streaming, and frame processing
"""
import asyncio
import base64
import io
import json
import logging
import ssl
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Set

import aiohttp
import aiohttp_cors
import av
import cv2
import numpy as np
import redis.asyncio as redis
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCDataChannel,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from prometheus_client import Counter, Gauge, Histogram

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
active_connections = Gauge("webrtc_active_connections", "Active WebRTC connections")
frames_processed = Counter("webrtc_frames_processed_total", "Total frames processed")
frame_processing_time = Histogram(
    "webrtc_frame_processing_seconds", "Frame processing time"
)
queue_size = Gauge("webrtc_queue_size", "Number of frames in processing queue")

# Configuration
REDIS_URL = "redis://localhost:6379"
FRAME_QUEUE_KEY = "frame_queue"
RESULT_QUEUE_KEY = "result_queue:{}"
MAX_QUEUE_SIZE = 100
FRAME_SAMPLE_RATE = 5  # Sample every 5th frame for processing


class VideoTransformTrack(MediaStreamTrack):
    """
    Video track that processes frames through ML workers
    """

    kind = "video"

    def __init__(
        self, track: MediaStreamTrack, peer_id: str, redis_client: redis.Redis
    ):
        super().__init__()
        self.track = track
        self.peer_id = peer_id
        self.redis_client = redis_client
        self.frame_count = 0
        self.processed_frames = {}
        self.transform_enabled = True
        self.process_type = "style_transfer"  # Default processing type

    async def recv(self):
        """Receive and process video frames"""
        frame = await self.track.recv()
        self.frame_count += 1

        if not self.transform_enabled:
            return frame

        try:
            # Sample frames for processing
            if self.frame_count % FRAME_SAMPLE_RATE == 0:
                # Convert frame to numpy array
                img = frame.to_ndarray(format="bgr24")

                # Create frame data
                frame_id = f"{self.peer_id}:{self.frame_count}"
                _, buffer = cv2.imencode(".jpg", img)

                # Queue frame for processing
                frame_data = {
                    "frame_id": frame_id,
                    "peer_id": self.peer_id,
                    "process_type": self.process_type,
                    "timestamp": time.time(),
                    "frame_data": base64.b64encode(buffer).decode("utf-8"),
                }

                # Add to processing queue
                queue_length = await self.redis_client.llen(FRAME_QUEUE_KEY)
                if queue_length < MAX_QUEUE_SIZE:
                    await self.redis_client.lpush(
                        FRAME_QUEUE_KEY, json.dumps(frame_data)
                    )
                    queue_size.set(queue_length + 1)

            # Check for processed frames
            result_key = RESULT_QUEUE_KEY.format(self.peer_id)
            result = await self.redis_client.rpop(result_key)

            if result:
                result_data = json.loads(result)
                frame_id = result_data["frame_id"]
                processed_data = base64.b64decode(result_data["frame_data"])

                # Decode processed frame
                nparr = np.frombuffer(processed_data, np.uint8)
                processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Convert back to video frame
                new_frame = av.VideoFrame.from_ndarray(processed_img, format="bgr24")
                new_frame.pts = frame.pts
                new_frame.time_base = frame.time_base

                frames_processed.inc()
                return new_frame

        except Exception as e:
            logger.error(f"Error processing frame: {e}")

        return frame


class WebRTCConnection:
    """Manages a single WebRTC peer connection"""

    def __init__(self, peer_id: str, redis_client: redis.Redis):
        self.peer_id = peer_id
        self.pc = RTCPeerConnection()
        self.redis_client = redis_client
        self.tracks = {}
        self.data_channel: Optional[RTCDataChannel] = None

        # Setup handlers
        self.pc.on("track", self.on_track)
        self.pc.on("datachannel", self.on_datachannel)
        self.pc.on("connectionstatechange", self.on_connection_state_change)

    async def on_track(self, track: MediaStreamTrack):
        """Handle incoming media track"""
        logger.info(f"Track received: {track.kind} for peer {self.peer_id}")

        if track.kind == "video":
            # Create transform track
            transform_track = VideoTransformTrack(
                track, self.peer_id, self.redis_client
            )
            self.tracks[track.id] = transform_track

            # Add transformed track to peer connection
            self.pc.addTrack(transform_track)

            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.id} ended")

    async def on_datachannel(self, channel: RTCDataChannel):
        """Handle data channel for control messages"""
        logger.info(f"Data channel opened for peer {self.peer_id}")
        self.data_channel = channel

        @channel.on("message")
        async def on_message(message):
            try:
                data = json.loads(message)
                command = data.get("command")

                if command == "set_transform":
                    # Update transform type for all video tracks
                    transform_type = data.get("type", "style_transfer")
                    for track in self.tracks.values():
                        if isinstance(track, VideoTransformTrack):
                            track.process_type = transform_type

                elif command == "toggle_transform":
                    # Toggle transformation on/off
                    enabled = data.get("enabled", True)
                    for track in self.tracks.values():
                        if isinstance(track, VideoTransformTrack):
                            track.transform_enabled = enabled

                # Send acknowledgment
                if channel.readyState == "open":
                    await channel.send(
                        json.dumps(
                            {"type": "ack", "command": command, "status": "success"}
                        )
                    )

            except Exception as e:
                logger.error(f"Error handling data channel message: {e}")

    async def on_connection_state_change(self):
        """Handle connection state changes"""
        logger.info(f"Connection state for {self.peer_id}: {self.pc.connectionState}")

        if self.pc.connectionState == "failed":
            await self.close()
        elif self.pc.connectionState == "closed":
            active_connections.dec()

    async def close(self):
        """Close the peer connection"""
        await self.pc.close()
        for track in self.tracks.values():
            if hasattr(track, "stop"):
                track.stop()


class WebRTCServer:
    """Main WebRTC server"""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.connections: Dict[str, WebRTCConnection] = {}
        self.relay = MediaRelay()

    async def setup(self):
        """Initialize server components"""
        self.redis_client = await redis.from_url(REDIS_URL, decode_responses=False)
        logger.info("WebRTC server initialized")

    async def handle_offer(self, request):
        """Handle WebRTC offer"""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # Create peer ID
        peer_id = str(uuid.uuid4())
        logger.info(f"Creating peer connection for {peer_id}")

        # Create connection
        connection = WebRTCConnection(peer_id, self.redis_client)
        self.connections[peer_id] = connection
        active_connections.inc()

        # Handle offer
        await connection.pc.setRemoteDescription(offer)

        # Create answer
        answer = await connection.pc.createAnswer()
        await connection.pc.setLocalDescription(answer)

        return web.json_response(
            {
                "sdp": connection.pc.localDescription.sdp,
                "type": connection.pc.localDescription.type,
                "peer_id": peer_id,
            }
        )

    async def handle_health(self, request):
        """Health check endpoint"""
        return web.json_response(
            {
                "status": "healthy",
                "connections": len(self.connections),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def cleanup(self):
        """Cleanup connections"""
        for connection in self.connections.values():
            await connection.close()
        if self.redis_client:
            await self.redis_client.close()

    def create_app(self):
        """Create aiohttp application"""
        app = web.Application()

        # Setup CORS
        cors = aiohttp_cors.setup(
            app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*",
                )
            },
        )

        # Add routes
        app.router.add_post("/offer", self.handle_offer)
        app.router.add_get("/health", self.handle_health)

        # Configure CORS on all routes
        for route in list(app.router.routes()):
            cors.add(route)

        return app


async def main():
    """Main entry point"""
    server = WebRTCServer()
    await server.setup()

    app = server.create_app()

    try:
        # Run server
        runner = web.AppRunner(app)
        await runner.setup()

        # SSL context for HTTPS
        ssl_context = None
        try:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(
                "/mnt/c/Users/steve/Code/claude/ooblex/ssl/server.crt",
                "/mnt/c/Users/steve/Code/claude/ooblex/ssl/server.key",
            )
        except:
            logger.warning("SSL certificates not found, running without HTTPS")

        site = web.TCPSite(runner, "0.0.0.0", 8443, ssl_context=ssl_context)
        await site.start()

        logger.info(
            f"WebRTC server started on port 8443 (HTTPS: {ssl_context is not None})"
        )

        # Keep running
        await asyncio.Event().wait()

    except KeyboardInterrupt:
        logger.info("Shutting down WebRTC server")
    finally:
        await server.cleanup()
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
