"""
Modern WebRTC Gateway Service for Ooblex
Handles WebRTC signaling and media routing
"""
import asyncio
import json
import logging
import uuid
from typing import Dict, Set, Optional, Any
from datetime import datetime
import ssl
from pathlib import Path

from aiohttp import web
import websockets
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaPlayer, MediaRecorder
import redis.asyncio as redis
from aio_pika import connect_robust, Message, DeliveryMode
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from pyee.asyncio import AsyncIOEventEmitter
import av
import numpy as np

from config import settings
from logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Metrics
websocket_connections = Counter('webrtc_websocket_connections_total', 'Total WebSocket connections')
active_connections = Gauge('webrtc_active_connections', 'Active WebSocket connections')
peer_connections = Counter('webrtc_peer_connections_total', 'Total peer connections', ['status'])
media_streams = Gauge('webrtc_media_streams', 'Active media streams', ['type'])
signaling_time = Histogram('webrtc_signaling_duration_seconds', 'Signaling duration')
frame_processing_time = Histogram('webrtc_frame_processing_seconds', 'Frame processing time')


class VideoTransformTrack(MediaStreamTrack):
    """Video track that transforms frames in real-time"""
    
    kind = "video"
    
    def __init__(self, track: MediaStreamTrack, transform_type: str = None):
        super().__init__()
        self.track = track
        self.transform_type = transform_type
        self.frame_count = 0
        
    async def recv(self):
        """Receive and optionally transform video frame"""
        frame = await self.track.recv()
        
        if self.transform_type:
            # Here we would apply transformations
            # For now, just count frames
            self.frame_count += 1
            
            # Convert frame to numpy array for processing
            img = frame.to_ndarray(format="bgr24")
            
            # TODO: Apply ML transformations based on transform_type
            # For now, just add a simple overlay
            if self.transform_type == "debug":
                import cv2
                cv2.putText(img, f"Frame: {self.frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert back to video frame
            new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            
            return new_frame
        
        return frame


class PeerConnection:
    """Manages a single peer connection"""
    
    def __init__(self, peer_id: str, redis_client: redis.Redis):
        self.peer_id = peer_id
        self.pc = RTCPeerConnection()
        self.redis_client = redis_client
        self.tracks: Dict[str, MediaStreamTrack] = {}
        self.data_channel = None
        self.created_at = datetime.utcnow()
        
        # Setup event handlers
        self.pc.on("track", self.on_track)
        self.pc.on("datachannel", self.on_datachannel)
        self.pc.on("connectionstatechange", self.on_connection_state_change)
        
    async def on_track(self, track: MediaStreamTrack):
        """Handle incoming media track"""
        logger.info(f"Track received: {track.kind} for peer {self.peer_id}")
        self.tracks[track.id] = track
        media_streams.labels(type=track.kind).inc()
        
        if track.kind == "video":
            # Apply transformation if requested
            transform_type = await self.redis_client.get(f"transform:{self.peer_id}")
            if transform_type:
                track = VideoTransformTrack(track, transform_type)
            
            # Store frames in Redis for ML processing
            @track.on("ended")
            async def on_ended():
                logger.info(f"Track {track.id} ended")
                media_streams.labels(type="video").dec()
            
            # Sample frames periodically
            asyncio.create_task(self.sample_frames(track))
    
    async def sample_frames(self, track: MediaStreamTrack, interval: float = 1.0):
        """Sample frames from video track for ML processing"""
        try:
            while True:
                frame = await track.recv()
                
                # Convert to JPEG
                img = frame.to_ndarray(format="bgr24")
                _, buffer = cv2.imencode('.jpg', img)
                
                # Store in Redis
                await self.redis_client.setex(
                    f"frame:{self.peer_id}:latest",
                    60,  # 60 second TTL
                    buffer.tobytes()
                )
                
                # Publish frame available event
                await self.redis_client.publish(
                    f"frame_available:{self.peer_id}",
                    json.dumps({
                        "peer_id": self.peer_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                )
                
                await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Error sampling frames: {e}")
    
    async def on_datachannel(self, channel):
        """Handle data channel"""
        logger.info(f"Data channel opened for peer {self.peer_id}")
        self.data_channel = channel
        
        @channel.on("message")
        def on_message(message):
            logger.debug(f"Data channel message from {self.peer_id}: {message}")
            # Echo back for now
            channel.send(f"Echo: {message}")
    
    async def on_connection_state_change(self):
        """Handle connection state changes"""
        state = self.pc.connectionState
        logger.info(f"Connection state for {self.peer_id}: {state}")
        
        if state == "connected":
            peer_connections.labels(status="connected").inc()
        elif state == "failed":
            peer_connections.labels(status="failed").inc()
        elif state == "closed":
            peer_connections.labels(status="closed").inc()
            # Cleanup
            media_streams._metrics.clear()
    
    async def create_offer(self):
        """Create SDP offer"""
        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)
        return {
            "type": offer.type,
            "sdp": offer.sdp
        }
    
    async def create_answer(self):
        """Create SDP answer"""
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        return {
            "type": answer.type,
            "sdp": answer.sdp
        }
    
    async def set_remote_description(self, sdp: Dict[str, str]):
        """Set remote SDP"""
        description = RTCSessionDescription(
            sdp=sdp["sdp"],
            type=sdp["type"]
        )
        await self.pc.setRemoteDescription(description)
    
    async def add_ice_candidate(self, candidate: Dict[str, Any]):
        """Add ICE candidate"""
        # aiortc handles ICE candidates automatically
        pass
    
    async def close(self):
        """Close peer connection"""
        await self.pc.close()
        # Cleanup tracks
        for track in self.tracks.values():
            track.stop()


class WebRTCGateway:
    """Main WebRTC Gateway service"""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.rabbitmq_connection = None
        self.rabbitmq_channel = None
        self.peers: Dict[str, PeerConnection] = {}
        self.websockets: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.events = AsyncIOEventEmitter()
        
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
        
        logger.info("WebRTC Gateway initialized")
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        peer_id = str(uuid.uuid4())
        self.websockets[peer_id] = websocket
        websocket_connections.inc()
        active_connections.inc()
        
        logger.info(f"WebSocket connected: {peer_id}")
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "welcome",
                "peer_id": peer_id
            }))
            
            # Handle messages
            async for message in websocket:
                await self.handle_message(peer_id, json.loads(message))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket disconnected: {peer_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {peer_id}: {e}")
        finally:
            # Cleanup
            active_connections.dec()
            if peer_id in self.websockets:
                del self.websockets[peer_id]
            if peer_id in self.peers:
                await self.peers[peer_id].close()
                del self.peers[peer_id]
    
    async def handle_message(self, peer_id: str, message: Dict[str, Any]):
        """Handle signaling messages"""
        msg_type = message.get("type")
        
        with signaling_time.time():
            if msg_type == "offer":
                await self.handle_offer(peer_id, message)
            elif msg_type == "answer":
                await self.handle_answer(peer_id, message)
            elif msg_type == "ice-candidate":
                await self.handle_ice_candidate(peer_id, message)
            elif msg_type == "get-offer":
                await self.handle_get_offer(peer_id, message)
            elif msg_type == "transform":
                await self.handle_transform(peer_id, message)
            else:
                logger.warning(f"Unknown message type: {msg_type}")
    
    async def handle_offer(self, peer_id: str, message: Dict[str, Any]):
        """Handle SDP offer"""
        # Create peer connection
        peer = PeerConnection(peer_id, self.redis_client)
        self.peers[peer_id] = peer
        
        # Set remote description
        await peer.set_remote_description(message["sdp"])
        
        # Create answer
        answer = await peer.create_answer()
        
        # Send answer back
        await self.send_to_peer(peer_id, {
            "type": "answer",
            "sdp": answer
        })
    
    async def handle_answer(self, peer_id: str, message: Dict[str, Any]):
        """Handle SDP answer"""
        if peer_id in self.peers:
            await self.peers[peer_id].set_remote_description(message["sdp"])
    
    async def handle_ice_candidate(self, peer_id: str, message: Dict[str, Any]):
        """Handle ICE candidate"""
        if peer_id in self.peers:
            await self.peers[peer_id].add_ice_candidate(message["candidate"])
    
    async def handle_get_offer(self, peer_id: str, message: Dict[str, Any]):
        """Create and send offer"""
        # Create peer connection
        peer = PeerConnection(peer_id, self.redis_client)
        self.peers[peer_id] = peer
        
        # Add transceiver for video
        peer.pc.addTransceiver("video", direction="recvonly")
        
        # Create offer
        offer = await peer.create_offer()
        
        # Send offer
        await self.send_to_peer(peer_id, {
            "type": "offer",
            "sdp": offer
        })
    
    async def handle_transform(self, peer_id: str, message: Dict[str, Any]):
        """Handle transform request"""
        transform_type = message.get("transform")
        
        # Store transform type in Redis
        await self.redis_client.setex(
            f"transform:{peer_id}",
            3600,
            transform_type
        )
        
        # Send to RabbitMQ for processing
        task_data = {
            "task_id": f"{peer_id}:{transform_type}:{datetime.utcnow().timestamp()}",
            "peer_id": peer_id,
            "transform_type": transform_type,
            "parameters": message.get("parameters", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.rabbitmq_channel.default_exchange.publish(
            Message(
                body=json.dumps(task_data).encode(),
                delivery_mode=DeliveryMode.PERSISTENT
            ),
            routing_key="transforms"
        )
        
        # Acknowledge
        await self.send_to_peer(peer_id, {
            "type": "transform-ack",
            "transform": transform_type
        })
    
    async def send_to_peer(self, peer_id: str, message: Dict[str, Any]):
        """Send message to peer"""
        if peer_id in self.websockets:
            await self.websockets[peer_id].send(json.dumps(message))
    
    async def health_check(self, request):
        """HTTP health check endpoint"""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "peers": len(self.peers),
            "websockets": len(self.websockets)
        })
    
    async def run(self):
        """Run the gateway"""
        await self.setup()
        
        # Start metrics server
        start_http_server(9090)
        logger.info("Metrics server started on port 9090")
        
        # Setup SSL context
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(
            certfile=settings.ssl_cert_path,
            keyfile=settings.ssl_key_path
        )
        
        # Start HTTP server for health checks
        app = web.Application()
        app.router.add_get('/health', self.health_check)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', 8101)
        await site.start()
        
        logger.info("Health check server started on port 8101")
        
        # Start WebSocket server
        logger.info(f"Starting WebRTC Gateway on port {settings.webrtc_port}")
        
        async with websockets.serve(
            self.handle_websocket,
            "0.0.0.0",
            settings.webrtc_port,
            ssl=ssl_context
        ):
            await asyncio.Future()  # Run forever


async def main():
    """Main entry point"""
    gateway = WebRTCGateway()
    
    try:
        await gateway.run()
    except KeyboardInterrupt:
        logger.info("Shutting down WebRTC Gateway")
    finally:
        # Cleanup
        for peer in gateway.peers.values():
            await peer.close()
        if gateway.rabbitmq_connection:
            await gateway.rabbitmq_connection.close()
        if gateway.redis_client:
            await gateway.redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())