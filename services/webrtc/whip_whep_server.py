"""
WHIP/WHEP Server Implementation
WebRTC-HTTP Ingestion Protocol (WHIP) and WebRTC-HTTP Egress Protocol (WHEP)
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import redis.asyncio as redis
from aiohttp import web
from aiortc import (
    MediaStreamTrack,
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaBlackhole, MediaRecorder, MediaRelay
from config import settings
from logger import setup_logger
from webrtc_gateway import VideoTransformTrack

logger = setup_logger(__name__)


class WHIPResource:
    """Represents a WHIP resource (incoming stream)"""

    def __init__(self, resource_id: str, peer_connection: RTCPeerConnection):
        self.id = resource_id
        self.pc = peer_connection
        self.created_at = datetime.utcnow()
        self.ice_candidates: List[RTCIceCandidate] = []
        self.relay = MediaRelay()
        self.tracks: Dict[str, MediaStreamTrack] = {}
        self.subscribers: Set[str] = set()
        self.metadata: Dict[str, any] = {}
        self.recording: Optional[MediaRecorder] = None

    def add_track(self, track: MediaStreamTrack):
        """Add a track to the resource"""
        self.tracks[track.id] = track
        # Create relay track for distribution
        relay_track = self.relay.subscribe(track)
        return relay_track

    async def close(self):
        """Close the resource"""
        if self.recording:
            await self.recording.stop()
        await self.pc.close()


class WHEPResource:
    """Represents a WHEP resource (outgoing stream)"""

    def __init__(
        self, resource_id: str, peer_connection: RTCPeerConnection, source_id: str
    ):
        self.id = resource_id
        self.pc = peer_connection
        self.source_id = source_id
        self.created_at = datetime.utcnow()
        self.ice_candidates: List[RTCIceCandidate] = []

    async def close(self):
        """Close the resource"""
        await self.pc.close()


class WHIPWHEPServer:
    """WHIP/WHEP Server implementation"""

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.whip_resources: Dict[str, WHIPResource] = {}
        self.whep_resources: Dict[str, WHEPResource] = {}
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        """Setup HTTP routes for WHIP/WHEP"""
        # WHIP routes
        self.app.router.add_post("/whip", self.handle_whip_post)
        self.app.router.add_patch("/whip/{resource_id}", self.handle_whip_patch)
        self.app.router.add_delete("/whip/{resource_id}", self.handle_whip_delete)
        self.app.router.add_options("/whip", self.handle_options)
        self.app.router.add_options("/whip/{resource_id}", self.handle_options)

        # WHEP routes
        self.app.router.add_post("/whep", self.handle_whep_post)
        self.app.router.add_patch("/whep/{resource_id}", self.handle_whep_patch)
        self.app.router.add_delete("/whep/{resource_id}", self.handle_whep_delete)
        self.app.router.add_options("/whep", self.handle_options)
        self.app.router.add_options("/whep/{resource_id}", self.handle_options)

        # Stream listing
        self.app.router.add_get("/streams", self.handle_list_streams)

        # Health check
        self.app.router.add_get("/health", self.handle_health)

    async def handle_options(self, request: web.Request) -> web.Response:
        """Handle OPTIONS requests for CORS"""
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, PATCH, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Expose-Headers": "Location, Link",
            "Access-Control-Max-Age": "86400",
        }
        return web.Response(status=204, headers=headers)

    async def handle_whip_post(self, request: web.Request) -> web.Response:
        """Handle WHIP POST - create new ingestion"""
        try:
            # Parse SDP offer
            content_type = request.headers.get("Content-Type", "")
            if content_type != "application/sdp":
                return web.Response(status=415, text="Unsupported Media Type")

            sdp_offer = await request.text()

            # Create peer connection
            pc = RTCPeerConnection()
            resource_id = str(uuid.uuid4())
            resource = WHIPResource(resource_id, pc)

            # Parse authorization for stream key (optional)
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                stream_key = auth_header.split(" ")[1]
                resource.metadata["stream_key"] = stream_key

            # Handle incoming tracks
            @pc.on("track")
            async def on_track(track: MediaStreamTrack):
                logger.info(
                    f"WHIP: Track received: {track.kind} for resource {resource_id}"
                )

                # Add to resource
                relay_track = resource.add_track(track)

                # Apply transformations if requested
                transform_type = request.headers.get("X-Transform-Type")
                if transform_type:
                    track = VideoTransformTrack(track, transform_type)

                # Store in Redis for discovery
                await self.redis_client.setex(
                    f"whip:stream:{resource_id}",
                    3600,
                    json.dumps(
                        {
                            "id": resource_id,
                            "kind": track.kind,
                            "created_at": resource.created_at.isoformat(),
                            "metadata": resource.metadata,
                        }
                    ),
                )

                # Publish event
                await self.redis_client.publish(
                    "whip:stream:new", json.dumps({"resource_id": resource_id})
                )

                # Optional recording
                if request.headers.get("X-Record", "").lower() == "true":
                    resource.recording = MediaRecorder(f"recordings/{resource_id}.mp4")
                    resource.recording.addTrack(track)
                    await resource.recording.start()

            # Set remote description
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp_offer, type="offer")
            )

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Store resource
            self.whip_resources[resource_id] = resource

            # Prepare response
            headers = {
                "Content-Type": "application/sdp",
                "Location": f"/whip/{resource_id}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Location, Link",
            }

            # Add Link header for ICE servers
            if settings.WEBRTC_STUN_SERVERS:
                ice_servers = []
                for server in settings.WEBRTC_STUN_SERVERS.split(","):
                    ice_servers.append(f'<{server}>; rel="ice-server"')
                headers["Link"] = ", ".join(ice_servers)

            return web.Response(
                status=201, text=pc.localDescription.sdp, headers=headers
            )

        except Exception as e:
            logger.error(f"WHIP POST error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_whip_patch(self, request: web.Request) -> web.Response:
        """Handle WHIP PATCH - ICE candidates"""
        try:
            resource_id = request.match_info["resource_id"]
            resource = self.whip_resources.get(resource_id)

            if not resource:
                return web.Response(status=404, text="Resource not found")

            # Parse ICE candidates
            content_type = request.headers.get("Content-Type", "")
            if content_type == "application/trickle-ice-sdpfrag":
                # Handle trickle ICE
                ice_fragment = await request.text()
                # Parse and add ICE candidates
                # Note: aiortc handles ICE gathering automatically
                logger.info(f"WHIP: Received ICE fragment for {resource_id}")

            return web.Response(
                status=204, headers={"Access-Control-Allow-Origin": "*"}
            )

        except Exception as e:
            logger.error(f"WHIP PATCH error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_whip_delete(self, request: web.Request) -> web.Response:
        """Handle WHIP DELETE - stop ingestion"""
        try:
            resource_id = request.match_info["resource_id"]
            resource = self.whip_resources.get(resource_id)

            if not resource:
                return web.Response(status=404, text="Resource not found")

            # Close resource
            await resource.close()
            del self.whip_resources[resource_id]

            # Remove from Redis
            await self.redis_client.delete(f"whip:stream:{resource_id}")

            # Publish event
            await self.redis_client.publish(
                "whip:stream:end", json.dumps({"resource_id": resource_id})
            )

            return web.Response(
                status=200, headers={"Access-Control-Allow-Origin": "*"}
            )

        except Exception as e:
            logger.error(f"WHIP DELETE error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_whep_post(self, request: web.Request) -> web.Response:
        """Handle WHEP POST - create new egress"""
        try:
            # Parse request
            content_type = request.headers.get("Content-Type", "")
            if content_type != "application/sdp":
                return web.Response(status=415, text="Unsupported Media Type")

            sdp_offer = await request.text()

            # Get source stream ID from URL parameter or header
            source_id = request.rel_url.query.get("stream") or request.headers.get(
                "X-Stream-ID"
            )
            if not source_id:
                return web.Response(status=400, text="Missing stream ID")

            # Check if source exists
            source_resource = self.whip_resources.get(source_id)
            if not source_resource:
                return web.Response(status=404, text="Source stream not found")

            # Create peer connection
            pc = RTCPeerConnection()
            resource_id = str(uuid.uuid4())
            resource = WHEPResource(resource_id, pc, source_id)

            # Add relay tracks from source
            for track_id, track in source_resource.tracks.items():
                if track.kind == "video":
                    # Get relay track
                    relay_track = source_resource.relay.subscribe(track)
                    pc.addTrack(relay_track)
                elif track.kind == "audio":
                    relay_track = source_resource.relay.subscribe(track)
                    pc.addTrack(relay_track)

            # Set remote description
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp_offer, type="offer")
            )

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            # Store resource
            self.whep_resources[resource_id] = resource
            source_resource.subscribers.add(resource_id)

            # Update viewer count
            await self.redis_client.hincrby(
                f"whip:stream:{source_id}:stats", "viewers", 1
            )

            # Prepare response
            headers = {
                "Content-Type": "application/sdp",
                "Location": f"/whep/{resource_id}",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Expose-Headers": "Location, Link",
            }

            # Add Link headers
            links = []

            # ICE servers
            if settings.WEBRTC_STUN_SERVERS:
                for server in settings.WEBRTC_STUN_SERVERS.split(","):
                    links.append(f'<{server}>; rel="ice-server"')

            # Server-Sent Events for real-time updates
            links.append(
                f'</whep/{resource_id}/events>; rel="urn:ietf:params:whep:ext:core:server-sent-events"'
            )

            # Layer selection extension
            links.append(
                f'</whep/{resource_id}/layer>; rel="urn:ietf:params:whep:ext:core:layer"'
            )

            if links:
                headers["Link"] = ", ".join(links)

            return web.Response(
                status=201, text=pc.localDescription.sdp, headers=headers
            )

        except Exception as e:
            logger.error(f"WHEP POST error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_whep_patch(self, request: web.Request) -> web.Response:
        """Handle WHEP PATCH - ICE candidates or layer selection"""
        try:
            resource_id = request.match_info["resource_id"]
            resource = self.whep_resources.get(resource_id)

            if not resource:
                return web.Response(status=404, text="Resource not found")

            content_type = request.headers.get("Content-Type", "")

            if content_type == "application/trickle-ice-sdpfrag":
                # Handle trickle ICE
                ice_fragment = await request.text()
                logger.info(f"WHEP: Received ICE fragment for {resource_id}")

            elif content_type == "application/json":
                # Handle layer selection
                data = await request.json()
                if "layer" in data:
                    # Implement simulcast layer selection
                    logger.info(
                        f"WHEP: Layer selection for {resource_id}: {data['layer']}"
                    )

            return web.Response(
                status=204, headers={"Access-Control-Allow-Origin": "*"}
            )

        except Exception as e:
            logger.error(f"WHEP PATCH error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_whep_delete(self, request: web.Request) -> web.Response:
        """Handle WHEP DELETE - stop egress"""
        try:
            resource_id = request.match_info["resource_id"]
            resource = self.whep_resources.get(resource_id)

            if not resource:
                return web.Response(status=404, text="Resource not found")

            # Update viewer count
            source_resource = self.whip_resources.get(resource.source_id)
            if source_resource:
                source_resource.subscribers.discard(resource_id)
                await self.redis_client.hincrby(
                    f"whip:stream:{resource.source_id}:stats", "viewers", -1
                )

            # Close resource
            await resource.close()
            del self.whep_resources[resource_id]

            return web.Response(
                status=200, headers={"Access-Control-Allow-Origin": "*"}
            )

        except Exception as e:
            logger.error(f"WHEP DELETE error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_list_streams(self, request: web.Request) -> web.Response:
        """List available streams"""
        try:
            # Get all WHIP streams from Redis
            streams = []
            cursor = 0

            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, match="whip:stream:*", count=100
                )

                for key in keys:
                    if ":stats" not in key:
                        stream_data = await self.redis_client.get(key)
                        if stream_data:
                            stream_info = json.loads(stream_data)

                            # Get stats
                            stats_key = f"{key}:stats"
                            viewers = (
                                await self.redis_client.hget(stats_key, "viewers") or 0
                            )

                            stream_info["viewers"] = int(viewers)
                            stream_info["whep_url"] = (
                                f"/whep?stream={stream_info['id']}"
                            )
                            streams.append(stream_info)

                if cursor == 0:
                    break

            return web.json_response(
                {"streams": streams, "count": len(streams)},
                headers={"Access-Control-Allow-Origin": "*"},
            )

        except Exception as e:
            logger.error(f"List streams error: {e}")
            return web.Response(status=500, text=str(e))

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response(
            {
                "status": "healthy",
                "whip_resources": len(self.whip_resources),
                "whep_resources": len(self.whep_resources),
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    async def cleanup_expired_resources(self):
        """Cleanup expired resources"""
        while True:
            try:
                now = datetime.utcnow()
                max_age = timedelta(hours=1)

                # Cleanup WHIP resources
                expired_whip = []
                for resource_id, resource in self.whip_resources.items():
                    if (
                        now - resource.created_at > max_age
                        and len(resource.subscribers) == 0
                    ):
                        expired_whip.append(resource_id)

                for resource_id in expired_whip:
                    resource = self.whip_resources[resource_id]
                    await resource.close()
                    del self.whip_resources[resource_id]
                    await self.redis_client.delete(f"whip:stream:{resource_id}")
                    logger.info(f"Cleaned up expired WHIP resource: {resource_id}")

                # Cleanup WHEP resources
                expired_whep = []
                for resource_id, resource in self.whep_resources.items():
                    if now - resource.created_at > max_age:
                        expired_whep.append(resource_id)

                for resource_id in expired_whep:
                    resource = self.whep_resources[resource_id]
                    await resource.close()
                    del self.whep_resources[resource_id]
                    logger.info(f"Cleaned up expired WHEP resource: {resource_id}")

                await asyncio.sleep(60)  # Run every minute

            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)

    async def start(self, port: int = 8102):
        """Start the WHIP/WHEP server"""
        # Start cleanup task
        asyncio.create_task(self.cleanup_expired_resources())

        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()

        logger.info(f"WHIP/WHEP server started on port {port}")


# Standalone server runner
async def main():
    """Run WHIP/WHEP server standalone"""
    redis_client = await redis.from_url(settings.REDIS_URL)
    server = WHIPWHEPServer(redis_client)

    await server.start()

    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Shutting down WHIP/WHEP server")
    finally:
        await redis_client.close()


if __name__ == "__main__":
    asyncio.run(main())
