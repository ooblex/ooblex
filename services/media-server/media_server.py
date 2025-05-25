import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid

import aiohttp
from aiohttp import web
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRecorder, MediaRelay
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.mediastreams import VideoStreamTrack, AudioStreamTrack
import av
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Participant:
    """Represents a participant in a room"""
    id: str
    pc: RTCPeerConnection
    tracks: Dict[str, MediaStreamTrack] = field(default_factory=dict)
    simulcast_layers: Dict[str, List[MediaStreamTrack]] = field(default_factory=dict)
    recording: Optional[MediaRecorder] = None
    joined_at: datetime = field(default_factory=datetime.now)


@dataclass
class Room:
    """Represents a media room"""
    id: str
    mode: str  # 'sfu' or 'mcu'
    participants: Dict[str, Participant] = field(default_factory=dict)
    composite_track: Optional[VideoStreamTrack] = None
    recording_enabled: bool = False
    created_at: datetime = field(default_factory=datetime.now)


class CompositeVideoTrack(VideoStreamTrack):
    """MCU composite video track that combines multiple video streams"""
    
    def __init__(self, sources: List[VideoStreamTrack]):
        super().__init__()
        self.sources = sources
        self.width = 1280
        self.height = 720
        
    async def recv(self):
        """Receive composite frame"""
        pts, time_base = await self.next_timestamp()
        
        # Create composite frame
        composite = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Simple grid layout for now
        grid_size = int(np.ceil(np.sqrt(len(self.sources))))
        cell_width = self.width // grid_size
        cell_height = self.height // grid_size
        
        for idx, source in enumerate(self.sources):
            try:
                frame = await source.recv()
                
                # Calculate position in grid
                row = idx // grid_size
                col = idx % grid_size
                x = col * cell_width
                y = row * cell_height
                
                # Resize and place frame
                img = frame.to_ndarray(format="bgr24")
                resized = av.VideoFrame.from_ndarray(
                    cv2.resize(img, (cell_width, cell_height)),
                    format="bgr24"
                )
                
                composite[y:y+cell_height, x:x+cell_width] = resized.to_ndarray(format="bgr24")
                
            except Exception as e:
                logger.error(f"Error processing source {idx}: {e}")
                continue
        
        # Convert to frame
        new_frame = av.VideoFrame.from_ndarray(composite, format="bgr24")
        new_frame.pts = pts
        new_frame.time_base = time_base
        
        return new_frame


class MediaServer:
    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self.relay = MediaRelay()
        self.whip_endpoint = os.getenv("WHIP_ENDPOINT", "http://whip-server:8080")
        self.whep_endpoint = os.getenv("WHEP_ENDPOINT", "http://whep-server:8080")
        self.vdo_ninja_url = os.getenv("VDO_NINJA_URL", "http://vdo-bridge:8080")
        
    async def create_room(self, room_id: str, mode: str = "sfu", recording: bool = False) -> Room:
        """Create a new room"""
        if room_id in self.rooms:
            raise ValueError(f"Room {room_id} already exists")
            
        room = Room(
            id=room_id,
            mode=mode,
            recording_enabled=recording
        )
        self.rooms[room_id] = room
        
        logger.info(f"Created room {room_id} in {mode} mode")
        return room
        
    async def join_room(self, room_id: str, participant_id: str, offer: RTCSessionDescription) -> RTCSessionDescription:
        """Join a room with WebRTC offer"""
        room = self.rooms.get(room_id)
        if not room:
            raise ValueError(f"Room {room_id} not found")
            
        # Create peer connection
        pc = RTCPeerConnection()
        participant = Participant(id=participant_id, pc=pc)
        
        # Handle incoming tracks
        @pc.on("track")
        async def on_track(track):
            logger.info(f"Track received from {participant_id}: {track.kind}")
            participant.tracks[track.kind] = track
            
            if room.mode == "sfu":
                # In SFU mode, forward tracks to other participants
                await self._forward_track_sfu(room, participant_id, track)
            else:
                # In MCU mode, update composite
                await self._update_composite_mcu(room)
                
            # Start recording if enabled
            if room.recording_enabled and not participant.recording:
                await self._start_recording(participant)
                
        # Set remote description
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Add participant to room
        room.participants[participant_id] = participant
        
        # Send existing tracks to new participant
        if room.mode == "sfu":
            await self._send_existing_tracks_sfu(room, participant_id)
        elif room.mode == "mcu" and room.composite_track:
            # Send composite track
            pc.addTrack(room.composite_track)
            
        logger.info(f"Participant {participant_id} joined room {room_id}")
        return pc.localDescription
        
    async def leave_room(self, room_id: str, participant_id: str):
        """Leave a room"""
        room = self.rooms.get(room_id)
        if not room:
            return
            
        participant = room.participants.get(participant_id)
        if not participant:
            return
            
        # Stop recording
        if participant.recording:
            await participant.recording.stop()
            
        # Close peer connection
        await participant.pc.close()
        
        # Remove participant
        del room.participants[participant_id]
        
        # Update composite if MCU mode
        if room.mode == "mcu":
            await self._update_composite_mcu(room)
            
        logger.info(f"Participant {participant_id} left room {room_id}")
        
    async def _forward_track_sfu(self, room: Room, sender_id: str, track: MediaStreamTrack):
        """Forward track to other participants in SFU mode"""
        for participant_id, participant in room.participants.items():
            if participant_id != sender_id:
                try:
                    # Use relay for efficient forwarding
                    relayed_track = self.relay.subscribe(track)
                    participant.pc.addTrack(relayed_track)
                except Exception as e:
                    logger.error(f"Error forwarding track to {participant_id}: {e}")
                    
    async def _send_existing_tracks_sfu(self, room: Room, new_participant_id: str):
        """Send existing tracks to new participant in SFU mode"""
        new_participant = room.participants[new_participant_id]
        
        for participant_id, participant in room.participants.items():
            if participant_id != new_participant_id:
                for track_kind, track in participant.tracks.items():
                    try:
                        relayed_track = self.relay.subscribe(track)
                        new_participant.pc.addTrack(relayed_track)
                    except Exception as e:
                        logger.error(f"Error sending existing track to {new_participant_id}: {e}")
                        
    async def _update_composite_mcu(self, room: Room):
        """Update composite track in MCU mode"""
        video_tracks = []
        
        for participant in room.participants.values():
            if "video" in participant.tracks:
                video_tracks.append(participant.tracks["video"])
                
        if video_tracks:
            # Create new composite track
            room.composite_track = CompositeVideoTrack(video_tracks)
            
            # Send to all participants
            for participant in room.participants.values():
                # Remove old composite track
                for sender in participant.pc.getSenders():
                    if sender.track and isinstance(sender.track, CompositeVideoTrack):
                        await sender.replaceTrack(room.composite_track)
                        break
                else:
                    # Add new composite track
                    participant.pc.addTrack(room.composite_track)
                    
    async def _start_recording(self, participant: Participant):
        """Start recording for a participant"""
        tracks = list(participant.tracks.values())
        if tracks:
            filename = f"recordings/{participant.id}_{datetime.now().isoformat()}.mp4"
            os.makedirs("recordings", exist_ok=True)
            participant.recording = MediaRecorder(filename)
            for track in tracks:
                participant.recording.addTrack(track)
            await participant.recording.start()
            logger.info(f"Started recording for participant {participant.id}")
            
    async def integrate_whip(self, room_id: str, whip_resource_url: str):
        """Integrate with WHIP server"""
        room = self.rooms.get(room_id)
        if not room:
            raise ValueError(f"Room {room_id} not found")
            
        # Create WHEP client to pull from WHIP
        async with aiohttp.ClientSession() as session:
            # Create WHEP offer
            pc = RTCPeerConnection()
            
            # Add transceiver for receiving
            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")
            
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            
            # Send to WHEP endpoint
            headers = {"Content-Type": "application/sdp"}
            async with session.post(
                f"{self.whep_endpoint}/whep",
                data=offer.sdp,
                headers=headers
            ) as resp:
                if resp.status == 201:
                    answer_sdp = await resp.text()
                    answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
                    await pc.setRemoteDescription(answer)
                    
                    # Handle incoming tracks
                    @pc.on("track")
                    async def on_track(track):
                        logger.info(f"Received track from WHIP: {track.kind}")
                        # Forward to room participants
                        if room.mode == "sfu":
                            await self._forward_track_sfu(room, "whip", track)
                        else:
                            await self._update_composite_mcu(room)
                            
    async def integrate_vdo_ninja(self, room_id: str, vdo_room_id: str):
        """Integrate with VDO.Ninja room"""
        room = self.rooms.get(room_id)
        if not room:
            raise ValueError(f"Room {room_id} not found")
            
        async with aiohttp.ClientSession() as session:
            # Connect to VDO.Ninja bridge
            async with session.ws_connect(f"{self.vdo_ninja_url}/ws") as ws:
                # Join VDO.Ninja room
                await ws.send_json({
                    "action": "join",
                    "room": vdo_room_id,
                    "mode": "bridge"
                })
                
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        
                        if data.get("type") == "offer":
                            # Handle VDO.Ninja participant
                            pc = RTCPeerConnection()
                            
                            @pc.on("track")
                            async def on_track(track):
                                logger.info(f"Received track from VDO.Ninja: {track.kind}")
                                if room.mode == "sfu":
                                    await self._forward_track_sfu(room, f"vdo_{data.get('peer_id')}", track)
                                else:
                                    await self._update_composite_mcu(room)
                                    
                            await pc.setRemoteDescription(
                                RTCSessionDescription(sdp=data["sdp"], type="offer")
                            )
                            
                            answer = await pc.createAnswer()
                            await pc.setLocalDescription(answer)
                            
                            await ws.send_json({
                                "type": "answer",
                                "peer_id": data.get("peer_id"),
                                "sdp": pc.localDescription.sdp
                            })
                            
    async def get_room_stats(self, room_id: str) -> dict:
        """Get statistics for a room"""
        room = self.rooms.get(room_id)
        if not room:
            raise ValueError(f"Room {room_id} not found")
            
        stats = {
            "room_id": room_id,
            "mode": room.mode,
            "created_at": room.created_at.isoformat(),
            "participant_count": len(room.participants),
            "recording_enabled": room.recording_enabled,
            "participants": []
        }
        
        for participant_id, participant in room.participants.items():
            participant_stats = {
                "id": participant_id,
                "joined_at": participant.joined_at.isoformat(),
                "tracks": list(participant.tracks.keys()),
                "recording": participant.recording is not None
            }
            
            # Get connection stats
            pc_stats = await participant.pc.getStats()
            participant_stats["connection_stats"] = self._parse_connection_stats(pc_stats)
            
            stats["participants"].append(participant_stats)
            
        return stats
        
    def _parse_connection_stats(self, stats) -> dict:
        """Parse WebRTC connection statistics"""
        parsed = {
            "bitrate_audio": 0,
            "bitrate_video": 0,
            "packets_lost": 0,
            "jitter": 0
        }
        
        for stat in stats.values():
            if stat.type == "inbound-rtp":
                if stat.get("mediaType") == "audio":
                    parsed["bitrate_audio"] = stat.get("bitrate", 0)
                elif stat.get("mediaType") == "video":
                    parsed["bitrate_video"] = stat.get("bitrate", 0)
                parsed["packets_lost"] += stat.get("packetsLost", 0)
                parsed["jitter"] = max(parsed["jitter"], stat.get("jitter", 0))
                
        return parsed


# HTTP API endpoints
async def create_room_handler(request):
    """Create a new room"""
    data = await request.json()
    room_id = data.get("room_id", str(uuid.uuid4()))
    mode = data.get("mode", "sfu")
    recording = data.get("recording", False)
    
    try:
        server = request.app["media_server"]
        room = await server.create_room(room_id, mode, recording)
        
        return web.json_response({
            "room_id": room.id,
            "mode": room.mode,
            "recording_enabled": room.recording_enabled
        })
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
        

async def join_room_handler(request):
    """Join a room"""
    room_id = request.match_info["room_id"]
    data = await request.json()
    participant_id = data.get("participant_id", str(uuid.uuid4()))
    offer_sdp = data.get("offer")
    
    if not offer_sdp:
        return web.json_response({"error": "Missing offer"}, status=400)
        
    try:
        server = request.app["media_server"]
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        answer = await server.join_room(room_id, participant_id, offer)
        
        return web.json_response({
            "participant_id": participant_id,
            "answer": answer.sdp
        })
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=404)
        

async def leave_room_handler(request):
    """Leave a room"""
    room_id = request.match_info["room_id"]
    participant_id = request.match_info["participant_id"]
    
    server = request.app["media_server"]
    await server.leave_room(room_id, participant_id)
    
    return web.json_response({"status": "ok"})


async def room_stats_handler(request):
    """Get room statistics"""
    room_id = request.match_info["room_id"]
    
    try:
        server = request.app["media_server"]
        stats = await server.get_room_stats(room_id)
        return web.json_response(stats)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=404)
        

async def integrate_whip_handler(request):
    """Integrate with WHIP server"""
    room_id = request.match_info["room_id"]
    data = await request.json()
    whip_resource_url = data.get("whip_resource_url")
    
    if not whip_resource_url:
        return web.json_response({"error": "Missing whip_resource_url"}, status=400)
        
    try:
        server = request.app["media_server"]
        await server.integrate_whip(room_id, whip_resource_url)
        return web.json_response({"status": "ok"})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=404)
        

async def integrate_vdo_handler(request):
    """Integrate with VDO.Ninja"""
    room_id = request.match_info["room_id"]
    data = await request.json()
    vdo_room_id = data.get("vdo_room_id")
    
    if not vdo_room_id:
        return web.json_response({"error": "Missing vdo_room_id"}, status=400)
        
    try:
        server = request.app["media_server"]
        await server.integrate_vdo_ninja(room_id, vdo_room_id)
        return web.json_response({"status": "ok"})
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=404)


async def health_handler(request):
    """Health check endpoint"""
    return web.json_response({"status": "healthy"})


def create_app():
    """Create the web application"""
    app = web.Application()
    app["media_server"] = MediaServer()
    
    # Add routes
    app.router.add_post("/rooms", create_room_handler)
    app.router.add_post("/rooms/{room_id}/join", join_room_handler)
    app.router.add_post("/rooms/{room_id}/leave/{participant_id}", leave_room_handler)
    app.router.add_get("/rooms/{room_id}/stats", room_stats_handler)
    app.router.add_post("/rooms/{room_id}/integrate/whip", integrate_whip_handler)
    app.router.add_post("/rooms/{room_id}/integrate/vdo", integrate_vdo_handler)
    app.router.add_get("/health", health_handler)
    
    return app


if __name__ == "__main__":
    app = create_app()
    web.run_app(app, host="0.0.0.0", port=8080)