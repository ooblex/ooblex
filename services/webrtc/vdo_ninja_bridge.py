"""
VDO.Ninja Compatibility Bridge
Enables integration with VDO.Ninja for remote video production
"""
import asyncio
import json
import uuid
import hashlib
from typing import Dict, Optional, List, Any
from datetime import datetime
import logging
from urllib.parse import parse_qs, urlparse

import websockets
from aiohttp import web, ClientSession
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import redis.asyncio as redis

from config import settings
from logger import setup_logger
from webrtc_gateway import VideoTransformTrack

logger = setup_logger(__name__)


class VDONinjaRoom:
    """Represents a VDO.Ninja compatible room"""
    
    def __init__(self, room_id: str, password: Optional[str] = None):
        self.room_id = room_id
        self.password = password
        self.participants: Dict[str, 'VDONinjaParticipant'] = {}
        self.created_at = datetime.utcnow()
        self.settings = {
            'bitrate': 2500,
            'codec': 'vp8',
            'stereo': True,
            'enhance': True,
            'autostart': True
        }
    
    def add_participant(self, participant: 'VDONinjaParticipant'):
        """Add participant to room"""
        self.participants[participant.stream_id] = participant
    
    def remove_participant(self, stream_id: str):
        """Remove participant from room"""
        if stream_id in self.participants:
            del self.participants[stream_id]
    
    def get_room_state(self) -> Dict[str, Any]:
        """Get current room state for VDO.Ninja protocol"""
        return {
            'room': self.room_id,
            'participants': [
                {
                    'streamID': p.stream_id,
                    'label': p.label,
                    'director': p.is_director,
                    'scene': p.scene_id is not None,
                    'muted': p.muted,
                    'videoMuted': p.video_muted
                }
                for p in self.participants.values()
            ],
            'settings': self.settings
        }


class VDONinjaParticipant:
    """Represents a VDO.Ninja participant"""
    
    def __init__(self, stream_id: str, websocket: websockets.WebSocketServerProtocol):
        self.stream_id = stream_id
        self.websocket = websocket
        self.pc: Optional[RTCPeerConnection] = None
        self.tracks: Dict[str, MediaStreamTrack] = {}
        self.label = stream_id
        self.room_id: Optional[str] = None
        self.scene_id: Optional[str] = None
        self.is_director = False
        self.muted = False
        self.video_muted = False
        self.view_list: List[str] = []  # List of streams this participant is viewing
        self.created_at = datetime.utcnow()
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to participant"""
        try:
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to {self.stream_id}: {e}")
    
    async def close(self):
        """Close participant connection"""
        if self.pc:
            await self.pc.close()
        if self.websocket.open:
            await self.websocket.close()


class VDONinjaBridge:
    """Bridge between VDO.Ninja protocol and Ooblex"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.rooms: Dict[str, VDONinjaRoom] = {}
        self.participants: Dict[str, VDONinjaParticipant] = {}
        self.stream_to_participant: Dict[str, str] = {}
        
    async def handle_websocket(self, websocket, path):
        """Handle VDO.Ninja WebSocket connection"""
        # Parse query parameters
        query = parse_qs(urlparse(path).query)
        
        # Extract stream ID or generate one
        stream_id = query.get('streamID', [str(uuid.uuid4())])[0]
        room_id = query.get('room', [None])[0]
        scene_id = query.get('scene', [None])[0]
        director = query.get('director', ['false'])[0].lower() == 'true'
        label = query.get('label', [stream_id])[0]
        password = query.get('password', [None])[0]
        
        # Create participant
        participant = VDONinjaParticipant(stream_id, websocket)
        participant.room_id = room_id
        participant.scene_id = scene_id
        participant.is_director = director
        participant.label = label
        
        self.participants[stream_id] = participant
        
        logger.info(f"VDO.Ninja participant connected: {stream_id} (room: {room_id})")
        
        try:
            # Send initial handshake
            await participant.send_message({
                'streamID': stream_id,
                'type': 'connect',
                'version': '23'  # VDO.Ninja version compatibility
            })
            
            # Join room if specified
            if room_id:
                await self.join_room(participant, room_id, password)
            
            # Handle messages
            async for message in websocket:
                await self.handle_message(participant, json.loads(message))
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"VDO.Ninja participant disconnected: {stream_id}")
        except Exception as e:
            logger.error(f"VDO.Ninja error for {stream_id}: {e}")
        finally:
            await self.cleanup_participant(participant)
    
    async def handle_message(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle VDO.Ninja protocol message"""
        msg_type = message.get('type')
        
        if msg_type == 'offer':
            await self.handle_offer(participant, message)
        elif msg_type == 'answer':
            await self.handle_answer(participant, message)
        elif msg_type == 'candidate':
            await self.handle_ice_candidate(participant, message)
        elif msg_type == 'join':
            await self.handle_join_room(participant, message)
        elif msg_type == 'leave':
            await self.handle_leave_room(participant, message)
        elif msg_type == 'chat':
            await self.handle_chat(participant, message)
        elif msg_type == 'pong':
            # Heartbeat response
            pass
        elif msg_type == 'getStats':
            await self.handle_get_stats(participant, message)
        elif msg_type == 'requestStream':
            await self.handle_request_stream(participant, message)
        elif msg_type == 'settings':
            await self.handle_settings_update(participant, message)
        elif msg_type == 'mute':
            await self.handle_mute(participant, message)
        else:
            logger.debug(f"Unknown VDO.Ninja message type: {msg_type}")
    
    async def handle_offer(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle SDP offer from VDO.Ninja"""
        target_id = message.get('targetID')
        sdp = message.get('sdp')
        
        if not target_id or not sdp:
            return
        
        # Create peer connection
        pc = RTCPeerConnection()
        participant.pc = pc
        
        # Handle tracks
        @pc.on("track")
        async def on_track(track: MediaStreamTrack):
            participant.tracks[track.id] = track
            logger.info(f"VDO.Ninja track received: {track.kind} from {participant.stream_id}")
            
            # Apply Ooblex transformations if requested
            if participant.room_id and participant.room_id.startswith("ooblex-"):
                # Extract transformation type from room ID
                transform_type = participant.room_id.split("-", 1)[1]
                track = VideoTransformTrack(track, transform_type)
            
            # Store in Redis for discovery
            await self.redis_client.setex(
                f"vdo:stream:{participant.stream_id}",
                3600,
                json.dumps({
                    'stream_id': participant.stream_id,
                    'room_id': participant.room_id,
                    'label': participant.label,
                    'kind': track.kind,
                    'created_at': participant.created_at.isoformat()
                })
            )
        
        # Set remote description
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        # Send answer back
        await participant.send_message({
            'type': 'answer',
            'targetID': participant.stream_id,
            'sourceID': target_id,
            'sdp': pc.localDescription.sdp
        })
    
    async def handle_answer(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle SDP answer from VDO.Ninja"""
        sdp = message.get('sdp')
        if not sdp or not participant.pc:
            return
        
        await participant.pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type="answer")
        )
    
    async def handle_ice_candidate(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle ICE candidate from VDO.Ninja"""
        # VDO.Ninja uses automatic ICE gathering
        # aiortc handles this automatically
        pass
    
    async def join_room(self, participant: VDONinjaParticipant, room_id: str, password: Optional[str] = None):
        """Join a VDO.Ninja room"""
        # Create room if it doesn't exist
        if room_id not in self.rooms:
            self.rooms[room_id] = VDONinjaRoom(room_id, password)
        
        room = self.rooms[room_id]
        
        # Check password
        if room.password and room.password != password:
            await participant.send_message({
                'type': 'error',
                'message': 'Invalid room password'
            })
            return
        
        # Add participant to room
        room.add_participant(participant)
        participant.room_id = room_id
        
        # Notify participant of room state
        await participant.send_message({
            'type': 'roomState',
            **room.get_room_state()
        })
        
        # Notify other participants
        for other_id, other in room.participants.items():
            if other_id != participant.stream_id:
                # Notify existing participant of new member
                await other.send_message({
                    'type': 'participantJoined',
                    'streamID': participant.stream_id,
                    'label': participant.label,
                    'director': participant.is_director
                })
                
                # Notify new participant of existing member
                await participant.send_message({
                    'type': 'participantJoined',
                    'streamID': other.stream_id,
                    'label': other.label,
                    'director': other.is_director
                })
    
    async def handle_join_room(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle room join request"""
        room_id = message.get('room')
        password = message.get('password')
        
        if room_id:
            await self.join_room(participant, room_id, password)
    
    async def handle_leave_room(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle room leave request"""
        if participant.room_id and participant.room_id in self.rooms:
            room = self.rooms[participant.room_id]
            room.remove_participant(participant.stream_id)
            
            # Notify other participants
            for other in room.participants.values():
                await other.send_message({
                    'type': 'participantLeft',
                    'streamID': participant.stream_id
                })
            
            # Clean up empty room
            if not room.participants:
                del self.rooms[participant.room_id]
            
            participant.room_id = None
    
    async def handle_chat(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle chat message"""
        if not participant.room_id or participant.room_id not in self.rooms:
            return
        
        room = self.rooms[participant.room_id]
        chat_message = message.get('message', '')
        
        # Broadcast to all participants in room
        for other in room.participants.values():
            await other.send_message({
                'type': 'chat',
                'sourceID': participant.stream_id,
                'label': participant.label,
                'message': chat_message,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def handle_get_stats(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle stats request"""
        stats = {
            'type': 'stats',
            'streamID': participant.stream_id,
            'connected': participant.pc is not None,
            'tracks': len(participant.tracks),
            'room': participant.room_id,
            'uptime': (datetime.utcnow() - participant.created_at).total_seconds()
        }
        
        await participant.send_message(stats)
    
    async def handle_request_stream(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle stream request"""
        target_id = message.get('targetID')
        
        if target_id and target_id in self.participants:
            target = self.participants[target_id]
            
            # Notify target to start streaming to requester
            await target.send_message({
                'type': 'streamRequest',
                'sourceID': participant.stream_id,
                'label': participant.label
            })
    
    async def handle_settings_update(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle settings update"""
        if participant.room_id and participant.room_id in self.rooms:
            room = self.rooms[participant.room_id]
            
            # Update room settings if director
            if participant.is_director:
                for key, value in message.get('settings', {}).items():
                    room.settings[key] = value
                
                # Broadcast settings to all participants
                for other in room.participants.values():
                    await other.send_message({
                        'type': 'settings',
                        'settings': room.settings
                    })
    
    async def handle_mute(self, participant: VDONinjaParticipant, message: Dict[str, Any]):
        """Handle mute/unmute"""
        audio_muted = message.get('audio', participant.muted)
        video_muted = message.get('video', participant.video_muted)
        
        participant.muted = audio_muted
        participant.video_muted = video_muted
        
        # Notify room participants
        if participant.room_id and participant.room_id in self.rooms:
            room = self.rooms[participant.room_id]
            
            for other in room.participants.values():
                if other.stream_id != participant.stream_id:
                    await other.send_message({
                        'type': 'muteState',
                        'streamID': participant.stream_id,
                        'audio': audio_muted,
                        'video': video_muted
                    })
    
    async def cleanup_participant(self, participant: VDONinjaParticipant):
        """Clean up participant on disconnect"""
        # Leave room
        if participant.room_id:
            await self.handle_leave_room(participant, {'room': participant.room_id})
        
        # Close peer connection
        await participant.close()
        
        # Remove from tracking
        if participant.stream_id in self.participants:
            del self.participants[participant.stream_id]
        
        # Remove from Redis
        await self.redis_client.delete(f"vdo:stream:{participant.stream_id}")
    
    async def heartbeat_task(self):
        """Send periodic heartbeats to all participants"""
        while True:
            try:
                for participant in list(self.participants.values()):
                    try:
                        await participant.send_message({'type': 'ping'})
                    except:
                        # Connection lost, will be cleaned up by main handler
                        pass
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(30)


class VDONinjaMCU:
    """MCU (Multipoint Control Unit) for VDO.Ninja rooms"""
    
    def __init__(self, bridge: VDONinjaBridge):
        self.bridge = bridge
        self.composite_streams: Dict[str, Any] = {}
    
    async def create_composite_stream(self, room_id: str, layout: str = "grid"):
        """Create a composite stream from all room participants"""
        if room_id not in self.bridge.rooms:
            return None
        
        room = self.bridge.rooms[room_id]
        
        # TODO: Implement video compositing
        # This would use FFmpeg or GStreamer to create a composite stream
        # from all participant streams in the room
        
        composite_id = f"composite_{room_id}"
        self.composite_streams[composite_id] = {
            'room_id': room_id,
            'layout': layout,
            'created_at': datetime.utcnow()
        }
        
        return composite_id


async def start_vdo_ninja_bridge(redis_client: redis.Redis, port: int = 8103):
    """Start VDO.Ninja compatibility bridge"""
    bridge = VDONinjaBridge(redis_client)
    
    # Start heartbeat task
    asyncio.create_task(bridge.heartbeat_task())
    
    # Start WebSocket server
    async with websockets.serve(
        bridge.handle_websocket,
        "0.0.0.0",
        port,
        subprotocols=["vdo.ninja"]  # VDO.Ninja subprotocol
    ):
        logger.info(f"VDO.Ninja bridge started on port {port}")
        await asyncio.Future()  # Run forever


# HTTP API for VDO.Ninja management
class VDONinjaAPI:
    """REST API for VDO.Ninja room management"""
    
    def __init__(self, bridge: VDONinjaBridge):
        self.bridge = bridge
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes"""
        self.app.router.add_get('/api/vdo/rooms', self.list_rooms)
        self.app.router.add_post('/api/vdo/rooms', self.create_room)
        self.app.router.add_get('/api/vdo/rooms/{room_id}', self.get_room)
        self.app.router.add_delete('/api/vdo/rooms/{room_id}', self.delete_room)
        self.app.router.add_get('/api/vdo/streams', self.list_streams)
        self.app.router.add_post('/api/vdo/rooms/{room_id}/composite', self.create_composite)
    
    async def list_rooms(self, request: web.Request) -> web.Response:
        """List all VDO.Ninja rooms"""
        rooms = []
        for room_id, room in self.bridge.rooms.items():
            rooms.append({
                'room_id': room_id,
                'participants': len(room.participants),
                'created_at': room.created_at.isoformat(),
                'settings': room.settings
            })
        
        return web.json_response({'rooms': rooms})
    
    async def create_room(self, request: web.Request) -> web.Response:
        """Create a new VDO.Ninja room"""
        data = await request.json()
        room_id = data.get('room_id', str(uuid.uuid4()))
        password = data.get('password')
        
        if room_id in self.bridge.rooms:
            return web.json_response(
                {'error': 'Room already exists'},
                status=409
            )
        
        room = VDONinjaRoom(room_id, password)
        if 'settings' in data:
            room.settings.update(data['settings'])
        
        self.bridge.rooms[room_id] = room
        
        return web.json_response({
            'room_id': room_id,
            'created_at': room.created_at.isoformat(),
            'join_url': f"https://vdo.ninja/?room={room_id}"
        }, status=201)
    
    async def get_room(self, request: web.Request) -> web.Response:
        """Get room details"""
        room_id = request.match_info['room_id']
        
        if room_id not in self.bridge.rooms:
            return web.json_response(
                {'error': 'Room not found'},
                status=404
            )
        
        room = self.bridge.rooms[room_id]
        
        return web.json_response({
            'room_id': room_id,
            'state': room.get_room_state(),
            'created_at': room.created_at.isoformat()
        })
    
    async def delete_room(self, request: web.Request) -> web.Response:
        """Delete a room"""
        room_id = request.match_info['room_id']
        
        if room_id not in self.bridge.rooms:
            return web.json_response(
                {'error': 'Room not found'},
                status=404
            )
        
        room = self.bridge.rooms[room_id]
        
        # Disconnect all participants
        for participant in list(room.participants.values()):
            await participant.send_message({
                'type': 'roomClosed',
                'reason': 'Room deleted by administrator'
            })
            await participant.close()
        
        del self.bridge.rooms[room_id]
        
        return web.Response(status=204)
    
    async def list_streams(self, request: web.Request) -> web.Response:
        """List all VDO.Ninja streams"""
        streams = []
        
        for stream_id, participant in self.bridge.participants.items():
            streams.append({
                'stream_id': stream_id,
                'label': participant.label,
                'room_id': participant.room_id,
                'is_director': participant.is_director,
                'tracks': len(participant.tracks),
                'created_at': participant.created_at.isoformat()
            })
        
        return web.json_response({'streams': streams})
    
    async def create_composite(self, request: web.Request) -> web.Response:
        """Create composite stream for a room"""
        room_id = request.match_info['room_id']
        data = await request.json()
        layout = data.get('layout', 'grid')
        
        mcu = VDONinjaMCU(self.bridge)
        composite_id = await mcu.create_composite_stream(room_id, layout)
        
        if not composite_id:
            return web.json_response(
                {'error': 'Failed to create composite stream'},
                status=500
            )
        
        return web.json_response({
            'composite_id': composite_id,
            'stream_url': f"/whep?stream={composite_id}"
        }, status=201)