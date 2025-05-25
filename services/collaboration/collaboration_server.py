import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import websockets
import redis.asyncio as redis
from contextlib import asynccontextmanager

@dataclass
class Annotation:
    id: str
    user_id: str
    type: str  # 'drawing', 'text', 'arrow', 'rectangle', 'circle'
    data: dict
    timestamp: float
    stream_time: float  # position in video stream
    
@dataclass
class CursorPosition:
    user_id: str
    x: float
    y: float
    timestamp: float

@dataclass
class ChatMessage:
    id: str
    user_id: str
    message: str
    mentions: List[str]
    timestamp: float

@dataclass
class PlaybackState:
    is_playing: bool
    position: float
    speed: float
    timestamp: float
    user_id: str  # who changed it

class CollaborationSession:
    def __init__(self, session_id: str, stream_id: str, redis_client):
        self.session_id = session_id
        self.stream_id = stream_id
        self.redis = redis_client
        self.clients: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.user_info: Dict[str, dict] = {}
        self.annotations: List[Annotation] = []
        self.playback_state = PlaybackState(
            is_playing=False, 
            position=0.0, 
            speed=1.0, 
            timestamp=time.time(),
            user_id=""
        )
        self.recording_enabled = False
        self.recording_start_time = None
        self.recorded_events = []
        
    async def add_client(self, user_id: str, websocket, user_info: dict):
        """Add a new client to the session"""
        self.clients[user_id] = websocket
        self.user_info[user_id] = user_info
        
        # Send current state to new client
        await self._send_initial_state(user_id)
        
        # Notify others of new user
        await self._broadcast({
            'type': 'user_joined',
            'user_id': user_id,
            'user_info': user_info,
            'timestamp': time.time()
        }, exclude=user_id)
        
    async def remove_client(self, user_id: str):
        """Remove a client from the session"""
        if user_id in self.clients:
            del self.clients[user_id]
            del self.user_info[user_id]
            
            # Notify others of user leaving
            await self._broadcast({
                'type': 'user_left',
                'user_id': user_id,
                'timestamp': time.time()
            })
    
    async def handle_message(self, user_id: str, message: dict):
        """Handle incoming message from a client"""
        msg_type = message.get('type')
        
        handlers = {
            'annotation': self._handle_annotation,
            'cursor_move': self._handle_cursor_move,
            'chat': self._handle_chat,
            'playback_control': self._handle_playback_control,
            'delete_annotation': self._handle_delete_annotation,
            'clear_annotations': self._handle_clear_annotations,
            'start_recording': self._handle_start_recording,
            'stop_recording': self._handle_stop_recording
        }
        
        handler = handlers.get(msg_type)
        if handler:
            await handler(user_id, message)
        else:
            print(f"Unknown message type: {msg_type}")
    
    async def _handle_annotation(self, user_id: str, message: dict):
        """Handle new annotation"""
        annotation = Annotation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            type=message['annotation_type'],
            data=message['data'],
            timestamp=time.time(),
            stream_time=message.get('stream_time', self.playback_state.position)
        )
        
        self.annotations.append(annotation)
        
        # Store in Redis for persistence
        await self._store_annotation(annotation)
        
        # Broadcast to all clients
        await self._broadcast({
            'type': 'annotation_added',
            'annotation': asdict(annotation),
            'user_info': self.user_info[user_id]
        })
        
        # Record event if recording
        if self.recording_enabled:
            self._record_event('annotation_added', asdict(annotation))
    
    async def _handle_cursor_move(self, user_id: str, message: dict):
        """Handle cursor movement"""
        cursor = CursorPosition(
            user_id=user_id,
            x=message['x'],
            y=message['y'],
            timestamp=time.time()
        )
        
        # Broadcast to others (not to sender)
        await self._broadcast({
            'type': 'cursor_update',
            'cursor': asdict(cursor),
            'user_info': self.user_info[user_id]
        }, exclude=user_id)
    
    async def _handle_chat(self, user_id: str, message: dict):
        """Handle chat message"""
        chat_msg = ChatMessage(
            id=str(uuid.uuid4()),
            user_id=user_id,
            message=message['text'],
            mentions=message.get('mentions', []),
            timestamp=time.time()
        )
        
        # Store in Redis
        await self._store_chat_message(chat_msg)
        
        # Broadcast to all
        await self._broadcast({
            'type': 'chat_message',
            'message': asdict(chat_msg),
            'user_info': self.user_info[user_id]
        })
        
        # Record event if recording
        if self.recording_enabled:
            self._record_event('chat_message', asdict(chat_msg))
    
    async def _handle_playback_control(self, user_id: str, message: dict):
        """Handle playback control changes"""
        control_type = message['control_type']
        
        if control_type == 'play':
            self.playback_state.is_playing = True
        elif control_type == 'pause':
            self.playback_state.is_playing = False
        elif control_type == 'seek':
            self.playback_state.position = message['position']
        elif control_type == 'speed':
            self.playback_state.speed = message['speed']
        
        self.playback_state.timestamp = time.time()
        self.playback_state.user_id = user_id
        
        # Broadcast to all
        await self._broadcast({
            'type': 'playback_update',
            'playback_state': asdict(self.playback_state),
            'user_info': self.user_info[user_id]
        })
        
        # Record event if recording
        if self.recording_enabled:
            self._record_event('playback_update', asdict(self.playback_state))
    
    async def _handle_delete_annotation(self, user_id: str, message: dict):
        """Handle annotation deletion"""
        annotation_id = message['annotation_id']
        
        # Find and remove annotation
        self.annotations = [a for a in self.annotations if a.id != annotation_id]
        
        # Remove from Redis
        await self._delete_annotation(annotation_id)
        
        # Broadcast deletion
        await self._broadcast({
            'type': 'annotation_deleted',
            'annotation_id': annotation_id,
            'user_id': user_id
        })
    
    async def _handle_clear_annotations(self, user_id: str, message: dict):
        """Handle clearing all annotations"""
        self.annotations.clear()
        
        # Clear from Redis
        await self._clear_annotations()
        
        # Broadcast clear
        await self._broadcast({
            'type': 'annotations_cleared',
            'user_id': user_id
        })
    
    async def _handle_start_recording(self, user_id: str, message: dict):
        """Start recording session"""
        if not self.recording_enabled:
            self.recording_enabled = True
            self.recording_start_time = time.time()
            self.recorded_events = []
            
            await self._broadcast({
                'type': 'recording_started',
                'user_id': user_id,
                'timestamp': self.recording_start_time
            })
    
    async def _handle_stop_recording(self, user_id: str, message: dict):
        """Stop recording session"""
        if self.recording_enabled:
            self.recording_enabled = False
            recording_data = {
                'session_id': self.session_id,
                'stream_id': self.stream_id,
                'start_time': self.recording_start_time,
                'end_time': time.time(),
                'events': self.recorded_events
            }
            
            # Store recording
            await self._store_recording(recording_data)
            
            await self._broadcast({
                'type': 'recording_stopped',
                'user_id': user_id,
                'recording_id': recording_data['session_id'],
                'duration': recording_data['end_time'] - recording_data['start_time']
            })
    
    def _record_event(self, event_type: str, data: dict):
        """Record an event during session recording"""
        if self.recording_enabled:
            self.recorded_events.append({
                'type': event_type,
                'data': data,
                'timestamp': time.time() - self.recording_start_time
            })
    
    async def _send_initial_state(self, user_id: str):
        """Send current session state to new user"""
        websocket = self.clients[user_id]
        
        # Send current users
        await websocket.send(json.dumps({
            'type': 'initial_state',
            'users': self.user_info,
            'annotations': [asdict(a) for a in self.annotations],
            'playback_state': asdict(self.playback_state),
            'recording_enabled': self.recording_enabled
        }))
    
    async def _broadcast(self, message: dict, exclude: Optional[str] = None):
        """Broadcast message to all clients except excluded"""
        message_str = json.dumps(message)
        
        # Send to all connected clients
        disconnected = []
        for user_id, websocket in self.clients.items():
            if user_id != exclude:
                try:
                    await websocket.send(message_str)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(user_id)
        
        # Clean up disconnected clients
        for user_id in disconnected:
            await self.remove_client(user_id)
    
    # Redis storage methods
    async def _store_annotation(self, annotation: Annotation):
        """Store annotation in Redis"""
        key = f"collab:session:{self.session_id}:annotation:{annotation.id}"
        await self.redis.setex(key, 86400, json.dumps(asdict(annotation)))  # 24h TTL
    
    async def _delete_annotation(self, annotation_id: str):
        """Delete annotation from Redis"""
        key = f"collab:session:{self.session_id}:annotation:{annotation_id}"
        await self.redis.delete(key)
    
    async def _clear_annotations(self):
        """Clear all annotations from Redis"""
        pattern = f"collab:session:{self.session_id}:annotation:*"
        async for key in self.redis.scan_iter(match=pattern):
            await self.redis.delete(key)
    
    async def _store_chat_message(self, message: ChatMessage):
        """Store chat message in Redis"""
        key = f"collab:session:{self.session_id}:chat:{message.id}"
        await self.redis.setex(key, 86400, json.dumps(asdict(message)))  # 24h TTL
    
    async def _store_recording(self, recording_data: dict):
        """Store session recording"""
        key = f"collab:recording:{recording_data['session_id']}"
        await self.redis.setex(key, 604800, json.dumps(recording_data))  # 7 days TTL


class CollaborationServer:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.sessions: Dict[str, CollaborationSession] = {}
        
    async def start(self):
        """Initialize server resources"""
        self.redis = await redis.from_url(self.redis_url)
        
    async def stop(self):
        """Clean up server resources"""
        if self.redis:
            await self.redis.close()
    
    @asynccontextmanager
    async def lifespan(self):
        """Lifespan context manager"""
        await self.start()
        yield
        await self.stop()
    
    async def handle_websocket(self, websocket, path):
        """Handle new WebSocket connection"""
        try:
            # Wait for initialization message
            init_msg = await websocket.recv()
            init_data = json.loads(init_msg)
            
            session_id = init_data['session_id']
            stream_id = init_data['stream_id']
            user_id = init_data['user_id']
            user_info = init_data.get('user_info', {})
            
            # Get or create session
            if session_id not in self.sessions:
                self.sessions[session_id] = CollaborationSession(
                    session_id, stream_id, self.redis
                )
            
            session = self.sessions[session_id]
            
            # Add client to session
            await session.add_client(user_id, websocket, user_info)
            
            # Handle messages
            async for message in websocket:
                data = json.loads(message)
                await session.handle_message(user_id, data)
                
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            print(f"Error handling websocket: {e}")
        finally:
            # Remove client from session
            if 'session' in locals() and 'user_id' in locals():
                await session.remove_client(user_id)
                
                # Clean up empty sessions
                if not session.clients:
                    del self.sessions[session_id]
    
    async def get_session_recording(self, recording_id: str) -> Optional[dict]:
        """Retrieve a session recording"""
        key = f"collab:recording:{recording_id}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None
    
    async def list_recordings(self, stream_id: Optional[str] = None) -> List[dict]:
        """List available recordings"""
        recordings = []
        pattern = "collab:recording:*"
        
        async for key in self.redis.scan_iter(match=pattern):
            data = await self.redis.get(key)
            if data:
                recording = json.loads(data)
                if not stream_id or recording.get('stream_id') == stream_id:
                    recordings.append({
                        'recording_id': recording['session_id'],
                        'stream_id': recording['stream_id'],
                        'start_time': recording['start_time'],
                        'duration': recording['end_time'] - recording['start_time']
                    })
        
        return sorted(recordings, key=lambda x: x['start_time'], reverse=True)


async def main():
    """Run the collaboration server"""
    server = CollaborationServer()
    
    async with server.lifespan():
        # Start WebSocket server
        async with websockets.serve(
            server.handle_websocket, 
            "localhost", 
            8765,
            ping_interval=20,
            ping_timeout=10
        ):
            print("Collaboration server running on ws://localhost:8765")
            await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())