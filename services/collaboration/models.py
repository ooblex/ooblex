from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import time

class PermissionLevel(str, Enum):
    """Permission levels for collaboration sessions"""
    OWNER = "owner"      # Full control, can manage permissions
    EDITOR = "editor"    # Can add/edit/delete annotations, control playback
    VIEWER = "viewer"    # Can only view, no editing capabilities

class AnnotationType(str, Enum):
    """Types of annotations supported"""
    DRAWING = "drawing"
    TEXT = "text"
    ARROW = "arrow"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    HIGHLIGHT = "highlight"

@dataclass
class CollaborationSession:
    """Represents a collaboration session"""
    session_id: str
    stream_id: str
    session_name: Optional[str] = None
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    permissions: Dict[str, str] = field(default_factory=dict)  # user_id -> PermissionLevel
    settings: Dict = field(default_factory=dict)
    
    def dict(self):
        return {
            "session_id": self.session_id,
            "stream_id": self.stream_id,
            "session_name": self.session_name,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "permissions": self.permissions,
            "settings": self.settings
        }

@dataclass
class CollaborationPermission:
    """Permission entry for a user in a session"""
    user_id: str
    session_id: str
    permission_level: PermissionLevel
    granted_by: str
    granted_at: float = field(default_factory=time.time)
    
    def can_edit(self) -> bool:
        return self.permission_level in [PermissionLevel.OWNER, PermissionLevel.EDITOR]
    
    def can_manage_permissions(self) -> bool:
        return self.permission_level == PermissionLevel.OWNER

@dataclass
class SessionAnnotation:
    """Annotation in a collaboration session"""
    annotation_id: str
    session_id: str
    user_id: str
    annotation_type: AnnotationType
    data: Dict  # Type-specific data (points, text, coordinates, etc.)
    stream_time: float  # Position in the video stream
    created_at: float = field(default_factory=time.time)
    updated_at: Optional[float] = None
    
    def dict(self):
        return {
            "annotation_id": self.annotation_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "annotation_type": self.annotation_type,
            "data": self.data,
            "stream_time": self.stream_time,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

@dataclass
class SessionMessage:
    """Chat message in a collaboration session"""
    message_id: str
    session_id: str
    user_id: str
    message: str
    mentions: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    edited: bool = False
    edited_at: Optional[float] = None

@dataclass
class SessionEvent:
    """Event in a collaboration session (for recording/playback)"""
    event_id: str
    session_id: str
    event_type: str  # annotation_added, playback_changed, user_joined, etc.
    user_id: str
    data: Dict
    timestamp: float = field(default_factory=time.time)
    stream_time: float = 0.0

@dataclass
class SessionHistory:
    """History/recording of a collaboration session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    events: List[SessionEvent] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    
    def add_event(self, event: SessionEvent):
        self.events.append(event)
        if event.user_id not in self.participants:
            self.participants.append(event.user_id)
    
    def get_duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

@dataclass
class UserCursor:
    """User cursor position for real-time tracking"""
    user_id: str
    session_id: str
    x: float  # Normalized coordinate (0-1)
    y: float  # Normalized coordinate (0-1)
    timestamp: float = field(default_factory=time.time)

@dataclass
class PlaybackState:
    """Synchronized playback state"""
    session_id: str
    is_playing: bool = False
    position: float = 0.0  # Current position in seconds
    speed: float = 1.0     # Playback speed
    last_updated_by: str = ""
    last_updated_at: float = field(default_factory=time.time)
    
    def dict(self):
        return {
            "session_id": self.session_id,
            "is_playing": self.is_playing,
            "position": self.position,
            "speed": self.speed,
            "last_updated_by": self.last_updated_by,
            "last_updated_at": self.last_updated_at
        }