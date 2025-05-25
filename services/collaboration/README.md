# Ooblex Real-Time Collaboration

This module enables multiple users to collaborate on video streams in real-time with features like:

- Shared annotations (drawing, text, shapes)
- Synchronized playback controls
- Real-time cursor tracking
- Live chat with mentions
- Session recording and playback

## Architecture

### Components

1. **Collaboration Server** (`collaboration_server.py`)
   - WebSocket server for real-time communication
   - Manages collaboration sessions
   - Handles annotation synchronization
   - Records session events

2. **JavaScript Client** (`html/js/collaboration.js`)
   - Canvas overlay for annotations
   - WebRTC data channels for low latency
   - UI components for collaboration tools
   - Conflict resolution

3. **API Endpoints** (`services/api/main.py`)
   - REST API for session management
   - Permission control
   - Session history retrieval

4. **Data Models** (`models.py`)
   - Type definitions for collaboration objects
   - Permission levels
   - Annotation types

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start Redis server:
```bash
redis-server
```

3. Run the collaboration server:
```bash
python collaboration_server.py
```

4. Run the API server:
```bash
cd ../api
python main.py
```

## Usage

### JavaScript Client

```javascript
// Initialize collaboration client
const collab = new CollaborationClient({
    wsUrl: 'ws://localhost:8765',
    sessionId: 'my-session-123',
    streamId: 'stream-456',
    userId: 'user-789',
    userName: 'John Doe',
    videoElement: document.getElementById('video'),
    onReady: () => {
        console.log('Connected to collaboration session');
    }
});
```

### API Usage

Create a session:
```bash
curl -X POST http://localhost:8000/api/collaboration/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "stream-123",
    "session_name": "Team Review",
    "permissions": {
      "user-456": "editor"
    }
  }?user_id=user-123'
```

List sessions:
```bash
curl http://localhost:8000/api/collaboration/sessions?user_id=user-123
```

## Features

### Annotation Tools
- **Pen**: Freehand drawing
- **Arrow**: Directional arrows
- **Rectangle**: Box highlights
- **Circle**: Circular highlights
- **Text**: Text annotations

### Playback Synchronization
- Play/pause synchronization
- Seek position sync
- Playback speed control

### Permission Levels
- **Owner**: Full control, manage permissions
- **Editor**: Add/edit/delete annotations
- **Viewer**: View-only access

### Session Recording
- Record all collaboration events
- Playback recorded sessions
- Export session history

## WebSocket Protocol

### Client -> Server Messages

```json
{
  "type": "annotation",
  "annotation_type": "drawing",
  "data": {
    "color": "#FF0000",
    "points": [{"x": 0.1, "y": 0.2}]
  },
  "stream_time": 45.2
}
```

### Server -> Client Messages

```json
{
  "type": "annotation_added",
  "annotation": {
    "id": "anno-123",
    "user_id": "user-456",
    "type": "drawing",
    "data": {...}
  },
  "user_info": {
    "name": "John Doe",
    "color": "#4ECDC4"
  }
}
```

## Deployment

### Using systemd

1. Copy the service file:
```bash
sudo cp launch_scripts/collaboration.service /etc/systemd/system/
```

2. Enable and start:
```bash
sudo systemctl enable collaboration
sudo systemctl start collaboration
```

### Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "collaboration_server.py"]
```

## Security Considerations

1. **Authentication**: Implement proper user authentication
2. **Authorization**: Validate permissions for each action
3. **Rate Limiting**: Prevent annotation spam
4. **Input Validation**: Sanitize all user inputs
5. **HTTPS/WSS**: Use encrypted connections in production

## Performance Optimization

1. **Annotation Batching**: Group rapid annotations
2. **Cursor Throttling**: Limit cursor update frequency
3. **Canvas Optimization**: Use requestAnimationFrame
4. **Redis Persistence**: Configure appropriate persistence
5. **WebSocket Compression**: Enable per-message deflate

## Future Enhancements

1. Voice/video chat integration
2. Advanced shape tools (polygons, curves)
3. Annotation templates
4. AI-powered auto-annotations
5. Integration with video analytics
6. Mobile app support