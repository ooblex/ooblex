#!/bin/bash
set -e

# WebRTC Stack Installation Script
# Installs modern WebRTC dependencies and tools
# Supports Ubuntu 20.04+ and container environments

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Configuration
NODE_VERSION="20"  # LTS version
PYTHON_VERSION="3"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_error "This script must be run as root"
   exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    print_error "Cannot detect OS version"
    exit 1
fi

print_status "Installing WebRTC stack on $OS $VER"

# Update system
print_status "Updating system packages..."
apt-get update -y

# Install basic dependencies
print_status "Installing basic dependencies..."
apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Node.js
print_status "Installing Node.js $NODE_VERSION..."
curl -fsSL https://deb.nodesource.com/setup_${NODE_VERSION}.x | bash -
apt-get install -y nodejs

# Verify Node.js installation
NODE_VER=$(node --version)
NPM_VER=$(npm --version)
print_status "Node.js $NODE_VER and npm $NPM_VER installed"

# Install Python and pip
print_status "Installing Python dependencies..."
apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-pip \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv

# Update pip
python${PYTHON_VERSION} -m pip install --upgrade pip

# Install Python WebRTC packages
print_status "Installing Python WebRTC packages..."
python${PYTHON_VERSION} -m pip install --upgrade \
    websockets \
    aiohttp \
    aiortc \
    pyee \
    dataclasses-json \
    av \
    opencv-python \
    numpy \
    pillow

# Install additional Python packages for signaling
print_status "Installing Python signaling packages..."
python${PYTHON_VERSION} -m pip install --upgrade \
    simple-websocket-server \
    websocket-client \
    amqpstorm \
    redis \
    asyncio \
    uvloop \
    fastapi \
    uvicorn[standard]

# Install GStreamer for WebRTC
print_status "Installing GStreamer WebRTC components..."
apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-nice \
    gstreamer1.0-pulseaudio \
    libgstrtspserver-1.0-0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    python3-gst-1.0

# Install coturn TURN server
print_status "Installing coturn TURN server..."
apt-get install -y coturn

# Configure coturn
print_status "Configuring coturn..."
cp /etc/turnserver.conf /etc/turnserver.conf.backup
cat > /etc/turnserver.conf << 'EOF'
# TURN server configuration
listening-port=3478
tls-listening-port=5349

# Use fingerprint in TURN messages
fingerprint

# Use long-term credential mechanism
lt-cred-mech

# User accounts
user=webrtc:webrtc123

# Realm
realm=example.com

# Log file
log-file=/var/log/turnserver.log
syslog

# Enable verbose logging
verbose

# Relay IP address (change to your server's IP)
# relay-ip=YOUR_SERVER_IP

# External IP (if behind NAT)
# external-ip=YOUR_EXTERNAL_IP

# Allowed peer IPs (optional)
# allowed-peer-ip=192.168.0.0-192.168.255.255

# Relay endpoints range
min-port=49152
max-port=65535

# SSL certificates (for TLS)
# cert=/etc/letsencrypt/live/yourdomain.com/fullchain.pem
# pkey=/etc/letsencrypt/live/yourdomain.com/privkey.pem

# Disable CLI
no-cli

# Disable web admin
# web-admin
# web-admin-ip=127.0.0.1
# web-admin-port=8080
EOF

# Enable coturn service
sed -i 's/#TURNSERVER_ENABLED=1/TURNSERVER_ENABLED=1/g' /etc/default/coturn

# Install Certbot for SSL certificates
print_status "Installing Certbot for SSL certificates..."
if [ "$VER" == "20.04" ] || [ "$VER" == "22.04" ]; then
    apt-get install -y snapd
    snap install core
    snap refresh core
    snap install --classic certbot
    ln -sf /snap/bin/certbot /usr/bin/certbot
else
    apt-get install -y certbot
fi

# Create WebRTC signaling server
print_status "Creating WebRTC signaling server..."
mkdir -p /opt/webrtc-signaling

cat > /opt/webrtc-signaling/signaling_server.py << 'EOF'
#!/usr/bin/env python3
"""
Modern WebRTC Signaling Server using WebSockets
Supports rooms, peer-to-peer connections, and basic authentication
"""

import asyncio
import json
import logging
import ssl
import uuid
from typing import Dict, Set
import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalingServer:
    def __init__(self):
        self.rooms: Dict[str, Set[WebSocketServerProtocol]] = {}
        self.peers: Dict[WebSocketServerProtocol, Dict] = {}
        
    async def register(self, websocket: WebSocketServerProtocol, room_id: str, peer_id: str = None):
        """Register a peer in a room"""
        if peer_id is None:
            peer_id = str(uuid.uuid4())
            
        # Store peer info
        self.peers[websocket] = {
            'id': peer_id,
            'room': room_id
        }
        
        # Add to room
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(websocket)
        
        # Notify others in room
        await self.broadcast_to_room(room_id, {
            'type': 'peer_joined',
            'peer_id': peer_id
        }, exclude=websocket)
        
        # Send room info to new peer
        room_peers = [
            self.peers[ws]['id'] 
            for ws in self.rooms[room_id] 
            if ws != websocket
        ]
        
        await websocket.send(json.dumps({
            'type': 'room_joined',
            'peer_id': peer_id,
            'peers': room_peers
        }))
        
        logger.info(f"Peer {peer_id} joined room {room_id}")
        return peer_id
        
    async def unregister(self, websocket: WebSocketServerProtocol):
        """Unregister a peer"""
        if websocket not in self.peers:
            return
            
        peer_info = self.peers[websocket]
        peer_id = peer_info['id']
        room_id = peer_info['room']
        
        # Remove from room
        if room_id in self.rooms:
            self.rooms[room_id].discard(websocket)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
            else:
                # Notify others
                await self.broadcast_to_room(room_id, {
                    'type': 'peer_left',
                    'peer_id': peer_id
                })
                
        # Remove peer info
        del self.peers[websocket]
        logger.info(f"Peer {peer_id} left room {room_id}")
        
    async def broadcast_to_room(self, room_id: str, message: dict, exclude: WebSocketServerProtocol = None):
        """Broadcast message to all peers in a room"""
        if room_id not in self.rooms:
            return
            
        message_str = json.dumps(message)
        tasks = []
        
        for websocket in self.rooms[room_id]:
            if websocket != exclude and websocket.open:
                tasks.append(websocket.send(message_str))
                
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
    async def relay_message(self, websocket: WebSocketServerProtocol, message: dict):
        """Relay a message to a specific peer"""
        if websocket not in self.peers:
            return
            
        sender_id = self.peers[websocket]['id']
        room_id = self.peers[websocket]['room']
        target_id = message.get('target')
        
        if not target_id:
            return
            
        # Find target peer
        for ws, peer_info in self.peers.items():
            if peer_info['id'] == target_id and peer_info['room'] == room_id:
                # Add sender info
                message['sender'] = sender_id
                await ws.send(json.dumps(message))
                break
                
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a client connection"""
        peer_id = None
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get('type')
                    
                    if msg_type == 'join':
                        room_id = data.get('room_id', 'default')
                        peer_id = await self.register(websocket, room_id, data.get('peer_id'))
                        
                    elif msg_type in ['offer', 'answer', 'ice_candidate']:
                        await self.relay_message(websocket, data)
                        
                    elif msg_type == 'broadcast':
                        if websocket in self.peers:
                            room_id = self.peers[websocket]['room']
                            data['sender'] = self.peers[websocket]['id']
                            await self.broadcast_to_room(room_id, data, exclude=websocket)
                            
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

async def main():
    server = SignalingServer()
    
    # Configure SSL context (optional)
    ssl_context = None
    # Uncomment to enable SSL
    # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # ssl_context.load_cert_chain('/path/to/cert.pem', '/path/to/key.pem')
    
    async with websockets.serve(
        server.handle_client,
        '0.0.0.0',
        8765,
        ssl=ssl_context
    ):
        logger.info("WebRTC Signaling Server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # Run forever

if __name__ == '__main__':
    asyncio.run(main())
EOF

chmod +x /opt/webrtc-signaling/signaling_server.py

# Create systemd service for signaling server
print_status "Creating systemd service for signaling server..."
cat > /etc/systemd/system/webrtc-signaling.service << EOF
[Unit]
Description=WebRTC Signaling Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/webrtc-signaling
ExecStart=/usr/bin/python${PYTHON_VERSION} /opt/webrtc-signaling/signaling_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create a simple WebRTC test client
print_status "Creating WebRTC test client..."
mkdir -p /var/www/html/webrtc

cat > /var/www/html/webrtc/index.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>WebRTC Test Client</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        video {
            width: 45%;
            margin: 10px;
            background: #000;
        }
        button {
            margin: 5px;
            padding: 10px;
        }
        #status {
            margin: 10px 0;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>WebRTC Test Client</h1>
    
    <div id="status">Status: Disconnected</div>
    
    <div>
        <input type="text" id="roomId" placeholder="Room ID" value="test-room">
        <button onclick="connect()">Connect</button>
        <button onclick="disconnect()">Disconnect</button>
    </div>
    
    <div>
        <button onclick="startCall()">Start Call</button>
        <button onclick="hangup()">Hang Up</button>
    </div>
    
    <div>
        <video id="localVideo" autoplay muted></video>
        <video id="remoteVideo" autoplay></video>
    </div>
    
    <script>
        let ws = null;
        let pc = null;
        let localStream = null;
        let peerId = null;
        let remotePeerId = null;
        
        const configuration = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                // Add your TURN server here
                // { urls: 'turn:your-server.com:3478', username: 'webrtc', credential: 'webrtc123' }
            ]
        };
        
        function updateStatus(message) {
            document.getElementById('status').textContent = 'Status: ' + message;
        }
        
        function connect() {
            const roomId = document.getElementById('roomId').value;
            const wsUrl = 'ws://localhost:8765';
            
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                updateStatus('Connected to signaling server');
                ws.send(JSON.stringify({
                    type: 'join',
                    room_id: roomId
                }));
            };
            
            ws.onmessage = async (event) => {
                const data = JSON.parse(event.data);
                
                switch(data.type) {
                    case 'room_joined':
                        peerId = data.peer_id;
                        updateStatus(`Joined room as ${peerId}`);
                        if (data.peers.length > 0) {
                            remotePeerId = data.peers[0];
                            updateStatus(`Found peer: ${remotePeerId}`);
                        }
                        break;
                        
                    case 'peer_joined':
                        remotePeerId = data.peer_id;
                        updateStatus(`Peer joined: ${remotePeerId}`);
                        break;
                        
                    case 'peer_left':
                        updateStatus(`Peer left: ${data.peer_id}`);
                        hangup();
                        break;
                        
                    case 'offer':
                        await handleOffer(data);
                        break;
                        
                    case 'answer':
                        await handleAnswer(data);
                        break;
                        
                    case 'ice_candidate':
                        await handleIceCandidate(data);
                        break;
                }
            };
            
            ws.onerror = (error) => {
                updateStatus('WebSocket error: ' + error);
            };
            
            ws.onclose = () => {
                updateStatus('Disconnected from signaling server');
            };
        }
        
        function disconnect() {
            if (ws) {
                ws.close();
                ws = null;
            }
            hangup();
        }
        
        async function startCall() {
            if (!remotePeerId) {
                alert('No remote peer available');
                return;
            }
            
            await setupPeerConnection();
            
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            
            ws.send(JSON.stringify({
                type: 'offer',
                target: remotePeerId,
                offer: offer
            }));
        }
        
        async function setupPeerConnection() {
            pc = new RTCPeerConnection(configuration);
            
            pc.onicecandidate = (event) => {
                if (event.candidate && remotePeerId) {
                    ws.send(JSON.stringify({
                        type: 'ice_candidate',
                        target: remotePeerId,
                        candidate: event.candidate
                    }));
                }
            };
            
            pc.ontrack = (event) => {
                document.getElementById('remoteVideo').srcObject = event.streams[0];
            };
            
            // Get user media
            localStream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true
            });
            
            document.getElementById('localVideo').srcObject = localStream;
            
            localStream.getTracks().forEach(track => {
                pc.addTrack(track, localStream);
            });
        }
        
        async function handleOffer(data) {
            remotePeerId = data.sender;
            await setupPeerConnection();
            
            await pc.setRemoteDescription(data.offer);
            const answer = await pc.createAnswer();
            await pc.setLocalDescription(answer);
            
            ws.send(JSON.stringify({
                type: 'answer',
                target: remotePeerId,
                answer: answer
            }));
        }
        
        async function handleAnswer(data) {
            await pc.setRemoteDescription(data.answer);
        }
        
        async function handleIceCandidate(data) {
            if (pc) {
                await pc.addIceCandidate(data.candidate);
            }
        }
        
        function hangup() {
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
            }
            
            if (pc) {
                pc.close();
                pc = null;
            }
            
            document.getElementById('localVideo').srcObject = null;
            document.getElementById('remoteVideo').srcObject = null;
        }
    </script>
</body>
</html>
EOF

# Create monitoring script
print_status "Creating WebRTC monitoring script..."
cat > /usr/local/bin/webrtc-monitor << 'EOF'
#!/bin/bash
# WebRTC Stack Monitoring Script

echo "WebRTC Stack Status"
echo "=================="
echo ""

# Check signaling server
echo "Signaling Server:"
if systemctl is-active --quiet webrtc-signaling; then
    echo "  Status: Running"
    echo "  Port: 8765"
else
    echo "  Status: Stopped"
fi
echo ""

# Check TURN server
echo "TURN Server (coturn):"
if systemctl is-active --quiet coturn; then
    echo "  Status: Running"
    echo "  Ports: 3478 (UDP/TCP), 5349 (TLS)"
else
    echo "  Status: Stopped"
fi
echo ""

# Check NGINX
echo "Web Server (nginx):"
if systemctl is-active --quiet nginx; then
    echo "  Status: Running"
    echo "  Test Client: http://localhost/webrtc/"
else
    echo "  Status: Not installed or stopped"
fi
echo ""

# Show network connections
echo "Active WebRTC Connections:"
ss -tunlp | grep -E "(8765|3478|5349)" | head -10
EOF

chmod +x /usr/local/bin/webrtc-monitor

# Enable services
print_status "Enabling services..."
systemctl daemon-reload
systemctl enable coturn
systemctl enable webrtc-signaling

# Start services
print_status "Starting services..."
systemctl start coturn
systemctl start webrtc-signaling

# Verify installation
print_status "Verifying installation..."

# Check Node.js
if node --version &>/dev/null; then
    print_status "Node.js installed successfully"
else
    print_error "Node.js installation failed"
fi

# Check Python packages
if python${PYTHON_VERSION} -c "import websockets, aiortc" 2>/dev/null; then
    print_status "Python WebRTC packages installed successfully"
else
    print_warning "Some Python packages may be missing"
fi

# Check services
if systemctl is-active --quiet webrtc-signaling; then
    print_status "WebRTC signaling server is running"
else
    print_error "WebRTC signaling server is not running"
fi

if systemctl is-active --quiet coturn; then
    print_status "TURN server is running"
else
    print_error "TURN server is not running"
fi

# Display information
print_status "Installation complete!"
echo ""
echo "WebRTC Stack Configuration:"
echo "  - Signaling Server: ws://localhost:8765"
echo "  - TURN Server: localhost:3478"
echo "  - Test Client: /var/www/html/webrtc/index.html"
echo ""
echo "Services:"
echo "  - Signaling: systemctl {start|stop|restart|status} webrtc-signaling"
echo "  - TURN: systemctl {start|stop|restart|status} coturn"
echo ""
echo "Configuration Files:"
echo "  - TURN: /etc/turnserver.conf"
echo "  - Signaling: /opt/webrtc-signaling/signaling_server.py"
echo ""
echo "SSL Certificates:"
echo "  - Generate: certbot certonly --standalone -d yourdomain.com"
echo ""
echo "Monitoring:"
echo "  - Status: webrtc-monitor"
echo "  - Logs: journalctl -u webrtc-signaling -f"
echo "  - TURN logs: journalctl -u coturn -f"
echo ""
echo "Python Usage:"
echo "  import asyncio"
echo "  import websockets"
echo "  from aiortc import RTCPeerConnection"
echo ""
echo "Security Notes:"
echo "  - Change default TURN credentials in /etc/turnserver.conf"
echo "  - Configure firewall rules for ports 3478, 5349, 8765"
echo "  - Use SSL/TLS for production deployments"

print_status "WebRTC stack installation completed successfully!"