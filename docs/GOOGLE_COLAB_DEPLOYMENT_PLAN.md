# Google Colab Deployment Plan for Ooblex

## Executive Summary

This document outlines the comprehensive plan to deploy Ooblex on Google Colab, enabling:
- **WebRTC video ingestion** from browsers/devices
- **GPU-accelerated AI processing** using Colab's free/paid GPUs
- **MJPEG stream output** for universal viewing compatibility

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Key Challenges & Solutions](#2-key-challenges--solutions)
3. [Implementation Components](#3-implementation-components)
4. [WebRTC Ingestion Strategy](#4-webrtc-ingestion-strategy)
5. [GPU Processing Pipeline](#5-gpu-processing-pipeline)
6. [MJPEG Output Solution](#6-mjpeg-output-solution)
7. [Colab Notebook Structure](#7-colab-notebook-structure)
8. [Dependencies & Setup](#8-dependencies--setup)
9. [Limitations & Workarounds](#9-limitations--workarounds)
10. [Cost Analysis](#10-cost-analysis)
11. [Implementation Timeline](#11-implementation-timeline)
12. [Alternative Approaches](#12-alternative-approaches)

---

## 1. Architecture Overview

### Target Architecture for Colab

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GOOGLE COLAB RUNTIME                          │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    Colab Notebook (Python)                       │ │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐ │
│  │  │  WebRTC       │  │  ML Worker    │  │  MJPEG Server         │ │
│  │  │  Signaling    │  │  (GPU)        │  │  (HTTP Streaming)     │ │
│  │  │  Server       │  │               │  │                       │ │
│  │  │  (aiortc)     │  │  TensorFlow   │  │  aiohttp server       │ │
│  │  │               │  │  PyTorch      │  │  multipart/x-mixed    │ │
│  │  │  STUN/TURN    │  │  MediaPipe    │  │  -replace             │ │
│  │  └───────┬───────┘  └───────┬───────┘  └───────────┬───────────┘ │
│  │          │                  │                      │             │ │
│  │          └──────────────────┴──────────────────────┘             │ │
│  │                             │                                     │ │
│  │                    ┌────────┴────────┐                           │ │
│  │                    │  In-Memory      │                           │ │
│  │                    │  Frame Queue    │                           │ │
│  │                    │  (asyncio.Queue)│                           │ │
│  │                    └─────────────────┘                           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                │                                     │
│  ┌─────────────────────────────┴────────────────────────────────┐   │
│  │                     NGROK TUNNEL                              │   │
│  │   TCP:8443 (WebRTC Signaling)    HTTP:8081 (MJPEG Stream)    │   │
│  └─────────────────────────────┬────────────────────────────────┘   │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │       INTERNET          │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Browser      │    │  OBS Studio       │    │  VLC/ffplay       │
│  (WebRTC      │    │  (WHIP output)    │    │  (MJPEG viewer)   │
│  getUserMedia)│    │                   │    │                   │
└───────────────┘    └───────────────────┘    └───────────────────┘
```

### Simplified Single-Process Architecture

Unlike the full microservices deployment, Colab requires a **monolithic** approach:

| Full Ooblex | Colab Version |
|-------------|---------------|
| Redis for frame queue | asyncio.Queue (in-memory) |
| RabbitMQ for tasks | Direct function calls |
| Multiple Docker containers | Single Python process |
| Kubernetes scaling | Single GPU instance |
| Nginx reverse proxy | ngrok tunneling |

---

## 2. Key Challenges & Solutions

### Challenge 1: No Inbound Network Access

**Problem:** Colab VMs cannot receive direct inbound connections.

**Solution:** Use **ngrok** or **cloudflared** tunneling.

```python
# Install and configure ngrok
!pip install pyngrok
from pyngrok import ngrok

# Create tunnels for WebRTC signaling and MJPEG
webrtc_tunnel = ngrok.connect(8443, "tcp")  # WebRTC signaling
mjpeg_tunnel = ngrok.connect(8081, "http")  # MJPEG streaming

print(f"WebRTC Signaling: {webrtc_tunnel.public_url}")
print(f"MJPEG Stream: {mjpeg_tunnel.public_url}/stream.mjpg")
```

### Challenge 2: WebRTC NAT Traversal

**Problem:** WebRTC requires STUN/TURN for NAT traversal. Colab is behind NAT.

**Understanding STUN vs TURN:**

| Protocol | Purpose | When It Works | Cost |
|----------|---------|---------------|------|
| **STUN** | Discovers public IP/port | Full cone, restricted cone, port-restricted NAT | Free (Google, Cloudflare) |
| **TURN** | Relays all media traffic | Symmetric NAT (like Colab), firewalls | Free via VDO.Ninja |

- **STUN alone** works ~80% of the time for typical home networks
- **TURN fallback** needed when both peers are behind symmetric NAT (Colab's case)
- WebRTC automatically tries STUN first, falls back to TURN if needed

**Solution: Use Free STUN + VDO.Ninja TURN Servers**

VDO.Ninja provides **free TURN servers** at `https://turnservers.vdo.ninja/` that can be fetched dynamically:

```python
import aiohttp
import asyncio

async def get_ice_servers():
    """Fetch free TURN servers from VDO.Ninja."""
    # Free STUN servers (Google, Cloudflare)
    ice_servers = [
        {"urls": "stun:stun.l.google.com:19302"},
        {"urls": "stun:stun1.l.google.com:19302"},
        {"urls": "stun:stun.cloudflare.com:3478"},
    ]

    # Fetch free TURN servers from VDO.Ninja
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://turnservers.vdo.ninja/") as resp:
                data = await resp.json()
                for server in data.get("servers", []):
                    ice_servers.append({
                        "urls": server["url"],
                        "username": server["username"],
                        "credential": server["credential"]
                    })
    except Exception as e:
        # Fallback to hardcoded VDO.Ninja TURN servers
        ice_servers.extend([
            {
                "urls": "turn:turn-use1.vdo.ninja:3478",
                "username": "vdoninja",
                "credential": "EastSideRepresentZ"
            },
            {
                "urls": "turn:turn-use2.vdo.ninja:3478",
                "username": "vdoninja",
                "credential": "pleaseUseYourOwn"
            }
        ])

    return ice_servers

# Example ICE configuration (hardcoded fallback)
ICE_SERVERS = [
    # Free STUN (address discovery)
    {"urls": "stun:stun.l.google.com:19302"},
    {"urls": "stun:stun.cloudflare.com:3478"},
    # Free TURN from VDO.Ninja (relay when STUN fails)
    {
        "urls": "turn:turn-use1.vdo.ninja:3478",
        "username": "vdoninja",
        "credential": "EastSideRepresentZ"
    },
    {
        "urls": "turns:turn-use1.vdo.ninja:3478",  # TLS variant
        "username": "vdoninja",
        "credential": "EastSideRepresentZ"
    }
]
```

**Why This Works:**
1. Browser/client tries **STUN** first (free, fast, direct connection)
2. If STUN fails (symmetric NAT), falls back to **TURN** relay
3. VDO.Ninja TURN servers are free and reliable
4. No paid TURN service needed for most use cases

### Challenge 3: Session Time Limits

**Problem:**
- Free Colab: ~12 hours max, disconnects after 90 min idle
- Colab Pro: ~24 hours, less aggressive disconnection

**Solutions:**
1. **Auto-reconnect logic** in client
2. **State checkpointing** to Google Drive
3. **Heartbeat mechanism** to prevent idle disconnect
4. **Colab Pro+** for longer sessions

```python
# Keep-alive mechanism
import time
from IPython.display import display, Javascript

def keep_alive():
    display(Javascript('''
        function ClickConnect(){
            console.log("Keeping session alive...");
            document.querySelector("colab-connect-button").click()
        }
        setInterval(ClickConnect, 60000)
    '''))
```

### Challenge 4: GPU Memory Management

**Problem:** Colab GPUs have limited VRAM (T4: 16GB, A100: 40GB).

**Solutions:**
1. **Model quantization** (INT8, FP16)
2. **Batch size optimization**
3. **Memory-efficient inference** (gradient checkpointing)
4. **Model unloading** when not in use

```python
# TensorFlow memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PyTorch memory optimization
import torch
torch.cuda.empty_cache()
```

### Challenge 5: Package Installation Time

**Problem:** Every Colab session requires fresh package installation.

**Solutions:**
1. **Cache packages to Google Drive**
2. **Use pre-built Docker images** (Colab Pro with custom containers)
3. **Minimal dependency installation**
4. **Parallel installation**

```python
# Cache packages to Google Drive
import sys
sys.path.insert(0, '/content/drive/MyDrive/colab_packages')

# Install with caching
!pip install --target=/content/drive/MyDrive/colab_packages aiortc aiohttp
```

---

## 3. Implementation Components

### Component 1: Colab Notebook Entry Point

**File:** `colab/ooblex_colab.ipynb`

```python
# Cell 1: GPU Check & Setup
!nvidia-smi
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 2: Install Dependencies
!pip install -q aiortc aiohttp opencv-python-headless pillow pyngrok
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -q mediapipe tensorflow

# Cell 3: Clone/Update Ooblex
!git clone https://github.com/ooblex/ooblex.git /content/ooblex 2>/dev/null || git -C /content/ooblex pull

# Cell 4: Start Services
%cd /content/ooblex
from colab.colab_server import OoblexColabServer
server = OoblexColabServer()
await server.start()
```

### Component 2: Unified Colab Server

**File:** `colab/colab_server.py`

```python
"""
Unified Ooblex server for Google Colab deployment.
Combines WebRTC ingestion, GPU processing, and MJPEG output in a single process.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import cv2

from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling
from aiohttp import web
import torch

logger = logging.getLogger(__name__)


@dataclass
class ColabConfig:
    """Configuration for Colab deployment."""
    webrtc_port: int = 8443
    mjpeg_port: int = 8081
    frame_queue_size: int = 30
    jpeg_quality: int = 85
    target_fps: int = 30
    model_name: str = "face_detection"  # Default model
    use_gpu: bool = True


class FrameProcessor:
    """GPU-accelerated frame processing."""

    def __init__(self, config: ColabConfig):
        self.config = config
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model(config.model_name)

    def _load_model(self, model_name: str):
        """Load ML model for processing."""
        if model_name == "face_detection":
            # Use MediaPipe for face detection (GPU accelerated)
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection
            self.model = self.mp_face.FaceDetection(min_detection_confidence=0.5)
        elif model_name == "style_transfer":
            # Load PyTorch style transfer model
            self.model = torch.hub.load('pytorch/vision', 'vgg19', pretrained=True)
            self.model = self.model.to(self.device).eval()
        # Add more models as needed

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with the loaded model."""
        if self.model is None:
            return frame

        # Example: Face detection with bounding boxes
        if hasattr(self, 'mp_face'):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.process(rgb_frame)

            if results.detections:
                h, w = frame.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        return frame


class WebRTCHandler:
    """Handle WebRTC connections and frame extraction."""

    def __init__(self, frame_queue: asyncio.Queue, config: ColabConfig):
        self.frame_queue = frame_queue
        self.config = config
        self.pcs: Dict[str, RTCPeerConnection] = {}

    async def handle_offer(self, request: web.Request) -> web.Response:
        """Handle WebRTC offer from client."""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        pc_id = f"pc_{id(pc)}"
        self.pcs[pc_id] = pc

        @pc.on("track")
        async def on_track(track):
            if track.kind == "video":
                asyncio.create_task(self._process_track(track, pc_id))

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ["failed", "closed"]:
                await self._cleanup_pc(pc_id)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response({
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "pc_id": pc_id
        })

    async def _process_track(self, track, pc_id: str):
        """Extract frames from video track and add to queue."""
        frame_count = 0
        while True:
            try:
                frame = await track.recv()
                frame_count += 1

                # Sample every Nth frame to reduce load
                if frame_count % 2 == 0:
                    img = frame.to_ndarray(format="bgr24")

                    # Non-blocking put to queue
                    try:
                        self.frame_queue.put_nowait(img)
                    except asyncio.QueueFull:
                        # Drop oldest frame
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(img)
                        except:
                            pass
            except Exception as e:
                logger.error(f"Track error: {e}")
                break

    async def _cleanup_pc(self, pc_id: str):
        """Cleanup peer connection."""
        if pc_id in self.pcs:
            await self.pcs[pc_id].close()
            del self.pcs[pc_id]


class MJPEGStreamer:
    """HTTP MJPEG streaming server."""

    def __init__(self, processed_queue: asyncio.Queue, config: ColabConfig):
        self.processed_queue = processed_queue
        self.config = config
        self.clients: list = []
        self.latest_frame: Optional[bytes] = None

    async def mjpeg_handler(self, request: web.Request) -> web.StreamResponse:
        """Handle MJPEG stream requests."""
        response = web.StreamResponse()
        response.content_type = 'multipart/x-mixed-replace; boundary=frame'
        await response.prepare(request)

        self.clients.append(response)

        try:
            while True:
                if self.latest_frame:
                    await response.write(
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        self.latest_frame +
                        b'\r\n'
                    )
                await asyncio.sleep(1 / self.config.target_fps)
        except ConnectionResetError:
            pass
        finally:
            self.clients.remove(response)

        return response

    async def snapshot_handler(self, request: web.Request) -> web.Response:
        """Handle single frame snapshot requests."""
        if self.latest_frame:
            return web.Response(body=self.latest_frame, content_type='image/jpeg')
        return web.Response(status=503, text="No frame available")

    async def update_frame(self, frame: np.ndarray):
        """Update the latest frame for streaming."""
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        self.latest_frame = jpeg.tobytes()


class OoblexColabServer:
    """Main server orchestrating all components."""

    def __init__(self, config: Optional[ColabConfig] = None):
        self.config = config or ColabConfig()
        self.frame_queue = asyncio.Queue(maxsize=self.config.frame_queue_size)
        self.processed_queue = asyncio.Queue(maxsize=self.config.frame_queue_size)

        self.processor = FrameProcessor(self.config)
        self.webrtc_handler = WebRTCHandler(self.frame_queue, self.config)
        self.mjpeg_streamer = MJPEGStreamer(self.processed_queue, self.config)

        self.running = False
        self.tunnels = {}

    async def start(self):
        """Start all services."""
        self.running = True

        # Start processing loop
        asyncio.create_task(self._processing_loop())

        # Setup HTTP server with WebRTC and MJPEG endpoints
        app = web.Application()
        app.router.add_post('/webrtc/offer', self.webrtc_handler.handle_offer)
        app.router.add_get('/stream.mjpg', self.mjpeg_streamer.mjpeg_handler)
        app.router.add_get('/snapshot.jpg', self.mjpeg_streamer.snapshot_handler)
        app.router.add_get('/health', self._health_handler)
        app.router.add_get('/', self._index_handler)

        # Start ngrok tunnels
        await self._setup_tunnels()

        # Start web server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.config.mjpeg_port)
        await site.start()

        logger.info(f"Server started on port {self.config.mjpeg_port}")
        self._print_urls()

        # Keep running
        while self.running:
            await asyncio.sleep(1)

    async def _processing_loop(self):
        """Main frame processing loop."""
        while self.running:
            try:
                # Get frame from queue
                frame = await asyncio.wait_for(
                    self.frame_queue.get(),
                    timeout=1.0
                )

                # Process with GPU
                processed = self.processor.process_frame(frame)

                # Update MJPEG streamer
                await self.mjpeg_streamer.update_frame(processed)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")

    async def _setup_tunnels(self):
        """Setup ngrok tunnels for external access."""
        try:
            from pyngrok import ngrok

            # Create HTTP tunnel for MJPEG
            http_tunnel = ngrok.connect(self.config.mjpeg_port, "http")
            self.tunnels['mjpeg'] = http_tunnel.public_url

            logger.info(f"MJPEG Tunnel: {http_tunnel.public_url}")

        except ImportError:
            logger.warning("pyngrok not installed, tunneling disabled")
        except Exception as e:
            logger.error(f"Tunnel setup failed: {e}")

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "queue_size": self.frame_queue.qsize(),
            "active_connections": len(self.webrtc_handler.pcs)
        })

    async def _index_handler(self, request: web.Request) -> web.Response:
        """Serve simple client page."""
        html = """
        <!DOCTYPE html>
        <html>
        <head><title>Ooblex Colab</title></head>
        <body>
            <h1>Ooblex on Google Colab</h1>
            <h2>MJPEG Stream</h2>
            <img src="/stream.mjpg" style="max-width: 100%;">
            <h2>WebRTC Input</h2>
            <video id="localVideo" autoplay muted style="max-width: 100%;"></video>
            <button onclick="startWebRTC()">Start Camera</button>
            <script>
            async function startWebRTC() {
                const stream = await navigator.mediaDevices.getUserMedia({video: true});
                document.getElementById('localVideo').srcObject = stream;

                const pc = new RTCPeerConnection({
                    iceServers: [{urls: 'stun:stun.l.google.com:19302'}]
                });

                stream.getTracks().forEach(track => pc.addTrack(track, stream));

                const offer = await pc.createOffer();
                await pc.setLocalDescription(offer);

                const response = await fetch('/webrtc/offer', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        sdp: pc.localDescription.sdp,
                        type: pc.localDescription.type
                    })
                });

                const answer = await response.json();
                await pc.setRemoteDescription(new RTCSessionDescription(answer));
            }
            </script>
        </body>
        </html>
        """
        return web.Response(text=html, content_type='text/html')

    def _print_urls(self):
        """Print access URLs."""
        print("\n" + "="*60)
        print("OOBLEX COLAB SERVER STARTED")
        print("="*60)
        if self.tunnels.get('mjpeg'):
            print(f"\nMJPEG Stream: {self.tunnels['mjpeg']}/stream.mjpg")
            print(f"Snapshot: {self.tunnels['mjpeg']}/snapshot.jpg")
            print(f"Web UI: {self.tunnels['mjpeg']}/")
        else:
            print(f"\nLocal MJPEG: http://localhost:{self.config.mjpeg_port}/stream.mjpg")
        print("\n" + "="*60)

    async def stop(self):
        """Stop all services."""
        self.running = False

        # Close all peer connections
        for pc_id in list(self.webrtc_handler.pcs.keys()):
            await self.webrtc_handler._cleanup_pc(pc_id)

        # Close ngrok tunnels
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except:
            pass
```

### Component 3: Client-Side WebRTC Page

**File:** `colab/static/client.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ooblex Colab Client</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .video-container { display: flex; gap: 20px; flex-wrap: wrap; }
        video, img { max-width: 100%; border: 2px solid #333; border-radius: 8px; }
        .panel { background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; margin: 5px; }
        button:hover { background: #007bff; color: white; }
        #status { padding: 10px; border-radius: 4px; margin: 10px 0; }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        .connecting { background: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <h1>Ooblex on Google Colab</h1>

    <div id="status" class="disconnected">Disconnected</div>

    <div class="panel">
        <h2>Configuration</h2>
        <label>Server URL: <input type="text" id="serverUrl" value="" placeholder="ngrok URL"></label>
        <br><br>
        <label>Processing Model:
            <select id="modelSelect">
                <option value="face_detection">Face Detection</option>
                <option value="background_blur">Background Blur</option>
                <option value="edge_detection">Edge Detection</option>
                <option value="style_transfer">Style Transfer</option>
            </select>
        </label>
    </div>

    <div class="panel">
        <h2>Controls</h2>
        <button onclick="startCamera()">Start Camera</button>
        <button onclick="stopCamera()">Stop Camera</button>
        <button onclick="connectWebRTC()">Connect WebRTC</button>
        <button onclick="disconnect()">Disconnect</button>
    </div>

    <div class="video-container">
        <div>
            <h3>Local Camera (Input)</h3>
            <video id="localVideo" autoplay muted playsinline></video>
        </div>
        <div>
            <h3>Processed Output (MJPEG)</h3>
            <img id="mjpegOutput" alt="Waiting for stream...">
        </div>
    </div>

    <div class="panel">
        <h2>Statistics</h2>
        <div id="stats">
            <p>Frames sent: <span id="framesSent">0</span></p>
            <p>Connection state: <span id="connectionState">-</span></p>
            <p>ICE state: <span id="iceState">-</span></p>
        </div>
    </div>

    <script>
        let localStream = null;
        let peerConnection = null;
        let framesSent = 0;

        const config = {
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' }
            ]
        };

        function updateStatus(status, className) {
            const el = document.getElementById('status');
            el.textContent = status;
            el.className = className;
        }

        async function startCamera() {
            try {
                localStream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 1280, height: 720, facingMode: 'user' },
                    audio: false
                });
                document.getElementById('localVideo').srcObject = localStream;
                updateStatus('Camera started', 'connected');
            } catch (err) {
                console.error('Camera error:', err);
                updateStatus('Camera error: ' + err.message, 'disconnected');
            }
        }

        function stopCamera() {
            if (localStream) {
                localStream.getTracks().forEach(track => track.stop());
                localStream = null;
                document.getElementById('localVideo').srcObject = null;
            }
            updateStatus('Camera stopped', 'disconnected');
        }

        async function connectWebRTC() {
            if (!localStream) {
                alert('Please start camera first');
                return;
            }

            const serverUrl = document.getElementById('serverUrl').value;
            if (!serverUrl) {
                alert('Please enter the server URL (ngrok URL)');
                return;
            }

            updateStatus('Connecting...', 'connecting');

            peerConnection = new RTCPeerConnection(config);

            // Add tracks
            localStream.getTracks().forEach(track => {
                peerConnection.addTrack(track, localStream);
            });

            // Monitor connection state
            peerConnection.onconnectionstatechange = () => {
                document.getElementById('connectionState').textContent = peerConnection.connectionState;
                if (peerConnection.connectionState === 'connected') {
                    updateStatus('Connected', 'connected');
                    // Start MJPEG stream
                    document.getElementById('mjpegOutput').src = serverUrl + '/stream.mjpg';
                } else if (peerConnection.connectionState === 'failed') {
                    updateStatus('Connection failed', 'disconnected');
                }
            };

            peerConnection.oniceconnectionstatechange = () => {
                document.getElementById('iceState').textContent = peerConnection.iceConnectionState;
            };

            // Create and send offer
            try {
                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);

                // Wait for ICE gathering
                await new Promise(resolve => {
                    if (peerConnection.iceGatheringState === 'complete') {
                        resolve();
                    } else {
                        peerConnection.onicegatheringstatechange = () => {
                            if (peerConnection.iceGatheringState === 'complete') resolve();
                        };
                    }
                });

                const response = await fetch(serverUrl + '/webrtc/offer', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sdp: peerConnection.localDescription.sdp,
                        type: peerConnection.localDescription.type,
                        model: document.getElementById('modelSelect').value
                    })
                });

                const answer = await response.json();
                await peerConnection.setRemoteDescription(new RTCSessionDescription(answer));

            } catch (err) {
                console.error('WebRTC error:', err);
                updateStatus('Connection error: ' + err.message, 'disconnected');
            }
        }

        function disconnect() {
            if (peerConnection) {
                peerConnection.close();
                peerConnection = null;
            }
            document.getElementById('mjpegOutput').src = '';
            updateStatus('Disconnected', 'disconnected');
        }
    </script>
</body>
</html>
```

---

## 4. WebRTC Ingestion Strategy

### Option A: Direct WebRTC with TURN (Recommended)

**Pros:**
- Low latency (~200-400ms)
- Bidirectional communication
- Standard browser APIs

**Cons:**
- Requires TURN server for reliable NAT traversal
- More complex setup

**Implementation:**

```python
# WebRTC configuration with TURN fallback
ICE_CONFIG = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:your-turn-server:3478"],
            "username": "user",
            "credential": "password"
        }
    ]
}
```

### Option B: HTTP-Based Frame Upload (Simpler)

**Pros:**
- Works through any firewall
- Simpler implementation
- No TURN server needed

**Cons:**
- Higher latency (~500ms-2s)
- More bandwidth usage

**Implementation:**

```python
# HTTP frame upload endpoint
@app.post("/upload_frame")
async def upload_frame(request: web.Request):
    data = await request.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    await frame_queue.put(frame)
    return web.Response(text="OK")
```

```javascript
// Client-side frame capture and upload
async function captureAndUpload() {
    const canvas = document.createElement('canvas');
    const video = document.getElementById('localVideo');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.8));
    await fetch(serverUrl + '/upload_frame', { method: 'POST', body: blob });
}

// Capture at 15 FPS
setInterval(captureAndUpload, 66);
```

### Option C: WHIP Protocol (Standards-Based)

**Pros:**
- HTTP-based WebRTC
- OBS Studio compatible
- Better enterprise support

**Cons:**
- Still needs TURN for media

**Implementation:**
Use existing `services/webrtc/whip_whep_server.py` adapted for Colab.

---

## 5. GPU Processing Pipeline

### Supported Processing Types

| Type | Framework | GPU Utilization | FPS (T4) |
|------|-----------|-----------------|----------|
| Face Detection | MediaPipe | Medium | 60+ |
| Background Blur | MediaPipe | Medium | 45+ |
| Style Transfer | PyTorch | High | 15-20 |
| Object Detection | TensorFlow | High | 30+ |
| Segmentation | ONNX | High | 25+ |
| Edge Detection | OpenCV | Low (CPU) | 100+ |

### GPU Memory Optimization

```python
# TensorFlow GPU configuration
import tensorflow as tf

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Or set hard limit
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB
    )

# PyTorch memory management
import torch
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

# Clear cache periodically
def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
```

### Model Loading Strategy

```python
class ModelManager:
    """Lazy load models to conserve GPU memory."""

    def __init__(self):
        self.models = {}
        self.current_model = None

    def get_model(self, model_name: str):
        if model_name not in self.models:
            self._load_model(model_name)

        # Unload other models if GPU memory is low
        if self._get_gpu_memory_used() > 0.8:  # 80% threshold
            self._unload_inactive_models(keep=model_name)

        return self.models[model_name]

    def _get_gpu_memory_used(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        return 0.0
```

---

## 6. MJPEG Output Solution

### Implementation Details

```python
class MJPEGServer:
    """MJPEG streaming server for processed frames."""

    def __init__(self, port: int = 8081, quality: int = 85):
        self.port = port
        self.quality = quality
        self.latest_frame: Optional[bytes] = None
        self.frame_event = asyncio.Event()

    async def stream_handler(self, request: web.Request) -> web.StreamResponse:
        """Handle MJPEG stream requests."""
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        await response.prepare(request)

        while True:
            await self.frame_event.wait()
            self.frame_event.clear()

            if self.latest_frame:
                try:
                    await response.write(
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n'
                        b'Content-Length: ' + str(len(self.latest_frame)).encode() + b'\r\n'
                        b'\r\n' + self.latest_frame + b'\r\n'
                    )
                except ConnectionResetError:
                    break

        return response

    def update_frame(self, frame: np.ndarray):
        """Encode and update latest frame."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        _, jpeg = cv2.imencode('.jpg', frame, encode_params)
        self.latest_frame = jpeg.tobytes()
        self.frame_event.set()
```

### MJPEG Consumption Options

**1. Browser (img tag):**
```html
<img src="https://your-ngrok-url.ngrok.io/stream.mjpg">
```

**2. VLC:**
```bash
vlc https://your-ngrok-url.ngrok.io/stream.mjpg
```

**3. FFmpeg:**
```bash
ffmpeg -i https://your-ngrok-url.ngrok.io/stream.mjpg -c:v libx264 output.mp4
```

**4. OpenCV:**
```python
cap = cv2.VideoCapture('https://your-ngrok-url.ngrok.io/stream.mjpg')
while True:
    ret, frame = cap.read()
    cv2.imshow('Stream', frame)
```

---

## 7. Colab Notebook Structure

### Complete Notebook: `colab/Ooblex_WebRTC_MJPEG.ipynb`

```python
# %% [markdown]
# # Ooblex on Google Colab
# Real-time AI video processing with WebRTC input and MJPEG output

# %% [markdown]
# ## 1. Check GPU Availability

# %%
!nvidia-smi
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Install Dependencies

# %%
# Core dependencies
!pip install -q aiortc aiohttp opencv-python-headless pillow numpy

# ML frameworks (use Colab's pre-installed when possible)
!pip install -q mediapipe

# Tunneling
!pip install -q pyngrok

# %% [markdown]
# ## 3. Configure ngrok (Required for external access)

# %%
# Get your free ngrok auth token from https://ngrok.com/
# Then run: ngrok config add-authtoken YOUR_TOKEN
from pyngrok import ngrok

# Set your ngrok auth token
NGROK_AUTH_TOKEN = ""  # @param {type:"string"}

if NGROK_AUTH_TOKEN:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    print("ngrok configured!")
else:
    print("WARNING: Set NGROK_AUTH_TOKEN for external access")

# %% [markdown]
# ## 4. Mount Google Drive (Optional - for model persistence)

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ## 5. Clone Ooblex Repository

# %%
!git clone https://github.com/ooblex/ooblex.git /content/ooblex 2>/dev/null || git -C /content/ooblex pull
%cd /content/ooblex

# %% [markdown]
# ## 6. Start Ooblex Server

# %%
import asyncio
import sys
sys.path.insert(0, '/content/ooblex/colab')

from colab_server import OoblexColabServer, ColabConfig

# Configure server
config = ColabConfig(
    webrtc_port=8443,
    mjpeg_port=8081,
    jpeg_quality=85,
    target_fps=30,
    model_name="face_detection",  # Options: face_detection, background_blur, edge_detection
    use_gpu=True
)

# Create and start server
server = OoblexColabServer(config)

# Run in background
import nest_asyncio
nest_asyncio.apply()

loop = asyncio.get_event_loop()
task = loop.create_task(server.start())

# %% [markdown]
# ## 7. Access Your Stream
#
# After starting the server, you'll see URLs printed above.
#
# - **MJPEG Stream**: Open in browser or VLC
# - **Web UI**: Use the built-in client interface
# - **Snapshot**: Get a single frame as JPEG

# %% [markdown]
# ## 8. Stop Server (when done)

# %%
await server.stop()

# %% [markdown]
# ## Alternative: HTTP-Based Frame Upload (No WebRTC)
#
# If WebRTC doesn't work due to NAT issues, use HTTP upload instead:

# %%
from aiohttp import web
import cv2
import numpy as np

frame_queue = asyncio.Queue(maxsize=30)

async def upload_handler(request):
    data = await request.read()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    try:
        frame_queue.put_nowait(frame)
    except asyncio.QueueFull:
        frame_queue.get_nowait()
        frame_queue.put_nowait(frame)
    return web.Response(text="OK")

# Add this route to the server
# app.router.add_post('/upload', upload_handler)
```

---

## 8. Dependencies & Setup

### Minimal Requirements

```
# Core
aiortc>=1.6.0          # WebRTC
aiohttp>=3.8.0         # HTTP server
opencv-python-headless # Image processing
pillow>=9.0.0          # Image encoding
numpy>=1.21.0          # Array operations

# Tunneling
pyngrok>=5.1.0         # ngrok Python wrapper

# GPU ML (usually pre-installed in Colab)
torch>=2.0.0           # PyTorch
tensorflow>=2.10.0     # TensorFlow
mediapipe>=0.10.0      # Google ML solutions

# Async support
nest-asyncio>=1.5.0    # Nested event loops for Jupyter
```

### Installation Script

```bash
#!/bin/bash
# install_colab_deps.sh

pip install -q \
    aiortc>=1.6.0 \
    aiohttp>=3.8.0 \
    opencv-python-headless \
    pillow>=9.0.0 \
    pyngrok>=5.1.0 \
    mediapipe>=0.10.0 \
    nest-asyncio>=1.5.0
```

---

## 9. Limitations & Workarounds

### Limitation 1: Session Timeout

| Tier | Max Duration | Idle Timeout |
|------|-------------|--------------|
| Free | ~12 hours | 90 minutes |
| Pro | ~24 hours | 90 minutes |
| Pro+ | ~24 hours | 90 minutes |

**Workaround:**
```python
# Anti-idle script (run in separate cell)
from IPython.display import display, Javascript
import time

def keep_alive():
    display(Javascript('''
        function ClickConnect(){
            var buttons = document.querySelectorAll("colab-connect-button");
            buttons.forEach(function(btn) { btn.click(); });
        }
        setInterval(ClickConnect, 60000);
    '''))

keep_alive()
```

### Limitation 2: No Persistent Public IP

**Workaround:** ngrok provides stable URLs during session.

### Limitation 3: Limited CPU for Non-GPU Tasks

**Workaround:** Offload all processing to GPU when possible.

### Limitation 4: WebRTC NAT Traversal

**Workaround Options:**
1. Use TURN server (Twilio, Xirsys)
2. Use HTTP frame upload fallback
3. Use cloudflared tunnel (alternative to ngrok)

```python
# Cloudflared alternative
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
!chmod +x cloudflared-linux-amd64
!./cloudflared-linux-amd64 tunnel --url http://localhost:8081 &
```

---

## 10. Cost Analysis

### Google Colab Tiers

| Feature | Free | Pro ($10/mo) | Pro+ ($50/mo) |
|---------|------|--------------|---------------|
| GPU Access | Limited T4 | Priority T4, V100 | Priority A100 |
| RAM | 12GB | 25GB | 52GB |
| Disk | 100GB | 100GB | 100GB |
| Timeout | 12h | 24h | 24h |
| Background | No | Yes | Yes |

### TURN Server Options

| Provider | Cost | Notes |
|----------|------|-------|
| **VDO.Ninja** | **Free** | Recommended! Fetch from `https://turnservers.vdo.ninja/` |
| Google STUN | Free | STUN only (no relay), works for most clients |
| Cloudflare STUN | Free | STUN only, `stun:stun.cloudflare.com:3478` |
| Twilio | 500 hrs free, then $0.0004/min | Enterprise option |
| Xirsys | 500MB free, then $0.08/GB | Enterprise option |
| Self-hosted (coturn) | Free (infra cost) | Full control, ~$5-20/month VPS |

**Recommendation:** Use VDO.Ninja free TURN servers - they're reliable and cost nothing.

### Total Monthly Cost Estimate

| Scenario | Cost |
|----------|------|
| **Free Colab + VDO.Ninja TURN** | **$0** |
| Colab Pro (better GPU access) | ~$10/month |
| Colab Pro+ (A100, background) | ~$50/month |

---

## 11. Implementation Timeline

### Phase 1: Core Infrastructure (1-2 days)
- [ ] Create `colab/` directory structure
- [ ] Implement `colab_server.py` unified server
- [ ] Create Jupyter notebook template
- [ ] Test basic HTTP/MJPEG flow

### Phase 2: WebRTC Integration (2-3 days)
- [ ] Integrate aiortc for WebRTC handling
- [ ] Implement signaling via HTTP POST
- [ ] Add TURN server configuration
- [ ] Test WebRTC → Frame Queue flow

### Phase 3: GPU Processing (1-2 days)
- [ ] Integrate MediaPipe models
- [ ] Add PyTorch/TensorFlow model support
- [ ] Implement model switching
- [ ] Optimize GPU memory usage

### Phase 4: Client & Documentation (1-2 days)
- [ ] Create web client HTML page
- [ ] Write user documentation
- [ ] Create demo video
- [ ] Add error handling and logging

### Phase 5: Testing & Optimization (1-2 days)
- [ ] Test various network conditions
- [ ] Optimize latency and FPS
- [ ] Add monitoring/metrics
- [ ] Handle edge cases

**Total Estimated Effort: 6-11 days**

---

## 12. Alternative Approaches

### Alternative 1: Google Cloud Run

**Pros:** Serverless, auto-scaling, persistent URLs
**Cons:** Cold start latency, GPU not available in all regions, cost

### Alternative 2: Google Cloud Vertex AI Workbench

**Pros:** Managed notebooks, persistent storage, better GPU options
**Cons:** Higher cost (~$0.20-0.50/hour)

### Alternative 3: Paperspace Gradient

**Pros:** Free GPU notebooks, persistent storage
**Cons:** Limited GPU availability, smaller community

### Alternative 4: AWS SageMaker Studio Lab

**Pros:** Free GPU (limited), persistent storage
**Cons:** GPU access requires approval, limited hours

### Comparison Matrix

| Platform | Free GPU | WebRTC Possible | Persistent URL | Cost |
|----------|----------|-----------------|----------------|------|
| Colab | Yes (limited) | Via tunnel | No | Free-$50/mo |
| Cloud Run | No | Yes | Yes | Pay-per-use |
| Vertex AI | Yes | Yes | Yes | $0.20-0.50/hr |
| Paperspace | Yes (limited) | Via tunnel | No | Free-$39/mo |
| SageMaker | Yes (limited) | Via tunnel | No | Free (limited) |

---

## Summary

Deploying Ooblex to Google Colab is **feasible** with the following key components:

1. **WebRTC Ingestion**: Use aiortc + ngrok tunnel + TURN server fallback
2. **GPU Processing**: Leverage Colab's T4/A100 with TensorFlow/PyTorch/MediaPipe
3. **MJPEG Output**: aiohttp server with ngrok HTTP tunnel

**Main Trade-offs:**
- Session time limits require reconnection logic
- NAT traversal needs TURN server for reliable WebRTC
- No persistent URLs (changes each session)

**Recommended Approach:**
- Start with HTTP frame upload for simplicity
- Add WebRTC once HTTP flow is working
- Use Colab Pro for longer sessions and better GPU access

---

## Next Steps

1. Review this plan
2. Approve implementation approach
3. Create the `colab/` directory structure
4. Implement components incrementally
5. Test and iterate

---

*Document created: January 2026*
*Last updated: January 2026*
