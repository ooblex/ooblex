"""
Ooblex Google Colab Demo Server

A lightweight, self-contained server for running Ooblex demos in Google Colab.
Supports webcam capture via JavaScript, GPU processing, and MJPEG output.

Usage in Colab:
    from ooblex_demo import OoblexDemo
    demo = OoblexDemo()
    demo.start()
"""

import asyncio
import base64
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Optional, Dict, Callable, List
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import mediapipe as mp
    # Check if the new or old API is available
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
    else:
        # Try new API (mediapipe >= 0.10.8)
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision
            MEDIAPIPE_AVAILABLE = True
            MEDIAPIPE_NEW_API = True
        except ImportError:
            MEDIAPIPE_AVAILABLE = False
            MEDIAPIPE_NEW_API = False
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_NEW_API = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DemoConfig:
    """Configuration for the demo server."""
    port: int = 8080
    jpeg_quality: int = 85
    max_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
    default_effect: str = "none"


class EffectsProcessor:
    """GPU-accelerated image effects processor."""

    # Available effects and their descriptions
    EFFECTS = {
        "none": "Original video (no processing)",
        "face_detection": "Detect and highlight faces",
        "edge_detection": "Canny edge detection",
        "cartoon": "Cartoon/comic book effect",
        "grayscale": "Black and white",
        "sepia": "Vintage sepia tone",
        "blur": "Gaussian blur",
        "pixelate": "Pixelation effect",
        "invert": "Invert colors",
        "mirror": "Horizontal flip",
        "emboss": "Emboss effect",
        "sketch": "Pencil sketch effect",
    }

    # GPU-accelerated effects (when MediaPipe available)
    GPU_EFFECTS = {
        "face_mesh": "468-point face mesh landmarks",
        "selfie_segment": "Background removal/blur",
        "pose_detection": "Full body pose estimation",
    }

    def __init__(self):
        self.face_cascade = None
        self.mp_face_mesh = None
        self.mp_selfie_seg = None
        self.mp_pose = None
        self._init_detectors()

    def _init_detectors(self):
        """Initialize detection models."""
        # OpenCV face cascade (always available)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("OpenCV face cascade loaded")
        except Exception as e:
            logger.warning(f"Could not load face cascade: {e}")

        # MediaPipe models (GPU accelerated)
        if MEDIAPIPE_AVAILABLE:
            try:
                # Try the legacy API first (mp.solutions)
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
                    self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        max_num_faces=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.mp_drawing = mp.solutions.drawing_utils
                    self.mp_drawing_styles = mp.solutions.drawing_styles
                    logger.info("MediaPipe Face Mesh loaded (legacy API)")
                else:
                    logger.info("MediaPipe Face Mesh not available (new API not yet supported)")
            except Exception as e:
                logger.warning(f"Could not load MediaPipe Face Mesh: {e}")

            try:
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'selfie_segmentation'):
                    self.mp_selfie_seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                        model_selection=1
                    )
                    logger.info("MediaPipe Selfie Segmentation loaded (legacy API)")
                else:
                    logger.info("MediaPipe Selfie Segmentation not available (new API not yet supported)")
            except Exception as e:
                logger.warning(f"Could not load MediaPipe Selfie Segmentation: {e}")

    def get_available_effects(self) -> Dict[str, str]:
        """Return dict of available effects and descriptions."""
        effects = dict(self.EFFECTS)
        if MEDIAPIPE_AVAILABLE:
            if self.mp_face_mesh:
                effects["face_mesh"] = self.GPU_EFFECTS["face_mesh"]
            if self.mp_selfie_seg:
                effects["selfie_segment"] = self.GPU_EFFECTS["selfie_segment"]
        return effects

    def process(self, frame: np.ndarray, effect: str) -> np.ndarray:
        """Apply effect to frame."""
        if effect == "none":
            return frame
        elif effect == "face_detection":
            return self._face_detection(frame)
        elif effect == "edge_detection":
            return self._edge_detection(frame)
        elif effect == "cartoon":
            return self._cartoon(frame)
        elif effect == "grayscale":
            return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        elif effect == "sepia":
            return self._sepia(frame)
        elif effect == "blur":
            return cv2.GaussianBlur(frame, (21, 21), 0)
        elif effect == "pixelate":
            return self._pixelate(frame, 10)
        elif effect == "invert":
            return cv2.bitwise_not(frame)
        elif effect == "mirror":
            return cv2.flip(frame, 1)
        elif effect == "emboss":
            return self._emboss(frame)
        elif effect == "sketch":
            return self._sketch(frame)
        elif effect == "face_mesh":
            return self._face_mesh(frame)
        elif effect == "selfie_segment":
            return self._selfie_segment(frame)
        else:
            logger.warning(f"Unknown effect: {effect}")
            return frame

    def _face_detection(self, frame: np.ndarray) -> np.ndarray:
        """Detect faces using OpenCV Haar cascade."""
        if self.face_cascade is None:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        output = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return output

    def _edge_detection(self, frame: np.ndarray) -> np.ndarray:
        """Canny edge detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def _cartoon(self, frame: np.ndarray) -> np.ndarray:
        """Cartoon effect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    def _sepia(self, frame: np.ndarray) -> np.ndarray:
        """Sepia tone effect."""
        kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        return cv2.transform(frame, kernel).clip(0, 255).astype(np.uint8)

    def _pixelate(self, frame: np.ndarray, pixel_size: int = 10) -> np.ndarray:
        """Pixelation effect."""
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def _emboss(self, frame: np.ndarray) -> np.ndarray:
        """Emboss effect."""
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        return cv2.filter2D(frame, -1, kernel) + 128

    def _sketch(self, frame: np.ndarray) -> np.ndarray:
        """Pencil sketch effect."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

    def _face_mesh(self, frame: np.ndarray) -> np.ndarray:
        """MediaPipe face mesh with 468 landmarks."""
        if not self.mp_face_mesh:
            return self._face_detection(frame)  # Fallback

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)
        output = frame.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
        return output

    def _selfie_segment(self, frame: np.ndarray) -> np.ndarray:
        """MediaPipe selfie segmentation with background blur."""
        if not self.mp_selfie_seg:
            return self._blur_background_simple(frame)  # Fallback

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_selfie_seg.process(rgb)

        # Create mask
        mask = results.segmentation_mask
        condition = mask > 0.5

        # Blur background
        blurred = cv2.GaussianBlur(frame, (55, 55), 0)
        output = np.where(condition[:, :, np.newaxis], frame, blurred)

        return output.astype(np.uint8)

    def _blur_background_simple(self, frame: np.ndarray) -> np.ndarray:
        """Simple background blur fallback."""
        return cv2.GaussianBlur(frame, (21, 21), 0)


class OoblexDemo:
    """
    One-click Ooblex demo for Google Colab.

    Provides:
    - Webcam capture via JavaScript
    - GPU-accelerated video processing
    - MJPEG stream output
    - Interactive effect switching
    """

    def __init__(self, config: Optional[DemoConfig] = None):
        self.config = config or DemoConfig()
        self.processor = EffectsProcessor()
        self.current_effect = self.config.default_effect
        self.latest_frame: Optional[np.ndarray] = None
        self.processed_frame: Optional[bytes] = None
        self.running = False
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self._lock = threading.Lock()
        self._server_thread = None
        self.tunnel_url = None

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode frame as JPEG."""
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality])
        return encoded.tobytes()

    def process_frame(self, frame: np.ndarray) -> bytes:
        """Process frame with current effect and encode."""
        processed = self.processor.process(frame, self.current_effect)

        # Add FPS overlay
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

        # Draw info overlay
        cv2.putText(processed, f"Effect: {self.current_effect}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed, f"FPS: {self.fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if CUDA_AVAILABLE:
            cv2.putText(processed, f"GPU: {torch.cuda.get_device_name(0)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return self._encode_frame(processed)

    def set_effect(self, effect: str):
        """Change the current processing effect."""
        if effect in self.processor.get_available_effects():
            self.current_effect = effect
            logger.info(f"Effect changed to: {effect}")
        else:
            logger.warning(f"Unknown effect: {effect}")

    def receive_frame(self, base64_data: str):
        """Receive a frame from JavaScript webcam capture."""
        try:
            # Decode base64 image
            img_data = base64.b64decode(base64_data.split(',')[1] if ',' in base64_data else base64_data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is not None:
                with self._lock:
                    self.latest_frame = frame
                    self.processed_frame = self.process_frame(frame)
        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    def get_mjpeg_frame(self) -> Optional[bytes]:
        """Get the latest processed frame for MJPEG streaming."""
        with self._lock:
            return self.processed_frame

    def _start_http_server(self):
        """Start HTTP server for MJPEG streaming."""
        from http.server import HTTPServer, BaseHTTPRequestHandler
        import json as json_module

        demo = self

        class MJPEGHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass  # Suppress logging

            def do_GET(self):
                if self.path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                    self.send_header('Cache-Control', 'no-cache')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()

                    while demo.running:
                        frame = demo.get_mjpeg_frame()
                        if frame:
                            try:
                                self.wfile.write(b'--frame\r\n')
                                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                                self.wfile.write(frame)
                                self.wfile.write(b'\r\n')
                            except:
                                break
                        time.sleep(1/30)

                elif self.path == '/snapshot.jpg':
                    frame = demo.get_mjpeg_frame()
                    if frame:
                        self.send_response(200)
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        self.wfile.write(frame)
                    else:
                        self.send_error(503, 'No frame available')

                elif self.path == '/effects':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json_module.dumps(demo.processor.get_available_effects()).encode())

                elif self.path == '/status':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    status = {
                        'running': demo.running,
                        'effect': demo.current_effect,
                        'fps': demo.fps,
                        'cuda_available': CUDA_AVAILABLE,
                        'mediapipe_available': MEDIAPIPE_AVAILABLE,
                    }
                    self.wfile.write(json_module.dumps(status).encode())

                elif self.path.startswith('/set_effect/'):
                    effect = self.path.split('/')[-1]
                    demo.set_effect(effect)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json_module.dumps({'effect': demo.current_effect}).encode())

                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == '/frame':
                    content_length = int(self.headers['Content-Length'])
                    body = self.rfile.read(content_length)
                    demo.receive_frame(body.decode())
                    self.send_response(200)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                else:
                    self.send_error(404)

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()

        # Try to bind to port, with fallback to alternative ports
        port = self.config.port
        server = None
        for attempt in range(10):  # Try up to 10 ports
            try:
                server = HTTPServer(('0.0.0.0', port), MJPEGHandler)
                self.config.port = port  # Update config with actual port
                logger.info(f"MJPEG server started on port {port}")
                break
            except OSError as e:
                if "Address already in use" in str(e):
                    logger.warning(f"Port {port} in use, trying {port + 1}")
                    port += 1
                else:
                    raise

        if server is None:
            logger.error("Could not find available port")
            return

        server.serve_forever()

    def _setup_tunnel(self) -> Optional[str]:
        """Setup tunnel for external access. Tries ngrok first, then localtunnel."""
        # Try ngrok first
        try:
            from pyngrok import ngrok
            tunnel = ngrok.connect(self.config.port, "http")
            self.tunnel_url = tunnel.public_url
            logger.info(f"ngrok tunnel URL: {self.tunnel_url}")
            return self.tunnel_url
        except ImportError:
            logger.info("pyngrok not installed, trying localtunnel...")
        except Exception as e:
            if "authtoken" in str(e).lower() or "authentication" in str(e).lower():
                logger.warning("ngrok requires auth token. Get free token at https://ngrok.com")
                logger.info("Trying localtunnel as alternative...")
            else:
                logger.warning(f"ngrok failed: {e}")

        # Try localtunnel as fallback (no auth required)
        try:
            import subprocess
            import re

            # Check if localtunnel is installed, if not install it
            result = subprocess.run(
                ["npx", "localtunnel", "--port", str(self.config.port)],
                capture_output=True,
                text=True,
                timeout=15
            )
            # Parse URL from output
            match = re.search(r'https://[^\s]+\.loca\.lt', result.stdout + result.stderr)
            if match:
                self.tunnel_url = match.group(0)
                logger.info(f"localtunnel URL: {self.tunnel_url}")
                return self.tunnel_url
        except Exception as e:
            logger.info(f"localtunnel not available: {e}")

        # No tunnel available - use Colab's built-in proxy
        logger.info("No external tunnel available. Using Colab's localhost proxy.")
        logger.info("For external access, set ngrok token or install localtunnel.")
        return None

    def start(self, use_tunnel: bool = True):
        """Start the demo server."""
        self.running = True

        # Start HTTP server in background thread
        self._server_thread = threading.Thread(target=self._start_http_server, daemon=True)
        self._server_thread.start()

        # Wait for server to start
        time.sleep(1)

        # Setup tunnel (optional)
        if use_tunnel:
            self._setup_tunnel()

        # Display the UI
        self._display_ui()

    def stop(self):
        """Stop the demo server."""
        self.running = False
        try:
            from pyngrok import ngrok
            ngrok.kill()
        except:
            pass
        logger.info("Demo stopped")

    def _display_ui(self):
        """Display the interactive UI in Colab."""
        from IPython.display import display, HTML, Javascript

        effects = self.processor.get_available_effects()
        effects_options = '\n'.join([f'<option value="{k}">{k}: {v}</option>' for k, v in effects.items()])

        stream_url = self.tunnel_url or f"http://localhost:{self.config.port}"

        html = f'''
        <style>
            .ooblex-container {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; }}
            .ooblex-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0; }}
            .ooblex-content {{ background: #f5f5f5; padding: 20px; border-radius: 0 0 10px 10px; }}
            .ooblex-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
            .ooblex-video {{ flex: 1; min-width: 300px; }}
            .ooblex-video video, .ooblex-video img {{ width: 100%; border-radius: 8px; background: #000; }}
            .ooblex-controls {{ background: white; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
            .ooblex-btn {{ padding: 10px 20px; font-size: 14px; cursor: pointer; border: none; border-radius: 5px; margin: 5px; }}
            .ooblex-btn-primary {{ background: #667eea; color: white; }}
            .ooblex-btn-primary:hover {{ background: #5a6fd6; }}
            .ooblex-btn-danger {{ background: #e74c3c; color: white; }}
            .ooblex-select {{ padding: 10px; font-size: 14px; border-radius: 5px; border: 1px solid #ddd; min-width: 200px; }}
            .ooblex-status {{ background: #e8f5e9; padding: 10px; border-radius: 5px; margin-top: 10px; }}
            .ooblex-urls {{ background: #fff3e0; padding: 10px; border-radius: 5px; margin-top: 10px; font-family: monospace; }}
        </style>

        <div class="ooblex-container">
            <div class="ooblex-header">
                <h2>Ooblex Demo - Real-time AI Video Processing</h2>
                <p>GPU: {'<b style="color:#4caf50">Enabled</b> (' + torch.cuda.get_device_name(0) + ')' if CUDA_AVAILABLE else '<b style="color:#ff9800">CPU Mode</b>'}</p>
            </div>

            <div class="ooblex-content">
                <div class="ooblex-controls">
                    <button class="ooblex-btn ooblex-btn-primary" onclick="startCamera()">Start Camera</button>
                    <button class="ooblex-btn ooblex-btn-danger" onclick="stopCamera()">Stop Camera</button>
                    <select class="ooblex-select" id="effectSelect" onchange="changeEffect(this.value)">
                        {effects_options}
                    </select>
                </div>

                <div class="ooblex-row">
                    <div class="ooblex-video">
                        <h4>Input (Your Webcam)</h4>
                        <video id="webcamVideo" autoplay muted playsinline></video>
                    </div>
                    <div class="ooblex-video">
                        <h4>Output (Processed)</h4>
                        <img id="outputStream" src="{stream_url}/snapshot.jpg" alt="Waiting for stream...">
                    </div>
                </div>

                <div class="ooblex-status" id="statusDiv">
                    Status: Ready to start
                </div>

                <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 12px;">
                    <b>Camera not working?</b><br>
                    • Click the camera icon in browser address bar to allow access<br>
                    • If in Colab: Try opening the tunnel URL directly in a new tab<br>
                    • Or use <b>Test Pattern</b> button below to demo without camera
                </div>

                <div style="margin-top: 10px;">
                    <button class="ooblex-btn" style="background:#4caf50;color:white" onclick="startTestPattern()">Use Test Pattern</button>
                </div>

                <div class="ooblex-urls">
                    <b>Stream URLs:</b><br>
                    MJPEG Stream: <a href="{stream_url}/stream.mjpg" target="_blank">{stream_url}/stream.mjpg</a><br>
                    Snapshot: <a href="{stream_url}/snapshot.jpg" target="_blank">{stream_url}/snapshot.jpg</a><br>
                    Effects API: <a href="{stream_url}/effects" target="_blank">{stream_url}/effects</a>
                </div>
            </div>
        </div>

        <script>
            let stream = null;
            let captureInterval = null;
            const SERVER_URL = "{stream_url}";

            async function startCamera() {{
                try {{
                    stream = await navigator.mediaDevices.getUserMedia({{
                        video: {{ width: {self.config.frame_width}, height: {self.config.frame_height}, facingMode: 'user' }},
                        audio: false
                    }});
                    document.getElementById('webcamVideo').srcObject = stream;
                    document.getElementById('statusDiv').innerHTML = 'Status: <b style="color:green">Camera active, streaming frames</b>';

                    // Start sending frames
                    const canvas = document.createElement('canvas');
                    canvas.width = {self.config.frame_width};
                    canvas.height = {self.config.frame_height};
                    const ctx = canvas.getContext('2d');
                    const video = document.getElementById('webcamVideo');

                    captureInterval = setInterval(async () => {{
                        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

                        try {{
                            await fetch(SERVER_URL + '/frame', {{
                                method: 'POST',
                                body: dataUrl,
                                mode: 'cors'
                            }});
                        }} catch(e) {{
                            console.log('Frame send error:', e);
                        }}
                    }}, {int(1000 / self.config.max_fps)});

                    // Start output stream
                    document.getElementById('outputStream').src = SERVER_URL + '/stream.mjpg';

                }} catch(err) {{
                    console.error('Camera error:', err);
                    document.getElementById('statusDiv').innerHTML = 'Status: <b style="color:red">Camera error: ' + err.message + '</b>';
                }}
            }}

            function stopCamera() {{
                if (stream) {{
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }}
                if (captureInterval) {{
                    clearInterval(captureInterval);
                    captureInterval = null;
                }}
                document.getElementById('webcamVideo').srcObject = null;
                document.getElementById('outputStream').src = SERVER_URL + '/snapshot.jpg';
                document.getElementById('statusDiv').innerHTML = 'Status: Camera stopped';
            }}

            async function changeEffect(effect) {{
                try {{
                    await fetch(SERVER_URL + '/set_effect/' + effect);
                    document.getElementById('statusDiv').innerHTML = 'Status: Effect changed to <b>' + effect + '</b>';
                }} catch(e) {{
                    console.log('Effect change error:', e);
                }}
            }}

            function startTestPattern() {{
                // Generate test pattern frames without camera
                document.getElementById('statusDiv').innerHTML = 'Status: <b style="color:green">Using test pattern (no camera)</b>';

                const canvas = document.createElement('canvas');
                canvas.width = {self.config.frame_width};
                canvas.height = {self.config.frame_height};
                const ctx = canvas.getContext('2d');

                let frame = 0;
                captureInterval = setInterval(async () => {{
                    // Draw colorful test pattern
                    const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
                    gradient.addColorStop(0, `hsl(${{frame % 360}}, 70%, 50%)`);
                    gradient.addColorStop(1, `hsl(${{(frame + 180) % 360}}, 70%, 50%)`);
                    ctx.fillStyle = gradient;
                    ctx.fillRect(0, 0, canvas.width, canvas.height);

                    // Draw moving circles
                    for (let i = 0; i < 5; i++) {{
                        ctx.beginPath();
                        const x = (Math.sin(frame * 0.02 + i) + 1) * canvas.width / 2;
                        const y = (Math.cos(frame * 0.03 + i * 2) + 1) * canvas.height / 2;
                        ctx.arc(x, y, 30 + i * 10, 0, Math.PI * 2);
                        ctx.fillStyle = `rgba(255,255,255,0.5)`;
                        ctx.fill();
                    }}

                    // Add text
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 24px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('OOBLEX TEST PATTERN', canvas.width/2, canvas.height/2);
                    ctx.font = '16px Arial';
                    ctx.fillText(`Frame: ${{frame}}`, canvas.width/2, canvas.height/2 + 30);

                    frame++;

                    // Show in input preview
                    document.getElementById('webcamVideo').style.display = 'none';
                    const preview = document.getElementById('testPreview') || createTestPreview();
                    preview.src = canvas.toDataURL('image/jpeg', 0.8);

                    // Send to server
                    try {{
                        await fetch(SERVER_URL + '/frame', {{
                            method: 'POST',
                            body: canvas.toDataURL('image/jpeg', 0.8),
                            mode: 'cors'
                        }});
                    }} catch(e) {{}}
                }}, {int(1000 / self.config.max_fps)});

                // Start output stream
                document.getElementById('outputStream').src = SERVER_URL + '/stream.mjpg';
            }}

            function createTestPreview() {{
                const img = document.createElement('img');
                img.id = 'testPreview';
                img.style.width = '100%';
                img.style.borderRadius = '8px';
                document.getElementById('webcamVideo').parentNode.appendChild(img);
                return img;
            }}
        </script>
        '''

        display(HTML(html))


def run_tests():
    """
    Run automated tests for CI/CD.
    Can be used to verify ooblex functionality in Colab.
    """
    import sys

    print("=" * 60)
    print("OOBLEX AUTOMATED TEST SUITE")
    print("=" * 60)

    results = []

    # Test 1: OpenCV import
    try:
        import cv2
        print(f"[PASS] OpenCV version: {cv2.__version__}")
        results.append(("OpenCV Import", True))
    except ImportError as e:
        print(f"[FAIL] OpenCV import: {e}")
        results.append(("OpenCV Import", False))

    # Test 2: NumPy import
    try:
        import numpy as np
        print(f"[PASS] NumPy version: {np.__version__}")
        results.append(("NumPy Import", True))
    except ImportError as e:
        print(f"[FAIL] NumPy import: {e}")
        results.append(("NumPy Import", False))

    # Test 3: GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[PASS] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"       CUDA version: {torch.version.cuda}")
            print(f"       GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            results.append(("CUDA GPU", True))
        else:
            print("[WARN] CUDA not available - CPU mode")
            results.append(("CUDA GPU", False))
    except ImportError:
        print("[WARN] PyTorch not installed")
        results.append(("CUDA GPU", False))

    # Test 4: MediaPipe
    try:
        import mediapipe
        print(f"[PASS] MediaPipe version: {mediapipe.__version__}")
        results.append(("MediaPipe", True))
    except ImportError:
        print("[WARN] MediaPipe not installed")
        results.append(("MediaPipe", False))

    # Test 5: Effects processor
    try:
        processor = EffectsProcessor()
        effects = processor.get_available_effects()
        print(f"[PASS] Effects available: {len(effects)}")
        for name, desc in effects.items():
            print(f"       - {name}: {desc}")
        results.append(("Effects Processor", True))
    except Exception as e:
        print(f"[FAIL] Effects processor: {e}")
        results.append(("Effects Processor", False))

    # Test 6: Process test image
    try:
        # Create test image
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.putText(test_img, "TEST", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        processor = EffectsProcessor()
        for effect in ["none", "grayscale", "edge_detection", "cartoon", "sepia"]:
            result = processor.process(test_img, effect)
            assert result.shape == test_img.shape, f"Shape mismatch for {effect}"
        print("[PASS] Image processing pipeline")
        results.append(("Image Processing", True))
    except Exception as e:
        print(f"[FAIL] Image processing: {e}")
        results.append(("Image Processing", False))

    # Test 7: JPEG encoding
    try:
        demo = OoblexDemo()
        encoded = demo._encode_frame(test_img)
        assert len(encoded) > 0, "Empty JPEG output"
        print(f"[PASS] JPEG encoding ({len(encoded)} bytes)")
        results.append(("JPEG Encoding", True))
    except Exception as e:
        print(f"[FAIL] JPEG encoding: {e}")
        results.append(("JPEG Encoding", False))

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    # Return exit code
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        sys.exit(run_tests())
    else:
        print("Usage: python ooblex_demo.py test")
        print("Or import and use OoblexDemo class in Colab")
