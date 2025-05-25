# Video Flow in Ooblex

## 🎥 How Video Works

Ooblex accepts live video, processes it with AI in real-time, and outputs the transformed video. Here's exactly how:

### 1. Video Input Methods

```
┌─────────────────────────────────────────────────────────┐
│                    VIDEO INPUTS                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  🎥 WebRTC Browser    📱 Mobile App    🎬 OBS/RTMP     │
│         ↓                    ↓               ↓          │
│    [WebRTC Gateway]    [WHIP Endpoint]  [RTMP Server]   │
│         ↓                    ↓               ↓          │
└─────────────────────────────────────────────────────────┘
                              ↓
```

### 2. Processing Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                  PROCESSING PIPELINE                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [Frame Decoder] → [ML Worker] → [Frame Encoder]        │
│        ↓               ↓              ↓                  │
│   Raw Frames      AI Processing   Processed Frames      │
│                   - Face Swap                            │
│                   - Style Transfer                       │
│                   - Background Blur                      │
│                   - Object Detection                     │
│                                                          │
└─────────────────────────────────────────────────────────┘
                              ↓
```

### 3. Video Output Methods

```
┌─────────────────────────────────────────────────────────┐
│                    VIDEO OUTPUTS                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  🖥️ WebRTC Preview   📹 MJPEG Stream   📺 HLS Stream   │
│   (Low Latency)      (Simple HTTP)    (CDN Ready)       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start - Live Video Demo

### Option 1: Browser WebRTC (Easiest)

1. Start Ooblex:
```bash
docker-compose up -d
```

2. Open the demo page:
```
https://localhost/demo
```

3. Click "Start Streaming" and select:
   - Your webcam
   - AI effect (face swap, style transfer, etc.)
   - Click "Apply"

4. You'll see:
   - Original video on the left
   - Processed video on the right
   - ~100ms latency

### Option 2: MJPEG Preview (Like Original Ooblex)

1. Stream video to Ooblex:
```bash
# Using ffmpeg from webcam
ffmpeg -f v4l2 -i /dev/video0 \
  -c:v libx264 -preset ultrafast \
  -f rtsp rtsp://localhost:8554/stream
```

2. View processed MJPEG stream:
```
http://localhost:8081/stream.mjpg
```

This gives you a simple HTTP stream you can embed anywhere:
```html
<img src="http://localhost:8081/stream.mjpg" />
```

### Option 3: OBS Studio Streaming

1. In OBS, add custom RTMP server:
   - Server: `rtmp://localhost:1935/live`
   - Stream Key: `mystream`

2. Start streaming in OBS

3. View your processed stream at:
   - MJPEG: `http://localhost:8081/mystream.mjpg`
   - HLS: `http://localhost:8084/mystream/playlist.m3u8`
   - WebRTC: `https://localhost/watch/mystream`

## 📊 Detailed Flow Example

Let's trace a video frame through the system:

```
1. Camera captures frame
   ↓
2. WebRTC encodes and sends to Ooblex (Port 8100)
   ↓
3. WebRTC Gateway receives frame
   ↓
4. Frame sent to Redis queue
   ↓
5. ML Worker picks up frame
   ↓
6. AI model processes frame (e.g., face swap)
   ↓
7. Processed frame sent back to Redis
   ↓
8. Output encoders pick up frame:
   - WebRTC: Sent back to browser
   - MJPEG: Added to HTTP stream  
   - HLS: Encoded to segments
   ↓
9. User sees processed video!
```

## 🎨 Available AI Effects

- **Face Swap**: Replace faces in real-time
- **Style Transfer**: Apply artistic styles
- **Background Blur/Replace**: Virtual backgrounds
- **Object Detection**: Highlight and track objects
- **Beauty Filter**: Skin smoothing, eye brightening
- **Emotion Detection**: Show emotion indicators
- **Hand Tracking**: Gesture recognition
- **Body Pose**: Skeleton tracking

## 🔧 Configuration

### Set Default Processing
```yaml
# .env file
DEFAULT_AI_MODEL=face_swap
DEFAULT_STYLE=cartoon
ENABLE_MJPEG=true
MJPEG_PORT=8081
MJPEG_QUALITY=85
```

### Multiple Streams
```yaml
# Each stream can have different processing
streams:
  webcam1:
    input: rtsp://camera1.local
    model: face_swap
    output: [mjpeg, webrtc]
  
  webcam2:
    input: rtsp://camera2.local  
    model: style_transfer
    style: vangogh
    output: [hls, mjpeg]
```

## 📱 Mobile App Usage

```swift
// iOS Swift
let ooblex = OoblexSDK.shared
ooblex.connect(to: "wss://your-server.com")
ooblex.startStreaming(effect: .faceSwap)
```

```kotlin
// Android Kotlin
val ooblex = OoblexSDK.getInstance()
ooblex.connect("wss://your-server.com")
ooblex.startStreaming(OoblexEffect.STYLE_TRANSFER)
```

## 🖥️ Web Integration

```html
<!-- Simple MJPEG embed -->
<img src="http://localhost:8081/stream.mjpg" width="640" height="480">

<!-- WebRTC with controls -->
<video id="processed-video" autoplay controls></video>
<script src="https://your-server.com/ooblex-sdk.js"></script>
<script>
  const ooblex = new OoblexClient();
  ooblex.connect('wss://your-server.com');
  ooblex.startProcessing({
    input: 'webcam',
    effect: 'face_swap',
    output: document.getElementById('processed-video')
  });
</script>
```

## 🎯 Latency Expectations

- **WebRTC to WebRTC**: 50-150ms (lowest latency)
- **WebRTC to MJPEG**: 100-300ms  
- **RTMP to HLS**: 2-6 seconds (CDN-optimized)
- **Edge Processing**: 10-50ms (runs in browser)

## 💡 Common Use Cases

1. **Live Streaming with Effects**
   - Stream from OBS → Process → Stream to Twitch/YouTube

2. **Video Calls with AI**
   - WebRTC in → Face beautification → WebRTC out

3. **Security Monitoring**
   - RTSP cameras → Object detection → Alert + MJPEG preview

4. **Virtual Production**
   - Multiple cameras → Background replacement → Combined output

## 🐛 Troubleshooting

**No video showing?**
- Check if all services are running: `docker-compose ps`
- Verify camera permissions in browser
- Check firewall allows UDP ports 10000-10100

**High latency?**
- Use WebRTC instead of RTMP for input
- Enable GPU acceleration: `docker-compose --profile gpu up`
- Reduce video resolution in settings

**MJPEG not working?**
- Ensure MJPEG service is enabled in .env
- Check http://localhost:8081/status
- Try different browser (some block mixed content)