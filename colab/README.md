# Ooblex Google Colab Integration

One-click demos and CI/CD testing for Ooblex on Google Colab with free GPU support.

## Quick Start

### Demo (One-Click)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ooblex/ooblex/blob/master/colab/Ooblex_Demo.ipynb)

1. Click the badge above (or open `Ooblex_Demo.ipynb` in Colab)
2. Go to Runtime → Change runtime type → **GPU**
3. Run all cells
4. Click "Start Camera" in the UI that appears
5. Select an effect from the dropdown

### CI/CD Testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ooblex/ooblex/blob/master/colab/Ooblex_CI_Tests.ipynb)

Run automated tests to verify Ooblex functionality:
- Environment checks (OpenCV, PyTorch, MediaPipe)
- GPU availability
- Effects processing
- Performance benchmarks

## Files

| File | Description |
|------|-------------|
| `Ooblex_Demo.ipynb` | Interactive demo notebook with webcam → GPU → MJPEG |
| `Ooblex_CI_Tests.ipynb` | Automated test suite for CI/CD |
| `ooblex_demo.py` | Core demo server module |
| `README.md` | This file |

## Architecture

```
Browser (Webcam)
    │
    │ JavaScript capture (30 FPS)
    │ Base64 JPEG frames
    ▼
┌─────────────────────────┐
│   Google Colab (GPU)    │
│                         │
│  ┌─────────────────┐   │
│  │ ooblex_demo.py  │   │
│  │                 │   │
│  │ • Frame decode  │   │
│  │ • GPU process   │   │
│  │ • MJPEG encode  │   │
│  └────────┬────────┘   │
│           │             │
│  ┌────────▼────────┐   │
│  │  ngrok tunnel   │   │
│  └────────┬────────┘   │
└───────────┼─────────────┘
            │
            │ HTTPS
            ▼
      MJPEG Stream
    (viewable anywhere)
```

## Available Effects

### CPU Effects (OpenCV)
- `none` - Passthrough
- `face_detection` - Haar cascade face detection
- `edge_detection` - Canny edge detection
- `cartoon` - Cartoon/comic effect
- `grayscale` - Black and white
- `sepia` - Vintage sepia tone
- `blur` - Gaussian blur
- `pixelate` - Pixelation
- `invert` - Color inversion
- `mirror` - Horizontal flip
- `emboss` - Emboss effect
- `sketch` - Pencil sketch

### GPU Effects (MediaPipe)
- `face_mesh` - 468-point facial landmarks
- `selfie_segment` - Background blur/removal

## API Endpoints

When the demo is running, these endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stream.mjpg` | GET | MJPEG video stream |
| `/snapshot.jpg` | GET | Single frame snapshot |
| `/effects` | GET | List available effects |
| `/set_effect/{name}` | GET | Change current effect |
| `/status` | GET | Server status JSON |
| `/frame` | POST | Submit frame (base64) |

## Usage Examples

### View MJPEG Stream

```bash
# VLC
vlc https://your-ngrok-url.ngrok.io/stream.mjpg

# ffplay
ffplay https://your-ngrok-url.ngrok.io/stream.mjpg

# Save to file
ffmpeg -i https://your-ngrok-url.ngrok.io/stream.mjpg -c copy output.mjpg
```

### Python Client

```python
import cv2

cap = cv2.VideoCapture('https://your-ngrok-url.ngrok.io/stream.mjpg')
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Ooblex', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Change Effect via API

```bash
curl https://your-ngrok-url.ngrok.io/set_effect/cartoon
```

## Running Tests Locally

```bash
cd colab
python ooblex_demo.py test
```

## Requirements

- Google Colab (free tier works!)
- GPU runtime recommended (T4 is sufficient)
- ngrok account for external access (free tier works)

## Limitations

- **Session timeout**: Free Colab disconnects after ~90 min idle
- **No persistent URLs**: ngrok URL changes each session
- **Single user**: Demo handles one webcam stream at a time

## Troubleshooting

### "No GPU found"
Go to Runtime → Change runtime type → Select "T4 GPU"

### "Camera not working"
- Allow camera permissions in browser
- Try a different browser (Chrome works best)
- Check that you're on HTTPS (required for getUserMedia)

### "Tunnel not connecting"
- Get a free ngrok token from https://ngrok.com
- Paste it in the notebook's ngrok configuration cell

### "Effects are slow"
- Ensure GPU runtime is selected
- Try lower resolution (modify `frame_width`/`frame_height` in config)
- Use simpler effects like `grayscale` or `edge_detection`

## License

Same as main Ooblex project - see repository root.
