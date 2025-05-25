# Ooblex Edge Computing Module

This module enables AI processing directly in the browser using WebAssembly, reducing latency and server load.

## Features

- **Face Detection**: Real-time face detection using a lightweight Viola-Jones cascade classifier
- **Style Transfer**: Apply artistic styles (sketch, oil painting, watercolor) to video streams
- **Background Blur**: Blur backgrounds in real-time for privacy or aesthetic effects
- **Hybrid Processing**: Automatic fallback to cloud processing when edge resources are insufficient

## Architecture

```
Browser
├── Main Thread (ooblex.v0.js)
│   ├── Edge Module API
│   ├── Performance Monitoring
│   └── Cloud Fallback Logic
│
├── Web Worker (edge_worker.js)
│   ├── WASM Module Loading
│   ├── AI Processing
│   └── Metrics Reporting
│
└── WASM Modules
    ├── face_detection.wasm
    ├── style_transfer_lite.wasm
    └── background_blur.wasm

Server (edge_server.py)
├── Module Distribution
├── Worker Coordination
├── Performance Monitoring
└── Cloud Fallback Routing
```

## Building WASM Modules

### Prerequisites

1. Install Emscripten:
```bash
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
source ./emsdk_env.sh
```

2. Build modules:
```bash
cd wasm_modules
make all
```

### Build Options

- `make all` - Build all modules with optimization
- `make debug` - Build with debugging symbols
- `make size` - Build optimized for size
- `make face_detection` - Build only face detection module
- `make style_transfer` - Build only style transfer module
- `make background_blur` - Build only background blur module

## Usage

### Client-Side Integration

1. Initialize edge computing:
```javascript
// Initialize on page load
await Ooblex.Edge.init({
    serverUrl: 'https://api.ooblex.com:8090',
    enableEdge: true,
    fallbackToCloud: true,
    performanceThreshold: 100 // ms
});
```

2. Enable edge processing for video:
```javascript
// After creating video stream
const media = Ooblex.Media.connect();

// Enable face detection
await media.enableEdgeFaceDetection();

// Enable background blur
await media.enableEdgeBackgroundBlur();

// Enable style transfer
await media.enableEdgeStyleTransfer('sketch'); // or 'oil_painting', 'watercolor'

// Disable edge processing
media.disableEdgeProcessing();
```

3. Direct edge processing:
```javascript
// Process image data directly
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');
const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

// Detect faces
const faces = await Ooblex.Edge.detectFaces(imageData.data);

// Apply style transfer
const styledData = await Ooblex.Edge.applyStyleTransfer(
    imageData.data, 
    'oil_painting',
    { strength: 0.8 }
);

// Blur background
const blurredData = await Ooblex.Edge.blurBackground(imageData.data, {
    blurRadius: 15,
    blurStrength: 0.9
});
```

### Server Deployment

1. Install dependencies:
```bash
pip install aiohttp aioredis numpy
```

2. Start edge server:
```bash
cd services/edge-compute
python edge_server.py
```

Or use systemd:
```bash
sudo cp launch_scripts/edge.service /etc/systemd/system/
sudo systemctl enable edge
sudo systemctl start edge
```

## API Reference

### Edge Module (Client)

#### `Ooblex.Edge.init(config)`
Initialize edge computing with optional configuration.

#### `Ooblex.Edge.process(taskType, data, options)`
Process data using edge computing with automatic cloud fallback.

#### `Ooblex.Edge.detectFaces(imageData, options)`
Detect faces in image data. Returns array of face boundaries.

#### `Ooblex.Edge.applyStyleTransfer(imageData, style, options)`
Apply artistic style to image. Styles: 'sketch', 'oil_painting', 'watercolor'.

#### `Ooblex.Edge.blurBackground(imageData, options)`
Blur image background. Options include blurRadius and blurStrength.

### Edge Server API

#### `GET /api/edge/modules`
List available WASM modules.

#### `GET /api/edge/modules/{module_name}`
Download WASM module.

#### `POST /api/edge/process`
Coordinate processing between edge and cloud.

#### `GET /api/edge/capabilities`
Get current edge computing capabilities.

#### `GET /api/edge/metrics`
Get edge computing performance metrics.

## Performance Considerations

1. **Module Size**: Keep WASM modules under 1MB for fast loading
2. **Memory Usage**: Monitor browser memory usage, especially on mobile
3. **Frame Rate**: Process every nth frame if needed to maintain 30fps
4. **Fallback**: Always implement cloud fallback for unsupported devices

## Browser Compatibility

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

Mobile support:
- Chrome Android 57+
- Safari iOS 11+

## Troubleshooting

### WASM modules not loading
- Check Content-Type headers (should be application/wasm or application/javascript)
- Verify CORS headers are set correctly
- Check browser console for specific error messages

### Poor performance
- Reduce processing resolution
- Process fewer frames per second
- Check if device supports SIMD instructions
- Consider using cloud processing for complex tasks

### Memory issues
- Implement periodic cleanup of unused modules
- Limit number of concurrent processing tasks
- Monitor memory usage and adjust accordingly

## Future Enhancements

- [ ] TensorFlow.js integration for more complex models
- [ ] WebGPU support for GPU acceleration
- [ ] Model quantization for smaller file sizes
- [ ] Progressive model loading
- [ ] Offline support with service workers
- [ ] P2P model sharing between clients