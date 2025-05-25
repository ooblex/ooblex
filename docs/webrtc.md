# WebRTC Integration Guide

## Overview

Ooblex provides a comprehensive WebRTC integration supporting multiple protocols including traditional WebRTC, WHIP/WHEP, and VDO.Ninja integration. Our platform handles real-time video streaming with ultra-low latency while supporting AI processing, edge computing, and blockchain verification.

## Getting Started

### Prerequisites
- Modern web browser with WebRTC support
- HTTPS enabled web server (for production)
- API key for authentication
- Optional: Edge device for local processing

### Quick Start

#### Basic WebRTC Connection
```javascript
// Initialize Ooblex WebRTC client
const ooblex = new OoblexWebRTC({
    apiKey: 'YOUR_API_KEY',
    serverUrl: 'wss://webrtc.ooblex.com',
    iceServers: [
        { urls: 'stun:stun.ooblex.com:3478' },
        { urls: 'turn:turn.ooblex.com:3478', username: 'user', credential: 'pass' }
    ]
});

// Start streaming
async function startStream() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1920, height: 1080, frameRate: 30 },
        audio: true
    });
    
    const session = await ooblex.createSession({
        stream: stream,
        aiModels: ['face_detection', 'emotion'],
        edgeProcessing: true
    });
    
    session.on('ai-result', (data) => {
        console.log('AI Detection:', data);
    });
}
```

## Detailed Usage Examples

### WHIP (WebRTC-HTTP Ingestion Protocol)

WHIP provides a standardized way to ingest WebRTC streams:

```javascript
class WHIPClient {
    constructor(endpoint, bearerToken) {
        this.endpoint = endpoint;
        this.bearerToken = bearerToken;
    }
    
    async publish(stream) {
        // Create peer connection
        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        
        // Add tracks
        stream.getTracks().forEach(track => {
            pc.addTrack(track, stream);
        });
        
        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        
        // Wait for ICE gathering
        await new Promise(resolve => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                pc.addEventListener('icegatheringstatechange', () => {
                    if (pc.iceGatheringState === 'complete') {
                        resolve();
                    }
                });
            }
        });
        
        // Send offer to WHIP endpoint
        const response = await fetch(this.endpoint, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.bearerToken}`,
                'Content-Type': 'application/sdp'
            },
            body: pc.localDescription.sdp
        });
        
        if (!response.ok) {
            throw new Error(`WHIP request failed: ${response.status}`);
        }
        
        // Set remote description
        const answerSdp = await response.text();
        await pc.setRemoteDescription({
            type: 'answer',
            sdp: answerSdp
        });
        
        return pc;
    }
}

// Usage
const whipClient = new WHIPClient(
    'https://api.ooblex.com/whip/stream_123',
    'YOUR_API_KEY'
);

const stream = await navigator.mediaDevices.getUserMedia({
    video: true,
    audio: true
});

const pc = await whipClient.publish(stream);
```

### WHEP (WebRTC-HTTP Egress Protocol)

WHEP enables standardized playback of WebRTC streams:

```javascript
class WHEPClient {
    constructor(endpoint, bearerToken) {
        this.endpoint = endpoint;
        this.bearerToken = bearerToken;
    }
    
    async view() {
        // Create peer connection
        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        
        // Add transceiver for receiving
        pc.addTransceiver('video', { direction: 'recvonly' });
        pc.addTransceiver('audio', { direction: 'recvonly' });
        
        // Create offer
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        
        // Send offer to WHEP endpoint
        const response = await fetch(this.endpoint, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${this.bearerToken}`,
                'Content-Type': 'application/sdp'
            },
            body: pc.localDescription.sdp
        });
        
        if (!response.ok) {
            throw new Error(`WHEP request failed: ${response.status}`);
        }
        
        // Set remote description
        const answerSdp = await response.text();
        await pc.setRemoteDescription({
            type: 'answer',
            sdp: answerSdp
        });
        
        // Wait for streams
        return new Promise((resolve) => {
            pc.addEventListener('track', (event) => {
                if (event.streams && event.streams[0]) {
                    resolve(event.streams[0]);
                }
            });
        });
    }
}

// Usage
const whepClient = new WHEPClient(
    'https://api.ooblex.com/whep/stream_123',
    'YOUR_API_KEY'
);

const remoteStream = await whepClient.view();
videoElement.srcObject = remoteStream;
```

### VDO.Ninja Integration

Seamlessly integrate with VDO.Ninja for enhanced collaboration:

```javascript
class VDONinjaIntegration {
    constructor(ooblexConfig) {
        this.ooblex = new OoblexWebRTC(ooblexConfig);
        this.vdoNinjaUrl = 'https://vdo.ninja';
    }
    
    async createRoom(roomConfig) {
        // Create Ooblex stream
        const streamId = await this.ooblex.createStream({
            name: roomConfig.name,
            aiModels: roomConfig.aiModels,
            recording: roomConfig.recording
        });
        
        // Generate VDO.Ninja compatible URL
        const vdoParams = new URLSearchParams({
            room: streamId,
            password: roomConfig.password || '',
            bitrate: roomConfig.bitrate || 2500,
            quality: roomConfig.quality || 2,
            stereo: roomConfig.stereo || true,
            novideo: false,
            noaudio: false
        });
        
        return {
            streamId: streamId,
            directorUrl: `${this.vdoNinjaUrl}/?director=${streamId}`,
            guestUrl: `${this.vdoNinjaUrl}/?room=${streamId}&${vdoParams}`,
            ooblexDashboard: `https://dashboard.ooblex.com/streams/${streamId}`
        };
    }
    
    async bridgeVDOtoOoblex(vdoStreamId, options = {}) {
        // Create bridge between VDO.Ninja and Ooblex
        const bridge = new RTCPeerConnection();
        
        // Capture VDO.Ninja stream
        const vdoStream = await this.captureVDOStream(vdoStreamId);
        
        // Process through Ooblex
        const processedStream = await this.ooblex.processStream(vdoStream, {
            aiModels: options.aiModels || ['face_detection'],
            edgeProcessing: options.edgeProcessing || true,
            enhance: options.enhance || {
                denoise: true,
                superResolution: false,
                backgroundBlur: false
            }
        });
        
        return processedStream;
    }
}
```

### Advanced WebRTC Features

#### Simulcast Support
```javascript
const pc = new RTCPeerConnection();

// Configure simulcast
const transceiver = pc.addTransceiver(track, {
    direction: 'sendonly',
    streams: [stream],
    sendEncodings: [
        { rid: 'high', maxBitrate: 1000000 },
        { rid: 'medium', maxBitrate: 500000, scaleResolutionDownBy: 2 },
        { rid: 'low', maxBitrate: 200000, scaleResolutionDownBy: 4 }
    ]
});
```

#### DataChannel for AI Results
```javascript
// Create data channel for real-time AI results
const dataChannel = pc.createDataChannel('ai-results', {
    ordered: true,
    maxRetransmits: 3
});

dataChannel.onopen = () => {
    console.log('AI results channel opened');
};

dataChannel.onmessage = (event) => {
    const aiData = JSON.parse(event.data);
    if (aiData.type === 'face_detection') {
        drawBoundingBoxes(aiData.faces);
    } else if (aiData.type === 'emotion') {
        updateEmotionDisplay(aiData.emotions);
    }
};
```

#### Edge Computing Integration
```javascript
class EdgeWebRTC {
    constructor(config) {
        this.edgeUrl = config.edgeUrl || 'wss://edge.local:8443';
        this.cloudUrl = config.cloudUrl || 'wss://cloud.ooblex.com';
        this.aiModels = config.aiModels || [];
    }
    
    async connect() {
        // Try edge first
        try {
            this.pc = await this.connectToEdge();
            console.log('Connected to edge device');
        } catch (e) {
            console.log('Edge unavailable, falling back to cloud');
            this.pc = await this.connectToCloud();
        }
        
        return this.pc;
    }
    
    async connectToEdge() {
        const pc = new RTCPeerConnection({
            iceServers: [{ urls: 'stun:stun.local:3478' }]
        });
        
        // Setup edge-specific processing
        const ws = new WebSocket(this.edgeUrl);
        
        ws.onopen = async () => {
            // Negotiate capabilities
            ws.send(JSON.stringify({
                type: 'capabilities',
                models: this.aiModels,
                processing: 'local'
            }));
        };
        
        return pc;
    }
}
```

## Configuration Options

### Client Configuration
```javascript
const config = {
    // WebRTC Configuration
    webrtc: {
        iceServers: [
            { urls: 'stun:stun.ooblex.com:3478' },
            { 
                urls: 'turn:turn.ooblex.com:3478',
                username: 'generated-username',
                credential: 'generated-password'
            }
        ],
        iceTransportPolicy: 'all', // 'all' or 'relay'
        bundlePolicy: 'max-bundle',
        rtcpMuxPolicy: 'require',
        iceCandidatePoolSize: 10
    },
    
    // Media Constraints
    media: {
        video: {
            width: { min: 640, ideal: 1920, max: 3840 },
            height: { min: 480, ideal: 1080, max: 2160 },
            frameRate: { min: 15, ideal: 30, max: 60 },
            facingMode: 'user', // 'user' or 'environment'
            aspectRatio: 16/9
        },
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 48000,
            channelCount: 2
        }
    },
    
    // Streaming Options
    streaming: {
        protocol: 'webrtc', // 'webrtc', 'whip', 'whep'
        codec: {
            video: 'h264', // 'h264', 'vp8', 'vp9', 'av1'
            audio: 'opus'  // 'opus', 'pcmu', 'pcma'
        },
        bitrate: {
            video: 2500000, // 2.5 Mbps
            audio: 128000   // 128 kbps
        },
        simulcast: true,
        adaptiveBitrate: true
    },
    
    // AI Processing
    ai: {
        models: ['face_detection', 'emotion', 'object_tracking'],
        processing: 'hybrid', // 'edge', 'cloud', 'hybrid'
        edgeDevice: 'auto', // 'auto', 'gpu', 'npu', 'cpu'
        updateInterval: 100, // ms
        confidenceThreshold: 0.7
    },
    
    // Advanced Features
    features: {
        recording: {
            enabled: false,
            format: 'webm',
            videoBitrate: 5000000,
            audioBitrate: 192000
        },
        screenshot: {
            enabled: true,
            format: 'png',
            quality: 0.9
        },
        virtualBackground: {
            enabled: false,
            type: 'blur', // 'blur', 'image', 'video'
            blurRadius: 10
        },
        networkAdaptation: {
            enabled: true,
            minBitrate: 200000,
            maxBitrate: 5000000,
            startBitrate: 1000000
        }
    }
};
```

### Server-Side Configuration
```python
# Janus Gateway Configuration for WebRTC
JANUS_CONFIG = {
    "general": {
        "configs_folder": "/etc/janus",
        "plugins_folder": "/usr/lib/janus/plugins",
        "transports_folder": "/usr/lib/janus/transports",
        "log_to_stdout": False,
        "debug_level": 4,
        "api_secret": "your-api-secret"
    },
    "nat": {
        "stun_server": "stun.ooblex.com",
        "stun_port": 3478,
        "nice_debug": False,
        "ice_lite": False,
        "ice_tcp": True
    },
    "media": {
        "ipv6": True,
        "min_port": 10000,
        "max_port": 60000,
        "dtls_mtu": 1200,
        "rtp_port_range": "10000-60000",
        "video_codecs": "h264,vp8,vp9,av1",
        "audio_codecs": "opus,pcmu,pcma"
    },
    "webrtc": {
        "ice_enforce_list": "eth0",
        "ice_ignore_list": "vmnet",
        "nat_1_1_mapping": "public.ip.address",
        "dtls_timeout": 500,
        "twcc_period": 100
    }
}
```

## Best Practices

### Connection Management

1. **Implement Reconnection Logic**
```javascript
class ResilientWebRTC {
    constructor(config) {
        this.config = config;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }
    
    async connect() {
        for (let attempt = 0; attempt < this.maxReconnectAttempts; attempt++) {
            try {
                await this.establishConnection();
                this.monitorConnection();
                return;
            } catch (error) {
                console.error(`Connection attempt ${attempt + 1} failed:`, error);
                if (attempt < this.maxReconnectAttempts - 1) {
                    await this.wait(this.reconnectDelay * Math.pow(2, attempt));
                }
            }
        }
        throw new Error('Failed to establish WebRTC connection');
    }
    
    monitorConnection() {
        this.pc.addEventListener('connectionstatechange', () => {
            console.log('Connection state:', this.pc.connectionState);
            if (this.pc.connectionState === 'failed') {
                this.reconnect();
            }
        });
        
        this.pc.addEventListener('iceconnectionstatechange', () => {
            console.log('ICE state:', this.pc.iceConnectionState);
            if (this.pc.iceConnectionState === 'disconnected') {
                setTimeout(() => {
                    if (this.pc.iceConnectionState === 'disconnected') {
                        this.reconnect();
                    }
                }, 5000);
            }
        });
    }
}
```

2. **Optimize Media Quality**
```javascript
async function optimizeVideoQuality(pc, targetBitrate) {
    const sender = pc.getSenders().find(s => s.track?.kind === 'video');
    if (!sender) return;
    
    const params = sender.getParameters();
    if (!params.encodings || params.encodings.length === 0) {
        params.encodings = [{}];
    }
    
    // Adjust bitrate based on network conditions
    params.encodings[0].maxBitrate = targetBitrate;
    params.encodings[0].scaleResolutionDownBy = targetBitrate < 500000 ? 2 : 1;
    
    await sender.setParameters(params);
}

// Monitor network quality
async function monitorNetworkQuality(pc) {
    const stats = await pc.getStats();
    let totalPacketsLost = 0;
    let totalPacketsSent = 0;
    
    stats.forEach(report => {
        if (report.type === 'outbound-rtp' && report.mediaType === 'video') {
            totalPacketsLost += report.packetsLost || 0;
            totalPacketsSent += report.packetsSent || 0;
        }
    });
    
    const lossRate = totalPacketsSent > 0 ? totalPacketsLost / totalPacketsSent : 0;
    
    if (lossRate > 0.05) { // 5% packet loss
        console.warn('High packet loss detected:', lossRate);
        // Reduce quality
        await optimizeVideoQuality(pc, 500000);
    }
}
```

### Security Best Practices

1. **Enable End-to-End Encryption**
```javascript
// Use insertable streams for E2EE
async function enableE2EE(pc, encryptionKey) {
    const senders = pc.getSenders();
    const receivers = pc.getReceivers();
    
    for (const sender of senders) {
        if (sender.track) {
            const streams = sender.createEncodedStreams();
            const transformStream = new TransformStream({
                transform: async (encodedFrame, controller) => {
                    // Encrypt frame
                    const encryptedData = await encrypt(
                        encodedFrame.data,
                        encryptionKey
                    );
                    encodedFrame.data = encryptedData;
                    controller.enqueue(encodedFrame);
                }
            });
            
            streams.readable
                .pipeThrough(transformStream)
                .pipeTo(streams.writable);
        }
    }
    
    for (const receiver of receivers) {
        const streams = receiver.createEncodedStreams();
        const transformStream = new TransformStream({
            transform: async (encodedFrame, controller) => {
                // Decrypt frame
                const decryptedData = await decrypt(
                    encodedFrame.data,
                    encryptionKey
                );
                encodedFrame.data = decryptedData;
                controller.enqueue(encodedFrame);
            }
        });
        
        streams.readable
            .pipeThrough(transformStream)
            .pipeTo(streams.writable);
    }
}
```

2. **Validate TURN Credentials**
```javascript
async function getSecureTurnCredentials() {
    const response = await fetch('/api/turn-credentials', {
        headers: {
            'Authorization': `Bearer ${apiKey}`
        }
    });
    
    const data = await response.json();
    
    // Credentials are time-limited
    return {
        urls: data.urls,
        username: data.username,
        credential: data.credential,
        credentialType: 'password',
        validUntil: data.validUntil
    };
}
```

### Performance Optimization

1. **Use Hardware Acceleration**
```javascript
// Request hardware-accelerated encoding
const videoConstraints = {
    width: 1920,
    height: 1080,
    frameRate: 30,
    // Request hardware encoding
    encodingParameters: {
        hardwareAcceleration: 'prefer-hardware'
    }
};

// Check if hardware acceleration is available
navigator.mediaDevices.getSupportedConstraints().then(constraints => {
    if (constraints.hardwareAcceleration) {
        console.log('Hardware acceleration supported');
    }
});
```

2. **Implement Adaptive Streaming**
```javascript
class AdaptiveStreaming {
    constructor(pc) {
        this.pc = pc;
        this.statsInterval = null;
        this.qualityLevels = [
            { resolution: 240, bitrate: 200000 },
            { resolution: 360, bitrate: 400000 },
            { resolution: 480, bitrate: 700000 },
            { resolution: 720, bitrate: 1500000 },
            { resolution: 1080, bitrate: 3000000 }
        ];
        this.currentLevel = 2;
    }
    
    start() {
        this.statsInterval = setInterval(() => {
            this.adjustQuality();
        }, 2000);
    }
    
    async adjustQuality() {
        const stats = await this.pc.getStats();
        const metrics = this.calculateMetrics(stats);
        
        if (metrics.rtt > 200 || metrics.packetLoss > 0.02) {
            // Decrease quality
            this.currentLevel = Math.max(0, this.currentLevel - 1);
        } else if (metrics.rtt < 50 && metrics.packetLoss < 0.001) {
            // Increase quality
            this.currentLevel = Math.min(
                this.qualityLevels.length - 1,
                this.currentLevel + 1
            );
        }
        
        await this.applyQualityLevel(this.currentLevel);
    }
}
```

## Troubleshooting

### Common Issues

#### Connection Failures
```javascript
// Debug connection issues
pc.addEventListener('icecandidateerror', (event) => {
    console.error('ICE candidate error:', {
        errorCode: event.errorCode,
        errorText: event.errorText,
        url: event.url,
        address: event.address,
        port: event.port
    });
});

// Check ICE gathering state
pc.addEventListener('icegatheringstatechange', () => {
    console.log('ICE gathering state:', pc.iceGatheringState);
    if (pc.iceGatheringState === 'complete') {
        const candidates = pc.localDescription.sdp.match(/a=candidate:.*/g);
        console.log('ICE candidates found:', candidates?.length || 0);
    }
});
```

#### Media Issues
```javascript
// Diagnose media problems
async function diagnoseMedia() {
    try {
        // Test camera access
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        console.log('Available cameras:', cameras);
        
        // Test with different constraints
        const constraints = [
            { video: true, audio: true },
            { video: { facingMode: 'user' }, audio: true },
            { video: { width: 640, height: 480 }, audio: true }
        ];
        
        for (const constraint of constraints) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraint);
                console.log('Success with constraints:', constraint);
                stream.getTracks().forEach(track => track.stop());
            } catch (e) {
                console.error('Failed with constraints:', constraint, e);
            }
        }
    } catch (error) {
        console.error('Media diagnosis failed:', error);
    }
}
```

#### Performance Issues
```javascript
// Performance monitoring
class PerformanceMonitor {
    constructor(pc) {
        this.pc = pc;
        this.metrics = [];
    }
    
    async collect() {
        const stats = await this.pc.getStats();
        const metric = {
            timestamp: Date.now(),
            bitrate: { video: 0, audio: 0 },
            framerate: 0,
            packetLoss: 0,
            jitter: 0,
            rtt: 0
        };
        
        stats.forEach(report => {
            if (report.type === 'outbound-rtp') {
                if (report.mediaType === 'video') {
                    metric.bitrate.video = report.bitrate || 0;
                    metric.framerate = report.framesPerSecond || 0;
                } else if (report.mediaType === 'audio') {
                    metric.bitrate.audio = report.bitrate || 0;
                }
            } else if (report.type === 'remote-inbound-rtp') {
                metric.packetLoss = report.packetsLost || 0;
                metric.jitter = report.jitter || 0;
                metric.rtt = report.roundTripTime || 0;
            }
        });
        
        this.metrics.push(metric);
        this.analyze();
    }
    
    analyze() {
        if (this.metrics.length < 10) return;
        
        const recent = this.metrics.slice(-10);
        const avgFramerate = recent.reduce((sum, m) => sum + m.framerate, 0) / 10;
        const avgBitrate = recent.reduce((sum, m) => sum + m.bitrate.video, 0) / 10;
        
        if (avgFramerate < 20) {
            console.warn('Low framerate detected:', avgFramerate);
        }
        if (avgBitrate < 500000) {
            console.warn('Low bitrate detected:', avgBitrate);
        }
    }
}
```

### Debug Tools

Enable verbose logging:
```javascript
// Enable WebRTC debugging
if (window.chrome && chrome.webstore) {
    // Chrome
    localStorage.debug = 'simple-peer:*';
} else if (window.navigator.userAgent.includes('Firefox')) {
    // Firefox
    window.mozRTCPeerConnection.generateCertificate({
        name: 'ECDSA',
        namedCurve: 'P-256'
    }).then(cert => {
        console.log('Certificate generated:', cert);
    });
}

// Custom debug logger
class WebRTCDebugger {
    constructor(pc) {
        this.pc = pc;
        this.events = [];
        this.setupListeners();
    }
    
    setupListeners() {
        const events = [
            'negotiationneeded', 'icecandidate', 'icecandidateerror',
            'track', 'iceconnectionstatechange', 'icegatheringstatechange',
            'connectionstatechange', 'signalingstatechange'
        ];
        
        events.forEach(event => {
            this.pc.addEventListener(event, (e) => {
                this.log(event, e);
            });
        });
    }
    
    log(event, data) {
        const entry = {
            timestamp: new Date().toISOString(),
            event: event,
            data: this.serializeEvent(data)
        };
        
        this.events.push(entry);
        console.log(`[WebRTC] ${event}:`, entry);
    }
    
    export() {
        return {
            events: this.events,
            finalState: {
                signalingState: this.pc.signalingState,
                iceConnectionState: this.pc.iceConnectionState,
                connectionState: this.pc.connectionState
            }
        };
    }
}
```

### Support Resources

- WebRTC API Documentation: https://docs.ooblex.com/webrtc
- Example Applications: https://github.com/ooblex/webrtc-examples
- Community Forum: https://forum.ooblex.com/webrtc
- Real-time Support: https://chat.ooblex.com