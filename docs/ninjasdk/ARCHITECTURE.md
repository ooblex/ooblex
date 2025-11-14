# NinjaSDK Integration Architecture

## Overview

This document provides a detailed technical overview of the NinjaSDK integration architecture in Ooblex.

## System Components

### 1. NinjaSDK Audio Ingestion Service (Node.js)

**Location**: `services/ninjasdk-audio-ingestion/`

**Purpose**: Establishes P2P WebRTC connections with clients and captures raw audio streams.

**Key Classes**:

- `NinjaSDKAudioService` - Main service orchestrator
- `AudioStreamManager` - Per-stream audio processing
- `RTCAudioSink` - Raw audio capture from WebRTC tracks

**Process Flow**:

```
1. Initialize NinjaSDK client
2. Join specified room
3. Listen for new stream events
4. Establish peer connection via quickView()
5. Attach RTCAudioSink to audio tracks
6. Capture raw audio samples (Float32/Int16)
7. Convert to PCM16 format
8. Resample to 16kHz mono
9. Buffer audio chunks
10. Flush to Redis when chunk size reached
11. Publish event to RabbitMQ
```

**Audio Processing**:

```javascript
// Format conversion
Float32Array → PCM16 Buffer
Int16Array → PCM16 Buffer

// Resampling
48kHz → 16kHz (linear interpolation)
44.1kHz → 16kHz

// Channel conversion
Stereo → Mono (averaging)

// Buffering
Accumulate until chunk duration reached
Default: 1000ms chunks
```

**Data Structures**:

```javascript
// Audio chunk metadata
{
  streamID: 'stream-123',
  timestamp: 1234567890,
  sampleRate: 16000,
  channels: 1,
  format: 'pcm16',
  durationMs: 1000.0,
  bytes: 32000
}

// Redis keys
audio:chunk:{streamID}:{timestamp} → Raw PCM16 data
audio:chunk:{streamID}:{timestamp}:meta → Metadata JSON
```

### 2. Whisper STT Worker (Python)

**Location**: `code/whisper_worker.py`

**Purpose**: Consumes audio chunks and transcribes using Whisper ASR.

**Key Classes**:

- `WhisperWorker` - Main worker class
- `AudioChunk` - Audio chunk data structure

**Process Flow**:

```
1. Connect to RabbitMQ and Redis
2. Load Whisper model (faster-whisper or openai-whisper)
3. Consume from 'audio-chunks' queue
4. Retrieve audio data from Redis
5. Convert PCM16 to Float32 numpy array
6. Run Whisper transcription
7. Extract text, confidence, language
8. Publish to 'stt-results' queue
```

**Whisper Model Selection**:

| Model | Parameters | Memory | Speed | Use Case |
|-------|-----------|---------|-------|----------|
| tiny | 39M | 1GB | 32x | Development, testing |
| base | 74M | 1GB | 16x | Production (recommended) |
| small | 244M | 2GB | 6x | Higher accuracy |
| medium | 769M | 5GB | 2x | Professional use |
| large | 1550M | 10GB | 1x | Maximum accuracy |

**Optimization Techniques**:

```python
# faster-whisper optimizations
- CTranslate2 backend (2-4x faster)
- Int8 quantization for CPU
- Float16 for GPU
- VAD filtering (skip silence)
- Beam search (accuracy vs speed)

# Memory management
- Chunk processing (no full file loading)
- Model caching
- Batch processing (future)
```

**Output Format**:

```json
{
  "streamID": "stream-123",
  "chunkID": "audio:chunk:stream-123:1234567890",
  "timestamp": 1234567890,
  "text": "This is the transcribed text",
  "language": "en",
  "language_probability": 0.99,
  "confidence": 0.95,
  "segments": 1,
  "audioDurationMs": 1000.0
}
```

### 3. LLM Response Worker (Python)

**Location**: `code/llm_worker.py`

**Purpose**: Generates AI responses to demonstrate cascading processing.

**Key Classes**:

- `LLMWorker` - Main worker
- `ConversationContext` - Per-stream conversation history
- `BaseLLMBackend` - Backend interface
- `OllamaBackend`, `TransformersBackend`, `LlamaCppBackend`, `MockLLMBackend` - Backend implementations

**Process Flow**:

```
1. Connect to RabbitMQ and Redis
2. Initialize LLM backend
3. Consume from 'stt-results' queue
4. Retrieve conversation context for stream
5. Add user message to context
6. Build messages array (system + history + user)
7. Generate LLM response
8. Add assistant message to context
9. Publish to 'llm-responses' queue
10. NinjaSDK service forwards to client via data channel
```

**LLM Backend Architecture**:

```python
class BaseLLMBackend:
    def initialize() -> None
    def generate(messages, max_tokens, temperature) -> str
    def cleanup() -> None

# Implementations
OllamaBackend → ollama.chat()
TransformersBackend → pipeline('text-generation')
LlamaCppBackend → Llama.create_chat_completion()
MockLLMBackend → Hardcoded responses (testing)
```

**Conversation Management**:

```python
# Context structure
{
  "stream_id": "stream-123",
  "messages": [
    {"role": "system", "content": "You are..."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    ...
  ],
  "created_at": "2024-01-15T10:00:00",
  "last_activity": "2024-01-15T10:05:00"
}

# Context length limiting
Keep last N message pairs (default: 5)
Prevents unbounded memory growth
Maintains conversation coherence
```

## Data Flow

### Audio Ingestion Flow

```
┌──────────────┐
│   Browser    │
└──────┬───────┘
       │ getUserMedia()
       ▼
┌──────────────┐
│ MediaStream  │
└──────┬───────┘
       │ publish()
       ▼
┌──────────────┐
│  NinjaSDK    │
│    P2P       │
└──────┬───────┘
       │ WebRTC Audio Track
       ▼
┌──────────────────┐
│ RTCAudioSink     │
│ ondata callback  │
└──────┬───────────┘
       │ { samples, sampleRate, channels }
       ▼
┌─────────────────────┐
│ AudioStreamManager  │
│ - toPCM16()         │
│ - resample()        │
│ - stereoToMono()    │
│ - buffer()          │
└──────┬──────────────┘
       │ PCM16 chunks
       ▼
┌──────────────┐
│    Redis     │
│  (30s TTL)   │
└──────┬───────┘
       │ chunk notification
       ▼
┌──────────────┐
│  RabbitMQ    │
│ audio-chunks │
└──────────────┘
```

### Transcription Flow

```
┌──────────────┐
│  RabbitMQ    │
│ audio-chunks │
└──────┬───────┘
       │ consume
       ▼
┌──────────────────┐
│ Whisper Worker   │
└──────┬───────────┘
       │ get chunk ID
       ▼
┌──────────────┐
│    Redis     │
│  audio data  │
└──────┬───────┘
       │ PCM16 bytes
       ▼
┌──────────────────┐
│ pcm16_to_float32 │
└──────┬───────────┘
       │ numpy array
       ▼
┌──────────────────┐
│ Whisper Model    │
│ transcribe()     │
└──────┬───────────┘
       │ segments
       ▼
┌──────────────────┐
│ Extract text     │
│ + confidence     │
└──────┬───────────┘
       │ transcription
       ▼
┌──────────────┐
│  RabbitMQ    │
│ stt-results  │
└──────┬───────┘
       │ consume
       ▼
┌──────────────────┐
│ NinjaSDK Service │
│ sendToClient()   │
└──────┬───────────┘
       │ data channel
       ▼
┌──────────────┐
│   Browser    │
│ onmessage    │
└──────────────┘
```

### LLM Response Flow (Optional)

```
┌──────────────┐
│  RabbitMQ    │
│ stt-results  │
└──────┬───────┘
       │ consume
       ▼
┌──────────────────┐
│  LLM Worker      │
└──────┬───────────┘
       │ get context
       ▼
┌──────────────────┐
│ Conversation     │
│ Context          │
└──────┬───────────┘
       │ add message
       ▼
┌──────────────────┐
│ Build prompt     │
│ (system+history) │
└──────┬───────────┘
       │ messages array
       ▼
┌──────────────────┐
│ LLM Backend      │
│ generate()       │
└──────┬───────────┘
       │ response text
       ▼
┌──────────────────┐
│ Update context   │
└──────┬───────────┘
       │ llm response
       ▼
┌──────────────┐
│  RabbitMQ    │
│llm-responses │
└──────┬───────┘
       │ consume
       ▼
┌──────────────────┐
│ NinjaSDK Service │
│ sendToClient()   │
└──────┬───────────┘
       │ data channel
       ▼
┌──────────────┐
│   Browser    │
│ onmessage    │
└──────────────┘
```

## Message Queue Schema

### audio-chunks Queue

**Producer**: NinjaSDK Audio Service
**Consumer**: Whisper Worker

```json
{
  "chunkID": "audio:chunk:stream-123:1234567890",
  "metadata": {
    "streamID": "stream-123",
    "timestamp": 1234567890,
    "sampleRate": 16000,
    "channels": 1,
    "format": "pcm16",
    "durationMs": 1000.0,
    "bytes": 32000
  }
}
```

### stt-results Queue

**Producer**: Whisper Worker
**Consumers**: NinjaSDK Service, LLM Worker

```json
{
  "streamID": "stream-123",
  "chunkID": "audio:chunk:stream-123:1234567890",
  "timestamp": 1234567890,
  "text": "transcribed text",
  "language": "en",
  "confidence": 0.95,
  "segments": 1,
  "audioDurationMs": 1000.0
}
```

### llm-responses Queue

**Producer**: LLM Worker
**Consumer**: NinjaSDK Service

```json
{
  "streamID": "stream-123",
  "timestamp": 1234567890,
  "userText": "user input",
  "response": "ai response",
  "type": "llm_response"
}
```

## Redis Data Structures

### Audio Chunks

```
Key: audio:chunk:{streamID}:{timestamp}
Type: String (binary)
Value: Raw PCM16 audio bytes
TTL: 300 seconds

Key: audio:chunk:{streamID}:{timestamp}:meta
Type: String (JSON)
Value: Chunk metadata
TTL: 300 seconds
```

### Stream Metadata

```
Key: stream:meta:{streamID}
Type: Hash
Fields:
  - startTime: ISO timestamp
  - lastActivity: ISO timestamp
  - totalChunks: integer
  - totalBytes: integer
TTL: 3600 seconds
```

## WebRTC Data Channels

### ooblex-stt Channel

**Direction**: Bidirectional
**Ordered**: Yes
**Reliable**: Yes

**Client → Server**:
```json
{
  "type": "command",
  "action": "start|stop|status"
}
```

**Server → Client**:
```json
// Welcome message
{
  "type": "welcome",
  "message": "Connected to Ooblex...",
  "timestamp": 1234567890
}

// Transcription
{
  "type": "transcription",
  "text": "transcribed text",
  "confidence": 0.95,
  "timestamp": 1234567890
}

// LLM response
{
  "type": "llm_response",
  "response": "ai response",
  "timestamp": 1234567890
}
```

## Scaling Architecture

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────┐
│                  Load Balancer                      │
└──────┬──────────────────────────────────────────────┘
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ NinjaSDK │   │ NinjaSDK │   │ NinjaSDK │   │ NinjaSDK │
│ Service  │   │ Service  │   │ Service  │   │ Service  │
│    #1    │   │    #2    │   │    #3    │   │    #4    │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                         │
                         ▼
                  ┌──────────┐
                  │  Redis   │
                  └────┬─────┘
                       │
                       ▼
                  ┌──────────┐
                  │ RabbitMQ │
                  └────┬─────┘
                       │
       ┌───────────────┼───────────────┬──────────────┐
       ▼               ▼               ▼              ▼
┌──────────┐    ┌──────────┐    ┌──────────┐   ┌──────────┐
│ Whisper  │    │ Whisper  │    │ Whisper  │   │ Whisper  │
│ Worker #1│    │ Worker #2│    │ Worker #3│   │ Worker #4│
│  (GPU)   │    │  (GPU)   │    │  (GPU)   │   │  (GPU)   │
└────┬─────┘    └────┬─────┘    └────┬─────┘   └────┬─────┘
     │               │               │              │
     └───────────────┴───────────────┴──────────────┘
                         │
                         ▼
                  ┌──────────┐
                  │  Redis   │
                  └────┬─────┘
                       │
                       ▼
                  ┌──────────┐
                  │ RabbitMQ │
                  └────┬─────┘
                       │
       ┌───────────────┴───────────────┐
       ▼                               ▼
┌──────────┐                    ┌──────────┐
│   LLM    │                    │   LLM    │
│Worker #1 │                    │Worker #2 │
└──────────┘                    └──────────┘
```

### Performance Characteristics

| Component | Instances | Capacity | Bottleneck |
|-----------|-----------|----------|------------|
| NinjaSDK Service | 4 | 400 streams | Network bandwidth |
| Whisper Worker (GPU) | 4 | 200 streams | GPU memory |
| Whisper Worker (CPU) | 10 | 50 streams | CPU cores |
| LLM Worker | 2 | 100 req/s | Model inference |
| Redis | 1 | 10K ops/s | Memory |
| RabbitMQ | 1 | 50K msg/s | Disk I/O |

## Security Considerations

### WebRTC Security

- **DTLS Encryption**: All WebRTC media encrypted by default
- **SRTP**: Secure RTP for media streams
- **Data Channel Encryption**: TLS-like security
- **ICE/STUN/TURN**: NAT traversal security

### Application Security

```python
# Input validation
- Sanitize stream IDs
- Validate audio formats
- Limit chunk sizes
- Rate limiting

# Access control
- Room passwords
- Stream authentication
- API key validation

# Data protection
- Redis TTL (auto-expiry)
- No persistent audio storage
- Ephemeral processing
```

### Privacy Considerations

- **No Audio Storage**: Audio deleted after processing (5 min TTL)
- **No Transcript Storage**: Transcripts not persisted
- **Ephemeral Contexts**: Conversation contexts in memory only
- **P2P Architecture**: Direct browser-to-service connections
- **No Third-Party APIs**: All processing local

## Monitoring & Observability

### Metrics to Track

```python
# Audio Ingestion
- Active streams
- Bytes received/s
- Audio buffer size
- Connection states

# Whisper Worker
- Chunks processed
- Average processing time
- RTF (Real-Time Factor)
- Queue depth

# LLM Worker
- Messages processed
- Response generation time
- Active conversations
- Token usage
```

### Logging

```javascript
// Structured logging
{
  "timestamp": "2024-01-15T10:00:00Z",
  "level": "info",
  "service": "ninjasdk-audio",
  "streamID": "stream-123",
  "event": "audio_chunk_flushed",
  "bytes": 32000,
  "duration_ms": 1000
}
```

### Health Checks

```bash
# NinjaSDK Service
GET /health → { "status": "healthy", "streams": 5 }

# Whisper Worker
RabbitMQ queue depth < 100

# LLM Worker
Response time < 5s
```

## Future Enhancements

1. **Streaming Transcription**: Real-time word-by-word transcription
2. **Speaker Diarization**: Identify different speakers
3. **Emotion Detection**: Detect emotional tone
4. **Multi-Language**: Auto-detect and switch languages
5. **Voice Activity Detection**: Skip silence intelligently
6. **Batch Processing**: Process multiple chunks together
7. **Result Caching**: Cache common phrases
8. **Model Quantization**: Further optimize inference
9. **WebAssembly**: Client-side processing option
10. **Recording**: Optional audio archival

## References

- [NinjaSDK GitHub](https://github.com/steveseguin/ninjasdk/)
- [Whisper Paper](https://arxiv.org/abs/2212.04356)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [WebRTC Specification](https://www.w3.org/TR/webrtc/)
- [RTCAudioSink](https://www.w3.org/TR/webrtc/#dom-rtcaudiosink)
