# HLS/DASH Streaming Service

This service transcodes WebRTC and other media streams into HLS (HTTP Live Streaming) and DASH (Dynamic Adaptive Streaming over HTTP) formats for massive scalability and CDN distribution.

## Features

- **Multi-format Support**: Transcode from WebRTC, RTMP, RTSP, or any FFmpeg-supported source
- **Adaptive Bitrate**: Automatic generation of multiple quality levels (1080p, 720p, 480p, 360p, 240p)
- **Low Latency**: Support for Low-Latency HLS (LL-HLS) with 2-second segments
- **CDN-Ready**: Optimized for CDN edge caching with proper cache headers
- **Auto-cleanup**: Automatic removal of stale streams and old segments
- **Monitoring**: Built-in health checks and Prometheus metrics

## Architecture

```
WebRTC/RTMP Source → FFmpeg Transcoding → HLS/DASH Segments → Nginx → CDN → Viewers
                            ↓
                     Multiple Qualities
                     (1080p/720p/480p/360p/240p)
```

## API Endpoints

### Start Stream
```bash
POST /streams
{
  "stream_id": "unique-stream-id",
  "source_url": "http://media-server:8080/stream/12345",
  "qualities": ["720p", "480p", "360p"],
  "enable_hls": true,
  "enable_dash": true,
  "low_latency": true
}
```

### Stop Stream
```bash
DELETE /streams/{stream_id}
```

### Get Stream Info
```bash
GET /streams/{stream_id}
```

### List Active Streams
```bash
GET /streams
```

### Stream Heartbeat
```bash
POST /streams/{stream_id}/heartbeat
```

## Playback URLs

Once a stream is started, it can be accessed at:

- **HLS Playlist**: `http://your-domain/streams/{stream_id}/playlist.m3u8`
- **DASH Manifest**: `http://your-domain/streams/{stream_id}/manifest.mpd`
- **CDN Edge**: `http://your-domain:8084/{stream_id}/playlist.m3u8`

## Example Usage

### Using the Example Client

```bash
# Start a stream from WebRTC source
python example_client.py start my-stream http://media-server:8080/stream/room123

# Monitor stream with heartbeats
python example_client.py monitor my-stream 300

# List all active streams
python example_client.py list

# Stop a stream
python example_client.py stop my-stream
```

### Direct API Usage

```bash
# Start stream
curl -X POST http://localhost:8083/streams \
  -H "Content-Type: application/json" \
  -d '{
    "stream_id": "test-stream",
    "source_url": "rtmp://localhost/live/test",
    "qualities": ["720p", "480p"],
    "enable_hls": true,
    "enable_dash": false,
    "low_latency": true
  }'

# Get stream info
curl http://localhost:8083/streams/test-stream

# Stop stream
curl -X DELETE http://localhost:8083/streams/test-stream
```

## Player Integration

### HLS.js (Web)

```html
<video id="video" controls></video>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<script>
  var video = document.getElementById('video');
  var videoSrc = 'http://your-domain/streams/test-stream/playlist.m3u8';
  
  if (Hls.isSupported()) {
    var hls = new Hls({
      lowLatencyMode: true,
      backBufferLength: 90
    });
    hls.loadSource(videoSrc);
    hls.attachMedia(video);
  } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
    video.src = videoSrc;
  }
</script>
```

### DASH.js (Web)

```html
<video id="video" controls></video>
<script src="https://cdn.dashjs.org/latest/dash.all.min.js"></script>
<script>
  var url = "http://your-domain/streams/test-stream/manifest.mpd";
  var player = dashjs.MediaPlayer().create();
  player.initialize(document.querySelector("#video"), url, true);
  player.updateSettings({
    'streaming': {
      'lowLatencyEnabled': true,
      'liveDelay': 3,
      'liveCatchUpMinDrift': 0.05,
      'liveCatchUpPlaybackRate': 0.5
    }
  });
</script>
```

### VLC Player

```bash
# HLS
vlc http://your-domain/streams/test-stream/playlist.m3u8

# DASH
vlc http://your-domain/streams/test-stream/manifest.mpd
```

## Configuration

Environment variables:

- `SEGMENT_DURATION`: Segment duration in seconds (default: 2)
- `PLAYLIST_SIZE`: Number of segments in playlist (default: 10)
- `LOW_LATENCY_HLS`: Enable LL-HLS features (default: true)
- `CLEANUP_INTERVAL`: Cleanup check interval in seconds (default: 300)
- `STALE_THRESHOLD`: Time before marking stream as stale (default: 600)

## CDN Integration

The service provides a dedicated CDN endpoint on port 8084 with:

- Immutable caching for segments (1 year)
- Short TTL for manifests (1 second)
- Proper CORS headers
- ETag support
- Byte-range requests
- Optimized for edge caching

### CloudFront Configuration

```json
{
  "Origins": [{
    "DomainName": "your-streaming-server.com",
    "OriginPath": "",
    "CustomOriginConfig": {
      "OriginProtocolPolicy": "http-only",
      "HTTPPort": 8084
    }
  }],
  "CacheBehaviors": [{
    "PathPattern": "*.m3u8",
    "TargetOriginId": "streaming-origin",
    "ViewerProtocolPolicy": "redirect-to-https",
    "CachePolicyId": "4135ea2d-6df8-44a3-9df3-4b5a84be39ad",
    "OriginRequestPolicyId": "88a5eaf4-2fd4-4709-b370-b4c650ea3fcf"
  }, {
    "PathPattern": "*.ts",
    "TargetOriginId": "streaming-origin",
    "ViewerProtocolPolicy": "redirect-to-https",
    "CachePolicyId": "658327cA-f89d-4fab-a63d-7e88639e58f6"
  }]
}
```

## Performance Considerations

1. **FFmpeg Process Management**: Each stream spawns FFmpeg processes for transcoding
2. **Disk I/O**: Segments are written to disk; use SSD for best performance
3. **Network Bandwidth**: Calculate bandwidth as sum of all quality bitrates × viewers
4. **CPU Usage**: Transcoding is CPU-intensive; use hardware acceleration when available

## Monitoring

The service exposes Prometheus metrics at `/metrics`:

- `streaming_active_streams`: Number of active streams
- `streaming_ffmpeg_processes`: Number of FFmpeg processes
- `streaming_cpu_percent`: CPU usage percentage
- `streaming_memory_bytes`: Memory usage in bytes

## Troubleshooting

### Stream not starting
- Check source URL is accessible
- Verify FFmpeg can decode the source format
- Check logs for FFmpeg errors

### High latency
- Ensure `LOW_LATENCY_HLS=true`
- Reduce `SEGMENT_DURATION` (minimum 1 second)
- Check network path to CDN

### Playback issues
- Verify segments are being created in `/var/www/streams/{stream_id}/`
- Check CORS headers are present
- Ensure player supports the codec profile