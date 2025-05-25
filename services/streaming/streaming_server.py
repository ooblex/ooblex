#!/usr/bin/env python3
"""
HLS/DASH Streaming Server
Transcodes WebRTC streams into HLS/DASH for massive scalability
"""

import asyncio
import json
import logging
import os
import shutil
import time
from typing import Dict, List, Optional, Set
from pathlib import Path
import aiofiles
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import subprocess
import signal
import psutil

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
MEDIA_SERVER_URL = os.getenv('MEDIA_SERVER_URL', 'http://media-server:8080')
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', '/var/www/streams'))
SEGMENT_DURATION = int(os.getenv('SEGMENT_DURATION', '2'))
PLAYLIST_SIZE = int(os.getenv('PLAYLIST_SIZE', '10'))
LOW_LATENCY_HLS = os.getenv('LOW_LATENCY_HLS', 'true').lower() == 'true'
CLEANUP_INTERVAL = int(os.getenv('CLEANUP_INTERVAL', '300'))  # 5 minutes
STALE_THRESHOLD = int(os.getenv('STALE_THRESHOLD', '600'))  # 10 minutes

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Quality presets for adaptive bitrate
QUALITY_PRESETS = {
    '1080p': {
        'width': 1920,
        'height': 1080,
        'bitrate': '5000k',
        'audio_bitrate': '192k',
        'profile': 'high',
        'level': '4.2'
    },
    '720p': {
        'width': 1280,
        'height': 720,
        'bitrate': '2800k',
        'audio_bitrate': '128k',
        'profile': 'main',
        'level': '3.1'
    },
    '480p': {
        'width': 854,
        'height': 480,
        'bitrate': '1400k',
        'audio_bitrate': '128k',
        'profile': 'main',
        'level': '3.0'
    },
    '360p': {
        'width': 640,
        'height': 360,
        'bitrate': '800k',
        'audio_bitrate': '96k',
        'profile': 'baseline',
        'level': '3.0'
    },
    '240p': {
        'width': 426,
        'height': 240,
        'bitrate': '400k',
        'audio_bitrate': '64k',
        'profile': 'baseline',
        'level': '3.0'
    }
}

class StreamRequest(BaseModel):
    stream_id: str
    source_url: str
    qualities: List[str] = ['720p', '480p', '360p']
    enable_hls: bool = True
    enable_dash: bool = True
    low_latency: bool = True

class StreamInfo(BaseModel):
    stream_id: str
    status: str
    hls_url: Optional[str]
    dash_url: Optional[str]
    qualities: List[str]
    created_at: float
    last_update: float

class StreamingServer:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.active_streams: Dict[str, dict] = {}
        self.processes: Dict[str, List[subprocess.Popen]] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        
    async def startup(self):
        """Initialize connections and start background tasks"""
        self.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        self.cleanup_task = asyncio.create_task(self.cleanup_loop())
        logger.info("Streaming server started")
        
    async def shutdown(self):
        """Clean up resources"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            
        # Stop all active streams
        for stream_id in list(self.active_streams.keys()):
            await self.stop_stream(stream_id)
            
        if self.redis:
            await self.redis.close()
            
        logger.info("Streaming server stopped")
        
    async def start_stream(self, request: StreamRequest) -> StreamInfo:
        """Start transcoding a stream to HLS/DASH"""
        stream_id = request.stream_id
        
        if stream_id in self.active_streams:
            raise HTTPException(status_code=400, detail="Stream already active")
            
        # Create stream directory
        stream_dir = OUTPUT_DIR / stream_id
        stream_dir.mkdir(exist_ok=True)
        
        # Initialize stream info
        stream_info = {
            'stream_id': stream_id,
            'source_url': request.source_url,
            'qualities': request.qualities,
            'enable_hls': request.enable_hls,
            'enable_dash': request.enable_dash,
            'low_latency': request.low_latency and LOW_LATENCY_HLS,
            'created_at': time.time(),
            'last_update': time.time(),
            'status': 'starting'
        }
        
        self.active_streams[stream_id] = stream_info
        self.processes[stream_id] = []
        
        # Start transcoding processes
        try:
            if request.enable_hls:
                await self._start_hls_transcoding(stream_id, request)
                
            if request.enable_dash:
                await self._start_dash_transcoding(stream_id, request)
                
            # Update status
            stream_info['status'] = 'active'
            stream_info['last_update'] = time.time()
            
            # Store in Redis
            await self.redis.setex(
                f"stream:{stream_id}",
                STALE_THRESHOLD,
                json.dumps(stream_info)
            )
            
            # Return stream info
            return StreamInfo(
                stream_id=stream_id,
                status=stream_info['status'],
                hls_url=f"/streams/{stream_id}/playlist.m3u8" if request.enable_hls else None,
                dash_url=f"/streams/{stream_id}/manifest.mpd" if request.enable_dash else None,
                qualities=request.qualities,
                created_at=stream_info['created_at'],
                last_update=stream_info['last_update']
            )
            
        except Exception as e:
            logger.error(f"Failed to start stream {stream_id}: {e}")
            await self.stop_stream(stream_id)
            raise HTTPException(status_code=500, detail=str(e))
            
    async def _start_hls_transcoding(self, stream_id: str, request: StreamRequest):
        """Start FFmpeg process for HLS transcoding"""
        stream_dir = OUTPUT_DIR / stream_id
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']
        
        # Input options
        cmd.extend([
            '-re',  # Read input at native frame rate
            '-i', request.source_url,
            '-map', '0:v:0',  # Map first video stream
            '-map', '0:a:0',  # Map first audio stream
        ])
        
        # Add video outputs for each quality
        for i, quality in enumerate(request.qualities):
            preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['480p'])
            
            # Video encoding settings
            cmd.extend([
                f'-c:v:{i}', 'libx264',
                f'-b:v:{i}', preset['bitrate'],
                f'-maxrate:v:{i}', preset['bitrate'],
                f'-bufsize:v:{i}', str(int(preset['bitrate'][:-1]) * 2) + 'k',
                f'-profile:v:{i}', preset['profile'],
                f'-level:v:{i}', preset['level'],
                f'-g:v:{i}', str(SEGMENT_DURATION * 30),  # GOP size
                f'-keyint_min:v:{i}', str(SEGMENT_DURATION * 15),
                f'-sc_threshold:v:{i}', '0',
                f'-s:v:{i}', f"{preset['width']}x{preset['height']}",
                
                # Audio encoding settings
                f'-c:a:{i}', 'aac',
                f'-b:a:{i}', preset['audio_bitrate'],
                f'-ac:a:{i}', '2',
                f'-ar:a:{i}', '48000'
            ])
            
        # HLS output settings
        hls_flags = ['delete_segments', 'append_list']
        if request.low_latency and LOW_LATENCY_HLS:
            hls_flags.extend(['low_latency', 'temp_file'])
            
        cmd.extend([
            '-f', 'hls',
            '-hls_time', str(SEGMENT_DURATION),
            '-hls_list_size', str(PLAYLIST_SIZE),
            '-hls_flags', '+'.join(hls_flags),
            '-hls_segment_type', 'mpegts',
            '-hls_segment_filename', str(stream_dir / 'segment_%v_%03d.ts'),
            '-master_pl_name', 'playlist.m3u8',
            '-var_stream_map', ' '.join([f'v:{i},a:{i}' for i in range(len(request.qualities))]),
            str(stream_dir / 'stream_%v.m3u8')
        ])
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        self.processes[stream_id].append(process)
        logger.info(f"Started HLS transcoding for stream {stream_id}")
        
    async def _start_dash_transcoding(self, stream_id: str, request: StreamRequest):
        """Start FFmpeg process for DASH transcoding"""
        stream_dir = OUTPUT_DIR / stream_id
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning']
        
        # Input options
        cmd.extend([
            '-re',
            '-i', request.source_url,
        ])
        
        # Add outputs for each quality
        for quality in request.qualities:
            preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS['480p'])
            
            cmd.extend([
                '-map', '0:v:0',
                '-map', '0:a:0',
                '-c:v', 'libx264',
                '-b:v', preset['bitrate'],
                '-maxrate', preset['bitrate'],
                '-bufsize', str(int(preset['bitrate'][:-1]) * 2) + 'k',
                '-profile:v', preset['profile'],
                '-level', preset['level'],
                '-g', str(SEGMENT_DURATION * 30),
                '-keyint_min', str(SEGMENT_DURATION * 15),
                '-sc_threshold', '0',
                '-s', f"{preset['width']}x{preset['height']}",
                '-c:a', 'aac',
                '-b:a', preset['audio_bitrate'],
                '-ac', '2',
                '-ar', '48000',
            ])
            
        # DASH output settings
        dash_options = [
            '-f', 'dash',
            '-seg_duration', str(SEGMENT_DURATION),
            '-window_size', str(PLAYLIST_SIZE),
            '-remove_at_exit', '1',
            '-use_template', '1',
            '-use_timeline', '1',
            '-adaptation_sets', 'id=0,streams=v id=1,streams=a',
        ]
        
        if request.low_latency:
            dash_options.extend([
                '-ldash', '1',
                '-streaming', '1',
                '-utc_timing_url', 'https://time.akamai.com/?iso'
            ])
            
        cmd.extend(dash_options)
        cmd.append(str(stream_dir / 'manifest.mpd'))
        
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid
        )
        
        self.processes[stream_id].append(process)
        logger.info(f"Started DASH transcoding for stream {stream_id}")
        
    async def stop_stream(self, stream_id: str):
        """Stop transcoding a stream"""
        if stream_id not in self.active_streams:
            raise HTTPException(status_code=404, detail="Stream not found")
            
        # Stop all processes
        if stream_id in self.processes:
            for process in self.processes[stream_id]:
                try:
                    # Send SIGTERM to process group
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=5)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    # Force kill if needed
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                        
            del self.processes[stream_id]
            
        # Remove from active streams
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            
        # Remove from Redis
        await self.redis.delete(f"stream:{stream_id}")
        
        # Clean up files (optional, keep for VOD)
        # stream_dir = OUTPUT_DIR / stream_id
        # if stream_dir.exists():
        #     shutil.rmtree(stream_dir)
            
        logger.info(f"Stopped stream {stream_id}")
        
    async def get_stream_info(self, stream_id: str) -> StreamInfo:
        """Get information about a stream"""
        if stream_id in self.active_streams:
            info = self.active_streams[stream_id]
        else:
            # Check Redis
            data = await self.redis.get(f"stream:{stream_id}")
            if not data:
                raise HTTPException(status_code=404, detail="Stream not found")
            info = json.loads(data)
            
        return StreamInfo(
            stream_id=stream_id,
            status=info['status'],
            hls_url=f"/streams/{stream_id}/playlist.m3u8" if info.get('enable_hls') else None,
            dash_url=f"/streams/{stream_id}/manifest.mpd" if info.get('enable_dash') else None,
            qualities=info['qualities'],
            created_at=info['created_at'],
            last_update=info['last_update']
        )
        
    async def list_streams(self) -> List[StreamInfo]:
        """List all active streams"""
        streams = []
        
        # Get from memory
        for stream_id, info in self.active_streams.items():
            streams.append(StreamInfo(
                stream_id=stream_id,
                status=info['status'],
                hls_url=f"/streams/{stream_id}/playlist.m3u8" if info.get('enable_hls') else None,
                dash_url=f"/streams/{stream_id}/manifest.mpd" if info.get('enable_dash') else None,
                qualities=info['qualities'],
                created_at=info['created_at'],
                last_update=info['last_update']
            ))
            
        # Get from Redis
        keys = await self.redis.keys("stream:*")
        for key in keys:
            stream_id = key.split(":", 1)[1]
            if stream_id not in self.active_streams:
                data = await self.redis.get(key)
                if data:
                    info = json.loads(data)
                    streams.append(StreamInfo(
                        stream_id=stream_id,
                        status=info['status'],
                        hls_url=f"/streams/{stream_id}/playlist.m3u8" if info.get('enable_hls') else None,
                        dash_url=f"/streams/{stream_id}/manifest.mpd" if info.get('enable_dash') else None,
                        qualities=info['qualities'],
                        created_at=info['created_at'],
                        last_update=info['last_update']
                    ))
                    
        return streams
        
    async def update_stream_status(self, stream_id: str):
        """Update stream last_update timestamp"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]['last_update'] = time.time()
            
            # Update Redis
            await self.redis.setex(
                f"stream:{stream_id}",
                STALE_THRESHOLD,
                json.dumps(self.active_streams[stream_id])
            )
            
    async def cleanup_loop(self):
        """Background task to clean up stale streams"""
        while True:
            try:
                await asyncio.sleep(CLEANUP_INTERVAL)
                
                current_time = time.time()
                stale_streams = []
                
                # Check for stale streams
                for stream_id, info in self.active_streams.items():
                    if current_time - info['last_update'] > STALE_THRESHOLD:
                        stale_streams.append(stream_id)
                        
                # Stop stale streams
                for stream_id in stale_streams:
                    logger.warning(f"Cleaning up stale stream {stream_id}")
                    try:
                        await self.stop_stream(stream_id)
                    except Exception as e:
                        logger.error(f"Error cleaning up stream {stream_id}: {e}")
                        
                # Clean up old files
                await self._cleanup_old_files()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                
    async def _cleanup_old_files(self):
        """Clean up old segment files"""
        current_time = time.time()
        
        for stream_dir in OUTPUT_DIR.iterdir():
            if not stream_dir.is_dir():
                continue
                
            # Skip active streams
            if stream_dir.name in self.active_streams:
                continue
                
            # Check if directory is old enough to delete
            try:
                mtime = stream_dir.stat().st_mtime
                if current_time - mtime > STALE_THRESHOLD * 2:
                    logger.info(f"Removing old stream directory: {stream_dir}")
                    shutil.rmtree(stream_dir)
            except Exception as e:
                logger.error(f"Error checking directory {stream_dir}: {e}")

# Create FastAPI app
app = FastAPI(title="HLS/DASH Streaming Server")
server = StreamingServer()

@app.on_event("startup")
async def startup_event():
    await server.startup()

@app.on_event("shutdown")
async def shutdown_event():
    await server.shutdown()

@app.post("/streams", response_model=StreamInfo)
async def create_stream(request: StreamRequest):
    """Start a new streaming session"""
    return await server.start_stream(request)

@app.delete("/streams/{stream_id}")
async def delete_stream(stream_id: str):
    """Stop a streaming session"""
    await server.stop_stream(stream_id)
    return {"status": "stopped"}

@app.get("/streams/{stream_id}", response_model=StreamInfo)
async def get_stream(stream_id: str):
    """Get stream information"""
    return await server.get_stream_info(stream_id)

@app.get("/streams", response_model=List[StreamInfo])
async def list_streams():
    """List all active streams"""
    return await server.list_streams()

@app.post("/streams/{stream_id}/heartbeat")
async def stream_heartbeat(stream_id: str):
    """Update stream heartbeat"""
    await server.update_stream_status(stream_id)
    return {"status": "ok"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "streams": len(server.active_streams)}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    metrics_text = []
    
    # Active streams gauge
    metrics_text.append(f"# HELP streaming_active_streams Number of active streams")
    metrics_text.append(f"# TYPE streaming_active_streams gauge")
    metrics_text.append(f"streaming_active_streams {len(server.active_streams)}")
    
    # Process count
    total_processes = sum(len(procs) for procs in server.processes.values())
    metrics_text.append(f"# HELP streaming_ffmpeg_processes Number of FFmpeg processes")
    metrics_text.append(f"# TYPE streaming_ffmpeg_processes gauge")
    metrics_text.append(f"streaming_ffmpeg_processes {total_processes}")
    
    # CPU and memory usage
    process = psutil.Process()
    metrics_text.append(f"# HELP streaming_cpu_percent CPU usage percentage")
    metrics_text.append(f"# TYPE streaming_cpu_percent gauge")
    metrics_text.append(f"streaming_cpu_percent {process.cpu_percent()}")
    
    metrics_text.append(f"# HELP streaming_memory_bytes Memory usage in bytes")
    metrics_text.append(f"# TYPE streaming_memory_bytes gauge")
    metrics_text.append(f"streaming_memory_bytes {process.memory_info().rss}")
    
    return PlainTextResponse("\n".join(metrics_text))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8083)