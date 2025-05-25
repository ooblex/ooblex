#!/usr/bin/env python3
"""
Example client for the HLS/DASH streaming service
Demonstrates how to start, monitor, and stop streams
"""

import asyncio
import httpx
import json
import sys

STREAMING_API_URL = "http://localhost:8083"

async def start_stream(stream_id: str, source_url: str):
    """Start a new streaming session"""
    async with httpx.AsyncClient() as client:
        payload = {
            "stream_id": stream_id,
            "source_url": source_url,
            "qualities": ["1080p", "720p", "480p", "360p"],
            "enable_hls": True,
            "enable_dash": True,
            "low_latency": True
        }
        
        response = await client.post(
            f"{STREAMING_API_URL}/streams",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"Stream started successfully!")
            print(f"Stream ID: {data['stream_id']}")
            print(f"HLS URL: http://localhost{data['hls_url']}")
            print(f"DASH URL: http://localhost{data['dash_url']}")
            print(f"Qualities: {', '.join(data['qualities'])}")
            return data
        else:
            print(f"Failed to start stream: {response.text}")
            return None

async def get_stream_info(stream_id: str):
    """Get information about a stream"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{STREAMING_API_URL}/streams/{stream_id}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return None

async def list_streams():
    """List all active streams"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{STREAMING_API_URL}/streams")
        
        if response.status_code == 200:
            streams = response.json()
            if streams:
                print("\nActive streams:")
                for stream in streams:
                    print(f"- {stream['stream_id']} (Status: {stream['status']})")
                    if stream.get('hls_url'):
                        print(f"  HLS: http://localhost{stream['hls_url']}")
                    if stream.get('dash_url'):
                        print(f"  DASH: http://localhost{stream['dash_url']}")
            else:
                print("No active streams")
            return streams
        else:
            print(f"Failed to list streams: {response.text}")
            return []

async def stop_stream(stream_id: str):
    """Stop a streaming session"""
    async with httpx.AsyncClient() as client:
        response = await client.delete(f"{STREAMING_API_URL}/streams/{stream_id}")
        
        if response.status_code == 200:
            print(f"Stream {stream_id} stopped successfully")
            return True
        else:
            print(f"Failed to stop stream: {response.text}")
            return False

async def send_heartbeat(stream_id: str):
    """Send heartbeat to keep stream alive"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{STREAMING_API_URL}/streams/{stream_id}/heartbeat"
        )
        return response.status_code == 200

async def monitor_stream(stream_id: str, duration: int = 60):
    """Monitor a stream and send heartbeats"""
    print(f"\nMonitoring stream {stream_id} for {duration} seconds...")
    print("Press Ctrl+C to stop")
    
    start_time = asyncio.get_event_loop().time()
    heartbeat_interval = 30  # seconds
    last_heartbeat = 0
    
    try:
        while asyncio.get_event_loop().time() - start_time < duration:
            current_time = asyncio.get_event_loop().time()
            
            # Send heartbeat
            if current_time - last_heartbeat >= heartbeat_interval:
                if await send_heartbeat(stream_id):
                    print(f"Heartbeat sent at {int(current_time - start_time)}s")
                    last_heartbeat = current_time
                else:
                    print("Failed to send heartbeat")
                    
            # Check stream status
            info = await get_stream_info(stream_id)
            if info:
                print(f"Stream status: {info['status']}")
            else:
                print("Stream not found")
                break
                
            await asyncio.sleep(5)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

async def main():
    print("HLS/DASH Streaming Service Example Client")
    print("=========================================")
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python example_client.py start <stream_id> <source_url>")
        print("  python example_client.py stop <stream_id>")
        print("  python example_client.py info <stream_id>")
        print("  python example_client.py list")
        print("  python example_client.py monitor <stream_id> [duration]")
        print("\nExample:")
        print("  python example_client.py start test-stream http://media-server:8080/stream/12345")
        print("  python example_client.py start webcam-1 rtmp://localhost/live/webcam")
        print("  python example_client.py monitor test-stream 300")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "start" and len(sys.argv) >= 4:
        stream_id = sys.argv[2]
        source_url = sys.argv[3]
        await start_stream(stream_id, source_url)
        
    elif command == "stop" and len(sys.argv) >= 3:
        stream_id = sys.argv[2]
        await stop_stream(stream_id)
        
    elif command == "info" and len(sys.argv) >= 3:
        stream_id = sys.argv[2]
        info = await get_stream_info(stream_id)
        if info:
            print(f"\nStream Information:")
            print(json.dumps(info, indent=2))
        else:
            print(f"Stream {stream_id} not found")
            
    elif command == "list":
        await list_streams()
        
    elif command == "monitor" and len(sys.argv) >= 3:
        stream_id = sys.argv[2]
        duration = int(sys.argv[3]) if len(sys.argv) >= 4 else 60
        await monitor_stream(stream_id, duration)
        
    else:
        print(f"Invalid command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())