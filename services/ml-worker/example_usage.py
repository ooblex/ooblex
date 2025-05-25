#!/usr/bin/env python3
"""
Example of how to use the ML worker effects in your application
"""
import json
import time
import asyncio
import redis.asyncio as redis
from aio_pika import connect_robust, Message
import uuid
import cv2
import numpy as np


async def submit_ml_task(channel, redis_client, stream_token, process_type, parameters=None):
    """Submit an ML processing task"""
    task_id = str(uuid.uuid4())
    
    task_data = {
        "task_id": task_id,
        "stream_token": stream_token,
        "process_type": process_type,
        "parameters": parameters or {}
    }
    
    # Publish to RabbitMQ
    message = Message(
        json.dumps(task_data).encode(),
        delivery_mode=2  # Persistent
    )
    
    await channel.default_exchange.publish(
        message,
        routing_key="tasks"
    )
    
    print(f"Submitted task {task_id} for {process_type}")
    return task_id


async def wait_for_result(redis_client, task_id, timeout=10):
    """Wait for task completion and get result"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        task_status = await redis_client.get(f"task:{task_id}")
        if task_status:
            status_data = json.loads(task_status)
            if status_data["status"] == "completed":
                return status_data
            elif status_data["status"] == "failed":
                raise Exception(f"Task failed: {status_data.get('error')}")
        
        await asyncio.sleep(0.1)
    
    raise TimeoutError(f"Task {task_id} timed out")


async def example_workflow():
    """Example workflow using ML worker effects"""
    
    # Connect to Redis
    redis_client = await redis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=False
    )
    
    # Connect to RabbitMQ
    connection = await connect_robust("amqp://guest:guest@localhost/")
    channel = await connection.channel()
    
    # Create a test image
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(test_image, (100, 100), (300, 300), (0, 0, 255), -1)
    cv2.circle(test_image, (400, 200), 80, (0, 255, 0), -1)
    cv2.putText(test_image, "ML Worker Test", (150, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Encode and store in Redis
    stream_token = "test_stream_" + str(uuid.uuid4())
    _, encoded = cv2.imencode('.jpg', test_image)
    await redis_client.setex(f"frame:{stream_token}:latest", 60, encoded.tobytes())
    
    print(f"Created test stream: {stream_token}")
    
    # Test different effects
    effects = [
        ("cartoon", {}),
        ("sepia", {}),
        ("edge_detection", {"low_threshold": 50, "high_threshold": 150}),
        ("background_blur", {}),
        ("pixelate", {"pixel_size": 15}),
        ("vintage", {}),
    ]
    
    for effect_type, params in effects:
        print(f"\nTesting {effect_type} effect...")
        
        # Submit task
        task_id = await submit_ml_task(
            channel, redis_client, stream_token, effect_type, params
        )
        
        # Wait for result
        try:
            result = await wait_for_result(redis_client, task_id)
            print(f"Task completed in {result['duration']:.2f}s")
            
            # Get processed frame
            result_key = result['result_key']
            processed_frame = await redis_client.get(result_key)
            
            if processed_frame:
                # Decode and save
                nparr = np.frombuffer(processed_frame, np.uint8)
                processed_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cv2.imwrite(f"example_{effect_type}.jpg", processed_image)
                print(f"Saved example_{effect_type}.jpg")
            
        except Exception as e:
            print(f"Error processing {effect_type}: {e}")
    
    # Cleanup
    await connection.close()
    await redis_client.close()
    
    print("\nExample workflow completed!")


if __name__ == "__main__":
    asyncio.run(example_workflow())