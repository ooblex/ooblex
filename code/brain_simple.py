"""
Simple ML Worker with OpenCV Effects (No Models Required)

This worker provides several image processing effects that demonstrate
the Ooblex pipeline without requiring any AI model downloads.

Effects available:
- face_detection: Detect faces using OpenCV Haar cascades
- background_blur: Blur background (simple Gaussian blur)
- edge_detection: Canny edge detection
- cartoon: Cartoon effect using bilateral filter
- grayscale: Convert to grayscale
- sepia: Sepia tone effect
- denoise: Non-local means denoising
- pixelate_faces: Pixelate detected faces (privacy filter)
- mirror: Horizontal flip
- invert: Invert colors

These effects run fast and don't require GPU.
"""

import json
import logging
import os

import cv2
import numpy as np
import redis
from amqpstorm import Message, UriConnection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (same as original)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672")

# Connect to Redis
r = redis.Redis.from_url(REDIS_URL)

# Connect to RabbitMQ
while True:
    try:
        mainConnection = UriConnection(RABBITMQ_URL)
        break
    except:
        logger.warning("Waiting for RabbitMQ...")
        import time

        time.sleep(2)

mainChannel_out = mainConnection.channel()
mainChannel_in = mainConnection.channel()


# Load face detection cascade (no download needed - comes with OpenCV)
face_cascade = None
try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    logger.info("Face detection cascade loaded")
except Exception as e:
    logger.warning(f"Could not load face cascade: {e}")


def apply_effect(image, effect_name):
    """
    Apply an image processing effect to a frame

    Args:
        image: numpy array (BGR format from OpenCV)
        effect_name: string name of effect

    Returns:
        processed image as numpy array
    """

    if effect_name == "face_detection":
        return effect_face_detection(image)
    elif effect_name == "background_blur":
        return effect_background_blur(image)
    elif effect_name == "edge_detection":
        return effect_edge_detection(image)
    elif effect_name == "cartoon":
        return effect_cartoon(image)
    elif effect_name == "grayscale":
        return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif effect_name == "sepia":
        return effect_sepia(image)
    elif effect_name == "denoise":
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    elif effect_name == "pixelate_faces":
        return effect_pixelate_faces(image)
    elif effect_name == "mirror":
        return cv2.flip(image, 1)
    elif effect_name == "invert":
        return cv2.bitwise_not(image)
    elif effect_name == "blur":
        return cv2.GaussianBlur(image, (21, 21), 0)
    else:
        # Unknown effect - return original
        logger.warning(f"Unknown effect: {effect_name}, returning original")
        return image


def effect_face_detection(image):
    """Detect faces and draw bounding boxes"""
    if face_cascade is None:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around faces
    output = image.copy()
    for x, y, w, h in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

    return output


def effect_background_blur(image):
    """Simple background blur effect"""
    # Blur the entire image
    blurred = cv2.GaussianBlur(image, (21, 21), 0)

    # For demo, just return blurred image
    # In real implementation, would do segmentation first
    return blurred


def effect_edge_detection(image):
    """Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # Convert back to BGR for display
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def effect_cartoon(image):
    """Cartoon effect using bilateral filter and edge detection"""
    # Edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
    )

    # Bilateral filter for color smoothing
    color = cv2.bilateralFilter(image, 9, 300, 300)

    # Combine edges and color
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon


def effect_sepia(image):
    """Sepia tone effect"""
    kernel = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )

    sepia = cv2.transform(image, kernel)
    # Clip values to 0-255
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    return sepia


def effect_pixelate_faces(image):
    """Pixelate detected faces for privacy"""
    if face_cascade is None:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    output = image.copy()

    for x, y, w, h in faces:
        # Extract face region
        face_region = output[y : y + h, x : x + w]

        # Pixelate by downsampling and upsampling
        small = cv2.resize(
            face_region, (w // 10, h // 10), interpolation=cv2.INTER_LINEAR
        )
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Replace face region
        output[y : y + h, x : x + w] = pixelated

    return output


def sendMessage(msg, token):
    """Send message back to client via RabbitMQ broadcast"""
    data = {"msg": msg, "key": token}
    msg_json = json.dumps(data)
    message = Message.create(mainChannel_out, msg_json)
    message.publish("broadcast-all")


def processTask(msg0):
    """Process a task from the queue"""
    msg = msg0.body
    msg = json.loads(msg)

    stream_key = msg.get("streamKey", "")
    task = msg.get("task", "")
    redis_id = msg.get("redisID", "")

    logger.info(f"Processing task: {task} for stream: {stream_key}")

    # Map old task names to new effect names
    effect_map = {
        "FaceOn": "face_detection",
        "TrumpOn": "pixelate_faces",  # Fun replacement for missing model
        "TaylorOn": "cartoon",  # Fun replacement
        "BlurOn": "background_blur",
        "EdgeOn": "edge_detection",
        "CartoonOn": "cartoon",
        "GrayOn": "grayscale",
        "SepiaOn": "sepia",
        "DenoiseOn": "denoise",
        "MirrorOn": "mirror",
        "InvertOn": "invert",
    }

    effect_name = effect_map.get(task, task.lower().replace("on", ""))

    try:
        # Get frame from Redis
        image_data = r.get(redis_id)
        if image_data is None:
            logger.warning(f"Frame not found in Redis: {redis_id}")
            return

        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            logger.warning(f"Could not decode image: {redis_id}")
            return

        # Apply effect
        processed = apply_effect(image, effect_name)

        # Encode result
        success, encoded = cv2.imencode(
            ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if not success:
            logger.warning("Could not encode processed image")
            return

        # Store result in Redis
        result_id = f"processed_{redis_id}"
        r.setex(result_id, 30, encoded.tobytes())  # 30 second TTL

        # Send completion message
        sendMessage(f"Processed with {effect_name}", stream_key)

        logger.info(f"Successfully processed {redis_id} with {effect_name}")

    except Exception as e:
        logger.error(f"Error processing task: {e}", exc_info=True)
        sendMessage(f"Error: {str(e)}", stream_key)


def main():
    """Main worker loop"""
    logger.info("Simple ML Worker starting...")
    logger.info(
        "Available effects: face_detection, background_blur, edge_detection, cartoon, grayscale, sepia, denoise, pixelate_faces, mirror, invert, blur"
    )
    logger.info("No AI models required - using OpenCV only!")

    # Declare task queue
    childChannel = mainConnection.channel()
    childChannel.queue.declare("tf-task", arguments={"x-message-ttl": 10000})

    # Start consuming tasks
    try:
        childChannel.basic.consume(processTask, "tf-task", no_ack=True)
        logger.info("Worker ready, waiting for tasks...")
        childChannel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
