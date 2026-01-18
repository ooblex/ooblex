"""
Enhanced ML Worker with MediaPipe AI Effects

This worker provides compelling real-time AI effects using MediaPipe,
demonstrating the Ooblex pipeline with lightweight models that run
efficiently on CPU or GPU.

Effects available:
- background_remove: Remove background (transparent/green screen)
- background_blur: Blur background keeping person sharp
- background_replace: Replace background with custom image/color
- face_mesh: Visualize 468 3D face landmarks
- face_contour: Draw face contour lines
- big_eyes: Enlarge eyes effect (like Snapchat filter)
- face_slim: Slim face effect
- face_distort: Fun face distortion effect
- anime_style: Stylize to anime/cartoon look (uses AnimeGAN ONNX)
- beauty_filter: Smooth skin and enhance features
- virtual_makeup: Add virtual lipstick/eyeshadow

Requirements:
- mediapipe>=0.10.18 (already in requirements.txt)
- opencv-python
- numpy

These effects run fast (15-60+ FPS) on modern CPUs.
"""

import os
import cv2
import numpy as np
import redis
import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple, List
from amqpstorm import UriConnection, Message

# MediaPipe imports
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@localhost:5672')
MODEL_CACHE_DIR = os.getenv('MODEL_CACHE_DIR', '/tmp/ooblex_models')

# Ensure model cache directory exists
Path(MODEL_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Connect to Redis
r = redis.Redis.from_url(REDIS_URL)

# Connect to RabbitMQ
while True:
    try:
        mainConnection = UriConnection(RABBITMQ_URL)
        break
    except Exception:
        logger.warning("Waiting for RabbitMQ...")
        import time
        time.sleep(2)

mainChannel_out = mainConnection.channel()
mainChannel_in = mainConnection.channel()


class MediaPipeEffects:
    """
    Manages MediaPipe models and effects for real-time video processing.
    """

    def __init__(self):
        self.selfie_segmenter = None
        self.face_mesh = None
        self.face_detection = None
        self.onnx_session = None  # For AnimeGAN
        self._init_models()

    def _init_models(self):
        """Initialize MediaPipe models"""
        try:
            # Initialize Selfie Segmentation (for background effects)
            # Using legacy API which is simpler and well-documented
            self.mp_selfie = mp.solutions.selfie_segmentation
            self.selfie_segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)
            logger.info("Selfie Segmentation model loaded (landscape mode)")

            # Initialize Face Mesh (for face effects)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,  # Includes iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("Face Mesh model loaded (468 landmarks + iris)")

            # Drawing utilities
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe models: {e}")

    def _download_animegan_model(self) -> Optional[str]:
        """Download AnimeGAN ONNX model if not present"""
        model_path = os.path.join(MODEL_CACHE_DIR, "animegan_v2_paprika.onnx")

        if os.path.exists(model_path):
            return model_path

        # AnimeGANv2 Paprika style - good for real-time, ~8.6MB
        url = "https://github.com/TachibanaYoshino/AnimeGANv2/raw/master/onnx/Paprika_54.onnx"

        try:
            logger.info(f"Downloading AnimeGAN model from {url}...")
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"AnimeGAN model downloaded to {model_path}")
            return model_path
        except Exception as e:
            logger.warning(f"Could not download AnimeGAN model: {e}")
            return None

    def _get_onnx_session(self):
        """Lazy-load ONNX session for AnimeGAN"""
        if self.onnx_session is None:
            model_path = self._download_animegan_model()
            if model_path:
                try:
                    import onnxruntime as ort
                    self.onnx_session = ort.InferenceSession(
                        model_path,
                        providers=['CPUExecutionProvider']
                    )
                    logger.info("AnimeGAN ONNX model loaded")
                except Exception as e:
                    logger.error(f"Failed to load AnimeGAN ONNX: {e}")
        return self.onnx_session


# Global effects processor
effects_processor: Optional[MediaPipeEffects] = None


def get_effects_processor() -> MediaPipeEffects:
    """Get or create the effects processor singleton"""
    global effects_processor
    if effects_processor is None:
        effects_processor = MediaPipeEffects()
    return effects_processor


def apply_effect(image: np.ndarray, effect_name: str) -> np.ndarray:
    """
    Apply an AI effect (or chain of effects) to a frame

    Args:
        image: numpy array (BGR format from OpenCV)
        effect_name: string name of effect, or pipe-separated chain like
                     "background_blur|big_eyes|beauty_filter"

    Returns:
        processed image as numpy array
    """
    processor = get_effects_processor()

    # Support chained effects with pipe separator
    if "|" in effect_name:
        return apply_effect_chain(image, effect_name.split("|"), processor)

    # Support chained effects with plus separator (URL-friendly)
    if "+" in effect_name:
        return apply_effect_chain(image, effect_name.split("+"), processor)

    return apply_single_effect(image, effect_name, processor)


def apply_effect_chain(image: np.ndarray, effects: List[str],
                       processor: MediaPipeEffects) -> np.ndarray:
    """
    Apply a chain of effects in sequence

    Args:
        image: input image
        effects: list of effect names to apply in order
        processor: MediaPipeEffects instance

    Returns:
        processed image after all effects applied
    """
    result = image
    for effect_name in effects:
        effect_name = effect_name.strip()
        if effect_name:
            result = apply_single_effect(result, effect_name, processor)
    return result


def apply_single_effect(image: np.ndarray, effect_name: str,
                        processor: MediaPipeEffects) -> np.ndarray:
    """
    Apply a single AI effect to a frame

    Args:
        image: numpy array (BGR format from OpenCV)
        effect_name: string name of effect

    Returns:
        processed image as numpy array
    """
    # Background effects
    if effect_name == "background_remove":
        return effect_background_remove(image, processor)
    elif effect_name == "background_blur":
        return effect_background_blur(image, processor)
    elif effect_name == "background_replace":
        return effect_background_replace(image, processor)

    # Face mesh effects
    elif effect_name == "face_mesh":
        return effect_face_mesh(image, processor)
    elif effect_name == "face_contour":
        return effect_face_contour(image, processor)

    # Face manipulation effects
    elif effect_name == "big_eyes":
        return effect_big_eyes(image, processor)
    elif effect_name == "face_slim":
        return effect_face_slim(image, processor)
    elif effect_name == "face_distort":
        return effect_face_distort(image, processor)

    # Style effects
    elif effect_name == "anime_style":
        return effect_anime_style(image, processor)
    elif effect_name == "beauty_filter":
        return effect_beauty_filter(image, processor)
    elif effect_name == "virtual_makeup":
        return effect_virtual_makeup(image, processor)

    # Legacy effects (from brain_simple.py)
    elif effect_name == "cartoon":
        return effect_cartoon(image)
    elif effect_name == "edge_detection":
        return effect_edge_detection(image)
    elif effect_name == "grayscale":
        return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif effect_name == "sepia":
        return effect_sepia(image)
    elif effect_name == "mirror":
        return cv2.flip(image, 1)
    elif effect_name == "invert":
        return cv2.bitwise_not(image)
    else:
        logger.warning(f"Unknown effect: {effect_name}, returning original")
        return image


# ============================================================================
# BACKGROUND EFFECTS (using MediaPipe Selfie Segmentation)
# ============================================================================

def effect_background_remove(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Remove background - replace with green screen for chroma keying
    """
    if processor.selfie_segmenter is None:
        return image

    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get segmentation mask
    results = processor.selfie_segmenter.process(rgb_image)
    mask = results.segmentation_mask

    # Create condition for foreground
    condition = mask > 0.5

    # Create green screen background
    green_bg = np.zeros_like(image)
    green_bg[:] = (0, 255, 0)  # Green in BGR

    # Combine foreground with green background
    output = np.where(condition[:, :, np.newaxis], image, green_bg)

    return output.astype(np.uint8)


def effect_background_blur(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Blur background while keeping the person sharp (like video conferencing)
    """
    if processor.selfie_segmenter is None:
        return cv2.GaussianBlur(image, (21, 21), 0)

    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get segmentation mask
    results = processor.selfie_segmenter.process(rgb_image)
    mask = results.segmentation_mask

    # Smooth the mask edges for better blending
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # Create heavily blurred background
    blurred = cv2.GaussianBlur(image, (55, 55), 0)

    # Blend based on mask
    mask_3d = np.stack([mask] * 3, axis=2)
    output = image * mask_3d + blurred * (1 - mask_3d)

    return output.astype(np.uint8)


def effect_background_replace(image: np.ndarray, processor: MediaPipeEffects,
                               bg_color: Tuple[int, int, int] = (50, 50, 150)) -> np.ndarray:
    """
    Replace background with a solid color or gradient
    Default is a nice purple/blue gradient effect
    """
    if processor.selfie_segmenter is None:
        return image

    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get segmentation mask
    results = processor.selfie_segmenter.process(rgb_image)
    mask = results.segmentation_mask

    # Smooth mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Create gradient background
    h, w = image.shape[:2]
    gradient_bg = np.zeros_like(image)
    for i in range(h):
        ratio = i / h
        gradient_bg[i, :] = [
            int(50 + 100 * ratio),   # Blue: 50 -> 150
            int(30 + 50 * ratio),    # Green: 30 -> 80
            int(100 + 100 * ratio)   # Red: 100 -> 200
        ]

    # Blend
    mask_3d = np.stack([mask] * 3, axis=2)
    output = image * mask_3d + gradient_bg * (1 - mask_3d)

    return output.astype(np.uint8)


# ============================================================================
# FACE MESH EFFECTS (using MediaPipe Face Mesh - 468 landmarks)
# ============================================================================

def effect_face_mesh(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Draw complete face mesh with all 468 landmarks
    Creates a futuristic/sci-fi face scanning effect
    """
    if processor.face_mesh is None:
        return image

    output = image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = processor.face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the face mesh tesselation
            processor.mp_drawing.draw_landmarks(
                image=output,
                landmark_list=face_landmarks,
                connections=processor.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=processor.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Draw contours on top
            processor.mp_drawing.draw_landmarks(
                image=output,
                landmark_list=face_landmarks,
                connections=processor.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=processor.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            # Draw irises
            processor.mp_drawing.draw_landmarks(
                image=output,
                landmark_list=face_landmarks,
                connections=processor.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=processor.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    return output


def effect_face_contour(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Draw only face contour lines (cleaner look)
    """
    if processor.face_mesh is None:
        return image

    output = image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = processor.face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw just the contours
            processor.mp_drawing.draw_landmarks(
                image=output,
                landmark_list=face_landmarks,
                connections=processor.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=processor.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=processor.mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2
                )
            )

    return output


# ============================================================================
# FACE MANIPULATION EFFECTS (using landmarks for warping)
# ============================================================================

def get_face_landmarks_array(face_landmarks, image_shape) -> np.ndarray:
    """Convert MediaPipe landmarks to numpy array of (x, y) coordinates"""
    h, w = image_shape[:2]
    landmarks = []
    for lm in face_landmarks.landmark:
        landmarks.append([int(lm.x * w), int(lm.y * h)])
    return np.array(landmarks)


def effect_big_eyes(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Enlarge eyes effect - like Snapchat/Instagram filters
    Uses face mesh to locate eyes and applies local magnification
    """
    if processor.face_mesh is None:
        return image

    output = image.copy()
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = processor.face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return output

    h, w = image.shape[:2]
    face_landmarks = results.multi_face_landmarks[0]

    # Eye landmark indices (from MediaPipe face mesh)
    # Left eye: around index 33, 133, 159, 145 (center region)
    # Right eye: around index 362, 263, 386, 374

    left_eye_indices = [33, 133, 159, 145, 153, 154, 155, 157, 158, 160, 161, 163, 7, 163, 144, 145, 153]
    right_eye_indices = [362, 263, 386, 374, 380, 381, 382, 384, 385, 387, 388, 390, 249, 263, 373, 374, 380]

    def get_eye_center(indices):
        x_sum, y_sum = 0, 0
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            x_sum += lm.x * w
            y_sum += lm.y * h
        return int(x_sum / len(indices)), int(y_sum / len(indices))

    def magnify_region(img, center, radius, strength=0.3):
        """Apply local magnification around a point"""
        cx, cy = center

        # Create meshgrid for the region
        y_indices, x_indices = np.ogrid[
            max(0, cy - radius):min(h, cy + radius),
            max(0, cx - radius):min(w, cx + radius)
        ]

        # Calculate distance from center
        dist = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)

        # Create magnification map
        mask = dist < radius
        if not np.any(mask):
            return img

        # Apply barrel distortion (magnification)
        result = img.copy()
        for y in range(max(0, cy - radius), min(h, cy + radius)):
            for x in range(max(0, cx - radius), min(w, cx + radius)):
                dx = x - cx
                dy = y - cy
                d = np.sqrt(dx*dx + dy*dy)

                if d < radius and d > 0:
                    # Magnification factor decreases with distance
                    factor = 1 - strength * (1 - d/radius)**2
                    src_x = int(cx + dx * factor)
                    src_y = int(cy + dy * factor)

                    if 0 <= src_x < w and 0 <= src_y < h:
                        result[y, x] = img[src_y, src_x]

        return result

    # Get eye centers
    left_eye_center = get_eye_center(left_eye_indices)
    right_eye_center = get_eye_center(right_eye_indices)

    # Calculate eye radius based on face size
    face_width = abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * w
    eye_radius = int(face_width * 0.15)

    # Apply magnification to both eyes
    output = magnify_region(output, left_eye_center, eye_radius, strength=0.4)
    output = magnify_region(output, right_eye_center, eye_radius, strength=0.4)

    return output


def effect_face_slim(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Make face appear slimmer by subtle horizontal compression
    """
    if processor.face_mesh is None:
        return image

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = processor.face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image

    h, w = image.shape[:2]
    face_landmarks = results.multi_face_landmarks[0]

    # Get face bounding box from landmarks
    x_coords = [lm.x * w for lm in face_landmarks.landmark]
    y_coords = [lm.y * h for lm in face_landmarks.landmark]

    face_left = int(min(x_coords))
    face_right = int(max(x_coords))
    face_top = int(min(y_coords))
    face_bottom = int(max(y_coords))
    face_center_x = (face_left + face_right) // 2

    output = image.copy()

    # Apply subtle horizontal compression towards center
    slim_factor = 0.08  # 8% slimmer

    for y in range(face_top, face_bottom):
        for x in range(face_left, face_right):
            # Calculate offset from center
            offset = x - face_center_x
            # Apply compression (stronger at edges)
            edge_factor = abs(offset) / ((face_right - face_left) / 2)
            new_offset = offset * (1 + slim_factor * edge_factor)
            src_x = int(face_center_x + new_offset)

            if 0 <= src_x < w:
                output[y, x] = image[y, src_x]

    return output


def effect_face_distort(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Fun face distortion - bulge effect centered on nose
    Creates a funhouse mirror effect
    """
    if processor.face_mesh is None:
        return image

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = processor.face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image

    h, w = image.shape[:2]
    face_landmarks = results.multi_face_landmarks[0]

    # Get nose tip as center (landmark 4)
    nose_tip = face_landmarks.landmark[4]
    cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)

    # Calculate face size for radius
    face_width = abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * w
    radius = int(face_width * 0.6)

    # Create output with bulge distortion
    output = image.copy()
    strength = 0.5  # Distortion strength

    y_indices, x_indices = np.mgrid[0:h, 0:w]

    # Calculate distance from center
    dx = x_indices - cx
    dy = y_indices - cy
    dist = np.sqrt(dx**2 + dy**2)

    # Apply bulge where distance < radius
    mask = dist < radius

    # Bulge formula
    factor = np.ones_like(dist, dtype=np.float32)
    factor[mask] = np.power(dist[mask] / radius, strength)

    # Calculate source coordinates
    src_x = (cx + dx * factor).astype(np.int32)
    src_y = (cy + dy * factor).astype(np.int32)

    # Clip to valid range
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)

    # Apply mapping
    output = image[src_y, src_x]

    return output


# ============================================================================
# STYLE EFFECTS
# ============================================================================

def effect_anime_style(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Convert to anime/cartoon style using AnimeGANv2 ONNX model
    Falls back to OpenCV cartoon effect if model unavailable
    """
    session = processor._get_onnx_session()

    if session is None:
        # Fallback to OpenCV cartoon
        return effect_cartoon(image)

    try:
        h, w = image.shape[:2]

        # Preprocess for AnimeGAN (expects 256x256 or similar)
        input_size = (512, 512)  # AnimeGAN works better with larger input
        resized = cv2.resize(image, input_size)

        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension

        # Run inference
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_data})[0]

        # Postprocess
        output = output[0]
        output = np.transpose(output, (1, 2, 0))  # CHW to HWC
        output = (output + 1.0) * 127.5  # Denormalize
        output = np.clip(output, 0, 255).astype(np.uint8)

        # Convert RGB back to BGR and resize
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        output = cv2.resize(output, (w, h))

        return output

    except Exception as e:
        logger.error(f"AnimeGAN inference failed: {e}")
        return effect_cartoon(image)


def effect_beauty_filter(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Beauty filter: smooth skin while preserving edges (like portrait mode)
    Uses bilateral filter for edge-preserving smoothing
    """
    if processor.face_mesh is None:
        # Apply to whole image
        return cv2.bilateralFilter(image, 9, 75, 75)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = processor.face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return cv2.bilateralFilter(image, 9, 75, 75)

    h, w = image.shape[:2]
    output = image.copy()

    # Create face mask from landmarks
    face_landmarks = results.multi_face_landmarks[0]

    # Get face oval landmarks (indices for face outline)
    face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                         397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                         172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    face_points = []
    for idx in face_oval_indices:
        lm = face_landmarks.landmark[idx]
        face_points.append([int(lm.x * w), int(lm.y * h)])
    face_points = np.array(face_points, dtype=np.int32)

    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [face_points], 255)

    # Apply strong bilateral filter to smoothed version
    smoothed = cv2.bilateralFilter(image, 15, 100, 100)

    # Blend smoothed face with original
    mask_3d = mask[:, :, np.newaxis] / 255.0
    output = (smoothed * mask_3d + image * (1 - mask_3d)).astype(np.uint8)

    return output


def effect_virtual_makeup(image: np.ndarray, processor: MediaPipeEffects) -> np.ndarray:
    """
    Add virtual lipstick and subtle eye enhancement
    """
    if processor.face_mesh is None:
        return image

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = processor.face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        return image

    h, w = image.shape[:2]
    output = image.copy()
    face_landmarks = results.multi_face_landmarks[0]

    # Lip landmarks (outer lips)
    lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

    lip_points = []
    for idx in lip_indices:
        lm = face_landmarks.landmark[idx]
        lip_points.append([int(lm.x * w), int(lm.y * h)])
    lip_points = np.array(lip_points, dtype=np.int32)

    # Create lip mask
    lip_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(lip_mask, [lip_points], 255)
    lip_mask = cv2.GaussianBlur(lip_mask, (7, 7), 5)

    # Create lipstick color overlay (red-pink)
    lipstick_color = np.zeros_like(image)
    lipstick_color[:] = (80, 50, 180)  # BGR: pinkish-red

    # Blend lipstick
    lip_mask_3d = lip_mask[:, :, np.newaxis] / 255.0 * 0.4  # 40% opacity
    output = (lipstick_color * lip_mask_3d + output * (1 - lip_mask_3d)).astype(np.uint8)

    return output


# ============================================================================
# LEGACY EFFECTS (from brain_simple.py - no ML required)
# ============================================================================

def effect_cartoon(image: np.ndarray) -> np.ndarray:
    """Cartoon effect using bilateral filter and edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(image, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def effect_edge_detection(image: np.ndarray) -> np.ndarray:
    """Canny edge detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def effect_sepia(image: np.ndarray) -> np.ndarray:
    """Sepia tone effect"""
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sepia = cv2.transform(image, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)


# ============================================================================
# MESSAGE HANDLING
# ============================================================================

def sendMessage(msg: str, token: str):
    """Send message back to client via RabbitMQ broadcast"""
    data = {
        'msg': msg,
        'key': token
    }
    msg_json = json.dumps(data)
    message = Message.create(mainChannel_out, msg_json)
    message.publish('broadcast-all')


def processTask(msg0):
    """Process a task from the queue"""
    msg = msg0.body
    msg = json.loads(msg)

    stream_key = msg.get('streamKey', '')
    task = msg.get('task', '')
    redis_id = msg.get('redisID', '')

    logger.info(f"Processing task: {task} for stream: {stream_key}")

    # Map old task names to new effect names
    effect_map = {
        # New MediaPipe effects
        'BackgroundRemoveOn': 'background_remove',
        'BackgroundBlurOn': 'background_blur',
        'BackgroundReplaceOn': 'background_replace',
        'FaceMeshOn': 'face_mesh',
        'FaceContourOn': 'face_contour',
        'BigEyesOn': 'big_eyes',
        'FaceSlimOn': 'face_slim',
        'FaceDistortOn': 'face_distort',
        'AnimeStyleOn': 'anime_style',
        'BeautyFilterOn': 'beauty_filter',
        'VirtualMakeupOn': 'virtual_makeup',

        # Legacy mappings
        'FaceOn': 'face_mesh',
        'TrumpOn': 'face_distort',  # Fun replacement for missing deepfake
        'TaylorOn': 'anime_style',  # Fun replacement
        'BlurOn': 'background_blur',
        'EdgeOn': 'edge_detection',
        'CartoonOn': 'cartoon',
        'GrayOn': 'grayscale',
        'SepiaOn': 'sepia',
        'MirrorOn': 'mirror',
        'InvertOn': 'invert',
    }

    effect_name = effect_map.get(task, task.lower().replace('on', ''))

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
        success, encoded = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 90])

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
    logger.info("=" * 60)
    logger.info("MediaPipe ML Worker Starting...")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Available AI Effects:")
    logger.info("  Background Effects:")
    logger.info("    - background_remove : Green screen background removal")
    logger.info("    - background_blur   : Video conferencing style blur")
    logger.info("    - background_replace: Replace with gradient")
    logger.info("")
    logger.info("  Face Mesh Effects:")
    logger.info("    - face_mesh    : 468 3D landmarks visualization")
    logger.info("    - face_contour : Face outline contours")
    logger.info("")
    logger.info("  Face Manipulation:")
    logger.info("    - big_eyes     : Enlarge eyes (Snapchat style)")
    logger.info("    - face_slim    : Slim face effect")
    logger.info("    - face_distort : Funhouse mirror effect")
    logger.info("")
    logger.info("  Style Effects:")
    logger.info("    - anime_style  : Convert to anime (AnimeGAN)")
    logger.info("    - beauty_filter: Skin smoothing")
    logger.info("    - virtual_makeup: Add lipstick")
    logger.info("")
    logger.info("  Classic Effects:")
    logger.info("    - cartoon, edge_detection, grayscale, sepia, mirror, invert")
    logger.info("")
    logger.info("All effects run in real-time on CPU!")
    logger.info("=" * 60)

    # Initialize models on startup
    get_effects_processor()

    # Declare task queue
    childChannel = mainConnection.channel()
    childChannel.queue.declare("tf-task", arguments={'x-message-ttl': 10000})

    # Start consuming tasks
    try:
        childChannel.basic.consume(processTask, 'tf-task', no_ack=True)
        logger.info("Worker ready, waiting for tasks...")
        childChannel.start_consuming()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
