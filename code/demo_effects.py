#!/usr/bin/env python3
"""
Standalone Demo for Ooblex AI Effects

Run this script to test the MediaPipe-based AI effects with your webcam.
Shows original and processed video side-by-side.

Usage:
    python demo_effects.py                    # Use default effect (face_mesh)
    python demo_effects.py background_blur    # Use specific effect
    python demo_effects.py --list             # List all available effects

Controls:
    - Press 1-9 or 0 to switch effects
    - Press 'n' for next effect
    - Press 'p' for previous effect
    - Press 's' to save a screenshot
    - Press 'q' to quit

Requirements:
    pip install mediapipe opencv-python numpy
"""

import sys
import cv2
import numpy as np
import time
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import effects from brain_mediapipe (but we'll also include standalone versions)
try:
    from brain_mediapipe import (
        MediaPipeEffects,
        effect_background_remove,
        effect_background_blur,
        effect_background_replace,
        effect_face_mesh,
        effect_face_contour,
        effect_big_eyes,
        effect_face_slim,
        effect_face_distort,
        effect_anime_style,
        effect_beauty_filter,
        effect_virtual_makeup,
        effect_cartoon,
        effect_edge_detection,
        effect_sepia,
    )
    USING_BRAIN_MEDIAPIPE = True
except ImportError:
    USING_BRAIN_MEDIAPIPE = False
    print("Note: Running in standalone mode (brain_mediapipe not found)")

import mediapipe as mp


class StandaloneEffects:
    """Standalone effects processor when brain_mediapipe is not available"""

    def __init__(self):
        # Initialize Selfie Segmentation
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.selfie_segmenter = self.mp_selfie.SelfieSegmentation(model_selection=1)

        # Initialize Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Drawing utilities
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        print("MediaPipe models loaded successfully")


# Available effects with descriptions
EFFECTS = [
    ("background_blur", "Blur background (video conference style)"),
    ("background_remove", "Green screen background removal"),
    ("background_replace", "Replace background with gradient"),
    ("face_mesh", "468 3D face landmarks visualization"),
    ("face_contour", "Face contour lines only"),
    ("big_eyes", "Enlarge eyes (Snapchat style)"),
    ("face_slim", "Slim face effect"),
    ("face_distort", "Funhouse mirror bulge effect"),
    ("anime_style", "Anime/cartoon style (AnimeGAN)"),
    ("beauty_filter", "Skin smoothing beauty filter"),
    ("virtual_makeup", "Virtual lipstick"),
    ("cartoon", "OpenCV cartoon effect"),
    ("edge_detection", "Canny edge detection"),
    ("sepia", "Sepia tone filter"),
    ("grayscale", "Black and white"),
    ("mirror", "Horizontal flip"),
    ("invert", "Invert colors"),
]

# Preset effect chains for compelling demos
EFFECT_CHAINS = [
    ("background_blur|big_eyes|beauty_filter", "Video Call Pro (blur + big eyes + smooth skin)"),
    ("background_remove|face_mesh", "AR Developer (green screen + face tracking)"),
    ("background_blur|anime_style", "Anime Meeting (blur + anime style)"),
    ("beauty_filter|virtual_makeup|face_contour", "Virtual Glam (beauty + makeup + contour)"),
    ("background_replace|big_eyes|face_slim", "Portrait Mode+ (gradient bg + enhance)"),
    ("face_distort|edge_detection", "Funhouse X-Ray (distort + edges)"),
    ("cartoon|sepia", "Vintage Toon (cartoon + sepia tone)"),
    ("background_blur|face_mesh|invert", "Cyber Vision (blur + mesh + invert)"),
]


def apply_effect_standalone(image, effect_name, processor):
    """Apply effect using standalone processor - supports chained effects"""

    # Handle chained effects
    if '|' in effect_name or '+' in effect_name:
        effects = effect_name.replace('+', '|').split('|')
        result = image
        for eff in effects:
            result = apply_effect_standalone(result, eff.strip(), processor)
        return result

    if effect_name == "background_blur":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = processor.selfie_segmenter.process(rgb)
        mask = results.segmentation_mask
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        blurred = cv2.GaussianBlur(image, (55, 55), 0)
        mask_3d = np.stack([mask] * 3, axis=2)
        return (image * mask_3d + blurred * (1 - mask_3d)).astype(np.uint8)

    elif effect_name == "background_remove":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = processor.selfie_segmenter.process(rgb)
        mask = results.segmentation_mask
        condition = mask > 0.5
        green_bg = np.zeros_like(image)
        green_bg[:] = (0, 255, 0)
        return np.where(condition[:, :, np.newaxis], image, green_bg).astype(np.uint8)

    elif effect_name == "background_replace":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = processor.selfie_segmenter.process(rgb)
        mask = results.segmentation_mask
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        h, w = image.shape[:2]
        gradient_bg = np.zeros_like(image)
        for i in range(h):
            ratio = i / h
            gradient_bg[i, :] = [int(50 + 100 * ratio), int(30 + 50 * ratio), int(100 + 100 * ratio)]
        mask_3d = np.stack([mask] * 3, axis=2)
        return (image * mask_3d + gradient_bg * (1 - mask_3d)).astype(np.uint8)

    elif effect_name == "face_mesh":
        output = image.copy()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = processor.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                processor.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=processor.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=processor.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                processor.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=processor.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=processor.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        return output

    elif effect_name == "face_contour":
        output = image.copy()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = processor.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                processor.mp_drawing.draw_landmarks(
                    image=output,
                    landmark_list=face_landmarks,
                    connections=processor.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=processor.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=processor.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
        return output

    elif effect_name == "face_distort":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = processor.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return image
        h, w = image.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        nose_tip = face_landmarks.landmark[4]
        cx, cy = int(nose_tip.x * w), int(nose_tip.y * h)
        face_width = abs(face_landmarks.landmark[234].x - face_landmarks.landmark[454].x) * w
        radius = int(face_width * 0.6)
        y_indices, x_indices = np.mgrid[0:h, 0:w]
        dx = x_indices - cx
        dy = y_indices - cy
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist < radius
        factor = np.ones_like(dist, dtype=np.float32)
        factor[mask] = np.power(dist[mask] / radius, 0.5)
        src_x = np.clip((cx + dx * factor).astype(np.int32), 0, w - 1)
        src_y = np.clip((cy + dy * factor).astype(np.int32), 0, h - 1)
        return image[src_y, src_x]

    elif effect_name == "cartoon":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 300, 300)
        return cv2.bitwise_and(color, color, mask=edges)

    elif effect_name == "edge_detection":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif effect_name == "sepia":
        kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
        return np.clip(cv2.transform(image, kernel), 0, 255).astype(np.uint8)

    elif effect_name == "grayscale":
        return cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    elif effect_name == "mirror":
        return cv2.flip(image, 1)

    elif effect_name == "invert":
        return cv2.bitwise_not(image)

    elif effect_name == "beauty_filter":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = processor.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return cv2.bilateralFilter(image, 9, 75, 75)
        h, w = image.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        face_points = np.array([[int(face_landmarks.landmark[idx].x * w),
                                 int(face_landmarks.landmark[idx].y * h)]
                                for idx in face_oval_indices], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [face_points], 255)
        smoothed = cv2.bilateralFilter(image, 15, 100, 100)
        mask_3d = mask[:, :, np.newaxis] / 255.0
        return (smoothed * mask_3d + image * (1 - mask_3d)).astype(np.uint8)

    else:
        # For effects not implemented in standalone, return original
        return image


def run_demo(initial_effect="face_mesh"):
    """Run the webcam demo with single effects and effect chains"""

    print("\n" + "=" * 60)
    print("  Ooblex AI Effects Demo")
    print("=" * 60)

    # Initialize processor
    if USING_BRAIN_MEDIAPIPE:
        print("Using brain_mediapipe module")
        processor = MediaPipeEffects()
    else:
        print("Using standalone effects")
        processor = StandaloneEffects()

    # Mode: single effects or chains
    chain_mode = False
    chain_idx = 0

    # Find initial effect index
    effect_names = [e[0] for e in EFFECTS]

    # Check if initial_effect is a chain
    if '|' in initial_effect or '+' in initial_effect:
        chain_mode = True
        # Try to find it in preset chains
        chain_names = [e[0] for e in EFFECT_CHAINS]
        if initial_effect in chain_names:
            chain_idx = chain_names.index(initial_effect)
        else:
            # Add it as a custom chain
            EFFECT_CHAINS.append((initial_effect, f"Custom: {initial_effect}"))
            chain_idx = len(EFFECT_CHAINS) - 1
        current_idx = 0
    else:
        try:
            current_idx = effect_names.index(initial_effect)
        except ValueError:
            print(f"Warning: Unknown effect '{initial_effect}', using 'face_mesh'")
            current_idx = effect_names.index("face_mesh")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Make sure a webcam is connected and accessible")
        return

    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nWebcam: {width}x{height}")

    print("\nControls:")
    print("  1-9, 0 : Select effect (0=10th effect)")
    print("  n/p    : Next/Previous effect")
    print("  c      : Toggle chain mode (preset effect combos)")
    print("  s      : Save screenshot")
    print("  q      : Quit")
    if chain_mode:
        print("\nStarting with effect chain:", EFFECT_CHAINS[chain_idx][0])
    else:
        print("\nStarting with effect:", EFFECTS[current_idx][0])
    print("-" * 60)

    # FPS tracking
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Get current effect (single or chain)
        if chain_mode:
            effect_name = EFFECT_CHAINS[chain_idx][0]
            effect_desc = EFFECT_CHAINS[chain_idx][1]
        else:
            effect_name = EFFECTS[current_idx][0]
            effect_desc = EFFECTS[current_idx][1]

        start_time = time.time()

        if USING_BRAIN_MEDIAPIPE:
            # Use the full brain_mediapipe implementation with chain support
            from brain_mediapipe import apply_effect as apply_effect_mp
            processed = apply_effect_mp(frame, effect_name)
        else:
            processed = apply_effect_standalone(frame, effect_name, processor)

        process_time = (time.time() - start_time) * 1000  # ms

        # Update FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_frame_count
            fps_frame_count = 0
            fps_start_time = time.time()

        # Create side-by-side display
        # Add labels
        original_labeled = frame.copy()
        processed_labeled = processed.copy()

        # Add effect name and FPS overlay
        cv2.putText(original_labeled, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show effect name (truncate if too long for chains)
        display_name = effect_name if len(effect_name) <= 35 else effect_name[:32] + "..."
        mode_label = "[CHAIN]" if chain_mode else "[SINGLE]"
        cv2.putText(processed_labeled, f"{mode_label} {display_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(processed_labeled, f"FPS: {fps} | {process_time:.1f}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show description for chains
        if chain_mode:
            cv2.putText(processed_labeled, effect_desc, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Combine side by side
        combined = np.hstack([original_labeled, processed_labeled])

        # Resize if too large
        max_width = 1600
        if combined.shape[1] > max_width:
            scale = max_width / combined.shape[1]
            combined = cv2.resize(combined, (max_width, int(combined.shape[0] * scale)))

        cv2.imshow("Ooblex AI Effects Demo (Press 'q' to quit)", combined)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            # Toggle chain mode
            chain_mode = not chain_mode
            if chain_mode:
                print(f"\n[CHAIN MODE] {EFFECT_CHAINS[chain_idx][0]}")
                print(f"  -> {EFFECT_CHAINS[chain_idx][1]}")
            else:
                print(f"\n[SINGLE MODE] {EFFECTS[current_idx][0]}")
                print(f"  -> {EFFECTS[current_idx][1]}")
        elif key == ord('n'):
            if chain_mode:
                chain_idx = (chain_idx + 1) % len(EFFECT_CHAINS)
                print(f"Chain: {EFFECT_CHAINS[chain_idx][0]}")
                print(f"  -> {EFFECT_CHAINS[chain_idx][1]}")
            else:
                current_idx = (current_idx + 1) % len(EFFECTS)
                print(f"Effect: {EFFECTS[current_idx][0]} - {EFFECTS[current_idx][1]}")
        elif key == ord('p'):
            if chain_mode:
                chain_idx = (chain_idx - 1) % len(EFFECT_CHAINS)
                print(f"Chain: {EFFECT_CHAINS[chain_idx][0]}")
                print(f"  -> {EFFECT_CHAINS[chain_idx][1]}")
            else:
                current_idx = (current_idx - 1) % len(EFFECTS)
                print(f"Effect: {EFFECTS[current_idx][0]} - {EFFECTS[current_idx][1]}")
        elif key == ord('s'):
            safe_name = effect_name.replace('|', '_').replace('+', '_')
            filename = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_name}.png"
            cv2.imwrite(filename, combined)
            print(f"Saved: {filename}")
        elif key >= ord('1') and key <= ord('9'):
            idx = key - ord('1')
            if chain_mode:
                if idx < len(EFFECT_CHAINS):
                    chain_idx = idx
                    print(f"Chain: {EFFECT_CHAINS[chain_idx][0]}")
                    print(f"  -> {EFFECT_CHAINS[chain_idx][1]}")
            else:
                if idx < len(EFFECTS):
                    current_idx = idx
                    print(f"Effect: {EFFECTS[current_idx][0]} - {EFFECTS[current_idx][1]}")
        elif key == ord('0'):
            if chain_mode:
                if 9 < len(EFFECT_CHAINS):
                    chain_idx = 9
                    print(f"Chain: {EFFECT_CHAINS[chain_idx][0]}")
            else:
                if 9 < len(EFFECTS):
                    current_idx = 9
                    print(f"Effect: {EFFECTS[current_idx][0]} - {EFFECTS[current_idx][1]}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDemo ended")


def list_effects():
    """Print all available effects"""
    print("\n" + "=" * 70)
    print("  SINGLE EFFECTS (press 1-9, 0 to select)")
    print("=" * 70)
    for i, (name, desc) in enumerate(EFFECTS):
        key = str((i + 1) % 10) if i < 10 else "-"
        print(f"  [{key}] {name:20s} : {desc}")

    print("\n" + "=" * 70)
    print("  EFFECT CHAINS (press 'c' to toggle chain mode, then 1-9)")
    print("=" * 70)
    for i, (chain, desc) in enumerate(EFFECT_CHAINS):
        key = str((i + 1) % 10) if i < 10 else "-"
        print(f"  [{key}] {desc}")
        print(f"      -> {chain}")

    print("\n" + "-" * 70)
    print("Background effects use MediaPipe Selfie Segmentation (454KB, <1ms)")
    print("Face effects use MediaPipe Face Mesh (468 3D landmarks, ~2ms)")
    print("Anime style uses AnimeGANv2 ONNX model (~8MB, downloads automatically)")
    print("\nCustom chains: separate effects with | or +")
    print("  Example: python demo_effects.py 'background_blur|big_eyes|sepia'")
    print("-" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list" or sys.argv[1] == "-l":
            list_effects()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(__doc__)
            list_effects()
        else:
            run_demo(sys.argv[1])
    else:
        list_effects()
        print("\nStarting demo with 'face_mesh' effect...\n")
        run_demo("face_mesh")
