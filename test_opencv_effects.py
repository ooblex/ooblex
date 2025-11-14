#!/usr/bin/env python3
"""
Test that OpenCV effects actually transform images
"""

import os
import sys

sys.path.append("services/ml-worker")

import cv2
import numpy as np
from opencv_effects import OpenCVProcessor


def create_test_image():
    """Create a test image with a face-like pattern"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255

    # Draw a "face"
    # Head
    cv2.circle(img, (320, 200), 80, (200, 180, 170), -1)
    # Eyes
    cv2.circle(img, (290, 180), 15, (50, 50, 50), -1)
    cv2.circle(img, (350, 180), 15, (50, 50, 50), -1)
    # Mouth
    cv2.ellipse(img, (320, 220), (40, 20), 0, 0, 180, (50, 50, 50), 2)

    # Add some background
    cv2.rectangle(img, (100, 300), (540, 450), (100, 200, 100), -1)
    cv2.putText(
        img, "Test Image", (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
    )

    return img


def images_are_different(img1, img2, threshold=0.01):
    """Check if two images are significantly different"""
    diff = cv2.absdiff(img1, img2)
    avg_diff = np.mean(diff)
    return avg_diff > threshold


def test_effects():
    """Test each effect to ensure it actually transforms the image"""
    processor = OpenCVProcessor()
    test_img = create_test_image()

    effects = [
        "edge_detection",
        "cartoon",
        "blur_background",
        "sepia",
        "grayscale",
        "vintage",
        "pixelate",
        "emboss",
        "sharpen",
        "blur",
    ]

    print("Testing OpenCV Effects")
    print("=" * 50)

    results = []
    for effect in effects:
        try:
            if effect == "blur_background":
                # Special case - needs face detection
                result = processor.blur_background(test_img, blur_strength=25)
            else:
                # Use generic apply_effect
                result = processor.apply_effect(test_img, effect)

            # Check if image was actually transformed
            is_different = images_are_different(test_img, result)

            # Additional checks for specific effects
            if effect == "grayscale" and len(result.shape) == 3:
                # Should be grayscale (all channels same)
                is_grayscale = np.allclose(
                    result[:, :, 0], result[:, :, 1]
                ) and np.allclose(result[:, :, 1], result[:, :, 2])
                status = "✓" if is_grayscale else "✗"
                extra = (
                    " (converted to grayscale)"
                    if is_grayscale
                    else " (FAILED to convert)"
                )
            elif effect == "edge_detection":
                # Should have mostly black with white edges
                black_pixels = np.sum(result < 50) / result.size
                status = "✓" if black_pixels > 0.5 else "✗"
                extra = f" ({black_pixels:.1%} black pixels)"
            else:
                status = "✓" if is_different else "✗"
                extra = " (image transformed)" if is_different else " (NO CHANGE)"

            results.append((effect, status == "✓"))
            print(f"{status} {effect:<20} {extra}")

            # Save sample outputs
            if is_different:
                cv2.imwrite(f"test_output_{effect}.jpg", result)

        except Exception as e:
            print(f"✗ {effect:<20} (ERROR: {e})")
            results.append((effect, False))

    # Test face detection
    print("\nTesting Face Detection:")
    faces = processor.detect_faces(test_img)
    if len(faces) > 0:
        print(f"✓ Detected {len(faces)} face(s)")
        # Draw rectangles
        result = test_img.copy()
        for x, y, w, h in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.imwrite("test_output_face_detection.jpg", result)
    else:
        print("✗ No faces detected (expected with simple test image)")

    # Summary
    print("\n" + "=" * 50)
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Summary: {passed}/{total} effects working")

    if passed > total * 0.7:  # If 70% work
        print("✅ OpenCV effects are working properly!")
        return True
    else:
        print("❌ Many effects are not working")
        return False


if __name__ == "__main__":
    # Save original and test image
    test_img = create_test_image()
    cv2.imwrite("test_input.jpg", test_img)

    success = test_effects()

    print("\nTest images saved:")
    print("  Input: test_input.jpg")
    print("  Outputs: test_output_*.jpg")

    sys.exit(0 if success else 1)
