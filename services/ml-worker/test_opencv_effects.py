#!/usr/bin/env python3
"""
Test OpenCV effects directly
"""
import cv2
import numpy as np
import asyncio
from ml_worker_opencv import OpenCVProcessor
import os
import sys


async def test_effects():
    """Test all OpenCV effects"""
    processor = OpenCVProcessor()
    
    # Create a test image with some features
    print("Creating test image...")
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles
    cv2.rectangle(test_image, (50, 50), (200, 200), (0, 0, 255), -1)  # Red
    cv2.rectangle(test_image, (250, 50), (400, 200), (0, 255, 0), -1)  # Green
    cv2.rectangle(test_image, (450, 50), (600, 200), (255, 0, 0), -1)  # Blue
    
    # Add some text
    cv2.putText(test_image, "OpenCV Test", (200, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    
    # Add circles
    cv2.circle(test_image, (150, 400), 50, (255, 255, 0), -1)  # Yellow
    cv2.circle(test_image, (320, 400), 50, (255, 0, 255), -1)  # Magenta
    cv2.circle(test_image, (490, 400), 50, (0, 255, 255), -1)  # Cyan
    
    # Save original
    cv2.imwrite("test_original.jpg", test_image)
    print("Saved test_original.jpg")
    
    # Test each effect
    effects = [
        ("grayscale", processor.convert_grayscale, {}),
        ("sepia", processor.apply_sepia, {}),
        ("edge_detection", processor.edge_detection, {"low_threshold": 50, "high_threshold": 150}),
        ("cartoon", processor.cartoon_effect, {}),
        ("vintage", processor.apply_vintage, {}),
        ("blur", processor.apply_blur, {"kernel_size": 21}),
        ("sharpen", processor.apply_sharpen, {}),
        ("pixelate", processor.pixelate, {"pixel_size": 10}),
        ("emboss", processor.apply_emboss, {}),
    ]
    
    for effect_name, effect_func, params in effects:
        print(f"\nApplying {effect_name} effect...")
        try:
            if params:
                result = await effect_func(test_image, **params)
            else:
                result = await effect_func(test_image)
            
            output_file = f"test_{effect_name}.jpg"
            cv2.imwrite(output_file, result)
            print(f"Saved {output_file}")
        except Exception as e:
            print(f"Error applying {effect_name}: {e}")
    
    # Test with a real image if available
    real_image_path = "../../assets/jpeg.jpg"
    if os.path.exists(real_image_path):
        print("\n\nTesting with real image...")
        real_image = cv2.imread(real_image_path)
        
        if real_image is not None:
            # Test face detection
            print("Testing face detection...")
            try:
                face_result, face_data = await processor.detect_faces(real_image)
                cv2.imwrite("test_face_detection.jpg", face_result)
                print(f"Saved test_face_detection.jpg - Found {len(face_data)} faces")
            except Exception as e:
                print(f"Face detection error: {e}")
            
            # Test background blur
            print("Testing background blur...")
            try:
                blur_result = await processor.blur_background(real_image)
                cv2.imwrite("test_background_blur.jpg", blur_result)
                print("Saved test_background_blur.jpg")
            except Exception as e:
                print(f"Background blur error: {e}")
            
            # Test cartoon on real image
            print("Testing cartoon effect on real image...")
            try:
                cartoon_result = await processor.cartoon_effect(real_image)
                cv2.imwrite("test_cartoon_real.jpg", cartoon_result)
                print("Saved test_cartoon_real.jpg")
            except Exception as e:
                print(f"Cartoon effect error: {e}")
    else:
        print(f"\nReal image not found at {real_image_path}")
    
    print("\n\nAll tests completed!")


if __name__ == "__main__":
    asyncio.run(test_effects())