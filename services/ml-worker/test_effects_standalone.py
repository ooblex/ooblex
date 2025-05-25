#!/usr/bin/env python3
"""
Standalone test for OpenCV effects - no dependencies required except OpenCV
"""
import cv2
import numpy as np
import os
import sys
import time
import urllib.request


class SimpleEffects:
    """Simplified version of effects for testing"""
    
    def __init__(self):
        # Download cascade if needed
        self.cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(self.cascade_path):
            print("Downloading face cascade...")
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, self.cascade_path)
        
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
    
    def apply_cartoon(self, image):
        """Simple cartoon effect"""
        # Convert to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply median blur
        gray = cv2.medianBlur(gray, 5)
        
        # Detect edges
        edges = cv2.adaptiveThreshold(gray, 255, 
                                     cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 9, 10)
        
        # Convert back to color
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Apply bilateral filter
        smooth = cv2.bilateralFilter(image, 15, 80, 80)
        
        # Combine
        result = cv2.bitwise_and(smooth, edges)
        
        return result
    
    def apply_sepia(self, image):
        """Sepia tone effect"""
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        
        result = cv2.transform(image, kernel)
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    def detect_blur_faces(self, image):
        """Detect faces and blur background"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return cv2.GaussianBlur(image, (21, 21), 0)
        
        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for (x, y, w, h) in faces:
            # Draw on image
            cv2.rectangle(image.copy(), (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Expand region
            expand = int(w * 0.3)
            x1 = max(0, x - expand)
            y1 = max(0, y - expand)
            x2 = min(image.shape[1], x + w + expand)
            y2 = min(image.shape[0], y + h + expand)
            
            # Elliptical mask
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = ((x2 - x1) // 2, (y2 - y1) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Blur mask
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        
        # Create blurred version
        blurred = cv2.GaussianBlur(image, (31, 31), 0)
        
        # Combine
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (image * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
        
        return result


def process_image(image_path, effects):
    """Process a single image with all effects"""
    print(f"\nProcessing {image_path}...")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    # Get base name
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Apply effects
    print("Applying cartoon effect...")
    cartoon = effects.apply_cartoon(image)
    cv2.imwrite(f"{base_name}_cartoon.jpg", cartoon)
    
    print("Applying sepia effect...")
    sepia = effects.apply_sepia(image)
    cv2.imwrite(f"{base_name}_sepia.jpg", sepia)
    
    print("Detecting faces and blurring background...")
    face_blur = effects.detect_blur_faces(image)
    cv2.imwrite(f"{base_name}_face_blur.jpg", face_blur)
    
    print("Applying edge detection...")
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{base_name}_edges.jpg", edges_color)
    
    print("Creating pixelated version...")
    small = cv2.resize(image, (image.shape[1]//10, image.shape[0]//10))
    pixelated = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{base_name}_pixelated.jpg", pixelated)
    
    print(f"Completed processing {image_path}")


def create_test_image():
    """Create a synthetic test image"""
    print("Creating synthetic test image...")
    
    # Create blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Add shapes
    cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 255), -1)
    cv2.circle(img, (320, 150), 80, (0, 255, 0), -1)
    cv2.ellipse(img, (500, 150), (60, 100), 45, 0, 360, (255, 0, 0), -1)
    
    # Add text
    cv2.putText(img, "OpenCV Effects Test", (100, 300), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Add gradient
    for i in range(100):
        cv2.line(img, (0, 350 + i), (640, 350 + i), 
                (i*2, i*2, 255-i*2), 1)
    
    cv2.imwrite("synthetic_test.jpg", img)
    return "synthetic_test.jpg"


def main():
    """Main function"""
    print("OpenCV Effects Test - Standalone Version")
    print("=" * 50)
    
    effects = SimpleEffects()
    
    # Create synthetic test image
    test_path = create_test_image()
    process_image(test_path, effects)
    
    # Process any command line images
    if len(sys.argv) > 1:
        for image_path in sys.argv[1:]:
            if os.path.exists(image_path):
                process_image(image_path, effects)
            else:
                print(f"File not found: {image_path}")
    
    # Try to find example image
    example_paths = [
        "../../assets/jpeg.jpg",
        "../../../assets/jpeg.jpg",
        "assets/jpeg.jpg",
        "jpeg.jpg"
    ]
    
    for path in example_paths:
        if os.path.exists(path):
            print(f"\nFound example image at {path}")
            process_image(path, effects)
            break
    
    print("\n" + "=" * 50)
    print("Test completed! Check the output images.")


if __name__ == "__main__":
    main()