# OpenCV Video Effects

This ML worker now includes real-time video effects using OpenCV. These effects work without requiring heavy ML models and provide immediate visual transformations.

## Available Effects

### Face Detection (`face_detection`)
- Detects faces in the video stream
- Draws bounding boxes around detected faces
- Also detects eyes within face regions
- Returns metadata with face coordinates

### Background Blur (`background_blur`)
- Detects faces and keeps them sharp
- Blurs the background for a portrait mode effect
- Uses smooth transitions between sharp and blurred areas

### Edge Detection (`edge_detection`)
- Applies Canny edge detection algorithm
- Configurable thresholds:
  - `low_threshold`: default 50
  - `high_threshold`: default 150

### Color Effects
- **Sepia** (`sepia`): Vintage brown-tone effect
- **Grayscale** (`grayscale`): Black and white conversion
- **Vintage** (`vintage`): Old photo effect with vignette and color shift

### Artistic Effects
- **Cartoon** (`cartoon`): Converts video to cartoon-like appearance
- **Emboss** (`emboss`): 3D embossed effect
- **Pixelate** (`pixelate`): Retro pixelated effect
  - `pixel_size`: default 10

### Enhancement Effects
- **Blur** (`blur`): Gaussian blur
  - `kernel_size`: default 15 (must be odd)
- **Sharpen** (`sharpen`): Enhances image details

## Testing

### Standalone Test (No dependencies)
```bash
python test_effects_standalone.py [image_path]
```

### Full Test Suite
```bash
python test_opencv_effects.py
```

### Simple ML Worker
For testing without heavy ML dependencies:
```bash
python ml_worker_simple.py
```

## Implementation Details

All effects are implemented in `ml_worker_opencv.py` using pure OpenCV operations:

1. **Real-time Performance**: All effects are optimized for real-time processing
2. **Fallback Support**: The main ML worker uses these as fallbacks when models aren't available
3. **Cascade Downloads**: Face detection cascades are automatically downloaded when needed

## Adding New Effects

To add a new effect:

1. Add the effect method to `OpenCVProcessor` class
2. Add the effect type handling in the worker's `process_task` method
3. Test with the standalone script

Example:
```python
async def apply_new_effect(self, image: np.ndarray) -> np.ndarray:
    """Apply your custom effect"""
    # Your OpenCV code here
    result = cv2.someOperation(image)
    return result
```