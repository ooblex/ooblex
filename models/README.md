# Ooblex ML Models

This directory contains the AI/ML models used by Ooblex for video processing.

## Model Structure

```
models/
├── face_swap.onnx        # Face swapping model
├── style_transfer.onnx   # Style transfer model
├── yolov8.onnx          # Object detection model
├── background_blur.onnx  # Background blur model
└── checkpoints/         # Model checkpoints
```

## Getting Started

### Download Pre-trained Models

For demo purposes, you can use these placeholder commands:

```bash
# Create dummy model files for testing
touch face_swap.onnx
touch style_transfer.onnx
touch yolov8.onnx
touch background_blur.onnx
```

### Real Models

For production use, you'll need to:

1. Train or download appropriate models
2. Convert them to ONNX format for optimal performance
3. Place them in this directory with the correct names

## Model Requirements

- **Face Swap**: Input shape (1, 3, 256, 256), Output shape (1, 3, 256, 256)
- **Style Transfer**: Input shape (1, 3, 512, 512), Output shape (1, 3, 512, 512)
- **Object Detection**: YOLO v8 compatible format
- **Background Blur**: Segmentation model with mask output

## Model Conversion

Example PyTorch to ONNX conversion:

```python
import torch

# Load your trained model
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 256, 256)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## Note

The ML worker service uses MediaPipe for basic face detection and segmentation when models are not available, providing fallback functionality.