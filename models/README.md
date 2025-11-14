# Ooblex Models Directory

## ⚠️ Original Models No Longer Available

The original TensorFlow face swap models from 2020 are **no longer available**:
- Too large to host on GitHub (~500MB each)
- Original Google Drive / S3 links are inactive
- Trained specifically for the 2020 demo

---

## ✅ Zero-Friction Demo (No Models Required!)

You can run Ooblex **right now** without downloading any models:

```bash
# Use simple OpenCV effects
python3 code/brain_simple.py
```

### Available Effects (No Downloads)

| Effect | Command | Description |
|--------|---------|-------------|
| Face Detection | `FaceOn` | Detect faces with boxes |
| Pixelate Faces | `TrumpOn` | Privacy filter |
| Cartoon | `CartoonOn` | Comic book style |
| Background Blur | `BlurOn` | Blur frame |
| Edge Detection | `EdgeOn` | Canny edges |
| Grayscale | `GrayOn` | Black & white |

**All run on CPU at 30-100+ FPS!**

---

## 🎨 Adding Your Own Models

See full guide above for:
- Supported frameworks (TensorFlow, PyTorch, ONNX, etc.)
- Model structure and metadata
- Download scripts
- Conversion to ONNX
- Training your own models

---

**Start with simple effects, add AI later!**
