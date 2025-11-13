#!/bin/bash
#
# Ooblex Model Download Script
#
# The original 2020 TensorFlow face swap models are no longer available.
# This script provides alternatives for getting models.
#

echo "=================================================="
echo "  Ooblex Model Download Script"
echo "=================================================="
echo

echo "⚠️  Note: Original 2020 models are NOT available"
echo "   (too large for GitHub, original links inactive)"
echo

echo "You have several options:"
echo

echo "1. Use built-in OpenCV effects (NO DOWNLOAD NEEDED)"
echo "   - Face detection, blur, edges, cartoon, etc."
echo "   - Run: python3 code/brain_simple.py"
echo "   - Works immediately!"
echo

echo "2. Download popular open-source models:"
echo

# Face Detection (included with OpenCV)
echo "   ✅ Face Detection: Included with OpenCV (no download)"

# MediaPipe for background removal
echo "   📦 Background Removal (MediaPipe):"
echo "      $ pip install mediapipe"

# YOLOv8 for object detection
echo "   📦 Object Detection (YOLOv8):"
echo "      $ pip install ultralytics"
echo "      $ python3 -c 'from ultralytics import YOLO; YOLO(\"yolov8n.pt\")'"

# Style transfer
echo "   📦 Style Transfer (ONNX):"
echo "      $ mkdir -p models/style_transfer"
echo "      $ wget https://github.com/onnx/models/raw/main/vision/style_transfer/fast_neural_style/model/mosaic-8.onnx \\"
echo "        -O models/style_transfer/mosaic.onnx"

echo

echo "3. Train your own models:"
echo "   - See ml_models/scripts/ for training examples"
echo "   - Use your own datasets"
echo "   - Export to ONNX for best compatibility"
echo

echo "4. Community models:"
echo "   - Check GitHub Discussions for shared models"
echo "   - Hugging Face Model Hub"
echo "   - TensorFlow Hub / PyTorch Hub"
echo

echo "=================================================="
echo

read -p "Download MediaPipe + YOLOv8? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing MediaPipe..."
    pip install mediapipe

    echo "Installing YOLOv8..."
    pip install ultralytics
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

    echo
    echo "✅ Models downloaded!"
    echo
    echo "Now create your own worker using these models."
    echo "See models/README.md for examples."
else
    echo
    echo "No problem! Use brain_simple.py for instant effects."
    echo "Run: python3 code/brain_simple.py"
fi

echo
