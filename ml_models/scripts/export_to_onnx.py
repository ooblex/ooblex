"""
Export trained PyTorch models to ONNX format for production deployment
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxsim import simplify
from train_background_removal import BackgroundRemovalNet

# Import model architectures
from train_face_swap import FaceSwapAutoencoder
from train_style_transfer import StyleTransferNet


def verify_onnx_model(onnx_path: str, test_input: np.ndarray) -> bool:
    """Verify ONNX model works correctly"""
    try:
        # Check model is valid
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Test inference
        ort_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        input_name = ort_session.get_inputs()[0].name
        outputs = ort_session.run(None, {input_name: test_input})

        print(f"✓ Model {onnx_path} verified successfully")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {outputs[0].shape}")
        return True

    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False


def optimize_onnx_model(input_path: str, output_path: str):
    """Optimize ONNX model for inference"""
    try:
        # Load model
        model = onnx.load(input_path)

        # Simplify
        model_simp, check = simplify(model)

        if check:
            onnx.save(model_simp, output_path)

            # Compare file sizes
            original_size = os.path.getsize(input_path) / 1024 / 1024
            optimized_size = os.path.getsize(output_path) / 1024 / 1024

            print(
                f"✓ Model optimized: {original_size:.2f} MB -> {optimized_size:.2f} MB"
            )
            print(f"  Reduction: {(1 - optimized_size/original_size) * 100:.1f}%")
        else:
            print("✗ Model simplification failed")

    except Exception as e:
        print(f"✗ Optimization error: {e}")


def export_face_swap_model(checkpoint_path: str, output_dir: str):
    """Export face swap model to ONNX"""
    print("\nExporting Face Swap Model...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = FaceSwapAutoencoder(latent_dim=512)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export encoder A
    dummy_input = torch.randn(1, 3, 256, 256)
    encoder_path = os.path.join(output_dir, "face_swap_encoder_A.onnx")
    torch.onnx.export(
        model.encoder_A,
        dummy_input,
        encoder_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["latent"],
        dynamic_axes={"image": {0: "batch_size"}, "latent": {0: "batch_size"}},
        verbose=False,
    )

    # Export decoder B
    dummy_latent = torch.randn(1, 512)
    decoder_path = os.path.join(output_dir, "face_swap_decoder_B.onnx")
    torch.onnx.export(
        model.decoder_B,
        dummy_latent,
        decoder_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["latent"],
        output_names=["image"],
        dynamic_axes={"latent": {0: "batch_size"}, "image": {0: "batch_size"}},
        verbose=False,
    )

    # Verify models
    test_image = np.random.randn(1, 3, 256, 256).astype(np.float32)
    verify_onnx_model(encoder_path, test_image)

    test_latent = np.random.randn(1, 512).astype(np.float32)
    verify_onnx_model(decoder_path, test_latent)

    # Optimize models
    optimize_onnx_model(
        encoder_path, os.path.join(output_dir, "face_swap_encoder_A_opt.onnx")
    )
    optimize_onnx_model(
        decoder_path, os.path.join(output_dir, "face_swap_decoder_B_opt.onnx")
    )

    # Save metadata
    metadata = {
        "model_type": "face_swap",
        "input_size": [256, 256],
        "latent_dim": 512,
        "version": "1.0",
        "exported_from": checkpoint_path,
    }

    with open(os.path.join(output_dir, "face_swap_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def export_style_transfer_model(checkpoint_path: str, output_dir: str):
    """Export style transfer model to ONNX"""
    print("\nExporting Style Transfer Model...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = StyleTransferNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export model
    dummy_input = torch.randn(1, 3, 512, 512)
    model_path = os.path.join(output_dir, "style_transfer.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["styled_image"],
        dynamic_axes={
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "styled_image": {0: "batch_size", 2: "height", 3: "width"},
        },
        verbose=False,
    )

    # Verify model
    test_image = np.random.randn(1, 3, 512, 512).astype(np.float32)
    verify_onnx_model(model_path, test_image)

    # Optimize model
    optimize_onnx_model(model_path, os.path.join(output_dir, "style_transfer_opt.onnx"))

    # Save metadata
    metadata = {
        "model_type": "style_transfer",
        "input_size": "dynamic",
        "style": checkpoint.get("style", "unknown"),
        "version": "1.0",
        "exported_from": checkpoint_path,
    }

    with open(os.path.join(output_dir, "style_transfer_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def export_background_removal_model(checkpoint_path: str, output_dir: str):
    """Export background removal model to ONNX"""
    print("\nExporting Background Removal Model...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = BackgroundRemovalNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export model
    dummy_input = torch.randn(1, 3, 512, 512)
    model_path = os.path.join(output_dir, "background_removal.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["mask"],
        dynamic_axes={
            "image": {0: "batch_size", 2: "height", 3: "width"},
            "mask": {0: "batch_size", 2: "height", 3: "width"},
        },
        verbose=False,
    )

    # Verify model
    test_image = np.random.randn(1, 3, 512, 512).astype(np.float32)
    verify_onnx_model(model_path, test_image)

    # Optimize model
    optimize_onnx_model(
        model_path, os.path.join(output_dir, "background_removal_opt.onnx")
    )

    # Save metadata
    metadata = {
        "model_type": "background_removal",
        "input_size": "dynamic",
        "output_type": "binary_mask",
        "version": "1.0",
        "exported_from": checkpoint_path,
    }

    with open(os.path.join(output_dir, "background_removal_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def create_model_package(models_dir: str, output_path: str):
    """Create a deployable model package"""
    import hashlib
    import zipfile

    print("\nCreating model package...")

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith((".onnx", ".json")):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, models_dir)
                    zipf.write(file_path, arcname)

                    # Calculate hash
                    with open(file_path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    print(f"  Added: {arcname} (SHA256: {file_hash[:8]}...)")

    # Package metadata
    package_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n✓ Model package created: {output_path} ({package_size:.2f} MB)")


def benchmark_onnx_model(model_path: str, input_shape: tuple, num_runs: int = 100):
    """Benchmark ONNX model performance"""
    import time

    print(f"\nBenchmarking {model_path}...")

    # Create session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = ort.InferenceSession(model_path, providers=providers)

    # Get provider
    provider = ort_session.get_providers()[0]
    print(f"  Provider: {provider}")

    # Prepare input
    input_name = ort_session.get_inputs()[0].name
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        ort_session.run(None, {input_name: dummy_input})

    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        ort_session.run(None, {input_name: dummy_input})
        times.append(time.perf_counter() - start)

    # Statistics
    times_ms = np.array(times) * 1000
    print(f"  Average: {np.mean(times_ms):.2f} ms")
    print(f"  Median: {np.median(times_ms):.2f} ms")
    print(f"  Min: {np.min(times_ms):.2f} ms")
    print(f"  Max: {np.max(times_ms):.2f} ms")
    print(f"  FPS: {1000 / np.mean(times_ms):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX format")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory with model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./onnx",
        help="Output directory for ONNX models",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["face_swap", "style_transfer", "background_removal"],
        help="Models to export",
    )
    parser.add_argument(
        "--create-package", action="store_true", help="Create deployable package"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark exported models"
    )

    args = parser.parse_args()

    # Export models
    if "face_swap" in args.models:
        checkpoint = os.path.join(args.checkpoint_dir, "face_swap_best.pth")
        if os.path.exists(checkpoint):
            export_face_swap_model(
                checkpoint, os.path.join(args.output_dir, "face_swap")
            )

    if "style_transfer" in args.models:
        checkpoint = os.path.join(args.checkpoint_dir, "style_transfer_best.pth")
        if os.path.exists(checkpoint):
            export_style_transfer_model(
                checkpoint, os.path.join(args.output_dir, "style_transfer")
            )

    if "background_removal" in args.models:
        checkpoint = os.path.join(args.checkpoint_dir, "background_removal_best.pth")
        if os.path.exists(checkpoint):
            export_background_removal_model(
                checkpoint, os.path.join(args.output_dir, "background_removal")
            )

    # Create package
    if args.create_package:
        package_path = os.path.join(args.output_dir, "ooblex_models_v1.0.zip")
        create_model_package(args.output_dir, package_path)

    # Benchmark models
    if args.benchmark:
        print("\n" + "=" * 50)
        print("PERFORMANCE BENCHMARKS")
        print("=" * 50)

        # Face swap encoder
        encoder_path = os.path.join(
            args.output_dir, "face_swap", "face_swap_encoder_A_opt.onnx"
        )
        if os.path.exists(encoder_path):
            benchmark_onnx_model(encoder_path, (1, 3, 256, 256))

        # Style transfer
        style_path = os.path.join(
            args.output_dir, "style_transfer", "style_transfer_opt.onnx"
        )
        if os.path.exists(style_path):
            benchmark_onnx_model(style_path, (1, 3, 512, 512))

        # Background removal
        bg_path = os.path.join(
            args.output_dir, "background_removal", "background_removal_opt.onnx"
        )
        if os.path.exists(bg_path):
            benchmark_onnx_model(bg_path, (1, 3, 512, 512))

    print("\nExport completed!")


if __name__ == "__main__":
    main()
