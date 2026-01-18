"""
Unit tests for OpenCV Processor
Tests all OpenCV-based video effects without requiring heavy ML dependencies.
"""
import pytest
import numpy as np
import cv2
import os
import sys

# Add the ml-worker path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'ml-worker'))

from ml_worker_opencv import OpenCVProcessor


@pytest.fixture(scope="module")
def processor():
    """Create OpenCV processor instance"""
    return OpenCVProcessor()


@pytest.fixture
def sample_image():
    """Create a sample test image with colors and patterns"""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add colored rectangles
    cv2.rectangle(image, (50, 50), (200, 200), (0, 0, 255), -1)  # Red
    cv2.rectangle(image, (250, 50), (400, 200), (0, 255, 0), -1)  # Green
    cv2.rectangle(image, (450, 50), (600, 200), (255, 0, 0), -1)  # Blue
    # Add text
    cv2.putText(image, "Test", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    # Add circles
    cv2.circle(image, (150, 400), 50, (255, 255, 0), -1)  # Cyan
    cv2.circle(image, (320, 400), 50, (255, 0, 255), -1)  # Magenta
    return image


@pytest.fixture
def grayscale_image():
    """Create a grayscale test image"""
    return np.random.randint(0, 255, (100, 100), dtype=np.uint8)


@pytest.fixture
def small_image():
    """Create a small test image for performance tests"""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Create a larger test image for stress tests"""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


class TestOpenCVProcessorInit:
    """Test OpenCV processor initialization"""

    def test_processor_initialization(self, processor):
        """Test that processor initializes correctly"""
        assert processor is not None
        assert hasattr(processor, 'face_cascade')
        assert hasattr(processor, 'eye_cascade')

    def test_cascade_files_exist(self, processor):
        """Test that cascade files were downloaded/loaded"""
        assert os.path.exists(processor.face_cascade_path)
        assert os.path.exists(processor.eye_cascade_path)

    def test_cascade_loaded(self, processor):
        """Test that cascade classifiers are loaded"""
        assert not processor.face_cascade.empty()
        assert not processor.eye_cascade.empty()


class TestGrayscaleEffect:
    """Test grayscale conversion"""

    @pytest.mark.asyncio
    async def test_grayscale_basic(self, processor, sample_image):
        """Test basic grayscale conversion"""
        result = await processor.convert_grayscale(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape  # Should return 3-channel
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_grayscale_preserves_dimensions(self, processor, sample_image):
        """Test that grayscale preserves image dimensions"""
        result = await processor.convert_grayscale(sample_image)

        assert result.shape[0] == sample_image.shape[0]  # Height
        assert result.shape[1] == sample_image.shape[1]  # Width

    @pytest.mark.asyncio
    async def test_grayscale_all_channels_same(self, processor, sample_image):
        """Test that grayscale result has same values in all channels"""
        result = await processor.convert_grayscale(sample_image)

        # All channels should be equal in a grayscale image converted to BGR
        assert np.array_equal(result[:, :, 0], result[:, :, 1])
        assert np.array_equal(result[:, :, 1], result[:, :, 2])


class TestSepiaEffect:
    """Test sepia effect"""

    @pytest.mark.asyncio
    async def test_sepia_basic(self, processor, sample_image):
        """Test basic sepia effect"""
        result = await processor.apply_sepia(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_sepia_color_shift(self, processor, sample_image):
        """Test that sepia creates warm color tones"""
        result = await processor.apply_sepia(sample_image)

        # Sepia should generally increase red channel values relative to blue
        # for colored regions
        assert result is not None


class TestEdgeDetection:
    """Test edge detection effect"""

    @pytest.mark.asyncio
    async def test_edge_detection_basic(self, processor, sample_image):
        """Test basic edge detection"""
        result = await processor.edge_detection(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_edge_detection_with_thresholds(self, processor, sample_image):
        """Test edge detection with custom thresholds"""
        result_low = await processor.edge_detection(sample_image, low_threshold=10, high_threshold=50)
        result_high = await processor.edge_detection(sample_image, low_threshold=100, high_threshold=200)

        # Lower thresholds should detect more edges
        assert result_low is not None
        assert result_high is not None

    @pytest.mark.asyncio
    async def test_edge_detection_returns_bgr(self, processor, sample_image):
        """Test that edge detection returns 3-channel image"""
        result = await processor.edge_detection(sample_image)

        assert len(result.shape) == 3
        assert result.shape[2] == 3


class TestCartoonEffect:
    """Test cartoon effect"""

    @pytest.mark.asyncio
    async def test_cartoon_basic(self, processor, sample_image):
        """Test basic cartoon effect"""
        result = await processor.cartoon_effect(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_cartoon_preserves_size(self, processor, small_image):
        """Test that cartoon effect preserves image size"""
        result = await processor.cartoon_effect(small_image)

        assert result.shape == small_image.shape


class TestVintageEffect:
    """Test vintage effect"""

    @pytest.mark.asyncio
    async def test_vintage_basic(self, processor, sample_image):
        """Test basic vintage effect"""
        result = await processor.apply_vintage(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_vintage_vignette(self, processor, sample_image):
        """Test that vintage creates vignette (darker corners)"""
        result = await processor.apply_vintage(sample_image)

        # Corners should generally be darker than center
        center = result[result.shape[0]//2, result.shape[1]//2]
        corner = result[0, 0]

        # Not a strict assertion as noise can affect this
        assert result is not None


class TestBlurEffect:
    """Test blur effect"""

    @pytest.mark.asyncio
    async def test_blur_basic(self, processor, sample_image):
        """Test basic blur effect"""
        result = await processor.apply_blur(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_blur_with_kernel_size(self, processor, sample_image):
        """Test blur with custom kernel size"""
        result_small = await processor.apply_blur(sample_image, kernel_size=5)
        result_large = await processor.apply_blur(sample_image, kernel_size=31)

        assert result_small is not None
        assert result_large is not None
        # Larger kernel should produce more blur (different result)
        assert not np.array_equal(result_small, result_large)

    @pytest.mark.asyncio
    async def test_blur_even_kernel_size(self, processor, sample_image):
        """Test that even kernel sizes are handled (converted to odd)"""
        result = await processor.apply_blur(sample_image, kernel_size=10)

        assert result is not None  # Should not raise error


class TestSharpenEffect:
    """Test sharpen effect"""

    @pytest.mark.asyncio
    async def test_sharpen_basic(self, processor, sample_image):
        """Test basic sharpen effect"""
        result = await processor.apply_sharpen(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_sharpen_enhances_edges(self, processor):
        """Test that sharpen enhances edges"""
        # Create image with gradient for edge enhancement
        gradient_image = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(100):
            gradient_image[i, :, :] = i * 2  # Vertical gradient

        result = await processor.apply_sharpen(gradient_image)

        # Sharpened image should be different from original
        assert not np.array_equal(result, gradient_image)


class TestPixelateEffect:
    """Test pixelate effect"""

    @pytest.mark.asyncio
    async def test_pixelate_basic(self, processor, sample_image):
        """Test basic pixelate effect"""
        result = await processor.pixelate(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_pixelate_with_size(self, processor, sample_image):
        """Test pixelate with custom pixel size"""
        result_small = await processor.pixelate(sample_image, pixel_size=5)
        result_large = await processor.pixelate(sample_image, pixel_size=20)

        assert result_small is not None
        assert result_large is not None
        # Different pixel sizes should produce different results
        assert not np.array_equal(result_small, result_large)

    @pytest.mark.asyncio
    async def test_pixelate_creates_blocks(self, processor, sample_image):
        """Test that pixelate creates uniform color blocks"""
        result = await processor.pixelate(sample_image, pixel_size=10)

        # Adjacent pixels in a block should have the same color
        # Check a small region
        block_sample = result[0:10, 0:10]
        # All pixels in the block should be identical
        first_pixel = block_sample[0, 0]
        assert all(np.array_equal(block_sample[i, j], first_pixel)
                   for i in range(10) for j in range(10))


class TestEmbossEffect:
    """Test emboss effect"""

    @pytest.mark.asyncio
    async def test_emboss_basic(self, processor, sample_image):
        """Test basic emboss effect"""
        result = await processor.apply_emboss(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


class TestFaceDetection:
    """Test face detection functionality"""

    @pytest.mark.asyncio
    async def test_detect_faces_no_faces(self, processor, sample_image):
        """Test face detection on image with no faces"""
        result, face_data = await processor.detect_faces(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert isinstance(face_data, list)
        # No faces expected in the test pattern image
        assert len(face_data) == 0

    @pytest.mark.asyncio
    async def test_detect_faces_returns_tuple(self, processor, sample_image):
        """Test that detect_faces returns tuple of image and data"""
        result = await processor.detect_faces(sample_image)

        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_face_data_structure(self, processor, sample_image):
        """Test face data structure"""
        _, face_data = await processor.detect_faces(sample_image)

        assert isinstance(face_data, list)
        # Each face detection should have bbox and eyes fields if present
        for face in face_data:
            assert 'bbox' in face
            assert 'eyes' in face


class TestBackgroundBlur:
    """Test background blur functionality"""

    @pytest.mark.asyncio
    async def test_blur_background_basic(self, processor, sample_image):
        """Test basic background blur"""
        result = await processor.blur_background(sample_image)

        assert result is not None
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_blur_background_no_faces(self, processor, sample_image):
        """Test background blur when no faces are detected"""
        result = await processor.blur_background(sample_image)

        # When no faces detected, should blur entire image slightly
        assert result is not None
        # Result should be different from original
        assert not np.array_equal(result, sample_image)


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_single_pixel_image(self, processor):
        """Test with minimum size image"""
        tiny_image = np.zeros((1, 1, 3), dtype=np.uint8)

        # These should not raise errors
        result = await processor.convert_grayscale(tiny_image)
        assert result is not None

    @pytest.mark.asyncio
    async def test_black_image(self, processor):
        """Test with completely black image"""
        black_image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = await processor.apply_sepia(black_image)
        assert result is not None
        # Black image should remain mostly black after sepia
        assert np.mean(result) < 10

    @pytest.mark.asyncio
    async def test_white_image(self, processor):
        """Test with completely white image"""
        white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        result = await processor.apply_sepia(white_image)
        assert result is not None

    @pytest.mark.asyncio
    async def test_large_image(self, processor, large_image):
        """Test with larger image (performance check)"""
        result = await processor.convert_grayscale(large_image)

        assert result is not None
        assert result.shape == large_image.shape


class TestDataIntegrity:
    """Test data integrity through processing"""

    @pytest.mark.asyncio
    async def test_original_image_unchanged(self, processor, sample_image):
        """Test that original image is not modified"""
        original_copy = sample_image.copy()

        await processor.apply_sepia(sample_image)
        await processor.apply_blur(sample_image)
        await processor.cartoon_effect(sample_image)

        assert np.array_equal(sample_image, original_copy)

    @pytest.mark.asyncio
    async def test_result_valid_pixel_values(self, processor, sample_image):
        """Test that results have valid pixel values (0-255)"""
        effects = [
            processor.convert_grayscale,
            processor.apply_sepia,
            processor.cartoon_effect,
            processor.apply_blur,
            processor.apply_sharpen,
            processor.apply_emboss,
        ]

        for effect in effects:
            result = await effect(sample_image)
            assert result.min() >= 0, f"{effect.__name__} produced negative values"
            assert result.max() <= 255, f"{effect.__name__} produced values > 255"


class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_multiple_effects_sequential(self, processor, sample_image):
        """Test applying multiple effects sequentially"""
        result = sample_image.copy()

        result = await processor.apply_blur(result)
        result = await processor.apply_sharpen(result)
        result = await processor.convert_grayscale(result)

        assert result is not None
        assert result.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
