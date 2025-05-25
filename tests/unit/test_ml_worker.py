"""
Unit tests for ML Worker service
"""
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, AsyncMock
import json

from ml_worker.ml_worker import ModelManager, MLProcessor, MLWorker


class TestModelManager:
    """Test ModelManager functionality"""
    
    @pytest.mark.asyncio
    async def test_load_model_onnx(self, mocker):
        """Test loading ONNX model"""
        # Mock onnxruntime
        mock_ort = mocker.patch("ml_worker.ml_worker.ort")
        mock_session = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        
        manager = ModelManager(cache_size=2)
        model = await manager.load_model("test_model", "onnx")
        
        assert model == mock_session
        assert "test_model" in manager.models
        mock_ort.InferenceSession.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_model_torch(self, mocker):
        """Test loading PyTorch model"""
        # Mock torch
        mock_torch = mocker.patch("ml_worker.ml_worker.torch")
        mock_model = MagicMock()
        mock_torch.jit.load.return_value = mock_model
        
        manager = ModelManager(cache_size=2)
        model = await manager.load_model("test_model", "torch")
        
        assert model == mock_model
        assert "test_model" in manager.models
        mock_model.cuda.assert_called_once()
        mock_model.eval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_model_cache_eviction(self, mocker):
        """Test LRU cache eviction"""
        mock_ort = mocker.patch("ml_worker.ml_worker.ort")
        mock_ort.InferenceSession.return_value = MagicMock()
        
        manager = ModelManager(cache_size=2)
        
        # Load 3 models (cache size is 2)
        await manager.load_model("model1", "onnx")
        await manager.load_model("model2", "onnx")
        await manager.load_model("model3", "onnx")
        
        # First model should be evicted
        assert "model1" not in manager.models
        assert "model2" in manager.models
        assert "model3" in manager.models
    
    @pytest.mark.asyncio
    async def test_model_load_error(self, mocker):
        """Test model loading error handling"""
        mock_ort = mocker.patch("ml_worker.ml_worker.ort")
        mock_ort.InferenceSession.side_effect = Exception("Model not found")
        
        manager = ModelManager()
        
        with pytest.raises(Exception, match="Model not found"):
            await manager.load_model("missing_model", "onnx")


class TestMLProcessor:
    """Test MLProcessor functionality"""
    
    @pytest.fixture
    def ml_processor(self, mocker):
        """Create MLProcessor instance with mocked dependencies"""
        model_manager = MagicMock()
        processor = MLProcessor(model_manager)
        
        # Mock MediaPipe components
        processor.face_detection = MagicMock()
        processor.face_mesh = MagicMock()
        processor.selfie_segmentation = MagicMock()
        
        return processor
    
    @pytest.mark.asyncio
    async def test_process_face_swap_no_face(self, ml_processor, sample_image):
        """Test face swap when no face is detected"""
        # Mock no face detection
        ml_processor.face_detection.process.return_value = MagicMock(detections=None)
        
        result = await ml_processor.process_face_swap(sample_image)
        
        # Should return original image
        np.testing.assert_array_equal(result, sample_image)
    
    @pytest.mark.asyncio
    async def test_process_face_swap_with_face(self, ml_processor, sample_image, mocker):
        """Test face swap with face detected"""
        # Mock face detection
        detection = MagicMock()
        detection.location_data.relative_bounding_box.xmin = 0.2
        detection.location_data.relative_bounding_box.ymin = 0.2
        detection.location_data.relative_bounding_box.width = 0.6
        detection.location_data.relative_bounding_box.height = 0.6
        
        results = MagicMock(detections=[detection])
        ml_processor.face_detection.process.return_value = results
        
        # Mock model
        mock_model = MagicMock()
        mock_model.get_inputs.return_value = [MagicMock(name="input")]
        mock_model.run.return_value = [np.ones((1, 3, 256, 256), dtype=np.float32)]
        ml_processor.model_manager.load_model = AsyncMock(return_value=mock_model)
        
        result = await ml_processor.process_face_swap(sample_image)
        
        assert result.shape == sample_image.shape
        ml_processor.model_manager.load_model.assert_called_once_with("face_swap")
    
    @pytest.mark.asyncio
    async def test_process_style_transfer(self, ml_processor, sample_image, mocker):
        """Test style transfer processing"""
        # Mock model
        mock_model = MagicMock()
        mock_model.get_inputs.return_value = [MagicMock(name="input")]
        mock_model.run.return_value = [np.ones((1, 3, 512, 512), dtype=np.float32)]
        ml_processor.model_manager.load_model = AsyncMock(return_value=mock_model)
        
        result = await ml_processor.process_style_transfer(sample_image)
        
        assert result.shape == sample_image.shape
        ml_processor.model_manager.load_model.assert_called_once_with("style_transfer")
    
    @pytest.mark.asyncio
    async def test_process_background_removal(self, ml_processor, sample_image):
        """Test background removal"""
        # Mock segmentation
        mask = np.ones((100, 100), dtype=np.float32)
        results = MagicMock(segmentation_mask=mask)
        ml_processor.selfie_segmentation.process.return_value = results
        
        result = await ml_processor.process_background_removal(sample_image)
        
        # Result should have alpha channel
        assert result.shape == (100, 100, 4)
        assert result.dtype == np.uint8
    
    @pytest.mark.asyncio
    async def test_process_object_detection(self, ml_processor, sample_image):
        """Test object detection"""
        # Mock face detection
        detection = MagicMock()
        detection.location_data.relative_bounding_box.xmin = 0.2
        detection.location_data.relative_bounding_box.ymin = 0.2
        detection.location_data.relative_bounding_box.width = 0.6
        detection.location_data.relative_bounding_box.height = 0.6
        detection.score = [0.95]
        
        results = MagicMock(detections=[detection])
        ml_processor.face_detection.process.return_value = results
        
        annotated, detections = await ml_processor.process_object_detection(sample_image)
        
        assert annotated.shape == sample_image.shape
        assert len(detections["detections"]) == 1
        assert detections["detections"][0]["class"] == "face"
        assert detections["detections"][0]["confidence"] == 0.95


class TestMLWorker:
    """Test MLWorker functionality"""
    
    @pytest.fixture
    def ml_worker(self):
        """Create MLWorker instance"""
        return MLWorker()
    
    @pytest.mark.asyncio
    async def test_setup(self, ml_worker, mocker):
        """Test worker setup"""
        # Mock connections
        mock_redis = mocker.patch("ml_worker.ml_worker.redis.from_url")
        mock_rabbitmq = mocker.patch("ml_worker.ml_worker.connect_robust")
        
        mock_redis.return_value = AsyncMock()
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_connection.channel.return_value = mock_channel
        mock_rabbitmq.return_value = mock_connection
        
        await ml_worker.setup()
        
        assert ml_worker.redis_client is not None
        assert ml_worker.rabbitmq_connection is not None
        assert ml_worker.rabbitmq_channel is not None
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, ml_worker, mocker):
        """Test successful task processing"""
        # Setup worker
        ml_worker.redis_client = AsyncMock()
        ml_worker.processor = AsyncMock()
        
        # Mock Redis get
        image_data = cv2.imencode('.jpg', np.zeros((100, 100, 3), dtype=np.uint8))[1].tobytes()
        ml_worker.redis_client.get.return_value = image_data
        
        # Mock processor
        processed_image = np.ones((100, 100, 3), dtype=np.uint8)
        ml_worker.processor.process_face_swap.return_value = processed_image
        
        # Create message
        task_data = {
            "task_id": "test_task",
            "stream_token": "test_stream",
            "process_type": "face_swap",
            "parameters": {}
        }
        
        message = AsyncMock()
        message.body = json.dumps(task_data).encode()
        
        # Process task
        await ml_worker.process_task(message)
        
        # Verify Redis calls
        ml_worker.redis_client.get.assert_called_with("frame:test_stream:latest")
        ml_worker.redis_client.setex.assert_called()
        
        # Verify processor called
        ml_worker.processor.process_face_swap.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_task_no_frame(self, ml_worker, mocker):
        """Test task processing when no frame is available"""
        # Setup worker
        ml_worker.redis_client = AsyncMock()
        ml_worker.redis_client.get.return_value = None
        
        # Create message
        task_data = {
            "task_id": "test_task",
            "stream_token": "test_stream",
            "process_type": "face_swap",
            "parameters": {}
        }
        
        message = AsyncMock()
        message.body = json.dumps(task_data).encode()
        
        # Process task
        await ml_worker.process_task(message)
        
        # Verify error status was set
        calls = ml_worker.redis_client.setex.call_args_list
        assert any("failed" in str(call) for call in calls)
    
    @pytest.mark.asyncio
    async def test_update_gpu_metrics(self, ml_worker, mocker):
        """Test GPU metrics update"""
        # Mock torch
        mock_torch = mocker.patch("ml_worker.ml_worker.torch")
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        
        # Mock metrics
        mock_gauge = mocker.patch("ml_worker.ml_worker.gpu_memory_usage")
        
        # Set running flag
        ml_worker.running = True
        
        # Run update once
        update_task = asyncio.create_task(ml_worker.update_gpu_metrics())
        await asyncio.sleep(0.1)
        ml_worker.running = False
        
        try:
            await asyncio.wait_for(update_task, timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        # Verify GPU metrics were set
        mock_gauge.set.assert_called_with(1024 * 1024 * 1024)