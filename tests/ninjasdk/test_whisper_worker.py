"""
Unit tests for Whisper STT Worker
"""

import unittest
import numpy as np
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../code'))

from whisper_worker import WhisperWorker, AudioChunk


class TestAudioConversion(unittest.TestCase):
    """Test audio format conversion"""

    def test_pcm16_to_float32(self):
        """Test PCM16 to Float32 conversion"""
        worker = WhisperWorker()

        # Create test PCM16 data
        int16_array = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        pcm_bytes = int16_array.tobytes()

        # Convert
        float32_array = worker.pcm16_to_float32(pcm_bytes)

        # Verify
        self.assertEqual(len(float32_array), 5)
        self.assertAlmostEqual(float32_array[0], 0.0, places=4)
        self.assertAlmostEqual(float32_array[1], 0.5, places=2)
        self.assertAlmostEqual(float32_array[2], -0.5, places=2)
        self.assertAlmostEqual(float32_array[3], 1.0, places=4)
        self.assertAlmostEqual(float32_array[4], -1.0, places=4)

    def test_pcm16_to_float32_normalization(self):
        """Test normalization is in correct range"""
        worker = WhisperWorker()

        # Create random PCM16 data
        int16_array = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
        pcm_bytes = int16_array.tobytes()

        # Convert
        float32_array = worker.pcm16_to_float32(pcm_bytes)

        # Verify range
        self.assertTrue(np.all(float32_array >= -1.0))
        self.assertTrue(np.all(float32_array <= 1.0))


class TestWhisperWorker(unittest.TestCase):
    """Test Whisper Worker functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.worker = WhisperWorker()

    @patch('whisper_worker.redis')
    @patch('whisper_worker.pika')
    def test_initialization(self, mock_pika, mock_redis):
        """Test worker initialization"""
        mock_redis_client = MagicMock()
        mock_redis.Redis.from_url.return_value = mock_redis_client

        mock_connection = MagicMock()
        mock_pika.BlockingConnection.return_value = mock_connection

        # Note: Can't actually call initialize() without Whisper model
        # This is a placeholder for demonstration
        self.assertIsNotNone(self.worker)

    def test_audio_chunk_creation(self):
        """Test AudioChunk dataclass"""
        chunk = AudioChunk(
            chunk_id='test-chunk-123',
            stream_id='test-stream',
            timestamp=1234567890,
            sample_rate=16000,
            channels=1,
            format='pcm16',
            duration_ms=1000.0,
            bytes=32000,
            data=b'test_data'
        )

        self.assertEqual(chunk.chunk_id, 'test-chunk-123')
        self.assertEqual(chunk.stream_id, 'test-stream')
        self.assertEqual(chunk.sample_rate, 16000)
        self.assertEqual(chunk.channels, 1)

    @patch.object(WhisperWorker, 'transcribe_faster_whisper')
    def test_transcription_mock(self, mock_transcribe):
        """Test transcription with mock"""
        mock_transcribe.return_value = {
            'text': 'Hello world',
            'language': 'en',
            'language_probability': 0.99,
            'confidence': 0.95,
            'segments': 1,
        }

        audio_array = np.random.randn(16000).astype(np.float32)
        self.worker.use_faster_whisper = True

        result = self.worker.transcribe_audio(
            AudioChunk(
                chunk_id='test',
                stream_id='test',
                timestamp=0,
                sample_rate=16000,
                channels=1,
                format='pcm16',
                duration_ms=1000,
                bytes=32000,
                data=audio_array.tobytes()
            )
        )

        self.assertEqual(result['text'], 'Hello world')
        self.assertEqual(result['confidence'], 0.95)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_end_to_end_pipeline(self):
        """Test complete pipeline (requires real services)"""
        # This would require Redis, RabbitMQ, and Whisper model
        # For CI/CD, use mock services or docker-compose test environment
        pass


if __name__ == '__main__':
    unittest.main()
