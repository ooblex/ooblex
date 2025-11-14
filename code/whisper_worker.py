"""
Whisper Speech-to-Text Worker for Ooblex NinjaSDK Integration

This worker consumes audio chunks from Redis (produced by the NinjaSDK audio service),
transcribes them using faster-whisper, and publishes results to RabbitMQ.

Features:
- Real-time speech-to-text using faster-whisper
- Optimized for low-latency transcription
- GPU acceleration support
- Streaming audio chunk processing
- Confidence scoring
- Language detection
- Timestamp alignment

Pipeline:
Redis (audio chunks) → This Worker → Whisper → RabbitMQ (text results)
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import pika
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672")

# Whisper Configuration
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")  # auto, cpu, cuda
WHISPER_COMPUTE_TYPE = os.getenv(
    "WHISPER_COMPUTE_TYPE", "default"
)  # default, int8, float16


@dataclass
class AudioChunk:
    """Audio chunk metadata"""

    chunk_id: str
    stream_id: str
    timestamp: int
    sample_rate: int
    channels: int
    format: str
    duration_ms: float
    bytes: int
    data: bytes


class WhisperWorker:
    """
    Whisper STT Worker

    Processes audio chunks from NinjaSDK ingestion service and transcribes using Whisper.
    """

    def __init__(self):
        self.redis_client = None
        self.rabbit_connection = None
        self.rabbit_channel = None
        self.whisper_model = None
        self.stats = {
            "chunks_processed": 0,
            "total_audio_duration": 0,
            "total_processing_time": 0,
            "errors": 0,
        }

    def initialize(self):
        """Initialize connections and load models"""
        logger.info("Initializing Whisper STT Worker...")

        # Connect to Redis
        logger.info("Connecting to Redis...")
        self.redis_client = redis.Redis.from_url(REDIS_URL)
        self.redis_client.ping()
        logger.info("Redis connected")

        # Connect to RabbitMQ
        logger.info("Connecting to RabbitMQ...")
        parameters = pika.URLParameters(RABBITMQ_URL)
        self.rabbit_connection = pika.BlockingConnection(parameters)
        self.rabbit_channel = self.rabbit_connection.channel()
        self.rabbit_channel.queue_declare(queue="audio-chunks", durable=False)
        self.rabbit_channel.queue_declare(queue="stt-results", durable=False)
        logger.info("RabbitMQ connected")

        # Load Whisper model
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}...")
        self.load_whisper_model()
        logger.info("Whisper model loaded")

        logger.info("Whisper STT Worker ready!")

    def load_whisper_model(self):
        """Load Whisper model with faster-whisper or fallback to openai-whisper"""
        try:
            # Try faster-whisper first (recommended for production)
            from faster_whisper import WhisperModel

            # Determine device
            device = WHISPER_DEVICE
            if device == "auto":
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"

            # Determine compute type
            compute_type = WHISPER_COMPUTE_TYPE
            if compute_type == "default":
                compute_type = "float16" if device == "cuda" else "int8"

            logger.info(
                f"Using faster-whisper with device={device}, compute_type={compute_type}"
            )
            self.whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=device,
                compute_type=compute_type,
                download_root=os.getenv("WHISPER_CACHE_DIR", None),
            )
            self.use_faster_whisper = True

        except ImportError:
            logger.warning("faster-whisper not found, falling back to openai-whisper")
            try:
                import whisper

                self.whisper_model = whisper.load_model(WHISPER_MODEL)
                self.use_faster_whisper = False
            except ImportError:
                logger.error("Neither faster-whisper nor openai-whisper is installed!")
                logger.error("Install with: pip install faster-whisper")
                sys.exit(1)

    def start_consuming(self):
        """Start consuming audio chunks from RabbitMQ"""
        logger.info("Starting to consume audio chunks...")

        def callback(ch, method, properties, body):
            try:
                self.process_audio_chunk(body)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}", exc_info=True)
                self.stats["errors"] += 1
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.rabbit_channel.basic_qos(prefetch_count=1)
        self.rabbit_channel.basic_consume(
            queue="audio-chunks", on_message_callback=callback, auto_ack=False
        )

        logger.info("Waiting for audio chunks...")
        try:
            self.rabbit_channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.rabbit_channel.stop_consuming()

    def process_audio_chunk(self, message_body: bytes):
        """Process a single audio chunk"""
        start_time = time.time()

        # Parse message
        message = json.loads(message_body.decode("utf-8"))
        chunk_id = message["chunkID"]
        metadata = message["metadata"]

        logger.info(f"Processing audio chunk: {chunk_id}")

        # Retrieve audio data from Redis
        audio_data = self.redis_client.get(chunk_id)
        if not audio_data:
            logger.warning(f"Audio chunk not found in Redis: {chunk_id}")
            return

        # Create AudioChunk object
        audio_chunk = AudioChunk(
            chunk_id=chunk_id,
            stream_id=metadata["streamID"],
            timestamp=metadata["timestamp"],
            sample_rate=metadata["sampleRate"],
            channels=metadata["channels"],
            format=metadata["format"],
            duration_ms=metadata["durationMs"],
            bytes=metadata["bytes"],
            data=audio_data,
        )

        # Transcribe audio
        result = self.transcribe_audio(audio_chunk)

        if result and result["text"].strip():
            # Publish result to RabbitMQ
            self.publish_result(audio_chunk, result)

            processing_time = time.time() - start_time
            logger.info(
                f"Transcribed {chunk_id}: \"{result['text']}\" "
                f"(confidence: {result['confidence']:.2f}, "
                f"processing: {processing_time:.2f}s)"
            )

            # Update stats
            self.stats["chunks_processed"] += 1
            self.stats["total_audio_duration"] += audio_chunk.duration_ms / 1000
            self.stats["total_processing_time"] += processing_time

            # Log stats periodically
            if self.stats["chunks_processed"] % 10 == 0:
                self.log_stats()
        else:
            logger.debug(f"No speech detected in chunk {chunk_id}")

    def transcribe_audio(self, audio_chunk: AudioChunk) -> Optional[Dict[str, Any]]:
        """Transcribe audio using Whisper"""
        try:
            # Convert PCM16 bytes to float32 numpy array
            audio_array = self.pcm16_to_float32(audio_chunk.data)

            if self.use_faster_whisper:
                return self.transcribe_faster_whisper(audio_array)
            else:
                return self.transcribe_openai_whisper(audio_array)

        except Exception as e:
            logger.error(f"Transcription error: {e}", exc_info=True)
            return None

    def pcm16_to_float32(self, pcm_data: bytes) -> np.ndarray:
        """Convert PCM16 audio data to float32 numpy array normalized to [-1.0, 1.0]"""
        # Convert bytes to int16 array
        int16_array = np.frombuffer(pcm_data, dtype=np.int16)

        # Convert to float32 and normalize
        float32_array = int16_array.astype(np.float32) / 32768.0

        return float32_array

    def transcribe_faster_whisper(
        self, audio_array: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Transcribe using faster-whisper"""
        segments, info = self.whisper_model.transcribe(
            audio_array,
            beam_size=5,
            language="en",  # Set to None for auto-detection
            vad_filter=True,  # Voice Activity Detection
            vad_parameters=dict(
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=1000,
            ),
        )

        # Combine all segments
        text_parts = []
        avg_confidence = 0
        segment_count = 0

        for segment in segments:
            text_parts.append(segment.text)
            avg_confidence += segment.avg_logprob
            segment_count += 1

        if segment_count == 0:
            return None

        text = " ".join(text_parts).strip()
        confidence = (
            np.exp(avg_confidence / segment_count) if segment_count > 0 else 0.0
        )

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "confidence": confidence,
            "segments": segment_count,
        }

    def transcribe_openai_whisper(
        self, audio_array: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Transcribe using openai-whisper"""
        import whisper

        result = self.whisper_model.transcribe(
            audio_array,
            language="en",
            fp16=False,  # Use FP32 for CPU
        )

        text = result["text"].strip()
        if not text:
            return None

        return {
            "text": text,
            "language": result.get("language", "en"),
            "language_probability": 1.0,
            "confidence": 0.8,  # openai-whisper doesn't provide confidence
            "segments": len(result.get("segments", [])),
        }

    def publish_result(self, audio_chunk: AudioChunk, result: Dict[str, Any]):
        """Publish STT result to RabbitMQ"""
        message = {
            "streamID": audio_chunk.stream_id,
            "chunkID": audio_chunk.chunk_id,
            "timestamp": audio_chunk.timestamp,
            "text": result["text"],
            "language": result["language"],
            "confidence": result["confidence"],
            "segments": result["segments"],
            "audioDurationMs": audio_chunk.duration_ms,
        }

        self.rabbit_channel.basic_publish(
            exchange="",
            routing_key="stt-results",
            body=json.dumps(message).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=1,  # Non-persistent
            ),
        )

    def log_stats(self):
        """Log worker statistics"""
        if self.stats["chunks_processed"] == 0:
            return

        avg_processing_time = (
            self.stats["total_processing_time"] / self.stats["chunks_processed"]
        )
        rtf = avg_processing_time / (
            self.stats["total_audio_duration"] / self.stats["chunks_processed"]
        )

        logger.info(
            f"Stats: {self.stats['chunks_processed']} chunks processed, "
            f"avg processing time: {avg_processing_time:.2f}s, "
            f"RTF: {rtf:.2f}x, "
            f"errors: {self.stats['errors']}"
        )

    def cleanup(self):
        """Cleanup connections"""
        logger.info("Cleaning up...")

        if self.rabbit_connection:
            self.rabbit_connection.close()

        if self.redis_client:
            self.redis_client.close()

        logger.info("Cleanup complete")


def main():
    """Main entry point"""
    worker = WhisperWorker()

    try:
        worker.initialize()
        worker.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        worker.cleanup()


if __name__ == "__main__":
    main()
