"""
LLM Response Worker for Ooblex NinjaSDK Integration

This worker demonstrates the cascading nature of Ooblex processing by consuming
speech-to-text results and generating AI responses using a local LLM.

The responses are sent back to clients via WebRTC data channels, creating a
complete voice-to-AI-to-voice pipeline.

Features:
- Local LLM integration (Ollama, llama.cpp, transformers)
- Context-aware conversation management
- Streaming responses (optional)
- Multiple backend support
- Conversation history tracking
- Fallback responses

Pipeline:
RabbitMQ (STT results) → This Worker → LLM → RabbitMQ (responses) → Client
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

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

# LLM Configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")  # ollama, transformers, llamacpp, mock
LLM_MODEL = os.getenv("LLM_MODEL", "llama2")  # Model name depends on backend
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "256"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_CONTEXT_LENGTH = int(os.getenv("LLM_CONTEXT_LENGTH", "5"))  # Keep last N messages

# System prompt
SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are a helpful AI assistant integrated into Ooblex, a real-time video processing platform. "
    "Provide concise, friendly responses to user questions. Keep responses brief (1-2 sentences) "
    "as they will be spoken back to the user.",
)


@dataclass
class ConversationContext:
    """Conversation context for a stream"""

    stream_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    def add_message(self, role: str, content: str):
        """Add message to context"""
        self.messages.append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )
        self.last_activity = datetime.now()

        # Keep only last N messages
        if len(self.messages) > LLM_CONTEXT_LENGTH * 2:  # *2 for user+assistant pairs
            self.messages = self.messages[-LLM_CONTEXT_LENGTH * 2 :]

    def get_messages(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]


class LLMWorker:
    """
    LLM Response Worker

    Processes speech-to-text results and generates AI responses to demonstrate
    the cascading processing capabilities of Ooblex.
    """

    def __init__(self):
        self.redis_client = None
        self.rabbit_connection = None
        self.rabbit_channel = None
        self.llm_backend = None
        self.conversations = defaultdict(ConversationContext)
        self.stats = {
            "messages_processed": 0,
            "responses_generated": 0,
            "total_llm_time": 0,
            "errors": 0,
        }

    def initialize(self):
        """Initialize connections and LLM backend"""
        logger.info("Initializing LLM Response Worker...")

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
        self.rabbit_channel.queue_declare(queue="stt-results", durable=False)
        self.rabbit_channel.queue_declare(queue="llm-responses", durable=False)
        logger.info("RabbitMQ connected")

        # Initialize LLM backend
        logger.info(f"Initializing LLM backend: {LLM_BACKEND}...")
        self.initialize_llm_backend()
        logger.info("LLM backend initialized")

        logger.info("LLM Response Worker ready!")

    def initialize_llm_backend(self):
        """Initialize the selected LLM backend"""
        if LLM_BACKEND == "ollama":
            self.llm_backend = OllamaBackend(LLM_MODEL)
        elif LLM_BACKEND == "transformers":
            self.llm_backend = TransformersBackend(LLM_MODEL)
        elif LLM_BACKEND == "llamacpp":
            self.llm_backend = LlamaCppBackend(LLM_MODEL)
        elif LLM_BACKEND == "mock":
            self.llm_backend = MockLLMBackend()
        else:
            logger.error(f"Unknown LLM backend: {LLM_BACKEND}")
            sys.exit(1)

        self.llm_backend.initialize()

    def start_consuming(self):
        """Start consuming STT results from RabbitMQ"""
        logger.info("Starting to consume STT results...")

        def callback(ch, method, properties, body):
            try:
                self.process_stt_result(body)
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as e:
                logger.error(f"Error processing STT result: {e}", exc_info=True)
                self.stats["errors"] += 1
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.rabbit_channel.basic_qos(prefetch_count=1)
        self.rabbit_channel.basic_consume(
            queue="stt-results", on_message_callback=callback, auto_ack=False
        )

        logger.info("Waiting for STT results...")
        try:
            self.rabbit_channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
            self.rabbit_channel.stop_consuming()

    def process_stt_result(self, message_body: bytes):
        """Process a single STT result"""
        start_time = time.time()

        # Parse message
        message = json.loads(message_body.decode("utf-8"))
        stream_id = message["streamID"]
        text = message["text"].strip()
        confidence = message.get("confidence", 0.0)

        logger.info(f'STT from {stream_id}: "{text}" (confidence: {confidence:.2f})')

        # Skip low-confidence or empty transcriptions
        if not text or confidence < 0.3:
            logger.debug(f"Skipping low-confidence transcription: {confidence}")
            return

        self.stats["messages_processed"] += 1

        # Get or create conversation context
        if stream_id not in self.conversations:
            self.conversations[stream_id] = ConversationContext(stream_id=stream_id)

        context = self.conversations[stream_id]
        context.add_message("user", text)

        # Generate LLM response
        response = self.generate_response(context)

        if response:
            context.add_message("assistant", response)

            # Publish response
            self.publish_response(stream_id, text, response)

            llm_time = time.time() - start_time
            self.stats["responses_generated"] += 1
            self.stats["total_llm_time"] += llm_time

            logger.info(f'LLM Response to {stream_id}: "{response}" ({llm_time:.2f}s)')

            # Log stats periodically
            if self.stats["responses_generated"] % 10 == 0:
                self.log_stats()

    def generate_response(self, context: ConversationContext) -> Optional[str]:
        """Generate LLM response"""
        try:
            # Build conversation history
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + context.get_messages()

            # Generate response
            response = self.llm_backend.generate(
                messages, max_tokens=LLM_MAX_TOKENS, temperature=LLM_TEMPERATURE
            )

            return response.strip()

        except Exception as e:
            logger.error(f"LLM generation error: {e}", exc_info=True)
            return self.get_fallback_response()

    def get_fallback_response(self) -> str:
        """Return a fallback response on error"""
        fallbacks = [
            "I'm having trouble processing that right now.",
            "Could you please rephrase that?",
            "I didn't quite catch that.",
        ]
        import random

        return random.choice(fallbacks)

    def publish_response(self, stream_id: str, user_text: str, response: str):
        """Publish LLM response to RabbitMQ"""
        message = {
            "streamID": stream_id,
            "timestamp": int(time.time() * 1000),
            "userText": user_text,
            "response": response,
            "type": "llm_response",
        }

        self.rabbit_channel.basic_publish(
            exchange="",
            routing_key="llm-responses",
            body=json.dumps(message).encode("utf-8"),
            properties=pika.BasicProperties(
                delivery_mode=1,  # Non-persistent
            ),
        )

    def log_stats(self):
        """Log worker statistics"""
        if self.stats["responses_generated"] == 0:
            return

        avg_llm_time = self.stats["total_llm_time"] / self.stats["responses_generated"]

        logger.info(
            f"Stats: {self.stats['messages_processed']} messages, "
            f"{self.stats['responses_generated']} responses, "
            f"avg LLM time: {avg_llm_time:.2f}s, "
            f"active conversations: {len(self.conversations)}, "
            f"errors: {self.stats['errors']}"
        )

    def cleanup(self):
        """Cleanup connections"""
        logger.info("Cleaning up...")

        if self.rabbit_connection:
            self.rabbit_connection.close()

        if self.redis_client:
            self.redis_client.close()

        if self.llm_backend:
            self.llm_backend.cleanup()

        logger.info("Cleanup complete")


# ============================================================================
# LLM Backend Implementations
# ============================================================================


class BaseLLMBackend:
    """Base class for LLM backends"""

    def initialize(self):
        """Initialize backend"""
        pass

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """Generate response"""
        raise NotImplementedError

    def cleanup(self):
        """Cleanup resources"""
        pass


class OllamaBackend(BaseLLMBackend):
    """Ollama LLM backend"""

    def __init__(self, model: str):
        self.model = model
        self.client = None

    def initialize(self):
        """Initialize Ollama client"""
        try:
            import ollama

            self.client = ollama.Client()
            # Test connection
            self.client.list()
            logger.info(f"Ollama backend initialized with model: {self.model}")
        except ImportError:
            logger.error(
                "ollama-python not installed! Install with: pip install ollama"
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running: ollama serve")
            sys.exit(1)

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """Generate response using Ollama"""
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        )
        return response["message"]["content"]


class TransformersBackend(BaseLLMBackend):
    """Hugging Face Transformers backend"""

    def __init__(self, model: str):
        self.model = model
        self.pipeline = None

    def initialize(self):
        """Initialize transformers pipeline"""
        try:
            import torch
            from transformers import pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading transformers model: {self.model} on {device}")

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                device=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            logger.info("Transformers backend initialized")
        except ImportError:
            logger.error(
                "transformers not installed! Install with: pip install transformers torch"
            )
            sys.exit(1)

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """Generate response using transformers"""
        # Convert messages to prompt
        prompt = self.messages_to_prompt(messages)

        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            return_full_text=False,
        )

        return outputs[0]["generated_text"]

    def messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to a single prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            prompt_parts.append(f"{role}: {content}")
        return "\n".join(prompt_parts) + "\nASSISTANT: "


class LlamaCppBackend(BaseLLMBackend):
    """llama-cpp-python backend"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.llama = None

    def initialize(self):
        """Initialize llama.cpp"""
        try:
            from llama_cpp import Llama

            logger.info(f"Loading llama.cpp model: {self.model_path}")
            self.llama = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
            )
            logger.info("llama.cpp backend initialized")
        except ImportError:
            logger.error(
                "llama-cpp-python not installed! Install with: pip install llama-cpp-python"
            )
            sys.exit(1)

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """Generate response using llama.cpp"""
        response = self.llama.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["message"]["content"]


class MockLLMBackend(BaseLLMBackend):
    """Mock LLM backend for testing"""

    def initialize(self):
        logger.info("Mock LLM backend initialized (for testing only)")

    def generate(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """Generate mock response"""
        user_message = next((m["content"] for m in messages if m["role"] == "user"), "")

        responses = {
            "hello": "Hello! How can I help you today?",
            "how are you": "I'm doing great, thank you for asking!",
            "what is ooblex": "Ooblex is a real-time AI video processing platform with low-latency streaming.",
            "thank you": "You're welcome!",
        }

        # Find matching response
        user_lower = user_message.lower()
        for key, response in responses.items():
            if key in user_lower:
                return response

        # Default response
        return f"I heard you say: '{user_message}'. This is a mock LLM response for testing."


def main():
    """Main entry point"""
    worker = LLMWorker()

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
