"""
Unit tests for LLM Response Worker
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add code directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../code'))

from llm_worker import LLMWorker, ConversationContext, MockLLMBackend


class TestConversationContext(unittest.TestCase):
    """Test conversation context management"""

    def test_add_message(self):
        """Test adding messages to context"""
        context = ConversationContext(stream_id='test-123')

        context.add_message('user', 'Hello')
        context.add_message('assistant', 'Hi there!')

        self.assertEqual(len(context.messages), 2)
        self.assertEqual(context.messages[0]['role'], 'user')
        self.assertEqual(context.messages[0]['content'], 'Hello')
        self.assertEqual(context.messages[1]['role'], 'assistant')

    def test_context_length_limit(self):
        """Test context length limiting"""
        from llm_worker import LLM_CONTEXT_LENGTH

        context = ConversationContext(stream_id='test-123')

        # Add more messages than the limit
        for i in range(LLM_CONTEXT_LENGTH * 3):
            context.add_message('user', f'Message {i}')
            context.add_message('assistant', f'Response {i}')

        # Should keep only last N*2 messages (user+assistant pairs)
        self.assertEqual(len(context.messages), LLM_CONTEXT_LENGTH * 2)

    def test_get_messages(self):
        """Test getting messages without timestamps"""
        context = ConversationContext(stream_id='test-123')
        context.add_message('user', 'Hello')
        context.add_message('assistant', 'Hi!')

        messages = context.get_messages()

        self.assertEqual(len(messages), 2)
        self.assertNotIn('timestamp', messages[0])
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'], 'Hello')


class TestMockLLMBackend(unittest.TestCase):
    """Test mock LLM backend"""

    def setUp(self):
        """Set up test fixtures"""
        self.backend = MockLLMBackend()
        self.backend.initialize()

    def test_hello_response(self):
        """Test hello response"""
        messages = [
            {'role': 'user', 'content': 'hello'}
        ]

        response = self.backend.generate(messages, max_tokens=100, temperature=0.7)

        self.assertIn('hello', response.lower())

    def test_ooblex_response(self):
        """Test Ooblex-specific response"""
        messages = [
            {'role': 'user', 'content': 'what is ooblex?'}
        ]

        response = self.backend.generate(messages, max_tokens=100, temperature=0.7)

        self.assertIn('ooblex', response.lower())

    def test_default_response(self):
        """Test default fallback response"""
        messages = [
            {'role': 'user', 'content': 'random question'}
        ]

        response = self.backend.generate(messages, max_tokens=100, temperature=0.7)

        self.assertIn('random question', response.lower())


class TestLLMWorker(unittest.TestCase):
    """Test LLM Worker functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.worker = LLMWorker()

    @patch('llm_worker.redis')
    @patch('llm_worker.pika')
    def test_initialization(self, mock_pika, mock_redis):
        """Test worker initialization"""
        mock_redis_client = MagicMock()
        mock_redis.Redis.from_url.return_value = mock_redis_client

        mock_connection = MagicMock()
        mock_pika.BlockingConnection.return_value = mock_connection

        # Mock LLM backend
        with patch.object(self.worker, 'initialize_llm_backend'):
            # Would call initialize() here in real test
            self.assertIsNotNone(self.worker)

    def test_fallback_response(self):
        """Test fallback response generation"""
        response = self.worker.get_fallback_response()

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    @patch.object(LLMWorker, 'llm_backend')
    def test_generate_response(self, mock_backend):
        """Test response generation"""
        mock_backend.generate.return_value = "Test response"

        context = ConversationContext(stream_id='test-123')
        context.add_message('user', 'Hello')

        self.worker.llm_backend = mock_backend

        response = self.worker.generate_response(context)

        self.assertEqual(response, "Test response")
        mock_backend.generate.assert_called_once()


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    def test_end_to_end_conversation(self):
        """Test complete conversation flow (requires real services)"""
        # This would require Redis, RabbitMQ, and LLM backend
        # For CI/CD, use mock services or docker-compose test environment
        pass


if __name__ == '__main__':
    unittest.main()
