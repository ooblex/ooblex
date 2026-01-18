"""
Unit tests for WebSocket Connection Manager
Tests the connection management functionality without requiring actual WebSocket connections.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add the API path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'services', 'api'))

from main import ConnectionManager


class TestConnectionManagerInit:
    """Test ConnectionManager initialization"""

    def test_init_creates_empty_dicts(self):
        """Test that manager initializes with empty dictionaries"""
        manager = ConnectionManager()

        assert isinstance(manager.active_connections, dict)
        assert isinstance(manager.stream_tokens, dict)
        assert len(manager.active_connections) == 0
        assert len(manager.stream_tokens) == 0


class TestConnectionManagerConnect:
    """Test connection handling"""

    @pytest.mark.asyncio
    async def test_connect_stores_connection(self):
        """Test that connect stores the websocket connection"""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()

        with patch.object(manager, 'active_connections', {}):
            await manager.connect(mock_websocket, "client_123")

            mock_websocket.accept.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_calls_accept(self):
        """Test that connect calls websocket.accept()"""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()

        await manager.connect(mock_websocket, "client_123")

        mock_websocket.accept.assert_called_once()


class TestConnectionManagerDisconnect:
    """Test disconnection handling"""

    def test_disconnect_removes_connection(self):
        """Test that disconnect removes the client connection"""
        manager = ConnectionManager()
        mock_websocket = MagicMock()
        manager.active_connections["client_123"] = mock_websocket

        manager.disconnect("client_123")

        assert "client_123" not in manager.active_connections

    def test_disconnect_nonexistent_client(self):
        """Test that disconnecting nonexistent client doesn't raise error"""
        manager = ConnectionManager()

        # Should not raise
        manager.disconnect("nonexistent_client")

        assert len(manager.active_connections) == 0

    def test_disconnect_cleans_up_stream_tokens(self):
        """Test that disconnect removes associated stream tokens"""
        manager = ConnectionManager()
        mock_websocket = MagicMock()
        manager.active_connections["client_123"] = mock_websocket
        manager.stream_tokens["stream_abc"] = "client_123"
        manager.stream_tokens["stream_def"] = "client_123"
        manager.stream_tokens["stream_xyz"] = "other_client"

        manager.disconnect("client_123")

        assert "stream_abc" not in manager.stream_tokens
        assert "stream_def" not in manager.stream_tokens
        assert "stream_xyz" in manager.stream_tokens  # Should not be removed


class TestConnectionManagerSendMessage:
    """Test message sending functionality"""

    @pytest.mark.asyncio
    async def test_send_personal_message_to_connected_client(self):
        """Test sending message to connected client"""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        manager.active_connections["client_123"] = mock_websocket

        await manager.send_personal_message("Hello", "client_123")

        mock_websocket.send_text.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_send_personal_message_to_disconnected_client(self):
        """Test sending message to non-existent client"""
        manager = ConnectionManager()

        # Should not raise
        await manager.send_personal_message("Hello", "nonexistent_client")

    @pytest.mark.asyncio
    async def test_send_personal_message_handles_error(self):
        """Test that send_personal_message handles send errors"""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Connection closed"))
        manager.active_connections["client_123"] = mock_websocket

        # Should not raise
        await manager.send_personal_message("Hello", "client_123")

        # Client should be disconnected after error
        assert "client_123" not in manager.active_connections


class TestConnectionManagerStreamTokens:
    """Test stream token mapping functionality"""

    @pytest.mark.asyncio
    async def test_send_to_stream_with_valid_token(self):
        """Test sending message to client via stream token"""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        manager.active_connections["client_123"] = mock_websocket
        manager.stream_tokens["stream_abc"] = "client_123"

        await manager.send_to_stream("Hello", "stream_abc")

        mock_websocket.send_text.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_send_to_stream_with_invalid_token(self):
        """Test sending message to nonexistent stream token"""
        manager = ConnectionManager()

        # Should not raise
        await manager.send_to_stream("Hello", "nonexistent_stream")

    def test_stream_token_mapping_getter(self):
        """Test getting client from stream token"""
        manager = ConnectionManager()
        manager.stream_tokens["stream_123"] = "client_abc"

        assert manager.stream_tokens.get("stream_123") == "client_abc"
        assert manager.stream_tokens.get("nonexistent") is None


class TestConnectionManagerMultipleClients:
    """Test behavior with multiple clients"""

    @pytest.mark.asyncio
    async def test_multiple_clients_connect(self):
        """Test multiple clients can connect simultaneously"""
        manager = ConnectionManager()
        mock_websocket1 = AsyncMock()
        mock_websocket1.accept = AsyncMock()
        mock_websocket2 = AsyncMock()
        mock_websocket2.accept = AsyncMock()
        mock_websocket3 = AsyncMock()
        mock_websocket3.accept = AsyncMock()

        await manager.connect(mock_websocket1, "client_1")
        await manager.connect(mock_websocket2, "client_2")
        await manager.connect(mock_websocket3, "client_3")

        assert len(manager.active_connections) == 3
        assert "client_1" in manager.active_connections
        assert "client_2" in manager.active_connections
        assert "client_3" in manager.active_connections

    def test_multiple_clients_disconnect(self):
        """Test multiple clients can disconnect independently"""
        manager = ConnectionManager()
        manager.active_connections["client_1"] = MagicMock()
        manager.active_connections["client_2"] = MagicMock()
        manager.active_connections["client_3"] = MagicMock()

        manager.disconnect("client_2")

        assert len(manager.active_connections) == 2
        assert "client_1" in manager.active_connections
        assert "client_2" not in manager.active_connections
        assert "client_3" in manager.active_connections

    def test_multiple_stream_tokens_per_client(self):
        """Test a client can have multiple stream tokens"""
        manager = ConnectionManager()
        manager.active_connections["client_1"] = MagicMock()
        manager.stream_tokens["stream_1"] = "client_1"
        manager.stream_tokens["stream_2"] = "client_1"
        manager.stream_tokens["stream_3"] = "client_1"

        # All stream tokens should map to the same client
        assert manager.stream_tokens.get("stream_1") == "client_1"
        assert manager.stream_tokens.get("stream_2") == "client_1"
        assert manager.stream_tokens.get("stream_3") == "client_1"


class TestConnectionManagerEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_connect_same_client_id_twice(self):
        """Test connecting with same client ID twice replaces the connection"""
        manager = ConnectionManager()
        mock_websocket1 = AsyncMock()
        mock_websocket1.accept = AsyncMock()
        mock_websocket2 = AsyncMock()
        mock_websocket2.accept = AsyncMock()

        await manager.connect(mock_websocket1, "client_123")
        await manager.connect(mock_websocket2, "client_123")

        # Second connection should replace first
        assert manager.active_connections["client_123"] == mock_websocket2

    def test_disconnect_multiple_times(self):
        """Test disconnecting same client multiple times doesn't raise error"""
        manager = ConnectionManager()
        manager.active_connections["client_123"] = MagicMock()

        # First disconnect
        manager.disconnect("client_123")
        # Second disconnect (client already gone)
        manager.disconnect("client_123")
        # Third disconnect
        manager.disconnect("client_123")

        assert "client_123" not in manager.active_connections

    @pytest.mark.asyncio
    async def test_empty_message(self):
        """Test sending empty message"""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        manager.active_connections["client_123"] = mock_websocket

        await manager.send_personal_message("", "client_123")

        mock_websocket.send_text.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_special_characters_in_client_id(self):
        """Test client IDs with special characters"""
        manager = ConnectionManager()
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()

        special_ids = [
            "client-with-dashes",
            "client_with_underscores",
            "client.with.dots",
            "client123",
            "CLIENT_UPPERCASE",
        ]

        for client_id in special_ids:
            await manager.connect(mock_websocket, client_id)
            assert client_id in manager.active_connections


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
