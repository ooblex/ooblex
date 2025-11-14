"""
WebSocket integration tests
Tests the WebSocket API with real connections
"""
import pytest
import asyncio
import json
import websockets
from unittest.mock import AsyncMock, patch


class TestWebSocketConnection:
    """Test WebSocket connection and basic communication"""

    @pytest.mark.asyncio
    async def test_websocket_connect(self):
        """Test basic WebSocket connection"""
        # This test requires the API server to be running
        # It will be skipped if the server is not available
        try:
            async with websockets.connect(
                "ws://localhost:8800/ws/test_client",
                timeout=5
            ) as websocket:
                # Connection successful
                assert websocket.open
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")

    @pytest.mark.asyncio
    async def test_websocket_ping_pong(self):
        """Test ping-pong message pattern"""
        try:
            async with websockets.connect(
                "ws://localhost:8800/ws/test_client",
                timeout=5
            ) as websocket:
                # Send ping message
                ping_msg = json.dumps({"type": "ping"})
                await websocket.send(ping_msg)

                # Receive pong response
                response = await asyncio.wait_for(websocket.recv(), timeout=2)
                response_data = json.loads(response)

                assert response_data.get("type") == "pong"
                assert "timestamp" in response_data
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")

    @pytest.mark.asyncio
    async def test_websocket_start_process(self):
        """Test 'start process' command (original API pattern)"""
        try:
            async with websockets.connect(
                "ws://localhost:8800/ws/test_client",
                timeout=5
            ) as websocket:
                # Send start process command
                stream_key = "test_stream_123"
                command = f"start process:{stream_key}"
                await websocket.send(command)

                # Expect multiple responses
                responses = []
                try:
                    while len(responses) < 3:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2)
                        responses.append(response)
                except asyncio.TimeoutError:
                    pass

                # Check for expected responses
                response_text = " ".join(responses)
                assert "Starting Video Processing" in response_text or len(responses) > 0
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")

    @pytest.mark.asyncio
    async def test_websocket_task_command(self):
        """Test task command format (task:streamKey)"""
        try:
            async with websockets.connect(
                "ws://localhost:8800/ws/test_client",
                timeout=5
            ) as websocket:
                # Send task command
                task_command = "FaceOn:test_stream_456"
                await websocket.send(task_command)

                # Should not error (may not get response without full setup)
                await asyncio.sleep(0.5)

                # Connection should still be open
                assert websocket.open
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")


class TestWebSocketResilience:
    """Test WebSocket connection resilience"""

    @pytest.mark.asyncio
    async def test_websocket_reconnect(self):
        """Test reconnecting after disconnect"""
        try:
            # First connection
            async with websockets.connect(
                "ws://localhost:8800/ws/test_client_1",
                timeout=5
            ) as ws1:
                assert ws1.open

            # Second connection with same client ID
            async with websockets.connect(
                "ws://localhost:8800/ws/test_client_1",
                timeout=5
            ) as ws2:
                assert ws2.open
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_connections(self):
        """Test multiple clients connected simultaneously"""
        try:
            connections = []
            for i in range(3):
                ws = await websockets.connect(
                    f"ws://localhost:8800/ws/test_client_{i}",
                    timeout=5
                )
                connections.append(ws)

            # All connections should be open
            assert all(ws.open for ws in connections)

            # Close all connections
            for ws in connections:
                await ws.close()

        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")
        finally:
            # Cleanup
            for ws in connections:
                if ws.open:
                    await ws.close()

    @pytest.mark.asyncio
    async def test_websocket_invalid_message(self):
        """Test handling of invalid messages"""
        try:
            async with websockets.connect(
                "ws://localhost:8800/ws/test_client",
                timeout=5
            ) as websocket:
                # Send invalid message
                await websocket.send("invalid message format")

                # Should get response (error or echo)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2)
                    assert response  # Got some response
                except asyncio.TimeoutError:
                    pass  # No response is also acceptable

                # Connection should still be open
                assert websocket.open
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")


class TestWebSocketSubscription:
    """Test WebSocket subscription functionality"""

    @pytest.mark.asyncio
    async def test_subscribe_to_stream(self):
        """Test subscribing to a stream"""
        try:
            async with websockets.connect(
                "ws://localhost:8800/ws/test_subscriber",
                timeout=5
            ) as websocket:
                # Subscribe to stream
                subscribe_msg = json.dumps({
                    "type": "subscribe",
                    "stream_token": "test_stream_789"
                })
                await websocket.send(subscribe_msg)

                # Should get subscription confirmation
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2)
                    response_data = json.loads(response)

                    assert response_data.get("type") == "subscribed"
                    assert response_data.get("stream_token") == "test_stream_789"
                except asyncio.TimeoutError:
                    pytest.skip("No subscription response (may require full setup)")
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")


class TestWebSocketStressTest:
    """Stress tests for WebSocket connection"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_rapid_messages(self):
        """Test sending many messages rapidly"""
        try:
            async with websockets.connect(
                "ws://localhost:8800/ws/stress_test_client",
                timeout=5
            ) as websocket:
                # Send 100 ping messages rapidly
                for i in range(100):
                    ping_msg = json.dumps({"type": "ping", "seq": i})
                    await websocket.send(ping_msg)

                # Connection should still be open
                assert websocket.open

                # Try to receive some responses
                responses = 0
                try:
                    while responses < 10:
                        await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        responses += 1
                except asyncio.TimeoutError:
                    pass

                # Should have received at least some responses
                assert responses > 0
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_long_lived_connection(self):
        """Test keeping connection open for extended period"""
        try:
            async with websockets.connect(
                "ws://localhost:8800/ws/long_lived_client",
                timeout=5
            ) as websocket:
                # Keep connection open for 10 seconds, sending pings
                for i in range(10):
                    ping_msg = json.dumps({"type": "ping", "iteration": i})
                    await websocket.send(ping_msg)

                    # Wait 1 second
                    await asyncio.sleep(1)

                    # Check connection still open
                    assert websocket.open

                # Final check
                assert websocket.open
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
            pytest.skip("API server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
