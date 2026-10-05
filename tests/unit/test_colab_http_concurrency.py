"""Exercise production handlers and stdlib dispatch without opening sockets.

Run with OOBLEX_DEMO_SOURCE pointing at the exact demo source under review.
The in-memory server replaces only socket transport and the accept loop.
HTTP parsing, application handlers, and HTTPServer/ThreadingHTTPServer request
dispatch remain real. Image decoding is outside this server-concurrency test.
"""

import ast
import http.server
import io
import json
import logging
import os
from pathlib import Path
import threading
import time
import types
import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch


SOURCE = Path(os.environ.get("OOBLEX_DEMO_SOURCE", Path(__file__).resolve().parents[2] / "colab/ooblex_demo.py"))
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {"DemoConfig", "OoblexDemo"}]
namespace = {
    "dataclass": dataclass,
    "Optional": Optional,
    "np": types.SimpleNamespace(ndarray=object),
    "threading": threading,
    "time": time,
    "logger": logging.getLogger("concurrency-test"),
    "CUDA_AVAILABLE": False,
    "MEDIAPIPE_AVAILABLE": False,
}
exec(compile(tree, str(SOURCE), "exec"), namespace)
DemoConfig = namespace["DemoConfig"]
OoblexDemo = namespace["OoblexDemo"]


class Connection:
    def __init__(self, path, body=None):
        method = "POST" if body is not None else "GET"
        payload = body.encode() if body is not None else b""
        headers = f"{method} {path} HTTP/1.0\r\nHost: localhost\r\n"
        if body is not None:
            headers += f"Content-Length: {len(payload)}\r\n"
        self.input = io.BytesIO(headers.encode() + b"\r\n" + payload)
        self.output = bytearray()
        self.headers_written = threading.Event()
        self.closed = threading.Event()

    def makefile(self, mode, buffering=None):
        return self.input

    def sendall(self, data):
        self.output.extend(data)
        if b"\r\n\r\n" in self.output:
            self.headers_written.set()


class ConcurrencyTests(unittest.TestCase):
    def exercise(self, initial_frame, stream_count=1):
        demo = OoblexDemo.__new__(OoblexDemo)
        demo.config = DemoConfig()
        demo.running = True
        demo._lock = threading.Lock()
        demo.processed_frame = initial_frame
        demo.current_effect = "none"
        demo.fps = 0
        demo.processor = types.SimpleNamespace(get_available_effects=lambda: {"none": "Original", "mirror": "Mirror"})
        received = threading.Event()
        captured = []

        def receive_frame(body):
            captured.append(body)
            with demo._lock:
                demo.processed_frame = b"NEW_JPEG_FRAME"
            received.set()

        demo.receive_frame = receive_frame
        streams = [Connection("/stream.mjpg") for _ in range(stream_count)]
        post = Connection("/frame", "synthetic-jpeg-input")
        effect = Connection("/set_effect/mirror")
        status = Connection("/status")
        snapshot = Connection("/snapshot.jpg")
        effects = Connection("/effects")
        requests = streams + [post, effect, status, snapshot, effects]
        dispatcher_done = threading.Event()
        errors = []

        class TransportOnly:
            def __init__(self, address, handler):
                # No socket is created or bound. Preserve the production class's
                # process_request implementation and real HTTP request handler.
                self.RequestHandlerClass = handler
                self.server_address = address
                self.server_name = "localhost"
                self.server_port = address[1]

            def serve_forever(self):
                try:
                    for connection in requests:
                        self.process_request(connection, ("127.0.0.1", 1))
                        if connection not in streams:
                            connection.closed.wait(1)
                    dispatcher_done.set()
                except BaseException as exc:
                    errors.append(exc)

            def shutdown_request(self, request):
                request.closed.set()

            def handle_error(self, request, address):
                import sys
                errors.append(sys.exception())

        class SerialTransport(TransportOnly, http.server.HTTPServer):
            pass

        class ThreadedTransport(TransportOnly, http.server.ThreadingHTTPServer):
            pass

        with patch.object(http.server, "HTTPServer", SerialTransport), patch.object(http.server, "ThreadingHTTPServer", ThreadedTransport):
            thread = threading.Thread(target=demo._start_http_server, daemon=True)
            thread.start()
            try:
                self.assertTrue(streams[0].headers_written.wait(1), "stream handler did not start")
                self.assertTrue(received.wait(0.5), "active MJPEG stream blocks incoming /frame POST")
                for request in [post, effect, status, snapshot, effects]:
                    self.assertTrue(request.closed.wait(1), "control request blocked by stream")
                    self.assertIn(b"200 OK", request.output)
                self.assertEqual(captured, ["synthetic-jpeg-input"])
                self.assertEqual(demo.current_effect, "mirror")
                self.assertEqual(json.loads(bytes(status.output).split(b"\r\n\r\n", 1)[1])["running"], True)
                self.assertIn(b"NEW_JPEG_FRAME", snapshot.output)
                self.assertIn(b"Original", effects.output)
                for stream in streams:
                    self.assertTrue(stream.headers_written.wait(1))
                    self.assertFalse(stream.closed.is_set(), "stream must remain active while POST and controls finish")
                self.assertTrue(dispatcher_done.wait(1))
                self.assertEqual(errors, [])
            finally:
                demo.running = False
                thread.join(2)
                for stream in streams:
                    stream.closed.wait(1)
                self.assertFalse(thread.is_alive(), "test dispatcher did not terminate")

    def test_first_frame_arrives_while_empty_stream_is_open(self):
        self.exercise(None)

    def test_new_frames_and_controls_work_while_stream_is_open(self):
        self.exercise(b"OLD_JPEG_FRAME")

    def test_two_stream_readers_do_not_block_ingestion(self):
        self.exercise(b"OLD_JPEG_FRAME", stream_count=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
