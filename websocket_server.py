"""
================================================================================
NeuralStride AI - WebSocket Live Broadcaster
================================================================================

Runs a WebSocket server alongside main.py that broadcasts landmark data
in real-time to any connected browser client.

HOW IT WORKS:
    1. When created, starts a WebSocket server on localhost:8765 in a
       background thread (asyncio event loop)
    2. A 3JS person opens an HTML page in their browser that connects to
       ws://localhost:8765
    3. On connection, the server sends a metadata message (landmark names,
       skeleton connections) — same format as the JSONL header
    4. As the workout runs, main.py calls update() with each frame's JSON data
    5. The server broadcasts the latest frame to ALL connected clients at ~30fps
    6. The browser page receives frame-by-frame JSON and updates the 3D skeleton

THREADING:
    The WebSocket server runs in its own daemon thread with its own asyncio
    event loop. This is necessary because main.py's loop is synchronous
    (OpenCV cv2.waitKey) and cannot run asyncio natively.

    Communication between main thread and WebSocket thread:
        - main.py calls update(json_string) -> stores in self.latest_data
        - WebSocket thread reads self.latest_data -> broadcasts to clients
        - Thread-safe because GIL protects simple attribute assignment

MESSAGES SENT TO CLIENTS:
    1. On connect (one-time):
        {
            "type": "metadata",
            "landmark_names": { "0": "nose", "11": "left_shoulder", ... },
            "connections": [[0,1], [1,2], ...],
            "coordinate_system": { ... }
        }

    2. Every frame (repeated):
        {
            "type": "frame",
            "frame": 1,
            "timestamp": 0.033,
            "landmarks": {
                "0": { "x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.95 },
                ...
            }
        }

CLIENT-SIDE (for 3JS person):
    const ws = new WebSocket("ws://localhost:8765");
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "metadata") {
            // Save landmark names and connections, build skeleton
        } else if (data.type === "frame") {
            // Update 3D sphere positions using data.landmarks
        }
    };

USAGE (from main.py):
    broadcaster = WebSocketBroadcaster(host="localhost", port=8765)
    # ... inside the main loop ...
    broadcaster.update(frame_json_string)
    # ... after workout ends ...
    broadcaster.close()

DEPENDENCIES:
    pip install websockets
================================================================================
"""

import json
import asyncio
import threading

import websockets

from landmark_exporter import LANDMARK_NAMES, POSE_CONNECTIONS


METADATA = {
    "type": "metadata",
    "format": "NeuralStride WebSocket Stream v1",
    "landmark_count": 33,
    "landmark_names": LANDMARK_NAMES,
    "connections": POSE_CONNECTIONS,
    "coordinate_system": {
        "x": "normalized 0-1, left to right",
        "y": "normalized 0-1, top to bottom",
        "z": "depth, negative = closer to camera",
        "visibility": "0-1 confidence"
    }
}


class WebSocketBroadcaster:
    """
    WebSocket server that broadcasts landmark data to connected browser clients.

    Runs in a background thread. Call update() from the main thread with
    each frame's JSON string to broadcast it to all connected clients.
    """

    def __init__(self, host="localhost", port=8765):
        """
        Start the WebSocket server on a background thread.

        Args:
            host: Hostname to bind to. Default: "localhost" (local connections only).
            port: Port number. Default: 8765.
                  The 3JS person connects to ws://localhost:8765
        """
        self.host = host
        self.port = port
        self.clients = set()
        self.latest_data = None
        self.loop = None
        self.running = True

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        print(f"WebSocket server started: ws://{host}:{port}", flush=True)

    async def _start(self):
        """
        Coroutine that starts the WebSocket server and runs forever.
        Called via loop.run_until_complete() from the background thread.
        """
        async with websockets.serve(self._handler, self.host, self.port):
            # Keep the server running until self.running is set to False
            while self.running:
                await asyncio.sleep(1)

    def _run(self):
        """
        Background thread entry point.
        Creates an asyncio event loop and runs the WebSocket server.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._start())

    async def _handler(self, websocket):
        """
        Handles a single client connection.

        Flow:
            1. Send metadata immediately on connect (landmark names, connections)
            2. Add client to the broadcast set
            3. Loop: send latest frame data every ~33ms (30fps)
            4. Remove client when they disconnect
        """
        try:
            # Send metadata as the first message so the client knows
            # landmark names and skeleton connections before any frames arrive
            await websocket.send(json.dumps(METADATA))
            self.clients.add(websocket)

            # Broadcast loop: send latest frame data to this client at ~30fps
            while self.running:
                if self.latest_data:
                    await websocket.send(self.latest_data)
                # Sleep for ~1 frame duration (33ms at 30fps)
                await asyncio.sleep(0.033)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)

    def update(self, json_string):
        """
        Set the latest frame data to broadcast to all connected clients.

        Called from main.py every frame with the same JSON string
        that the landmark exporter builds for the JSONL file.
        No re-serialization needed — both file and WebSocket share the same JSON.

        Thread-safe: GIL protects this simple string assignment.

        Args:
            json_string: JSON string with frame data
                         e.g. '{"frame": 1, "timestamp": 0.033, "landmarks": {...}}'
        """
        self.latest_data = json_string

    def close(self):
        """
        Stop the WebSocket server and disconnect all clients.
        Called when the workout ends (after 'q' is pressed).
        """
        self.running = False
        # Close all client connections
        for client in list(self.clients):
            try:
                client.close()
            except Exception:
                pass
        self.clients.clear()
