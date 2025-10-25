import json
import os
import simpleaudio as sa
from collections import deque
from ..ws import sock
from .sessions import _session_path


def register_ws_routes():
    # No-op function to ensure module import and route registration via decorators
    pass


@sock.route('/ws/sessions/<session_id>')
def session_stream(ws, session_id):
    # Validate session exists (best-effort)
    try:
        if not os.path.exists(_session_path(session_id)):
            ws.send(json.dumps({"type": "error", "error": "session not found"}))
            return
    except Exception:
        pass

    # Optional: initial status
    try:
        ws.send(json.dumps({"type": "status", "state": "running", "sessionId": session_id}))
        print(f"[ws_sessions] WS connected for session {session_id}")
    except Exception:
        # If client closed immediately
        return

    # Retain recent PlayObject references to avoid premature GC
    recent_play_objects = deque(maxlen=64)
    frames = 0

    # Stream loop: text frames are control, binary frames are PCM16 played on server
    while True:
        try:
            msg = ws.receive()
        except Exception:
            break
        if msg is None:
            break
        if isinstance(msg, str):
            # Handle control JSON messages
            try:
                data = json.loads(msg)
                if isinstance(data, dict) and data.get("type") == "stop":
                    break
            except Exception:
                # Ignore malformed
                pass
            continue
        # Binary: play on server speakers (mono, 16-bit, 16 kHz)
        try:
            play_obj = sa.play_buffer(msg, num_channels=1, bytes_per_sample=2, sample_rate=16000)
            recent_play_objects.append(play_obj)
            frames += 1
            if frames % 50 == 0:
                print(f"[ws_sessions] received and played {frames} frames for session {session_id}")
        except Exception:
            # If playback fails, continue receiving
            pass


