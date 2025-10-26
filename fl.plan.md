<!-- 8232e0bf-a08d-46ec-b1b3-e3381d476862 b2660956-13f4-414e-82da-6454e475f6ff -->
# Bidirectional Audio Streaming for Active Sessions (WebSocket + PCM16)

### Key choices

- Transport: WebSocket (one connection per session) for bidirectional audio; simpler than WebRTC and enough for demo.
- Codec/format: PCM16, mono, 16 kHz, 20–40 ms frames. Payload = little-endian Int16 bytes. Optional JSON control messages.
- Flow: When a session is started, the app opens `ws://localhost:2025/ws/sessions/{id}`. The app streams mic frames → backend. Backend streams audio frames back (for now loopback/echo to validate pipeline; can be replaced with TTS/LLM output later).

### Backend changes

- Add dependency: `flask-sock` (and `simple-websocket`).
- Files:
  - `/Users/vs/Coding/janus-ai/backend/app/ws.py`: WebSocket setup and helpers
  - `/Users/vs/Coding/janus-ai/backend/app/routes/ws_sessions.py`: `@sock.route('/ws/sessions/<session_id>')` handler
- Behavior:
  - Accept binary frames (PCM16) from client; push into queue.
  - For MVP, echo frames back immediately (simple low-latency loopback) and broadcast status JSON periodically.
  - Close on `stop` message or socket close.

### iOS changes

- Files:
  - `swift-app/janusai/AudioStreamingClient.swift`: WebSocket client using `URLSessionWebSocketTask`
  - Wire `SessionRunningView` to:
    - Start: connect, start mic capture (AVAudioEngine), chunk to 20–40 ms PCM16 @16k, send frames.
    - Receive: read binary frames, schedule on `AVAudioPlayerNode`, compute RMS to drive `WaveformView`.
    - Stop: close socket, stop capture, stop player, call `/api/sessions/{id}/stop`.

### Protocol

- WebSocket Binary: raw PCM16, mono, 16 kHz, frame size: 320 samples (20 ms) or 640 (40 ms) → 640 or 1280 bytes.
- WebSocket Text JSON (control):
  - Client → server `{ "type": "start", "sampleRate": 16000 }`, `{ "type": "stop" }`
  - Server → client `{ "type": "status", "state": "running" }`

### Essential snippets

- Backend ws route (sketch):
  ```python
  from flask import current_app
  from flask_sock import Sock
  from .routes.sessions import _session_path
  
  sock = Sock()
  
  @sock.route('/ws/sessions/<session_id>')
  def session_stream(ws, session_id):
      # Optional: validate session exists and is running
      while True:
          msg = ws.receive()
          if msg is None:
              break
          if isinstance(msg, str):
              # handle control JSON
              continue
          # binary: echo back for MVP
          ws.send(msg)
  ```

- iOS capture (sketch of per-buffer send):
  ```swift
  // in AudioStreamingClient
  input.installTap(onBus: 0, bufferSize: 1024, format: format) { buf, _ in
      let pcm16 = Self.convertToPCM16Mono16k(buf)
      self.sendBinary(pcm16)
  }
  ```


### Navigation/UX

- `NewSessionView` Start Session: on success, navigate to `SessionRunningView(sessionId: ...)`.
- `SessionRunningView` is non-dismissible; “Stop” ends stream and calls stop endpoint.

### Testing path

1) Start backend, open app, create and start a session.

2) Speak: you should hear echo playback from backend; waveform animates.

3) Press Stop: socket closes, engine stops, returns to previous screen.

### Future swap-in

- Replace echo with TTS/LLM output (e.g., push TTS PCM chunks to same socket from backend worker) without changing client.

### To-dos

- [ ] Add flask-sock dependency and initialize Sock in app
- [ ] Create /ws/sessions/<id> route to echo audio frames
- [ ] Add AudioStreamingClient with WebSocket + AVAudioEngine
- [ ] Wire SessionRunningView to start/stop streaming with sessionId
- [ ] Manual test echo loop and waveform updates