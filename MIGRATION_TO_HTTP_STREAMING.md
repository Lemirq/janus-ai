# Migration from WebSocket to HTTP Streaming

This document summarizes the changes made to replace WebSocket communication with simple HTTP streaming.

## Changes Made

### Backend (Flask/Python)

#### 1. Removed WebSocket Dependencies

- **File**: `backend/pyproject.toml`
- Removed `flask-sock` and `simple-websocket` dependencies

#### 2. Deleted WebSocket Files

- **Deleted**: `backend/app/ws.py` - WebSocket initialization module
- **Deleted**: `backend/app/routes/ws_sessions.py` - WebSocket session handler

#### 3. Created HTTP Streaming Module

- **File**: `backend/app/routes/stream_sessions.py` (NEW)
- **Endpoints**:
  - `POST /api/sessions/<session_id>/upload_audio` - Receives PCM16 audio chunks from client
  - `GET /api/sessions/<session_id>/stream_audio` - Streams audio back to client as WAV
  - `POST /api/sessions/<session_id>/stop_stream` - Stops the audio stream

#### 4. Updated App Initialization

- **File**: `backend/app/__init__.py`
- Removed WebSocket initialization (`sock.init_app(app)`)
- Removed WebSocket route registration
- Added HTTP streaming blueprint registration

### Frontend (Swift/iOS)

#### 1. Replaced AudioStreamingClient

- **File**: `swift-app/janusai/AudioStreamingClient.swift`
- **Changes**:
  - Removed all WebSocket code (`URLSessionWebSocketTask`, `webSocket` property)
  - Added HTTP upload using `URLSession.uploadTask` for sending audio chunks
  - Added HTTP download using `URLSessionDataTask` for receiving streamed audio
  - Implemented `URLSessionDataDelegate` to handle incoming audio stream data
  - Changed from push-based WebSocket messages to:
    - **Upload**: POST requests with PCM16 audio data
    - **Download**: Long-running GET request that streams WAV audio

#### 2. Updated APIService

- **File**: `swift-app/janusai/APIService.swift`
- **Removed**: `webSocketURL(sessionId:)` method
- **Added**:
  - `streamAudioURL(sessionId:)` - Returns URL for audio download stream
  - `uploadAudioChunkURL(sessionId:)` - Returns URL for uploading audio chunks
  - `stopStreamURL(sessionId:)` - Returns URL for stopping the stream

#### 3. Updated SessionRunningView

- **File**: `swift-app/janusai/SessionRunningView.swift`
- Simplified `start(sessionId:)` to not require WebSocket URL parameter
- Now just passes `sessionId` directly to `AudioStreamingClient`

## How It Works

### Audio Upload (Client → Server)

1. Client captures audio using `AVAudioEngine` (same as before)
2. Audio is converted to PCM16, mono, 16kHz (same as before)
3. Instead of sending via WebSocket, chunks are uploaded via HTTP POST to `/upload_audio`
4. Server receives chunks, plays them locally with `simpleaudio`, and buffers them

### Audio Download (Server → Client)

1. Client initiates a long-running HTTP GET request to `/stream_audio`
2. Server responds with:
   - WAV header (44 bytes)
   - Continuous stream of PCM16 audio data
3. Flask uses `Response(stream_with_context(generator))` for chunked transfer encoding
4. Client's `URLSessionDataDelegate` receives data chunks as they arrive
5. Client schedules audio for playback using `AVAudioPlayerNode` (same as before)

## Benefits of HTTP Streaming

1. **Simpler**: No WebSocket library dependencies
2. **More Compatible**: Standard HTTP/1.1 chunked transfer encoding
3. **Better Debugging**: Can use standard HTTP tools to inspect traffic
4. **Easier Deployment**: Works through most proxies/load balancers without special WebSocket configuration
5. **Native Support**: Both Flask and Swift have excellent HTTP streaming support built-in

## Testing

To test the new implementation:

1. **Update Backend Dependencies**:

   ```bash
   cd backend
   uv sync
   ```

2. **Run Backend**:

   ```bash
   python run.py
   ```

3. **Run Swift App** in Xcode and start a session

4. **Expected Behavior**:
   - Audio should be captured and uploaded to server
   - Server should play received audio
   - Client should receive streamed audio from server (currently silence, can be replaced with TTS)

## Future Enhancements

The current implementation streams silence from server to client. To add actual audio generation:

1. Replace the silence generation in `stream_sessions.py:generate_wav_stream()`
2. Integrate with TTS engine or AI audio generation
3. Yield actual audio chunks instead of silence

## Notes

- Audio format remains PCM16, mono, 16kHz throughout
- Upload uses binary POST with `application/octet-stream`
- Download uses WAV format with `audio/wav` MIME type
- Both upload and download happen simultaneously (bidirectional audio)
