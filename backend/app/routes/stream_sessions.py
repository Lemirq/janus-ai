import json
import os
import time
import wave
import io
from datetime import datetime
from flask import Blueprint, Response, request, stream_with_context
from collections import deque
from .sessions import _session_path

bp = Blueprint('stream_sessions', __name__)

# Store active session audio buffers
# Format: {session_id: deque of PCM16 audio chunks}
_session_audio_buffers = {}
_session_states = {}  # Track session streaming state

# Track upload counts per session for logging
_upload_counts = {}

# Track WAV file handles for recording
_session_wav_files = {}

@bp.route('/sessions/<session_id>/upload_audio', methods=['POST'])
def upload_audio_chunk(session_id):
    """
    Receives audio chunks from the client (PCM16, mono, 16kHz).
    Saves them to a WAV file for the session.
    """
    # Validate session exists
    if not os.path.exists(_session_path(session_id)):
        return {"error": "session not found"}, 404
    
    # Get raw PCM16 data from request body
    pcm_data = request.get_data()
    
    if len(pcm_data) == 0:
        return {"error": "empty audio data"}, 400
    
    # Initialize buffer and WAV file for this session if needed
    if session_id not in _session_audio_buffers:
        _session_audio_buffers[session_id] = deque(maxlen=200)  # Keep last ~4 seconds at 50 chunks/sec
        _upload_counts[session_id] = 0
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Create WAV file for this session
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        wav_filename = os.path.join(uploads_dir, f'recording-{session_id}-{timestamp}.wav')
        wav_file = wave.open(wav_filename, 'wb')
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(16000)  # 16 kHz
        _session_wav_files[session_id] = wav_file
        
        print(f"[stream_sessions] Created recording file: {wav_filename}")
    
    # Increment upload count
    _upload_counts[session_id] += 1
    count = _upload_counts[session_id]
    
    # Add to buffer
    _session_audio_buffers[session_id].append(pcm_data)
    
    # Log first few and periodic chunks
    if count <= 3 or count % 50 == 0:
        bytes_preview = " ".join(f"{b:02x}" for b in pcm_data[:16])
        print(f"[stream_sessions] Received chunk #{count} for {session_id}: {len(pcm_data)} bytes, first 16: {bytes_preview}")
    
    # Write to WAV file
    try:
        if session_id in _session_wav_files:
            _session_wav_files[session_id].writeframes(pcm_data)
            if count <= 3 or count % 50 == 0:
                print(f"[stream_sessions] Wrote chunk #{count} to WAV file")
    except Exception as e:
        print(f"[stream_sessions] Error writing chunk #{count} to WAV: {e}")
    
    return {"status": "ok", "bytes_received": len(pcm_data)}, 200


@bp.route('/sessions/<session_id>/stream_audio')
def stream_audio(session_id):
    """
    Streams audio back to the client as a continuous WAV stream.
    The client can play this with AVPlayer or AVAudioEngine.
    """
    # Validate session exists
    if not os.path.exists(_session_path(session_id)):
        return {"error": "session not found"}, 404
    
    print(f"[stream_sessions] Starting audio stream for {session_id}")
    
    def generate_wav_stream():
        """
        Generator that yields WAV header followed by PCM16 chunks.
        This creates a continuous audio stream.
        """
        # Send WAV header first (44 bytes for a basic WAV with unknown size)
        # We'll use a "streaming" WAV format with max size
        sample_rate = 16000
        num_channels = 1
        bytes_per_sample = 2
        
        # Create WAV header
        wav_header = io.BytesIO()
        with wave.open(wav_header, 'wb') as wav:
            wav.setnchannels(num_channels)
            wav.setsampwidth(bytes_per_sample)
            wav.setframerate(sample_rate)
            # Write a dummy frame to create header, we'll stream the rest
            wav.writeframes(b'\x00\x00')
        
        # Get the header (modify to support streaming)
        header_bytes = wav_header.getvalue()
        # Modify the RIFF chunk size to maximum (0xFFFFFFFF for streaming)
        header_bytes = bytearray(header_bytes)
        header_bytes[4:8] = (0xFFFFFFFF).to_bytes(4, 'little')
        header_bytes[40:44] = (0xFFFFFFFF).to_bytes(4, 'little')
        
        print(f"[stream_sessions] Sending WAV header ({len(header_bytes[:44])} bytes) for {session_id}")
        yield bytes(header_bytes[:44])  # Send just the 44-byte header
        
        # Mark session as streaming
        _session_states[session_id] = {'streaming': True, 'start_time': time.time()}
        
        # Now stream audio chunks as they become available
        # This is a simple demo - in production you'd generate/synthesize audio
        frame_count = 0
        try:
            print(f"[stream_sessions] Beginning audio chunk stream for {session_id}")
            while _session_states.get(session_id, {}).get('streaming', False):
                # Check if we have audio to send (simulate server generating audio)
                # In a real app, this would come from TTS or AI response
                
                # For now, send silence in chunks to keep the stream alive
                # 1024 samples = 64ms at 16kHz
                silence_chunk = b'\x00\x00' * 1024
                yield silence_chunk
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"[stream_sessions] Streamed {frame_count} silence chunks for {session_id}")
                
                # Small delay to prevent CPU spinning
                time.sleep(0.064)  # ~64ms per chunk
                
                # Stop after reasonable time or if client disconnects
                if time.time() - _session_states[session_id]['start_time'] > 3600:  # 1 hour max
                    break
        except GeneratorExit:
            print(f"[stream_sessions] Client disconnected from stream for {session_id}")
        finally:
            # Cleanup
            print(f"[stream_sessions] Stopping stream for {session_id} (sent {frame_count} chunks)")
            if session_id in _session_states:
                _session_states[session_id]['streaming'] = False
    
    return Response(
        stream_with_context(generate_wav_stream()),
        mimetype='audio/wav',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering if behind nginx
        }
    )


@bp.route('/sessions/<session_id>/stop_stream', methods=['POST'])
def stop_stream(session_id):
    """
    Stops the audio stream for a session and closes the recording file.
    """
    print(f"[stream_sessions] Stopping stream for {session_id}")
    
    if session_id in _session_states:
        _session_states[session_id]['streaming'] = False
    
    # Close and finalize WAV file
    if session_id in _session_wav_files:
        try:
            _session_wav_files[session_id].close()
            print(f"[stream_sessions] Closed WAV file for {session_id}")
        except Exception as e:
            print(f"[stream_sessions] Error closing WAV file: {e}")
        del _session_wav_files[session_id]
    
    # Clean up buffers
    if session_id in _session_audio_buffers:
        del _session_audio_buffers[session_id]
    
    # Clean up upload counts
    if session_id in _upload_counts:
        total_chunks = _upload_counts[session_id]
        duration_seconds = (total_chunks * 3200) / (16000 * 2)  # 3200 bytes / (16000 Hz * 2 bytes/sample)
        print(f"[stream_sessions] Session {session_id} received {total_chunks} chunks (~{duration_seconds:.1f} seconds of audio)")
        del _upload_counts[session_id]
    
    return {"status": "stopped"}, 200

