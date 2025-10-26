import json
import os
import time
import wave
import io
import struct
import numpy as np
import sounddevice as sd
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

# Track packet directories for each session
_session_packet_dirs = {}

# Track audio output streams for playback
_session_audio_streams = {}

# Track latency statistics per session
_latency_stats = {}  # {session_id: {'min': ms, 'max': ms, 'sum': ms, 'count': n}}

@bp.route('/sessions/<session_id>/upload_audio', methods=['POST'])
def upload_audio_chunk(session_id):
    """
    Receives audio chunks from the client (PCM16, mono, 16kHz).
    Saves them to a WAV file for the session.
    """
    # Validate session exists
    if not os.path.exists(_session_path(session_id)):
        return {"error": "session not found"}, 404
    
    # Get raw data from request body (timestamp + PCM16 data)
    raw_data = request.get_data()
    
    if len(raw_data) < 8:
        return {"error": "invalid packet: too small"}, 400
    
    # Extract timestamp (first 8 bytes, Int64 little-endian, milliseconds)
    timestamp_ms = struct.unpack('<q', raw_data[:8])[0]
    
    # Calculate latency
    current_time_ms = int(time.time() * 1000)
    latency_ms = current_time_ms - timestamp_ms
    
    # Extract PCM audio data (remaining bytes)
    pcm_data = raw_data[8:]
    
    if len(pcm_data) == 0:
        return {"error": "empty audio data"}, 400
    
    # Initialize buffer and packet directory for this session if needed
    if session_id not in _session_audio_buffers:
        _session_audio_buffers[session_id] = deque(maxlen=200)  # Keep last ~4 seconds at 50 chunks/sec
        _upload_counts[session_id] = 0
        _latency_stats[session_id] = {'min': float('inf'), 'max': 0, 'sum': 0, 'count': 0}
        
        # Create packets directory for this session
        uploads_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        packet_dir = os.path.join(uploads_dir, f'packets-{session_id}-{timestamp}')
        os.makedirs(packet_dir, exist_ok=True)
        _session_packet_dirs[session_id] = packet_dir
        
        print(f"[stream_sessions] Created packet directory: {packet_dir}")
        
        # Initialize audio output stream for playback through speakers
        try:
            audio_stream = sd.OutputStream(
                samplerate=16000,
                channels=1,
                dtype='int16',
                blocksize=0  # Use variable block size
            )
            audio_stream.start()
            _session_audio_streams[session_id] = audio_stream
            print(f"[stream_sessions] Started audio playback stream for {session_id}")
        except Exception as e:
            print(f"[stream_sessions] Error starting audio playback: {e}")
    
    # Increment upload count
    _upload_counts[session_id] += 1
    count = _upload_counts[session_id]
    
    # Add to buffer
    _session_audio_buffers[session_id].append(pcm_data)
    
    # Update latency statistics
    stats = _latency_stats[session_id]
    stats['min'] = min(stats['min'], latency_ms)
    stats['max'] = max(stats['max'], latency_ms)
    stats['sum'] += latency_ms
    stats['count'] += 1
    avg_latency = stats['sum'] / stats['count']
    
    # Log first few and periodic chunks with latency
    if count <= 3 or count % 50 == 0:
        bytes_preview = " ".join(f"{b:02x}" for b in pcm_data[:16])
        print(f"[stream_sessions] Chunk #{count} | LATENCY: {latency_ms}ms (avg: {avg_latency:.1f}ms, min: {stats['min']}ms, max: {stats['max']}ms) | {len(pcm_data)} bytes")
    
    # Save packet to individual file
    packet_filename = None
    try:
        if session_id in _session_packet_dirs:
            packet_filename = os.path.join(
                _session_packet_dirs[session_id],
                f'packet_{count:06d}_latency{latency_ms}ms_ts{timestamp_ms}.pcm'
            )
            with open(packet_filename, 'wb') as f:
                f.write(pcm_data)
            if count <= 3 or count % 50 == 0:
                print(f"[stream_sessions] Saved chunk #{count} to {os.path.basename(packet_filename)}")
    except Exception as e:
        print(f"[stream_sessions] Error saving packet #{count}: {e}")
    
    # Play audio through speakers by reading the saved file
    try:
        if session_id in _session_audio_streams and packet_filename and os.path.exists(packet_filename):
            # Read the packet file we just saved
            with open(packet_filename, 'rb') as f:
                file_pcm_data = f.read()
            
            # Convert PCM bytes to numpy array for sounddevice
            audio_array = np.frombuffer(file_pcm_data, dtype=np.int16)
            _session_audio_streams[session_id].write(audio_array)
            if count <= 3 or count % 50 == 0:
                print(f"[stream_sessions] Played chunk #{count} from file through speakers")
    except Exception as e:
        print(f"[stream_sessions] Error playing audio chunk #{count}: {e}")
    
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
    
    # Clean up packet directory reference
    if session_id in _session_packet_dirs:
        packet_dir = _session_packet_dirs[session_id]
        print(f"[stream_sessions] Packets saved to: {packet_dir}")
        del _session_packet_dirs[session_id]
    
    # Stop and close audio playback stream
    if session_id in _session_audio_streams:
        try:
            _session_audio_streams[session_id].stop()
            _session_audio_streams[session_id].close()
            print(f"[stream_sessions] Closed audio playback stream for {session_id}")
        except Exception as e:
            print(f"[stream_sessions] Error closing audio playback: {e}")
        del _session_audio_streams[session_id]
    
    # Clean up buffers
    if session_id in _session_audio_buffers:
        del _session_audio_buffers[session_id]
    
    # Clean up upload counts and print latency stats
    if session_id in _upload_counts:
        total_chunks = _upload_counts[session_id]
        duration_seconds = (total_chunks * 3200) / (16000 * 2)  # 3200 bytes / (16000 Hz * 2 bytes/sample)
        print(f"[stream_sessions] Session {session_id} received {total_chunks} chunks (~{duration_seconds:.1f} seconds of audio)")
        del _upload_counts[session_id]
    
    # Print final latency statistics
    if session_id in _latency_stats:
        stats = _latency_stats[session_id]
        if stats['count'] > 0:
            avg_latency = stats['sum'] / stats['count']
            print(f"[stream_sessions] Latency stats for {session_id}: avg={avg_latency:.1f}ms, min={stats['min']}ms, max={stats['max']}ms, samples={stats['count']}")
        del _latency_stats[session_id]
    
    return {"status": "stopped"}, 200

