"""
Real-time audio transcription service using Whisper.
Processes audio in chunked WAV segments: every N PCM packets are written to a
temporary WAV file, transcribed, appended to a transcript, and then the WAV
file is deleted. This repeats continuously.
"""

import os
import wave
import threading
import queue
from typing import Optional
from huggingface_hub import InferenceClient

# Global transcript accumulator (consumed by the AI model)
T: str = ""
_T_LOCK = threading.Lock()

def get_live_transcript() -> str:
    """Safely get the current live transcript text."""
    with _T_LOCK:
        return T

class WhisperTranscriptionService:
    """Handles real-time transcription using Whisper large-v3-turbo"""
    
    def __init__(self, session_id: str, packet_dir: str, transcript_path: str, packets_per_transcription: int = 75, chunks_per_file: int = 50, save_full_recording: bool = False, full_recording_filename: str = 'combined.wav'):
        """
        Initialize transcription service for a session.
        
        Args:
            session_id: Session identifier
            packet_dir: Directory where packets and combined WAV will be stored
            transcript_path: Path to transcript text file
            packets_per_transcription: Number of packets to accumulate before transcribing (default 75 ≈ 1.5 seconds)
        """
        self.session_id = session_id
        self.packet_dir = packet_dir
        self.transcript_path = transcript_path
        # Deprecated in favor of chunks_per_file but kept for compatibility with callers
        self.packets_per_transcription = packets_per_transcription
        # Number of PCM packets to combine into a single WAV file for transcription
        self.chunks_per_file = chunks_per_file
        
        # Audio parameters (must match incoming PCM format)
        self.sample_rate = 24000
        self.channels = 1
        self.sample_width = 2  # 16-bit PCM = 2 bytes
        
        # Chunked WAV bookkeeping
        self._chunk_index = 0
        
        # Disable persistent full-session WAV recording by default
        self.save_full_recording = False
        self.full_recording_path = os.path.join(self.packet_dir, full_recording_filename)
        self._full_wav_file = None
        
        # Background transcription worker
        self._task_queue: queue.Queue[str] = queue.Queue()
        self._running = True
        self._worker = threading.Thread(target=self._transcription_worker, name=f"transcriber-{session_id}", daemon=True)
        self._worker.start()
        
        # Packet buffer
        self.packet_buffer = []
        self.packet_count = 0
        self.total_frames_written = 0
        
        # Initialize Whisper pipeline (lazy load on first use)
        self._pipe = None
        self._model_loaded = False
        
        # Initialize transcript file
        with open(self.transcript_path, 'w', encoding='utf-8') as f:
            f.write(f"# Transcript for session {session_id}\n\n")
        
        print(f"[TRX] init {session_id} -> {os.path.basename(self.transcript_path)}")
    
    def _write_wav_chunk(self, pcm_bytes: bytes) -> str:
        """Write a WAV file for the current chunk buffer and return its path."""
        self._chunk_index += 1
        wav_name = f"combined_{self._chunk_index:06d}.wav"
        wav_path = os.path.join(self.packet_dir, wav_name)
        with wave.open(wav_path, 'wb') as wav:
            wav.setnchannels(self.channels)
            wav.setsampwidth(self.sample_width)
            wav.setframerate(self.sample_rate)
            wav.writeframes(pcm_bytes)
        print(f"[TRX] chunk {os.path.basename(wav_path)} size={len(pcm_bytes)}")
        return wav_path

    def _transcription_worker(self):
        """Continuously transcribe WAV chunks from the queue and append results."""
        while self._running or not self._task_queue.empty():
            try:
                wav_path = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                transcript = self._transcribe_wav_file(wav_path)
                global T
                with _T_LOCK:
                    T = transcript
                if transcript:
                    self._append_transcript_text(transcript)
            except Exception as e:
                print(f"[TRX] worker_err {type(e).__name__}: {str(e)[:120]}")
            finally:
                try:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                        print(f"[TRX] del {os.path.basename(wav_path)}")
                except Exception as e:
                    print(f"[TRX] del_warn {os.path.basename(wav_path)}: {str(e)[:100]}")
                self._task_queue.task_done()
    
    def _get_pipeline(self):
        """Lazy init Hugging Face Inference client on first use"""
        if self._pipe is None:
            print(f"[TRX] hf_init")
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                print("[TRX] hf_warn no_token")
            self._pipe = InferenceClient(provider="hf-inference", api_key=hf_token)
            self._model_loaded = True
            print(f"[TRX] hf_ready")
        return self._pipe

    def _append_transcript_text(self, text: str) -> None:
        """Append transcribed text to both the file and global T safely."""
        if not text:
            return
        with open(self.transcript_path, 'a', encoding='utf-8') as f:
            f.write(text + " ")
        global T
        with _T_LOCK:
            if T:
                T = (T + " " + text).strip()
            else:
                T = text.strip()
    
    def add_packet(self, pcm_data: bytes) -> Optional[str]:
        """
        Add a PCM packet to the buffer and WAV file.
        Triggers transcription every N packets.
        
        Args:
            pcm_data: Raw PCM16 audio data
            
        Returns:
            Transcribed text if transcription was triggered, None otherwise
        """
        # Update stats
        frames = len(pcm_data) // self.sample_width
        self.total_frames_written += frames
        
        # Add to buffer
        self.packet_buffer.append(pcm_data)
        self.packet_count += 1
        
        # No full-session WAV recording
        
        # When enough packets are collected, write a WAV chunk and enqueue for transcription
        if self.packet_count >= self.chunks_per_file:
            combined_pcm = b''.join(self.packet_buffer)
            wav_path = self._write_wav_chunk(combined_pcm)
            # Clear buffer immediately to keep ingestion non-blocking
            self.packet_buffer.clear()
            self.packet_count = 0
            # Enqueue for background transcription
            self._task_queue.put(wav_path)
        
        return None
    
    def _transcribe_wav_file(self, wav_path: str) -> Optional[str]:
        """Transcribe a WAV file via HF Inference and return the text (or None)."""
        try:
            # Log basic params for visibility
            with wave.open(wav_path, 'rb') as wav:
                if (
                    wav.getnchannels() != self.channels or
                    wav.getframerate() != self.sample_rate or
                    wav.getsampwidth() != self.sample_width
                ):
                    print(f"[Transcription] Warning: WAV params mismatch for {wav_path}")
            client = self._get_pipeline()
            name = os.path.basename(wav_path)
            print(f"[TRX] hf_req {name}")
            result = client.automatic_speech_recognition(
                wav_path,
                model="openai/whisper-large-v3-turbo"
            )
            # Result may be a dict with 'text' or raw string
            transcript_text = None
            if isinstance(result, dict):
                transcript_text = (result.get("text") or "").strip()
            else:
                transcript_text = (str(result) or "").strip()
            if transcript_text:
                print(f"[TRX] hf_res {name} -> '{transcript_text[:48]}{'…' if len(transcript_text)>48 else ''}'")
                return transcript_text
            print(f"[TRX] hf_res_empty {name}")
            return None
        except Exception as e:
            print(f"[TRX] hf_err {os.path.basename(wav_path)} {type(e).__name__}: {str(e)[:140]}")
            return None
    
    def finalize(self) -> Optional[str]:
        """
        Finalize transcription - transcribe any remaining audio and close files.
        
        Returns:
            Final transcribed text or None
        """
        print(f"[TRX] finalize {self.session_id}")
        
        # If there is remaining audio, write a final chunk and enqueue
        if self.packet_buffer:
            print(f"[TRX] enqueue_rem {self.packet_count}")
            combined_pcm = b''.join(self.packet_buffer)
            wav_path = self._write_wav_chunk(combined_pcm)
            self.packet_buffer.clear()
            self.packet_count = 0
            self._task_queue.put(wav_path)
        
        # Stop worker after queue drains
        self._running = False
        self._task_queue.join()
        if self._worker.is_alive():
            self._worker.join(timeout=5)

        # Add final newline to transcript
        with open(self.transcript_path, 'a', encoding='utf-8') as f:
            f.write("\n\n# End of transcript\n")
        
        # No full-session WAV to close
        transcript_size = os.path.getsize(self.transcript_path)
        print(f"[TRX] done size={transcript_size}")
        
        return None
    
    def get_stats(self):
        """Get transcription statistics"""
        return {
            "session_id": self.session_id,
            "total_packets": self.packet_count,
            "total_frames": self.total_frames_written,
            "duration_seconds": self.total_frames_written / self.sample_rate,
            "transcript_path": self.transcript_path,
            "full_recording_path": None,
            "model_loaded": self._model_loaded
        }

