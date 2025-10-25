"""
Real-time transcription module using Higgs audio understanding
"""

import asyncio
import base64
from typing import Optional, List
from collections import deque
import numpy as np
from openai import AsyncOpenAI


class RealTimeTranscriber:
    """Handles real-time audio transcription using Higgs"""
    
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # Buffer for audio chunks
        self.audio_buffer = deque(maxlen=50)  # ~5 seconds at 100ms chunks
        self.processing = False
        self.min_chunk_size = int(config.sample_rate * 0.5)  # 0.5 seconds minimum
        
    async def process_chunk(self, audio_chunk: bytes) -> Optional[str]:
        """Process a single audio chunk and return transcript if available"""
        self.audio_buffer.append(audio_chunk)
        
        # Only process if we have enough audio and not already processing
        if len(self.audio_buffer) >= 5 and not self.processing:
            return await self._transcribe_buffer()
            
        return None
        
    async def _transcribe_buffer(self) -> Optional[str]:
        """Transcribe the current audio buffer"""
        self.processing = True
        
        try:
            # Combine audio chunks
            combined_audio = b''.join(self.audio_buffer)
            
            # Encode to base64
            audio_base64 = base64.b64encode(combined_audio).decode('utf-8')
            
            # Send to Higgs for transcription
            response = await self.client.chat.completions.create(
                model=self.config.transcription_model,
                messages=[
                    {"role": "system", "content": "Transcribe this audio accurately. Include punctuation."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "pcm"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=256,
                temperature=0.0
            )
            
            transcript = response.choices[0].message.content
            
            # Clear buffer after successful transcription
            self.audio_buffer.clear()
            
            return transcript if transcript else None
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
            
        finally:
            self.processing = False
            
    def add_audio_chunk(self, chunk: bytes):
        """Add audio chunk to buffer without blocking"""
        self.audio_buffer.append(chunk)
        
    async def force_transcribe(self) -> Optional[str]:
        """Force transcription of current buffer regardless of size"""
        if len(self.audio_buffer) > 0:
            return await self._transcribe_buffer()
        return None


class StreamingTranscriber:
    """Alternative streaming transcriber for continuous processing"""
    
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        self.current_segment = []
        self.silence_threshold = 0.1  # seconds
        
    async def process_stream(self, audio_stream):
        """Process continuous audio stream"""
        async for chunk in audio_stream:
            self.current_segment.append(chunk)
            
            # Check for silence or segment boundary
            if await self._is_segment_complete(chunk):
                transcript = await self._transcribe_segment()
                if transcript:
                    yield transcript
                self.current_segment = []
                
    async def _is_segment_complete(self, chunk: bytes) -> bool:
        """Detect if current segment is complete (silence detection)"""
        # Convert bytes to numpy array
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        
        # Simple energy-based silence detection
        energy = np.sqrt(np.mean(audio_data**2))
        
        # If energy is below threshold, consider it silence
        return energy < 100  # Threshold would be tuned in production
        
    async def _transcribe_segment(self) -> Optional[str]:
        """Transcribe current audio segment"""
        if not self.current_segment:
            return None
            
        combined_audio = b''.join(self.current_segment)
        audio_base64 = base64.b64encode(combined_audio).decode('utf-8')
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.transcription_model,
                messages=[
                    {"role": "system", "content": "Transcribe this audio segment."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "pcm"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=128,
                temperature=0.0
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Streaming transcription error: {e}")
            return None
