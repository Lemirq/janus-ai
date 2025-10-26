"""
Audio generator with prosody support using Higgs
"""

import base64
import wave
import io
from typing import List, Optional, Dict, Tuple
from openai import AsyncOpenAI
import asyncio
import numpy as np


class AudioGenerator:
    """Generates audio with prosody using Higgs model"""
    
    def __init__(self, config):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # Audio parameters
        self.sample_rate = 24000
        self.num_channels = 1
        self.sample_width = 2
        
        # Reference audio cache for voice cloning
        self.reference_cache = {}
        
    async def generate(self,
                      text: str,
                      prosody_tokens: List[int],
                      voice_profile: str = "en_woman") -> bytes:
        """Generate audio with prosody tokens"""
        
        # For initial version, we'll use the chat completion API
        # with reference audio for better control
        
        # Get reference audio for voice
        reference_audio, reference_text = await self._get_reference_audio(voice_profile)
        
        # Build messages with prosody context
        messages = self._build_prosody_messages(
            text, prosody_tokens, reference_audio, reference_text
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=messages,
                modalities=["text", "audio"],
                max_completion_tokens=4096,
                temperature=0.8,  # Balanced for naturalness
                top_p=0.95,
                stream=False,
                stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
                extra_body={"top_k": 50}
            )
            
            # Extract audio data
            audio_b64 = response.choices[0].message.audio.data
            audio_data = base64.b64decode(audio_b64)
            
            return audio_data
            
        except Exception as e:
            print(f"Audio generation error: {e}")
            # Fallback to simple generation
            return await self.generate_simple(text, voice_profile)
            
    async def generate_simple(self, text: str, voice: str = "en_woman") -> bytes:
        """Simple audio generation without complex prosody"""
        
        try:
            response = await self.client.audio.speech.create(
                model=self.config.generation_model,
                voice=voice,
                input=text,
                response_format="pcm"
            )
            
            return response.content
            
        except Exception as e:
            print(f"Simple audio generation error: {e}")
            # Return empty audio as last resort
            return self._generate_silence(1.0)
            
    async def generate_quick(self, text: str, voice: str = "en_woman") -> bytes:
        """Quick generation for stalling responses"""
        # Use simple generation for speed
        return await self.generate_simple(text, voice)
        
    async def generate_with_emotion(self,
                                   text: str,
                                   emotion: str,
                                   voice: str = "en_woman") -> bytes:
        """Generate audio with specific emotional tone"""
        
        # Map emotions to scene descriptions
        emotion_scenes = {
            'confident': "Speaker is confident and authoritative, speaking clearly.",
            'friendly': "Speaker is warm and friendly, with a smile in their voice.",
            'serious': "Speaker is serious and professional, with measured delivery.",
            'excited': "Speaker is enthusiastic and energetic.",
            'calm': "Speaker is calm and reassuring, speaking slowly."
        }
        
        scene_desc = emotion_scenes.get(emotion, emotion_scenes['friendly'])
        
        # Build system prompt with emotion
        system_prompt = f"""You are an AI assistant designed to convert text into speech.
<|scene_desc_start|>
{scene_desc}
<|scene_desc_end|>"""

        try:
            # Use reference audio if available
            reference_audio, reference_text = await self._get_reference_audio(voice)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": reference_text},
                {
                    "role": "assistant",
                    "content": [{
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(reference_audio).decode('utf-8'),
                            "format": "wav"
                        }
                    }]
                },
                {"role": "user", "content": text}
            ]
            
            response = await self.client.chat.completions.create(
                model=self.config.generation_model,
                messages=messages,
                modalities=["text", "audio"],
                max_completion_tokens=4096,
                temperature=0.9,
                top_p=0.95,
                stream=False,
                stop=["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
                extra_body={"top_k": 50}
            )
            
            audio_b64 = response.choices[0].message.audio.data
            return base64.b64decode(audio_b64)
            
        except Exception as e:
            print(f"Emotion audio generation error: {e}")
            return await self.generate_simple(text, voice)
            
    def _build_prosody_messages(self,
                               text: str,
                               prosody_tokens: List[int],
                               reference_audio: bytes,
                               reference_text: str) -> List[Dict]:
        """Build messages for prosody-aware generation"""
        
        # System prompt explaining prosody handling
        system_prompt = """You are an AI assistant that converts text to speech with prosody control.
When you see special prosody markers like <emph>, <pitch_high>, <pause_short>, apply the corresponding speech effects:
- <emph>: Emphasize the following word
- <pause_short>: Brief pause (0.5s)
- <pause_long>: Longer pause (1.5s)
- <pitch_high>: Higher pitch
- <pitch_low>: Lower pitch
- <pitch_rising>: Rising intonation
- <pitch_falling>: Falling intonation

<|scene_desc_start|>
Professional setting, clear audio quality.
<|scene_desc_end|>"""

        # Include reference for voice consistency
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": reference_text},
            {
                "role": "assistant",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(reference_audio).decode('utf-8'),
                        "format": "wav"
                    }
                }]
            },
            {"role": "user", "content": text}  # Text already has prosody markers
        ]
        
        return messages
        
    async def _get_reference_audio(self, voice_profile: str) -> Tuple[bytes, str]:
        """Get reference audio for voice cloning"""
        
        # Check cache first
        if voice_profile in self.reference_cache:
            return self.reference_cache[voice_profile]
            
        # Reference audio mapping
        reference_map = {
            'en_woman': ('belinda.wav', "Hello, this is a test of the voice generation system."),
            'en_man': ('en_man.wav', "Welcome to our presentation today."),
            'confident': ('chadwick.wav', "I'm absolutely certain this will work."),
            'friendly': ('mabel.wav', "It's so nice to meet you!"),
            'professional': ('broom_salesman.wav', "Let me show you our best options.")
        }
        
        if voice_profile in reference_map:
            filename, text = reference_map[voice_profile]
            # In production, load actual reference audio
            # For now, return placeholder
            audio_data = self._generate_silence(0.1)  # Placeholder
            
            self.reference_cache[voice_profile] = (audio_data, text)
            return audio_data, text
            
        # Default reference
        return self._generate_silence(0.1), "Hello, this is a voice test."
        
    def _generate_silence(self, duration: float) -> bytes:
        """Generate silent audio of specified duration"""
        num_samples = int(self.sample_rate * duration)
        silence = np.zeros(num_samples, dtype=np.int16)
        return silence.tobytes()
        
    async def save_to_file(self, audio_data: bytes, filename: str):
        """Save audio data to WAV file"""
        with wave.open(filename, 'wb') as wav:
            wav.setnchannels(self.num_channels)
            wav.setsampwidth(self.sample_width)
            wav.setframerate(self.sample_rate)
            wav.writeframes(audio_data)
            
    def combine_audio_segments(self, segments: List[bytes]) -> bytes:
        """Combine multiple audio segments"""
        if not segments:
            return self._generate_silence(0.1)
            
        # Convert to numpy arrays
        arrays = []
        for segment in segments:
            if segment:
                arr = np.frombuffer(segment, dtype=np.int16)
                arrays.append(arr)
                
        if not arrays:
            return self._generate_silence(0.1)
            
        # Concatenate
        combined = np.concatenate(arrays)
        return combined.tobytes()


class ProsodyAudioProcessor:
    """Post-processes audio to enhance prosody effects"""
    
    @staticmethod
    def apply_emphasis(audio_data: bytes, 
                      start_sample: int, 
                      end_sample: int,
                      emphasis_factor: float = 1.5) -> bytes:
        """Apply emphasis to a portion of audio"""
        
        # Convert to numpy
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        # Apply emphasis (simple amplitude boost)
        audio_array[start_sample:end_sample] *= emphasis_factor
        
        # Clip to prevent overflow
        audio_array = np.clip(audio_array, -32768, 32767)
        
        return audio_array.astype(np.int16).tobytes()
        
    @staticmethod
    def insert_pause(audio_data: bytes,
                    insert_position: int,
                    pause_duration: float,
                    sample_rate: int = 24000) -> bytes:
        """Insert pause at specified position"""
        
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Create pause
        pause_samples = int(pause_duration * sample_rate)
        pause = np.zeros(pause_samples, dtype=np.int16)
        
        # Split and insert
        before = audio_array[:insert_position]
        after = audio_array[insert_position:]
        
        combined = np.concatenate([before, pause, after])
        
        return combined.tobytes()
        
    @staticmethod
    def adjust_pitch(audio_data: bytes,
                    pitch_factor: float) -> bytes:
        """Simple pitch adjustment (would use proper DSP in production)"""
        
        # This is a placeholder - real implementation would use
        # proper pitch shifting algorithms like PSOLA or phase vocoder
        
        return audio_data  # Return unchanged for now
