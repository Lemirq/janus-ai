"""
Janus AI - Real-time Persuasion Assistant
Main entry point for the system
"""

import asyncio
import os
from typing import Dict, Optional
from dataclasses import dataclass

from core.transcription import RealTimeTranscriber
from core.sentiment_analyzer import SentimentAnalyzer
from core.persuasion_engine import PersuasionEngine, PersuasionObjective
from core.response_generator import ResponseGenerator
from core.audio_generator import AudioGenerator


@dataclass
class JanusConfig:
    """Configuration for Janus AI system"""
    api_key: str
    base_url: str = "https://hackathon.boson.ai/v1"
    transcription_model: str = "higgs-audio-understanding-Hackathon"
    generation_model: str = "higgs-audio-generation-Hackathon"
    reasoning_model: str = "Qwen3-32B-thinking-Hackathon"
    
    # Audio settings
    sample_rate: int = 24000
    chunk_duration_ms: int = 100  # Process audio in 100ms chunks
    
    # Persuasion settings
    enable_question_prediction: bool = True
    enable_smooth_stall: bool = True
    enable_audience_profiling: bool = True


class JanusAI:
    """Main Janus AI system coordinator"""
    
    def __init__(self, config: JanusConfig):
        self.config = config
        
        # Initialize components
        self.transcriber = RealTimeTranscriber(config)
        self.sentiment_analyzer = SentimentAnalyzer(config)
        self.persuasion_engine = PersuasionEngine(config)
        self.response_generator = ResponseGenerator(config)
        self.audio_generator = AudioGenerator(config)
        
        # State management
        self.conversation_history = []
        self.current_objective: Optional[PersuasionObjective] = None
        self.audience_profile: Optional[Dict] = None
        
    async def set_persuasion_objective(self, objective: PersuasionObjective):
        """Set the main persuasion objective for the conversation"""
        self.current_objective = objective
        self.persuasion_engine.set_objective(objective)
        
    async def set_audience_profile(self, profile: Dict):
        """Set audience profile for better targeting"""
        self.audience_profile = profile
        self.persuasion_engine.set_audience_profile(profile)
        
    async def process_audio_stream(self, audio_stream):
        """Main processing loop for real-time audio"""
        async for audio_chunk in audio_stream:
            # Real-time transcription
            transcript = await self.transcriber.process_chunk(audio_chunk)
            
            if transcript:
                # Analyze sentiment and detect questions
                analysis = await self.sentiment_analyzer.analyze(
                    transcript, 
                    self.conversation_history
                )
                
                # Check alignment with persuasion objective
                alignment = await self.persuasion_engine.check_alignment(
                    transcript,
                    analysis,
                    self.current_objective
                )
                
                # Generate response if needed
                if analysis.requires_response:
                    # Quick stall if enabled
                    if self.config.enable_smooth_stall and analysis.is_complex_question:
                        stall_audio = await self._generate_stall()
                        yield stall_audio
                    
                    # Generate full response
                    response = await self.response_generator.generate(
                        transcript=transcript,
                        analysis=analysis,
                        alignment=alignment,
                        objective=self.current_objective,
                        history=self.conversation_history
                    )
                    
                    # Convert to audio with prosody
                    audio_response = await self.audio_generator.generate(
                        response.text,
                        response.prosody_tokens,
                        response.voice_profile
                    )
                    
                    # Update conversation history
                    self.conversation_history.append({
                        'user': transcript,
                        'assistant': response.text,
                        'analysis': analysis,
                        'alignment': alignment
                    })
                    
                    yield audio_response
                    
    async def _generate_stall(self):
        """Generate a smooth stalling response"""
        stall_phrases = [
            "That's an excellent question...",
            "Let me think about that for a moment...",
            "I see what you're asking...",
            "That's definitely worth considering..."
        ]
        # Quick generation without complex prosody
        return await self.audio_generator.generate_quick(
            stall_phrases[0],  # Would randomly select in production
            voice="en_woman"
        )
        
    async def get_session_analytics(self):
        """Get analytics for the current session"""
        return await self.persuasion_engine.get_session_analytics(
            self.conversation_history,
            self.current_objective
        )


async def main():
    """Example usage of Janus AI"""
    config = JanusConfig(
        api_key=os.getenv("BOSON_API_KEY"),
        enable_smooth_stall=True,
        enable_question_prediction=True
    )
    
    janus = JanusAI(config)
    
    # Set persuasion objective
    objective = PersuasionObjective(
        main_goal="Sign the partnership agreement",
        key_points=[
            "Highlight cost savings of 30%",
            "Emphasize proven track record",
            "Address security concerns proactively",
            "Create urgency with limited-time offer"
        ],
        audience_triggers=["ROI", "reliability", "security"]
    )
    
    await janus.set_persuasion_objective(objective)
    
    # Set audience profile
    audience = {
        "type": "corporate_executive",
        "priorities": ["cost_reduction", "risk_mitigation"],
        "communication_style": "direct",
        "decision_timeline": "quarterly"
    }
    
    await janus.set_audience_profile(audience)
    
    # In production, this would connect to actual audio stream
    # For now, we'll simulate
    print("Janus AI initialized and ready for persuasion assistance")


if __name__ == "__main__":
    asyncio.run(main())
