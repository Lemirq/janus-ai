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


async def simulate_conversation(janus: JanusAI, output_dir: str = "output_audio"):
    """Simulate a sales conversation and generate audio responses"""
    import wave
    from pathlib import Path
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    print(f"\n[OUTPUT] Saving audio files to: {output_dir}/\n")
    
    # Simulated conversation: customer asking questions about a product/service
    simulated_conversation = [
        {
            "speaker": "Customer",
            "text": "I'm interested in your solution, but I'm concerned about the cost. What are we looking at price-wise?"
        },
        {
            "speaker": "Customer", 
            "text": "That sounds expensive. How long until we see ROI?"
        },
        {
            "speaker": "Customer",
            "text": "What about security? We've had data breaches before and our board is very sensitive about this."
        },
        {
            "speaker": "Customer",
            "text": "Interesting. Do you have case studies or references from similar companies?"
        },
        {
            "speaker": "Customer",
            "text": "Okay, I need to think about this and discuss with my team. When do we need to decide?"
        }
    ]
    
    print("Starting simulated sales conversation...\n")
    print("=" * 70)
    
    for i, exchange in enumerate(simulated_conversation, 1):
        print(f"\n[{exchange['speaker']}]: \"{exchange['text']}\"")
        print("-" * 70)
        
        # Simulate sentiment analysis (in real version, this would analyze audio)
        from core.sentiment_analyzer import ConversationAnalysis
        
        # Create mock analysis
        is_question = "?" in exchange['text']
        concerns = []
        if "cost" in exchange['text'].lower() or "expensive" in exchange['text'].lower():
            concerns.append("pricing")
        if "security" in exchange['text'].lower() or "breach" in exchange['text'].lower():
            concerns.append("security")
        if "time" in exchange['text'].lower() or "roi" in exchange['text'].lower():
            concerns.append("timeline")
            
        analysis = ConversationAnalysis(
            sentiment="questioning" if is_question else "interested",
            is_question=is_question,
            question_type="objection" if concerns else "clarification",
            detected_concerns=concerns,
            emotional_state="skeptical" if concerns else "engaged",
            requires_response=True,
            is_complex_question=len(concerns) > 0,
            key_topics=concerns
        )
        
        # Check alignment with persuasion objective
        alignment = await janus.persuasion_engine.check_alignment(
            exchange['text'],
            analysis,
            janus.current_objective
        )
        
        print(f"[ANALYSIS] Sentiment: {analysis.sentiment} | Concerns: {', '.join(concerns) if concerns else 'None'}")
        print(f"[STRATEGY] Alignment Score: {alignment.alignment_score:.2f} | Next Action: {alignment.next_best_action}")
        
        # Generate response
        print(f"[AI] Generating persuasive response...")
        response = await janus.response_generator.generate(
            transcript=exchange['text'],
            analysis=analysis,
            alignment=alignment,
            objective=janus.current_objective,
            history=janus.conversation_history
        )
        
        print(f"[RESPONSE] Text: \"{response.text}\"")
        print(f"[PROSODY] With markup: \"{response.prosody_text}\"")
        print(f"[TACTICS] Used: {', '.join(response.persuasion_tactics)}")
        
        # Generate audio
        print(f"[AUDIO] Generating speech with prosody...")
        audio_data = await janus.audio_generator.generate(
            response.prosody_text,  # Use prosody-marked text
            response.prosody_tokens,
            response.voice_profile
        )
        
        # Save audio file
        filename = f"{output_dir}/response_{i:02d}.wav"
        await janus.audio_generator.save_to_file(audio_data, filename)
        print(f"[SAVED] Audio file: {filename}")
        
        # Update conversation history
        janus.conversation_history.append({
            'user': exchange['text'],
            'assistant': response.text,
            'analysis': analysis,
            'alignment': alignment
        })
        
        print("=" * 70)
    
    # Generate session analytics
    print("\n\n" + "=" * 70)
    print("SESSION ANALYTICS")
    print("=" * 70)
    analytics = await janus.get_session_analytics()
    
    print(f"\nObjective Completion: {analytics['objective_completion']:.1%}")
    print(f"Average Alignment: {analytics['average_alignment']:.2f}")
    print(f"\nAddressed Points:")
    for point in analytics['addressed_points']:
        print(f"   - {point}")
    print(f"\nRemaining Points:")
    for point in analytics['remaining_points']:
        print(f"   - {point}")
    print(f"\nKey Moments: {len(analytics['key_moments'])}")
    for moment in analytics['key_moments'][:3]:  # Show first 3
        print(f"   - {moment['type']}: {moment['description']}")
    
    print("\n" + "=" * 70)
    print(f"Conversation complete! Audio files saved to: {output_dir}/")
    print("=" * 70)


async def main():
    """Example usage of Janus AI with actual audio generation"""
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Janus AI - Real-time Persuasion Assistant")
    parser.add_argument('--mode', choices=['demo', 'live'], default='demo',
                       help='Demo mode (simulated conversation) or live mode (real audio)')
    parser.add_argument('--output-dir', type=str, default='output_audio',
                       help='Directory to save audio files (default: output_audio)')
    parser.add_argument('--scenario', type=str, 
                       choices=['sales', 'negotiation', 'interview'],
                       default='sales',
                       help='Conversation scenario type')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("JANUS AI - Real-time Persuasion Assistant")
    print("="*70)
    
    # Initialize configuration
    api_key = os.getenv("BOSON_API_KEY")
    if not api_key:
        print("\n[ERROR] BOSON_API_KEY environment variable not set")
        print("Set it with: $env:BOSON_API_KEY=\"your-key\"")
        return
    
    config = JanusConfig(
        api_key=api_key,
        enable_smooth_stall=True,
        enable_question_prediction=True
    )
    
    print(f"\n[CONFIG] Configuration:")
    print(f"   - Mode: {args.mode}")
    print(f"   - Scenario: {args.scenario}")
    print(f"   - Output: {args.output_dir}")
    print(f"   - Models: {config.generation_model}, {config.reasoning_model}")
    
    # Initialize Janus AI
    print(f"\n[INIT] Initializing Janus AI components...")
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
    print(f"[OBJECTIVE] Goal: {objective.main_goal}")
    
    # Set audience profile
    audience = {
        "type": "corporate_executive",
        "priorities": ["cost_reduction", "risk_mitigation"],
        "communication_style": "direct",
        "decision_timeline": "quarterly"
    }
    
    await janus.set_audience_profile(audience)
    print(f"[AUDIENCE] Type: {audience['type']}")
    
    if args.mode == 'demo':
        # Run simulated conversation
        await simulate_conversation(janus, args.output_dir)
    else:
        # Live mode (not implemented yet)
        print("\n[WARNING] Live mode not implemented yet. Use --mode demo for now.")
        print("Live mode would require audio device integration.")


if __name__ == "__main__":
    asyncio.run(main())
