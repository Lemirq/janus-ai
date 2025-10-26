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


def clean_response(text: str) -> str:
    """Remove thinking tags and clean up response"""
    import re
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'^\s*Okay,.*?(Let me|I need to).*?$', '', cleaned, flags=re.MULTILINE)
    
    # Remove old tokens
    old_tokens = ['<STRONG>', '<SOFT>', '<SLOW>', '<FAST>', '<CONFIDENT>', 
                  '<FRIENDLY>', '<SERIOUS>', '<EXCITED>', '<CALM>', '<RESET>',
                  '<strong>', '<soft>', '<slow>', '<fast>', '<confident>',
                  '<friendly>', '<serious>', '<excited>', '<calm>', '<reset>']
    for token in old_tokens:
        cleaned = cleaned.replace(token, '')
    
    cleaned = ' '.join(cleaned.split()).strip()
    
    # Fallback extraction
    if not cleaned or len(cleaned) < 10:
        quote_match = re.search(r'"([^"]{10,})"', text)
        if quote_match:
            cleaned = quote_match.group(1)
    
    return cleaned


async def generate_single_response(janus, user_input: str, talking_points: list, output_file: str = "output/response.wav"):
    """Generate a single persuasive response"""
    import re
    from pathlib import Path
    from core.sentiment_analyzer import ConversationAnalysis
    from core.persuasion_engine import AlignmentAnalysis
    
    Path("output").mkdir(exist_ok=True)
    
    print(f"\n[INPUT] Their statement:")
    print(f"  \"{user_input}\"")
    print(f"\n[YOUR POINTS] To emphasize:")
    for i, point in enumerate(talking_points, 1):
        print(f"  {i}. {point}")
    
    print(f"\n{'='*70}")
    print("[PROCESSING] Generating strategic response...")
    print("="*70)
    
    # Create analysis
    is_question = "?" in user_input
    analysis = ConversationAnalysis(
        sentiment="questioning" if is_question else "interested",
        is_question=is_question,
        question_type="clarification" if is_question else None,
        detected_concerns=[],
        emotional_state="engaged",
        requires_response=True,
        is_complex_question=False,
        key_topics=[]
    )
    
    alignment = AlignmentAnalysis(
        alignment_score=0.7,
        addressed_points=[],
        remaining_points=talking_points,
        detected_opportunities=["address directly"],
        suggested_pivot=None,
        urgency_level="medium",
        next_best_action="respond persuasively"
    )
    
    # Generate response
    response = await janus.response_generator.generate(
        transcript=user_input,
        analysis=analysis,
        alignment=alignment,
        objective=janus.current_objective,
        history=[]
    )
    
    clean_text = clean_response(response.text)
    clean_prosody = clean_response(response.prosody_text)
    
    print(f"\n[THINKING PROCESS]")
    print("-"*70)
    thinking_match = re.search(r'<think>(.*?)</think>', response.text, re.DOTALL)
    if thinking_match:
        for line in thinking_match.group(1).strip().split('\n'):
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print("  (Model didn't show thinking)")
    
    print(f"\n{'-'*70}")
    print(f"[FINAL RESPONSE]")
    print("-"*70)
    print(f"  Plain: \"{clean_text}\"")
    print(f"\n  With Prosody: \"{clean_prosody}\"")
    
    print(f"\n[PERSUASION TACTICS]")
    print("-"*70)
    for tactic in response.persuasion_tactics:
        print(f"  - {tactic.replace('_', ' ').title()}")
    
    # Generate audio with retries
    print(f"\n{'='*70}")
    print(f"[GENERATING AUDIO]")
    print("="*70)
    
    if not clean_prosody or len(clean_prosody) < 10:
        print(f"\n[ERROR] Response too short")
        return
    
    if len(clean_prosody) > 300:
        sentences = re.split(r'[.!?]+\s+', clean_prosody)
        clean_prosody = '. '.join(sentences[:3]) + '.'
    
    max_retries = 3
    audio_data = None
    
    # Try streaming generation first
    streaming_success = False
    
    try:
        print(f"[ATTEMPT 1] Streaming generation with prosody...")
        audio_stream = janus.audio_generator.generate_streaming(
            clean_prosody,
            janus.response_generator.prosody_tokenizer.encode_with_prosody(clean_prosody),
            response.voice_profile
        )
        
        # Save with streaming
        await janus.audio_generator.save_streaming_to_file(audio_stream, output_file)
        
        # Check file size
        from pathlib import Path
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size
            duration = file_size / (24000 * 2)
            
            if duration < 20 and file_size > 1000:
                print(f"\n[SUCCESS] {output_file}")
                print(f"  Duration: {duration:.1f}s")
                print(f"  Size: {file_size / 1024:.1f} KB")
                streaming_success = True
            else:
                print(f"[WARNING] Generated audio invalid, trying batch mode...")
                streaming_success = False
    except Exception as e:
        print(f"[INFO] Streaming mode not available: {str(e)[:80]}")
        print(f"[INFO] Falling back to batch generation...")
        streaming_success = False
    
    # Fallback to batch mode if streaming failed
    if not streaming_success:
        audio_data = None
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    print(f"[ATTEMPT {attempt+1}] With prosody...")
                    audio_data = await janus.audio_generator.generate(
                        clean_prosody,
                        janus.response_generator.prosody_tokenizer.encode_with_prosody(clean_prosody),
                        response.voice_profile
                    )
                elif attempt == 1:
                    print(f"[ATTEMPT {attempt+1}] Simplified...")
                    simplified = re.sub(r'<pause_[a-z]+>|<pitch_[a-z]+>', '', clean_prosody, flags=re.IGNORECASE)
                    audio_data = await janus.audio_generator.generate_simple(simplified, response.voice_profile)
                else:
                    print(f"[ATTEMPT {attempt+1}] Plain text...")
                    plain = re.sub(r'<[a-z_]+>', '', clean_prosody, flags=re.IGNORECASE)
                    audio_data = await janus.audio_generator.generate_simple(plain, response.voice_profile)
                
                if audio_data and len(audio_data) > 1000:
                    duration = len(audio_data) / (24000 * 2)
                    if duration < 20:
                        break
                    print(f"[WARNING] Audio too long ({duration:.1f}s), retrying...")
                    audio_data = None
                    
            except Exception as e:
                print(f"[WARNING] Attempt {attempt+1} failed: {str(e)[:100]}")
                audio_data = None
            
            if audio_data is None and attempt < max_retries - 1:
                await asyncio.sleep(1)
        
        if audio_data and len(audio_data) > 1000:
            await janus.audio_generator.save_to_file(audio_data, output_file)
            print(f"\n[SUCCESS] {output_file}")
            print(f"  Duration: {len(audio_data) / (24000 * 2):.1f}s")
            print(f"  Size: {len(audio_data) / 1024:.1f} KB")
        else:
            print(f"\n[ERROR] Audio generation failed after {max_retries} attempts")
    
    print(f"\n{'='*70}\n")


async def interactive_mode(janus):
    """Interactive mode for multiple questions"""
    print("\n" + "="*70)
    print("JANUS AI - Interactive Mode")
    print("="*70)
    print("Enter 'quit' to exit\n")
    
    counter = 1
    
    while True:
        print(f"\n--- Response #{counter} ---\n")
        
        user_input = input("[Q] What did they say?\n> ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        print("\n[POINTS] Your talking points (one per line, empty to finish):")
        talking_points = []
        while True:
            point = input(f"  {len(talking_points)+1}. ")
            if not point.strip():
                break
            talking_points.append(point.strip())
        
        if not talking_points:
            print("\n[ERROR] Need at least one point!")
            continue
        
        output_file = f"output/response_{counter:03d}.wav"
        await generate_single_response(janus, user_input, talking_points, output_file)
        
        counter += 1


async def main():
    """Main entry point with simple demo interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Janus AI - Persuasive Response Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py --interactive
  
  # Single response
  python main.py -i "How much?" -p "30%% savings" "No fees"
  
  # Custom output
  python main.py -i "Is it secure?" -p "Military encryption" -o output/secure.wav
        """
    )
    
    # Simple demo arguments
    parser.add_argument('-i', '--input', type=str, help='What they said')
    parser.add_argument('-p', '--points', nargs='+', help='Your talking points')
    parser.add_argument('-o', '--output', type=str, default='output/response.wav', help='Output file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    # Legacy demo mode arguments  
    parser.add_argument('--mode', choices=['demo', 'live'], help='[DEPRECATED] Use -i/-p instead')
    parser.add_argument('--output-dir', type=str, help='[DEPRECATED] Use -o instead')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("JANUS AI - Persuasive Response Generator")
    print("="*70)
    
    # Check API key
    api_key = os.getenv("BOSON_API_KEY")
    if not api_key:
        print("\n[ERROR] BOSON_API_KEY not set!")
        print("Set it with: $env:BOSON_API_KEY=\"your-key\"")
        return
    
    # Initialize config
    config = JanusConfig(
        api_key=api_key,
        enable_smooth_stall=False,
        enable_question_prediction=False
    )
    
    print(f"\n[INIT] Initializing Janus AI...")
    janus = JanusAI(config)
    
    # Set objective based on talking points
    if args.input and args.points:
        objective = PersuasionObjective(
            main_goal="Persuade effectively",
            key_points=args.points,
            audience_triggers=["results", "value", "trust"]
        )
        await janus.set_persuasion_objective(objective)
    
    # Run appropriate mode
    if args.interactive:
        # Interactive mode
        await interactive_mode(janus)
    elif args.input and args.points:
        # Single response mode
        await generate_single_response(janus, args.input, args.points, args.output)
    elif args.mode == 'demo':
        # Legacy demo mode
        objective = PersuasionObjective(
            main_goal="Sign partnership",
            key_points=["30% savings", "Proven track record"],
            audience_triggers=["ROI", "reliability"]
        )
        await janus.set_persuasion_objective(objective)
        await simulate_conversation(janus, args.output_dir or 'output_audio')
    else:
        # Show help
        print("\nNo arguments provided. Use one of:")
        print("  python main.py -i \"Question\" -p \"Point 1\" \"Point 2\"")
        print("  python main.py --interactive")
        print("  python main.py --help")


if __name__ == "__main__":
    asyncio.run(main())
