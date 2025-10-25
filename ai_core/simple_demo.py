"""
Janus AI - Simple Command-Line Interface
Just input a question and your talking points, get audio response with prosody

Examples:
    # Interactive mode
    python simple_demo.py --interactive
    
    # Single response
    python simple_demo.py -i "Is this secure?" -p "Military-grade encryption" "Zero breaches"
    
    # With custom output
    python simple_demo.py --input "How much does this cost?" --points "30% savings compared to competitors" "Pay-as-you-go pricing" "No long-term contracts" --output pricing_response.wav
"""

import asyncio
import os
import sys
import re
from pathlib import Path

from main import JanusAI, JanusConfig
from core.persuasion_engine import PersuasionObjective, AlignmentAnalysis
from core.sentiment_analyzer import ConversationAnalysis


def clean_response(text: str) -> str:
    """Remove thinking tags and clean up response"""
    # Remove <think>...</think> blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # Remove any remaining angle brackets that aren't prosody markers
    prosody_markers = ['<EMPH>', '<STRONG>', '<SLOW>', '<FAST>', '<PAUSE_SHORT>', 
                       '<PAUSE_LONG>', '<PITCH_HIGH>', '<PITCH_LOW>', '<CONFIDENT>', 
                       '<FRIENDLY>', '<CALM>', '<RESET>']
    
    # Clean up whitespace
    cleaned = ' '.join(cleaned.split())
    cleaned = cleaned.strip()
    
    return cleaned


async def generate_response(user_input: str, talking_points: list, output_file: str = "response.wav"):
    """
    Generate a persuasive response with prosody
    
    Args:
        user_input: What the other person said
        talking_points: Your key points to address
        output_file: Where to save the audio
    """
    
    # Initialize Janus AI
    api_key = os.getenv("BOSON_API_KEY")
    if not api_key:
        print("\n[ERROR] BOSON_API_KEY not set!")
        print("Set it with: $env:BOSON_API_KEY=\"your-key\"")
        return
    
    config = JanusConfig(
        api_key=api_key,
        enable_smooth_stall=False,  # Disable for simple mode
        enable_question_prediction=False
    )
    
    print("\n" + "="*70)
    print("JANUS AI - Persuasive Response Generator")
    print("="*70)
    
    janus = JanusAI(config)
    
    # Set objective from talking points
    objective = PersuasionObjective(
        main_goal="Persuade effectively",
        key_points=talking_points,
        audience_triggers=["results", "value", "trust"]
    )
    
    await janus.set_persuasion_objective(objective)
    
    # Simple analysis (since we don't have real audio)
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
    
    # Mock alignment
    alignment = AlignmentAnalysis(
        alignment_score=0.7,
        addressed_points=[],
        remaining_points=talking_points,
        detected_opportunities=["address their concern directly"],
        suggested_pivot=None,
        urgency_level="medium",
        next_best_action="respond persuasively"
    )
    
    print(f"\n[INPUT] Their statement:")
    print(f"  \"{user_input}\"")
    print(f"\n[YOUR POINTS] To emphasize:")
    for i, point in enumerate(talking_points, 1):
        print(f"  {i}. {point}")
    
    print(f"\n{'='*70}")
    print("[PROCESSING] Generating strategic response...")
    print("="*70)
    
    # Generate response
    response = await janus.response_generator.generate(
        transcript=user_input,
        analysis=analysis,
        alignment=alignment,
        objective=janus.current_objective,
        history=[]
    )
    
    # Clean the response
    clean_text = clean_response(response.text)
    clean_prosody = clean_response(response.prosody_text)
    
    print(f"\n[THINKING PROCESS]")
    print("-"*70)
    # Extract thinking if present
    thinking_match = re.search(r'<think>(.*?)</think>', response.text, re.DOTALL)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # Format thinking nicely
        thinking_lines = thinking.split('\n')
        for line in thinking_lines:
            if line.strip():
                print(f"  {line.strip()}")
    else:
        print("  (Model didn't show thinking process)")
    
    print(f"\n{'-'*70}")
    print(f"[FINAL RESPONSE]")
    print("-"*70)
    print(f"  Plain: \"{clean_text}\"")
    print(f"\n  With Prosody: \"{clean_prosody}\"")
    
    print(f"\n[PERSUASION TACTICS USED]")
    print("-"*70)
    for tactic in response.persuasion_tactics:
        print(f"  - {tactic.replace('_', ' ').title()}")
    
    # Generate audio (only for clean response, not thinking)
    print(f"\n{'='*70}")
    print(f"[GENERATING AUDIO] Creating speech with prosody...")
    print("="*70)
    
    try:
        audio_data = await janus.audio_generator.generate(
            clean_prosody,  # Use cleaned prosody text
            janus.response_generator.prosody_tokenizer.encode_with_prosody(clean_prosody),
            response.voice_profile
        )
        
        # Save audio
        await janus.audio_generator.save_to_file(audio_data, output_file)
        
        print(f"\n[SUCCESS] Audio saved to: {output_file}")
        print(f"Duration: ~{len(audio_data) / (24000 * 2):.1f} seconds")
        
    except Exception as e:
        print(f"\n[ERROR] Audio generation failed: {e}")
        print("Response text is still valid above.")
    
    print("\n" + "="*70)
    print("Complete!")
    print("="*70 + "\n")


async def interactive_mode():
    """Interactive mode for multiple questions"""
    print("\n" + "="*70)
    print("JANUS AI - Interactive Persuasion Assistant")
    print("="*70)
    print("\nEnter 'quit' to exit")
    print("="*70)
    
    counter = 1
    
    while True:
        print(f"\n--- Response #{counter} ---\n")
        
        # Get user input
        user_input = input("[Q] What did they say?\n> ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        # Get talking points
        print("\n[POINTS] Enter your talking points (one per line, empty line to finish):")
        talking_points = []
        while True:
            point = input(f"  {len(talking_points)+1}. ")
            if not point.strip():
                break
            talking_points.append(point.strip())
        
        if not talking_points:
            print("\n[ERROR] You need at least one talking point!")
            continue
        
        # Generate output filename
        output_file = f"response_{counter:03d}.wav"
        
        # Generate response
        await generate_response(user_input, talking_points, output_file)
        
        counter += 1


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Janus AI - Simple Persuasive Response Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python simple_demo.py --interactive
  
  # Single response
  python simple_demo.py --input "How much does this cost?" --points "30%% savings" "Proven ROI"
  
  # With custom output
  python simple_demo.py -i "Is it secure?" -p "Military-grade encryption" "Zero breaches" -o secure_response.wav
        """
    )
    
    parser.add_argument('-i', '--input', type=str,
                       help='What the other person said')
    parser.add_argument('-p', '--points', nargs='+',
                       help='Your talking points to emphasize')
    parser.add_argument('-o', '--output', type=str, default='response.wav',
                       help='Output audio filename (default: response.wav)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        await interactive_mode()
    elif args.input and args.points:
        await generate_response(args.input, args.points, args.output)
    else:
        # Default demo
        print("\n" + "="*70)
        print("JANUS AI - Demo Mode")
        print("="*70)
        print("\nNo arguments provided. Running demo example...")
        print("Use --help for usage information")
        print("="*70)
        
        demo_input = "I'm concerned about the cost. How much are we talking?"
        demo_points = [
            "30% cost savings in first year",
            "Proven ROI within 6 months",
            "No hidden fees"
        ]
        
        await generate_response(demo_input, demo_points, "demo_response.wav")


if __name__ == "__main__":
    asyncio.run(main())

