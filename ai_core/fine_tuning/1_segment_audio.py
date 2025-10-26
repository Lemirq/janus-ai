"""
STEP 1: Automatic Audio Segmentation
Splits long audio file into training segments automatically
"""

import wave
import numpy as np
from pathlib import Path
import json
import re


def detect_pauses(audio_path, threshold_db=-35, min_pause_duration=0.8):
    """
    Detect pauses in audio using energy-based detection
    
    Args:
        audio_path: Path to WAV file
        threshold_db: Energy threshold for silence (default: -35dB)
        min_pause_duration: Minimum pause length to split on (default: 0.8s)
        
    Returns:
        List of (start_time, end_time) tuples for speech segments
    """
    print(f"\n[1/3] Loading audio: {audio_path}")
    
    with wave.open(audio_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        audio_data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16)
    
    duration = len(audio_data) / sample_rate
    print(f"      Duration: {duration:.1f} seconds")
    print(f"      Sample rate: {sample_rate} Hz")
    
    # Convert to float and normalize
    audio_float = audio_data.astype(np.float32) / 32768.0
    
    # Calculate energy in frames
    print(f"\n[2/3] Analyzing energy levels...")
    frame_length = int(sample_rate * 0.1)  # 100ms frames
    hop_length = int(sample_rate * 0.05)   # 50ms hop
    
    energy = []
    for i in range(0, len(audio_float) - frame_length, hop_length):
        frame = audio_float[i:i + frame_length]
        frame_energy = 20 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-10)
        energy.append(frame_energy)
    
    # Find pauses (low energy regions)
    is_pause = np.array(energy) < threshold_db
    
    # Group consecutive pauses
    segments = []
    in_segment = False
    segment_start = 0
    
    for i, pause in enumerate(is_pause):
        if not pause and not in_segment:
            # Start of speech segment
            segment_start = i * hop_length / sample_rate
            in_segment = True
        elif pause and in_segment:
            # Check if pause is long enough
            pause_start = i
            pause_duration = 0
            j = i
            while j < len(is_pause) and is_pause[j]:
                pause_duration += hop_length / sample_rate
                j += 1
            
            if pause_duration >= min_pause_duration:
                # End of segment
                segment_end = pause_start * hop_length / sample_rate
                segments.append((segment_start, segment_end))
                in_segment = False
    
    # Add final segment if exists
    if in_segment:
        segments.append((segment_start, len(audio_float) / sample_rate))
    
    print(f"      Found {len(segments)} speech segments")
    
    return segments


def segment_audio_and_text(audio_path, transcript_path, output_dir="training_data/segments"):
    """
    Automatically segment audio and align with transcript
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 1: AUTOMATIC AUDIO SEGMENTATION")
    print("="*70)
    
    # Read transcript
    print(f"\nLoading transcript: {transcript_path}")
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read()
    
    # Split transcript by prosody markers and sentences
    # Extract sentences with their prosody markers
    text_segments = []
    current_text = ""
    
    # Simple sentence splitting (keeps prosody markers with text)
    words = transcript.split()
    sentence = []
    
    for word in words:
        sentence.append(word)
        # End sentence at punctuation or after ~15 words
        if word.endswith(('.', '?', '!')) or len(sentence) > 15:
            text_segments.append(' '.join(sentence))
            sentence = []
    
    if sentence:  # Add remaining words
        text_segments.append(' '.join(sentence))
    
    # Detect audio segments
    audio_segments = detect_pauses(audio_path, threshold_db=-35, min_pause_duration=0.8)
    
    print(f"\n[3/3] Aligning text with audio...")
    print(f"      Text segments: {len(text_segments)}")
    print(f"      Audio segments: {len(audio_segments)}")
    
    # Align them (simple ratio-based alignment)
    aligned_pairs = []
    num_pairs = min(len(audio_segments), len(text_segments), 50)  # Limit to 50
    
    for i in range(num_pairs):
        # Simple 1:1 alignment
        text_idx = int(i * len(text_segments) / num_pairs)
        audio_idx = int(i * len(audio_segments) / num_pairs)
        
        if audio_idx < len(audio_segments) and text_idx < len(text_segments):
            aligned_pairs.append({
                'text': text_segments[text_idx],
                'audio_start': audio_segments[audio_idx][0],
                'audio_end': audio_segments[audio_idx][1],
                'index': i
            })
    
    # Load and segment audio
    print(f"\nExtracting {len(aligned_pairs)} segments...")
    with wave.open(audio_path, 'rb') as wav:
        params = wav.getparams()
        sample_rate = wav.getframerate()
        audio_data = wav.readframes(wav.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    training_examples = []
    
    for pair in aligned_pairs:
        # Extract audio segment
        start_frame = int(pair['audio_start'] * sample_rate)
        end_frame = int(pair['audio_end'] * sample_rate)
        segment_audio = audio_array[start_frame:end_frame]
        
        # Save audio segment
        audio_filename = f"segment_{pair['index']+1:03d}.wav"
        audio_path_full = f"{output_dir}/{audio_filename}"
        
        with wave.open(audio_path_full, 'wb') as seg_wav:
            seg_wav.setparams(params)
            seg_wav.writeframes(segment_audio.tobytes())
        
        duration = pair['audio_end'] - pair['audio_start']
        
        training_examples.append({
            'id': pair['index'] + 1,
            'text': pair['text'],
            'audio_file': audio_filename,
            'duration': duration
        })
        
        # Show progress every 10 segments
        if (pair['index'] + 1) % 10 == 0 or pair['index'] == 0:
            print(f"  [{pair['index']+1}/{len(aligned_pairs)}] {pair['text'][:60]}... ({duration:.1f}s)")
    
    # Save manifest
    manifest_path = f"{output_dir}/manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary_path = "training_data/segmentation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"Audio Segmentation Summary\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Total segments: {len(training_examples)}\n")
        f.write(f"Average duration: {np.mean([e['duration'] for e in training_examples]):.1f}s\n")
        f.write(f"Total duration: {sum([e['duration'] for e in training_examples]):.1f}s\n")
        f.write(f"Output directory: {output_dir}/\n")
    
    print(f"\n" + "="*70)
    print(f"✓ Created {len(training_examples)} training segments")
    print(f"✓ Saved to: {output_dir}/")
    print(f"✓ Manifest: {manifest_path}")
    print(f"✓ Summary: {summary_path}")
    print("="*70)
    
    print(f"\n✓ STEP 1 COMPLETE")
    print(f"Next: python 2_prepare_data.py")
    
    return training_examples


if __name__ == "__main__":
    audio_file = "jre_training_audio.wav/jre_training_audio.wav"
    transcript_file = "jre_training_transcript.txt"
    
    segments = segment_audio_and_text(audio_file, transcript_file, output_dir="training_data/segments")

