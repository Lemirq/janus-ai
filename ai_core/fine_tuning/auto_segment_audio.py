"""
Automatic audio segmentation for fine-tuning
Takes a long audio file and transcript, segments them automatically
"""

import wave
import numpy as np
from pathlib import Path
import json
import re


def detect_pauses(audio_path, threshold_db=-40, min_pause_duration=0.5):
    """
    Detect pauses in audio file automatically
    Returns list of (start_time, end_time) tuples for segments
    """
    with wave.open(audio_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        audio_data = np.frombuffer(wav.readframes(n_frames), dtype=np.int16)
    
    # Convert to float and normalize
    audio_float = audio_data.astype(np.float32) / 32768.0
    
    # Calculate energy in frames
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
            while i < len(is_pause) and is_pause[i]:
                pause_duration += hop_length / sample_rate
                i += 1
            
            if pause_duration >= min_pause_duration:
                # End of segment
                segment_end = pause_start * hop_length / sample_rate
                segments.append((segment_start, segment_end))
                in_segment = False
    
    # Add final segment if exists
    if in_segment:
        segments.append((segment_start, len(audio_float) / sample_rate))
    
    return segments


def segment_audio_and_text(audio_path, transcript_path, output_dir="training_segments"):
    """
    Automatically segment audio and align with transcript
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Read transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = f.read()
    
    # Split transcript by sentences (rough approximation)
    # Look for sentence boundaries
    sentences = re.split(r'([.!?])\s+', transcript)
    text_segments = []
    current = ""
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            current = sentences[i] + sentences[i+1]
        else:
            current = sentences[i]
        if current.strip():
            text_segments.append(current.strip())
    
    # Detect audio segments
    print(f"Analyzing audio file: {audio_path}")
    audio_segments = detect_pauses(audio_path, threshold_db=-35, min_pause_duration=0.8)
    
    print(f"Found {len(audio_segments)} audio segments")
    print(f"Found {len(text_segments)} text segments")
    
    # Align them (simple time-based alignment)
    # This is rough - better would use forced alignment tools
    aligned_pairs = []
    segment_ratio = len(audio_segments) / len(text_segments)
    
    for i, text in enumerate(text_segments[:len(audio_segments)]):
        audio_idx = int(i * segment_ratio)
        if audio_idx < len(audio_segments):
            aligned_pairs.append({
                'text': text,
                'audio_start': audio_segments[audio_idx][0],
                'audio_end': audio_segments[audio_idx][1],
                'index': i
            })
    
    # Save segments
    with wave.open(audio_path, 'rb') as wav:
        params = wav.getparams()
        sample_rate = wav.getframerate()
        audio_data = wav.readframes(wav.getnframes())
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
    
    training_examples = []
    
    for pair in aligned_pairs[:50]:  # Limit to 50 segments for demo
        # Extract audio segment
        start_frame = int(pair['audio_start'] * sample_rate)
        end_frame = int(pair['audio_end'] * sample_rate)
        segment_audio = audio_array[start_frame:end_frame]
        
        # Save audio segment
        audio_filename = f"{output_dir}/segment_{pair['index']:04d}.wav"
        with wave.open(audio_filename, 'wb') as seg_wav:
            seg_wav.setparams(params)
            seg_wav.writeframes(segment_audio.tobytes())
        
        training_examples.append({
            'id': pair['index'],
            'text': pair['text'],
            'audio_path': audio_filename,
            'duration': pair['audio_end'] - pair['audio_start']
        })
        
        print(f"Segment {pair['index']}: {pair['text'][:50]}... ({pair['audio_end'] - pair['audio_start']:.2f}s)")
    
    # Save manifest
    with open(f"{output_dir}/training_manifest.json", 'w') as f:
        json.dump(training_examples, f, indent=2)
    
    print(f"\nCreated {len(training_examples)} training segments")
    print(f"Saved to: {output_dir}/")
    print(f"Manifest: {output_dir}/training_manifest.json")
    
    return training_examples


if __name__ == "__main__":
    # Run segmentation
    audio_file = "jre_training_audio.wav/jre_training_audio.wav"
    transcript_file = "jre_training_transcript.txt"
    
    print("Automatic Audio Segmentation for Fine-tuning")
    print("=" * 60)
    
    segments = segment_audio_and_text(audio_file, transcript_file)
    
    print("\nReady for fine-tuning!")
    print("Next step: Run train_prosody.py with the segmented data")

