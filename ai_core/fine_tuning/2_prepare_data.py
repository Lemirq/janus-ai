"""
STEP 2: Prepare Training Dataset
Converts audio segments into PyTorch-compatible format
"""

import json
import wave
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_prosody_markers(text):
    """Extract prosody markers from text"""
    prosody_tokens = []
    markers = ['<emph>', '<pause_short>', '<pause_long>', '<pitch_high>', 
               '<pitch_low>', '<pitch_rising>', '<pitch_falling>']
    
    for marker in markers:
        if marker in text.lower():
            prosody_tokens.append(marker)
    
    return prosody_tokens


def prepare_dataset(segments_dir="training_data/segments", output_file="training_data/prosody_dataset.json"):
    """
    Prepare dataset from segmented audio
    """
    print("\n" + "="*70)
    print("STEP 2: PREPARE TRAINING DATASET")
    print("="*70)
    
    # Load manifest
    manifest_path = f"{segments_dir}/manifest.json"
    print(f"\n[1/4] Loading manifest: {manifest_path}")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    print(f"      Loaded {len(segments)} segments")
    
    # Process each segment
    print(f"\n[2/4] Processing segments...")
    training_data = []
    prosody_stats = {}
    
    for segment in tqdm(segments, desc="Processing"):
        # Extract prosody markers
        prosody_tokens = extract_prosody_markers(segment['text'])
        
        # Count prosody types
        for token in prosody_tokens:
            prosody_stats[token] = prosody_stats.get(token, 0) + 1
        
        # Create training example
        example = {
            'id': segment['id'],
            'text': segment['text'],
            'prosody_text': segment['text'],  # Already has markers
            'audio_path': f"{segments_dir}/{segment['audio_file']}",
            'prosody_tokens': prosody_tokens,
            'duration': segment['duration']
        }
        
        training_data.append(example)
    
    # Save dataset
    print(f"\n[3/4] Saving dataset...")
    Path("training_data").mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    # Create dataset info
    print(f"\n[4/4] Creating dataset info...")
    info = {
        'num_examples': len(training_data),
        'total_duration': sum(e['duration'] for e in training_data),
        'avg_duration': np.mean([e['duration'] for e in training_data]),
        'prosody_distribution': prosody_stats,
        'unique_prosody_types': len(prosody_stats)
    }
    
    info_path = "training_data/dataset_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    # Print summary
    print(f"\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    print(f"\nTotal examples: {info['num_examples']}")
    print(f"Total duration: {info['total_duration']:.1f} seconds ({info['total_duration']/60:.1f} minutes)")
    print(f"Average duration: {info['avg_duration']:.1f} seconds")
    print(f"\nProsody token distribution:")
    for token, count in sorted(prosody_stats.items(), key=lambda x: -x[1]):
        print(f"  {token}: {count} occurrences")
    
    print(f"\n✓ Dataset saved to: {output_file}")
    print(f"✓ Info saved to: {info_path}")
    print("="*70)
    
    print(f"\n✓ STEP 2 COMPLETE")
    print(f"Next: python 3_train_lora.py --epochs 3")
    
    return training_data


if __name__ == "__main__":
    dataset = prepare_dataset()
    print(f"\nDataset ready for training!")

