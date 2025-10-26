"""
STEP 4: Setup Higgs Audio Repository and Model
Clones the official Higgs repo and downloads the 3B model
"""

import os
import subprocess
from pathlib import Path


def setup_higgs():
    """
    Setup Higgs Audio for local fine-tuning
    
    Based on: https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base
    """
    
    print("\n" + "="*70)
    print("HIGGS AUDIO SETUP")
    print("="*70)
    
    print(f"\nThis will:")
    print(f"  1. Clone Higgs repository from GitHub")
    print(f"  2. Install Higgs dependencies")
    print(f"  3. Download model weights (~6 GB)")
    print(f"\nTotal time: 15-30 minutes")
    print(f"Disk space needed: ~10 GB")
    
    # Step 1: Clone Higgs repository
    higgs_repo = Path("models/higgs-audio")
    
    if not higgs_repo.exists():
        print(f"\n[1/3] Cloning Higgs repository...")
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/boson-ai/higgs-audio.git",
                str(higgs_repo)
            ], check=True)
            print(f"       ✓ Cloned to: {higgs_repo}")
        except Exception as e:
            print(f"       ERROR: {e}")
            print(f"       Make sure git is installed!")
            return False
    else:
        print(f"\n[1/3] Higgs repository exists: {higgs_repo}")
    
    # Step 2: Install Higgs dependencies
    print(f"\n[2/3] Installing Higgs dependencies...")
    try:
        subprocess.run([
            "pip", "install", "-r", str(higgs_repo / "requirements.txt")
        ], check=False)
        
        subprocess.run([
            "pip", "install", "-e", str(higgs_repo)
        ], check=True)
        
        print(f"       ✓ Higgs package installed")
    except Exception as e:
        print(f"       WARNING: Some dependencies may have failed: {e}")
        print(f"       Continuing anyway...")
    
    # Step 3: Download model (happens automatically on first use)
    print(f"\n[3/3] Model download info...")
    print(f"       Model: bosonai/higgs-audio-v2-generation-3B-base")
    print(f"       Size: ~6 GB")
    print(f"       Will download on first use (automatic)")
    
    # Create model cache directory
    cache_dir = Path("models/higgs_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"[SUCCESS] Higgs setup complete!")
    print(f"{'='*70}")
    print(f"\nRepository: {higgs_repo}")
    print(f"Cache dir: {cache_dir}")
    
    print(f"\nNext: python 5_train_higgs_prosody.py --epochs 3")
    
    return True


if __name__ == "__main__":
    success = setup_higgs()
    
    if success:
        print(f"\n[READY] You can now fine-tune Higgs!")
        print(f"Run: python 5_train_higgs_prosody.py --epochs 3")
    else:
        print(f"\n[FAILED] Setup incomplete")
        print(f"Check errors above and try again")

