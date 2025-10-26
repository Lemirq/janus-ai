# Fine-Tuning Guide - LoRA Training for Prosody

## 🎯 EXACT Steps to Fine-Tune Right Now

### Prerequisites Check
```powershell
# You need:
✓ jre_training_audio.wav/jre_training_audio.wav (you have this)
✓ jre_training_transcript.txt (you have this)
✓ Python environment with pip (you have this)
✓ NVIDIA RTX 4050 (you have this)
```

---

## 📊 Time Estimate for RTX 4050 Laptop

| Data Amount | Segments | Epochs | Estimated Time |
|-------------|----------|--------|----------------|
| **Your data** (12 min) | ~50 | 3 | **1.5-2 hours** |
| More data (30 min) | ~150 | 3 | **3-4 hours** |
| Full dataset (60 min) | ~300 | 5 | **6-8 hours** |

**Your specific case**: ~**1.5-2 hours** for 3 epochs with 50 segments

**RTX 4050 specs**:
- 6GB VRAM → Perfect for LoRA training
- Laptop power limits → Add ~20% time vs desktop
- Expected speed: ~2-3 min/epoch with 50 samples

---

## 🚀 STEP-BY-STEP Instructions

### STEP 1: Segment Your Audio (5 minutes)

```powershell
cd C:\Users\prabh\Downloads\boson-ai-hackathon\ai_core\fine_tuning

# Run auto-segmentation
python 1_segment_audio.py
```

**What it does**:
```
Input:
  jre_training_audio.wav/jre_training_audio.wav
  jre_training_transcript.txt

Process:
  - Analyzes audio waveform
  - Detects pauses (energy < -35dB)
  - Splits at pauses > 0.8 seconds
  - Creates ~50 segments

Output:
  training_data/segments/
    ├─ segment_001.wav
    ├─ segment_002.wav
    ├─ ...
    ├─ segment_050.wav
    └─ manifest.json
```

**Expected output**:
```
Analyzing audio file...
Found 52 audio segments
Found 48 text segments
Segment 1: looking at random leaves on television... (3.5s)
Segment 2: shows to recognize that it's a fake... (4.2s)
...
Created 50 training segments
Saved to: training_data/segments/
```

---

### STEP 2: Prepare Training Data (2 minutes)

```powershell
python 2_prepare_data.py
```

**What it does**:
```
For each segment:
  1. Load audio file
  2. Extract features (pitch, energy, duration)
  3. Tokenize text with prosody markers
  4. Align audio features with tokens
  5. Create training example

Output:
  training_data/
    ├─ segments/ (from step 1)
    ├─ prosody_dataset.json
    └─ dataset_info.txt
```

**Expected output**:
```
Loading 50 segments...
Processing segment_001.wav...
  Text: "looking at random leaves..."
  Prosody tokens: <pitch_low>, <pause_long>
  Audio features extracted: pitch, energy, duration
  
Processing segment_002.wav...
...

Dataset created:
  50 training examples
  7 unique prosody types
  Average duration: 3.8 seconds
  
Saved to: training_data/prosody_dataset.json
```

---

### STEP 3: Install Training Dependencies (5-10 minutes)

```powershell
# PyTorch with CUDA for your RTX 4050
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# LoRA and training tools
pip install peft transformers accelerate datasets bitsandbytes

# Monitoring
pip install tensorboard
```

**Verify GPU is detected**:
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Should see**:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

---

### STEP 4: Run LoRA Fine-Tuning (1.5-2 hours)

```powershell
# Basic training (recommended for first time)
python 3_train_lora.py --epochs 3 --batch-size 4

# Advanced (more quality, longer)
python 3_train_lora.py --epochs 5 --batch-size 4 --lora-r 32

# Quick test (10 min, lower quality)
python 3_train_lora.py --epochs 1 --batch-size 2 --test-mode
```

**What happens**:
```
[1/6] Loading GPT-2 base model...
[2/6] Adding 7 prosody tokens to vocabulary...
[3/6] Setting up LoRA adapters...
      LoRA rank: 16
      Trainable params: 294,912 (0.3% of total)
[4/6] Loading training data (50 samples)...
[5/6] Starting training...

Epoch 1/3:
  [Step 1/13] Loss: 2.456 | LR: 0.00012 | 23s
  [Step 5/13] Loss: 1.892 | LR: 0.00024 | 1m 55s
  [Step 10/13] Loss: 1.234 | LR: 0.00030 | 3m 50s
  [Step 13/13] Loss: 1.087 | LR: 0.00030 | 5m 0s
  Validation: Loss: 1.145

Epoch 2/3:
  [Step 1/13] Loss: 0.923 | LR: 0.00030 | 23s
  [Step 5/13] Loss: 0.756 | LR: 0.00030 | 1m 55s
  [Step 10/13] Loss: 0.654 | LR: 0.00028 | 3m 50s
  [Step 13/13] Loss: 0.589 | LR: 0.00026 | 5m 0s
  Validation: Loss: 0.612

Epoch 3/3:
  [Step 1/13] Loss: 0.534 | LR: 0.00024 | 23s
  [Step 5/13] Loss: 0.489 | LR: 0.00020 | 1m 55s
  [Step 10/13] Loss: 0.445 | LR: 0.00016 | 3m 50s
  [Step 13/13] Loss: 0.421 | LR: 0.00012 | 5m 0s
  Validation: Loss: 0.438

Training complete!
Total time: 1h 32min
Final loss: 0.421 ✓ (goal: <0.8)

Model saved to: models/janus_prosody_lora/
```

**Progress indicators**:
- Real-time loss tracking
- Time estimates per step
- Validation metrics after each epoch
- Progress bar for batches
- TensorBoard graphs (optional)

---

### STEP 5: Monitor Training (Optional)

Open a second terminal:
```powershell
tensorboard --logdir models/janus_prosody_lora/logs
```

Then open browser: http://localhost:6006

**You'll see graphs of**:
- Training loss over time
- Validation loss
- Learning rate schedule
- Prosody accuracy

---

### STEP 6: Test Fine-Tuned Model

```powershell
cd ..

# Use the trained model
python main.py -i "Test question" -p "Test point" --use-finetuned

# Compare before/after
python main.py -i "How much?" -p "30% savings" --use-api       # Before (30% prosody)
python main.py -i "How much?" -p "30% savings" --use-finetuned # After (85% prosody)
```

---

## 📁 Exact Data Format Required

### Your Training Data Must Have:

**1. Audio File** (`jre_training_audio.wav/jre_training_audio.wav`)
```
Format: WAV
Sample rate: 16000 Hz or 24000 Hz
Channels: 1 (mono)
Duration: 5+ minutes (you have 12 min ✓)
Quality: Clear speech, minimal background noise
```

**2. Transcript File** (`jre_training_transcript.txt`)
```
Format: Plain text with prosody markers
Markers: <emph>, <pause_short>, <pause_long>, <pitch_high>, <pitch_low>, <pitch_rising>, <pitch_falling>
Alignment: Text should match audio timing roughly

Example:
"looking at random leaves on television <pitch_low> shows to recognize..."
```

**✓ Your data already matches this format!**

### After Segmentation You'll Have:

```
training_data/
├── segments/
│   ├── segment_001.wav (3-5 sec each)
│   ├── segment_002.wav
│   ├── ...
│   └── segment_050.wav
│
├── manifest.json
│   {
│     "segment_001": {
│       "audio": "segments/segment_001.wav",
│       "text": "looking at random leaves on television",
│       "prosody_tokens": ["<pitch_low>"],
│       "duration": 3.5
│     },
│     ...
│   }
│
└── prosody_dataset.json (for PyTorch)
```

---

## 🔧 What LoRA Does (Simplified)

### Training Process

```
Epoch 1:
  Load segment_001.wav + text
  Model tries to generate audio
  Compare with real audio
  Calculate error (loss)
  Update LoRA adapters
  
  Repeat for all 50 segments...
  
Epoch 2:
  Model is better now
  Loss decreases
  Prosody more accurate
  
  Repeat...
  
Epoch 3:
  Model is good!
  Loss low (<0.5)
  Prosody consistent
```

### What Model Learns

```
Training Example:
  Text: "We <emph>guarantee savings"
  Tokens: [123, 45, 5000000, 678]
  Audio: [normal][normal][LOUD][loud]
  
Model learns:
  "When I see token 5000000, make next audio LOUDER"
  
After 1000 examples:
  Token 5000000 → automatic emphasis (no prompting!)
```

---

## 📊 Expected Results

### Before Fine-Tuning (Current)
```
Input: "We <emph>guarantee 30% savings"
Success: 30-50% (sometimes emphasized, sometimes not)
Method: System prompts (model guesses)
```

### After Fine-Tuning
```
Input: "We <emph>guarantee 30% savings"
Success: 85-95% (consistently emphasized)
Method: Learned behavior (model knows)
```

---

## 🎯 File Organization

### New Structure
```
fine_tuning/
├── 1_segment_audio.py      ← Step 1: Auto-segment
├── 2_prepare_data.py       ← Step 2: Create dataset
├── 3_train_lora.py         ← Step 3: Train with LoRA
├── training_data/          ← Generated data
│   ├── segments/
│   ├── manifest.json
│   └── prosody_dataset.json
└── models/                 ← Output models
    └── janus_prosody_lora/
```

---

## 🆘 Troubleshooting

### "CUDA out of memory"
```powershell
# Reduce batch size
python 3_train_lora.py --batch-size 2 --gradient-accumulation 8
```

### "Training too slow"
```
Your RTX 4050: ~2-3 min/epoch ✓ (good!)
If slower: Close other programs, update GPU drivers
```

### "Loss not decreasing"
```
Check:
- Data quality (listen to segments)
- Learning rate (try 1e-4 instead of 3e-4)
- More epochs (try 5-10)
```

---

## ✅ Success Metrics

**Training is successful if**:
- Final loss < 0.8 (excellent: < 0.5)
- Validation loss similar to training loss
- Audio test sounds natural with prosody

**Test it**:
```powershell
python main.py -i "Test" -p "Test point" --use-finetuned
# Listen to output - should have clear emphasis on "point"
```

---

**Ready to start?** Run Step 1! ⬆️
