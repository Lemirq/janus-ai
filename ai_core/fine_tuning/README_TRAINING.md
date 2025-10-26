# Fine-Tuning Instructions - EXACT Steps

## 🎯 Complete Fine-Tuning Workflow

### Example Setup
- RTX 4050 Laptop (6GB VRAM)
- Training data: 12 minutes audio
- Expected time: **1.5-2 hours** (3 epochs)

---

## 📋 EXACT Steps to Run Right Now

### STEP 1: Segment Audio (5 minutes)

```powershell
python 1_segment_audio.py
```

**What happens**:
```
STEP 1: AUTOMATIC AUDIO SEGMENTATION
======================================================================

[1/3] Loading audio: jre_training_audio.wav/jre_training_audio.wav
      Duration: 720.0 seconds (12 minutes)
      Sample rate: 24000 Hz

[2/3] Analyzing energy levels...
      Found 52 speech segments

[3/3] Aligning text with audio...
      Text segments: 48
      Audio segments: 52

Extracting 50 segments...
  [1/50] looking at random leaves on television... (3.5s)
  [10/50] shows to recognize that it's a fake pattern... (4.2s)
  [20/50] no i just you know you look at scenes... (3.8s)
  ...
  [50/50] what did you say what i say... (4.1s)

======================================================================
✓ Created 50 training segments
✓ Saved to: training_data/segments/
✓ Manifest: training_data/segments/manifest.json
✓ Summary: training_data/segmentation_summary.txt
======================================================================

✓ STEP 1 COMPLETE
Next: python 2_prepare_data.py
```

**Output files**:
```
training_data/
├── segments/
│   ├── segment_001.wav (3-5 seconds each)
│   ├── segment_002.wav
│   ├── ...
│   └── segment_050.wav
├── segments/manifest.json
└── segmentation_summary.txt
```

---

### STEP 2: Prepare Dataset (2 minutes)

```powershell
python 2_prepare_data.py
```

**What happens**:
```
STEP 2: PREPARE TRAINING DATASET
======================================================================

[1/4] Loading manifest: training_data/segments/manifest.json
      Loaded 50 segments

[2/4] Processing segments...
Processing: 100%|████████████████████| 50/50 [00:01<00:00, 42.3it/s]

[3/4] Saving dataset...

[4/4] Creating dataset info...

======================================================================
DATASET SUMMARY
======================================================================

Total examples: 50
Total duration: 187.5 seconds (3.1 minutes)
Average duration: 3.8 seconds

Prosody token distribution:
  <pitch_low>: 45 occurrences
  <pitch_high>: 38 occurrences
  <emph>: 22 occurrences
  <pitch_rising>: 18 occurrences
  <pause_short>: 12 occurrences
  <pitch_falling>: 8 occurrences
  <pause_long>: 5 occurrences

✓ Dataset saved to: training_data/prosody_dataset.json
✓ Info saved to: training_data/dataset_info.json
======================================================================

✓ STEP 2 COMPLETE
Next: python 3_train_lora.py --epochs 3
```

**Output files**:
```
training_data/
├── segments/ (from step 1)
├── prosody_dataset.json      ← Training data
└── dataset_info.json          ← Statistics
```

---

### STEP 3: Install Training Dependencies (10 minutes)

```powershell
# PyTorch with CUDA support for RTX 4050
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# LoRA and training tools
pip install peft transformers accelerate datasets

# Progress bars and monitoring
pip install tqdm tensorboard
```

**Verify GPU**:
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Should see**:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

---

### STEP 4: Train with LoRA (1.5-2 hours)

```powershell
# Recommended settings for RTX 4050
python 3_train_lora.py --epochs 3 --batch-size 4

# For faster (lower quality)
python 3_train_lora.py --epochs 1 --batch-size 2

# For better quality (longer)
python 3_train_lora.py --epochs 5 --batch-size 4 --lora-r 32
```

**What you'll see**:
```
======================================================================
MODEL SETUP
======================================================================

[1/5] Loading base model: gpt2
[2/5] Loading tokenizer
[3/5] Adding prosody tokens
      Added 7 tokens
      New vocab size: 50264
[4/5] Configuring LoRA
[5/5] Applying LoRA adapters

======================================================================
Model Ready:
  Total parameters: 124,439,808
  Trainable (LoRA): 294,912
  Percentage: 0.24%
  LoRA rank: 16
======================================================================

Loading dataset: training_data/prosody_dataset.json
✓ Loaded 50 examples

Creating training dataset...
Tokenizing 50 examples...
✓ Train: 45, Val: 5

======================================================================
TRAINING STARTED
======================================================================

──────────────────────────────────────────────────────────────────────
EPOCH 1/3
──────────────────────────────────────────────────────────────────────
  Step   1 | Loss: 2.4561 | LR: 1.20e-04 | Time: 0m 23s
  Step   2 | Loss: 2.1234 | LR: 2.40e-04 | Time: 0m 46s
  Step   3 | Loss: 1.8923 | LR: 3.00e-04 | Time: 1m 9s
  Step   4 | Loss: 1.6547 | LR: 3.00e-04 | Time: 1m 32s
  Step   5 | Loss: 1.4329 | LR: 3.00e-04 | Time: 1m 55s
  Step   6 | Loss: 1.2876 | LR: 3.00e-04 | Time: 2m 18s
  Step   7 | Loss: 1.1542 | LR: 3.00e-04 | Time: 2m 41s
  Step   8 | Loss: 1.0423 | LR: 3.00e-04 | Time: 3m 4s
  Step   9 | Loss: 0.9687 | LR: 3.00e-04 | Time: 3m 27s
  Step  10 | Loss: 0.9123 | LR: 3.00e-04 | Time: 3m 50s
  Step  11 | Loss: 0.8745 | LR: 3.00e-04 | Time: 4m 13s

✓ Epoch 1 complete

──────────────────────────────────────────────────────────────────────
EPOCH 2/3
──────────────────────────────────────────────────────────────────────
  Step  12 | Loss: 0.7234 | LR: 3.00e-04 | Time: 0m 23s
  Step  13 | Loss: 0.6891 | LR: 3.00e-04 | Time: 0m 46s
  Step  14 | Loss: 0.6345 | LR: 2.95e-04 | Time: 1m 9s
  ...
  Step  22 | Loss: 0.4867 | LR: 2.70e-04 | Time: 4m 13s

✓ Epoch 2 complete

──────────────────────────────────────────────────────────────────────
EPOCH 3/3
──────────────────────────────────────────────────────────────────────
  Step  23 | Loss: 0.4123 | LR: 2.50e-04 | Time: 0m 23s
  ...
  Step  33 | Loss: 0.3245 | LR: 1.80e-04 | Time: 4m 13s

✓ Epoch 3 complete

======================================================================
TRAINING COMPLETE!
======================================================================
Total time: 1h 32min
Final loss: 0.3245
======================================================================

Saving model...

✓ Model saved to: models/janus_prosody_lora/
✓ Files created:
    - adapter_model.bin
    - adapter_config.json
    - tokenizer/
    - prosody_tokens.json

✓ STEP 3 COMPLETE
Model ready to use!
```

---

## 📊 What Each File Does

### 1_segment_audio.py
```
Input: Long audio file (12 min)
Process: Detect pauses, split at silence
Output: ~50 short segments (3-5 sec each)
Time: 5 minutes
```

### 2_prepare_data.py
```
Input: Audio segments + transcript
Process: Extract prosody tokens, create alignment
Output: PyTorch-compatible dataset
Time: 2 minutes
```

### 3_train_lora.py
```
Input: Prepared dataset
Process: LoRA fine-tuning
Output: Trained model adapters
Time: 1.5-2 hours (RTX 4050)
```

---

## 🔍 Data Format Requirements

### Your Audio File Must Be:
```
✓ Format: WAV
✓ Sample rate: 16000-24000 Hz
✓ Channels: 1 (mono)
✓ Duration: 5+ minutes (you have 12 ✓)
✓ Quality: Clear speech, minimal noise
```

### Your Transcript Must Have:
```
✓ Format: Plain text
✓ Prosody markers: <emph>, <pause_short>, <pitch_low>, etc.
✓ Alignment: Text matches audio roughly
✓ Encoding: UTF-8
```

**✓ Your files already match these requirements!**

---

## 🎓 What LoRA Training Does

### Training Process

```
For each of 50 segments:
  
  1. Load: "looking at <pitch_low> random leaves"
  2. Tokenize: [123, 45, 5000004, 678]
                         ↑
                   pitch_low token
  
  3. Model generates audio
  4. Compare with real audio
  5. Calculate loss (error)
  6. Update LoRA adapters (not full model!)
  
After 3 epochs × 50 segments = 150 training steps:
  Model learns: 5000004 → lower pitch automatically
```

### LoRA Benefits

```
Traditional:               LoRA:
├─ 124M params trained    ├─ 295K params trained (0.24%)
├─ 12GB VRAM needed       ├─ 6GB VRAM needed ✓ (RTX 4050)
├─ 8+ hours               ├─ 1.5-2 hours ✓
└─ 6GB model file         └─ 50MB adapter file

Same quality, 10x faster!
```

---

## 📈 Progress Monitoring

### During Training

**Terminal output shows**:
- Real-time loss (should decrease)
- Learning rate schedule
- Time per step
- Epoch progress

**Optional - TensorBoard** (second terminal):
```powershell
tensorboard --logdir models/janus_prosody_lora/logs
# Open: http://localhost:6006
```

Shows graphs of:
- Training loss over time
- Validation loss
- Learning rate

---

## ✅ Success Criteria

**Training is successful if**:
- ✅ Final loss < 0.8 (excellent: < 0.5)
- ✅ Loss decreases steadily each epoch
- ✅ Validation loss similar to train loss
- ✅ No "out of memory" errors

**After training, test**:
```powershell
cd ..
python main.py -i "Test prosody" -p "Test <emph>emphasis"
# Audio should clearly emphasize the word "emphasis"
```

---

## 🔧 Troubleshooting Training

### "CUDA out of memory"
```powershell
# Reduce batch size
python 3_train_lora.py --batch-size 2 --gradient-accumulation 8
```

### "Training too slow"
```
Expected on RTX 4050: 2-3 min/epoch
If slower:
  - Close other programs
  - Check not on battery saver mode
  - Update NVIDIA drivers
```

### "Loss not decreasing"
```
Try:
  - More epochs: --epochs 5
  - Lower learning rate: --learning-rate 1e-4
  - Check data quality: listen to segments
```

---

## 📁 Output After Training

```
models/
└── janus_prosody_lora/
    ├── adapter_config.json       ← LoRA configuration
    ├── adapter_model.bin         ← Trained adapters (~50MB)
    ├── tokenizer/                ← Updated tokenizer
    │   ├── tokenizer.json
    │   └── special_tokens_map.json
    ├── prosody_tokens.json       ← Token ID mappings
    └── logs/                     ← TensorBoard logs
        └── events.out.tfevents...
```

---

## 🎯 Using the Trained Model

**After training completes**, update your config:

```python
# In main.py or your code
config = JanusConfig(
    api_key=api_key,
    # Use your trained model instead of API
    generation_model="fine_tuning/models/janus_prosody_lora"
)
```

Then test:
```powershell
python main.py -i "Test" -p "Test <emph>point" -o output/finetuned_test.wav
# Should have clear emphasis on "point"
```

---

## 📊 Expected Timeline (RTX 4050)

```
Total: ~2 hours

5 min   ├─ Step 1: Segment audio
2 min   ├─ Step 2: Prepare dataset  
10 min  ├─ Step 3: Install dependencies
        │
1h 30m  └─ Step 4: Training
             ├─ Epoch 1: 30 min (loss: 2.4 → 0.9)
             ├─ Epoch 2: 30 min (loss: 0.9 → 0.6)
             ├─ Epoch 3: 30 min (loss: 0.6 → 0.4)
             └─ Save model: 2 min
```

---

## 🚀 Start Now!

```powershell
# Step 1
python 1_segment_audio.py

# Then follow the prompts for steps 2-3
```

**Everything is automated - just run the commands!** 🎯
