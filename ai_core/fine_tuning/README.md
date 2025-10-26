# Fine-Tuning Setup - Complete Guide

## ⚡ OPTION 1: Use Pre-Trained Model (Recommended)

### Download and Install (5 minutes)

**Skip the 2-hour training** and use the pre-trained model:

```powershell
# Step 1: Create models folder
cd C:\Users\<your-username>\...\janus-ai\ai_core\fine_tuning
mkdir models

# Step 2: Download pre-trained model
```

**Download from Google Drive**: [janus_prosody_lora.zip](https://drive.google.com/file/d/18uoP8ecfsCwDTEdTJZKzwhGTulTSoTsb/view?usp=sharing)

```powershell
# Step 3: Extract ZIP into the models folder
# Right-click the ZIP → Extract All → Select fine_tuning/models/

# Or PowerShell:
Expand-Archive -Path janus_prosody_lora.zip -DestinationPath models/

# Step 4: Verify structure
dir models\janus_prosody_lora
# Should show: adapter_model.safetensors, adapter_config.json, etc.
```

**That's it!** The trained model is ready to use.

---

## 🔧 OPTION 2: Train From Scratch (Advanced)

### Full Training Setup (RTX 4050: ~2 hours total)

### Install Dependencies

```powershell
cd fine_tuning

# Install everything needed for GPU training
pip install -r requirements.txt
```

**This installs**:
- PyTorch with CUDA 11.8 (GPU support)
- PEFT/LoRA for efficient training
- Transformers, Accelerate
- TensorBoard for monitoring
- All training utilities

---

## ✅ Verify GPU is Working

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Should show**:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

**If it shows `False`**:
- Your PyTorch is CPU-only version
- Run: `pip uninstall torch torchvision torchaudio`
- Then: `pip install -r requirements.txt` again

---

## 📋 Training Steps

```powershell
# STEP 1: Segment audio (5 min)
python 1_segment_audio.py

# STEP 2: Prepare dataset (2 min)
python 2_prepare_data.py

# STEP 3: Train with LoRA (1.5-2 hours on GPU)
python 3_train_lora.py --epochs 3 --batch-size 4
```

---

## 🎯 GPU vs CPU Performance

### RTX 4050 (GPU)
```
Batch size: 4
Step time: 2-3 seconds
Epoch time: ~30 minutes
Total (3 epochs): 1.5-2 hours ✓
```

### CPU Only
```
Batch size: 1-2 (limited by RAM)
Step time: 20-30 seconds
Epoch time: ~2-3 hours
Total (3 epochs): 6-8 hours ✗
```

**GPU is 4x faster!** Use the GPU version.

---

## 🔧 Troubleshooting

### "CUDA available: False"

**Problem**: PyTorch CPU-only version installed

**Fix**:
```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"

**Fix**:
```powershell
python 3_train_lora.py --batch-size 2 --gradient-accumulation 8
```

### "No NVIDIA driver"

**Fix**: Install NVIDIA drivers
- Download from: https://www.nvidia.com/download/index.aspx
- Select: RTX 4050 Laptop
- Install and restart

---

## 📊 What Gets Installed

```
torch (2.4GB) - CUDA 11.8 version
peft (50MB) - LoRA implementation
transformers (400MB) - Model handling
accelerate (20MB) - Training optimization
tensorboard (50MB) - Progress monitoring
tqdm (1MB) - Progress bars
+ dependencies
```

**Total**: ~3-4 GB download

---

## ⚡ Quick Check

After installing, verify:

```powershell
# Check PyTorch
python -c "import torch; print(torch.__version__)"
# Should show: 2.x.x+cu118 (NOT +cpu)

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"
# Should show: True

# Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should show: NVIDIA GeForce RTX 4050 Laptop GPU
```

---

## 🎯 Ready to Train

Once all checks pass:

```powershell
python 3_train_lora.py --epochs 3 --batch-size 4
```

**Training will use GPU automatically!**

**Time**: ~1.5-2 hours (with GPU) vs 6-8 hours (CPU only)

---

**Install**: `pip install -r requirements.txt`

**Verify**: Check CUDA is available

**Train**: Run script and enjoy 4x speedup!
