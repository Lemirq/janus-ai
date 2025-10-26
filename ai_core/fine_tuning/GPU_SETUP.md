# GPU Setup for Training

## ❌ Problem Identified

Your PyTorch is **CPU-only version**:
```
PyTorch version: 2.8.0+cpu  ← Problem!
CUDA available: False       ← Should be True
```

But your GPU is working fine:
```
NVIDIA GeForce RTX 4050 Laptop GPU ✓
CUDA Version: 12.9 ✓
Driver: 577.03 ✓
```

**Solution**: Install PyTorch with GPU support

---

## 🔧 EXACT Fix

```powershell
# 1. Uninstall CPU version
pip uninstall -y torch torchvision torchaudio

# 2. Install GPU version (CUDA 11.8 recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Verify GPU is detected
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True

# 4. Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA GeForce RTX 4050 Laptop GPU
```

---

## 🎯 Alternative: Use requirements.txt

```powershell
# This file already has the correct PyTorch version
pip install -r requirements.txt
```

**Note**: If torch is already installed (CPU version), uninstall it first!

---

## 📊 Performance Difference

### Current (CPU)
```
Training test (10 samples, 1 epoch):
  Time: 43 seconds
  Speed: ~4.8 seconds/step
  
Full training (98 samples, 3 epochs):
  Estimated: 6-8 hours
```

### After GPU Fix
```
Training test (10 samples, 1 epoch):
  Time: ~10-12 seconds (4x faster!)
  Speed: ~1.2 seconds/step
  
Full training (98 samples, 3 epochs):
  Estimated: 1.5-2 hours (4x faster!)
```

**You'll save 5-6 hours with GPU!**

---

## ⚙️ Why This Happened

When you installed PyTorch, you likely ran:
```powershell
pip install torch  # Gets CPU version by default
```

To get GPU version, you MUST specify the index URL:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ Verification Checklist

After installing GPU version:

```powershell
# 1. Check PyTorch version
python -c "import torch; print(torch.__version__)"
# Should show: 2.x.x+cu118 (or cu121)
# Should NOT show: +cpu

# 2. Check CUDA
python -c "import torch; print(torch.cuda.is_available())"
# Should show: True

# 3. Check GPU name
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Should show: NVIDIA GeForce RTX 4050 Laptop GPU

# 4. Check GPU count
python -c "import torch; print('GPUs:', torch.cuda.device_count())"
# Should show: GPUs: 1
```

---

## 🚀 After Fix

```powershell
# Run test to confirm GPU is used
python 3_train_lora.py --test-mode --batch-size 4

# Should see:
# - Much faster (~10 sec vs 43 sec)
# - GPU utilization in nvidia-smi
# - Lower time per step
```

**Then run full training**:
```powershell
python 3_train_lora.py --epochs 3 --batch-size 4
```

**Expected**: 1.5-2 hours (not 6-8 hours!)

---

## 💡 Pro Tip

While training, open another terminal:
```powershell
# Monitor GPU usage
nvidia-smi -l 1
```

**Should show**:
- GPU Utilization: 80-95%
- Memory Used: ~4-5 GB / 6 GB
- Power: 60-80W

If utilization is 0%, GPU isn't being used!

---

## 🎯 Summary

**Issue**: PyTorch CPU version installed
**Fix**: Uninstall and reinstall with `--index-url https://download.pytorch.org/whl/cu118`
**Result**: 4x faster training (1.5-2 hrs vs 6-8 hrs)

**Install now**:
```powershell
pip uninstall -y torch torchvision torchaudio
pip install -r requirements.txt
```

**Verify and train!**
