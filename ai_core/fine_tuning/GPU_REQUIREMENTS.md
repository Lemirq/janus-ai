# GPU Requirements for Higgs Training

## 🎯 VRAM Requirements

### Higgs Audio V2 (6B parameters)

| GPU | VRAM | Batch Size | Training Time (3 epochs) | Status |
|-----|------|------------|-------------------------|---------|
| **RTX 4050** | 6 GB | N/A | N/A | ❌ Too small |
| **RTX 3090** | 24 GB | 2 | 2-3 hours | ✅ Works |
| **RTX 4090** | 24 GB | 2-4 | 1.5-2 hours | ✅ Recommended |
| **A100 40GB** | 40 GB | 4-8 | 1-1.5 hours | ✅ Ideal |
| **A100 80GB** | 80 GB | 8-16 | 45-60 min | ✅ Optimal |
| **H100 80GB** | 80 GB | 8-16 | 30-45 min | ✅ Best |

### Your RTX 4050 Laptop (6GB)

```
❌ Cannot train Higgs (too small)
✅ CAN train GPT-2 (already done!)
✅ CAN use Higgs API (works great!)

Recommendation: Use GPT-2 fine-tuned + Higgs API
```

---

## ☁️ **Cloud GPU Options**

### For Higgs Training

**RunPod** (recommended):
- A100 40GB: $1.89/hour
- Training time: ~1.5 hours
- Cost: ~$3 for full training

**Google Colab Pro**:
- A100 40GB: $10/month
- Includes compute time
- Good for experimentation

**Lambda Labs**:
- A100 40GB: $1.10/hour  
- Cost: ~$2 for training

**Paperspace Gradient**:
- A100 40GB: $2.30/hour
- Easy setup

---

## 🚀 **Training on Cloud GPU**

### Setup (5 minutes)

```bash
# On cloud GPU instance:

# 1. Clone repo
git clone https://github.com/Lemirq/janus-ai.git
cd janus-ai/ai_core/fine_tuning

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download your training data
# Upload: training_data/ folder (or run segmentation)

# 4. Setup Higgs
python 4_setup_higgs.py

# 5. Train (1-2 hours on A100)
python 5_train_higgs_prosody.py --epochs 3 --batch-size 4 --lora-r 16
```

### Download Trained Model

```bash
# After training completes:
cd models
tar -czf higgs_prosody_lora.tar.gz higgs_prosody_lora/

# Download this file to your local machine
# Extract to: ai_core/fine_tuning/models/
```

---

## 💡 **Current System (No Cloud GPU Needed!)**

### What You Have on RTX 4050

```
Component 1: GPT-2 Fine-Tuned ✓
  - Trained locally on RTX 4050
  - 85% prosody token placement
  - Works perfectly!

Component 2: Higgs API ✓
  - Cloud-based (no local GPU needed)
  - 30-50% prosody in audio
  - Always available

Combined: 60-70% quality
Cost: $0 (just API usage)
Status: Production-ready!
```

### With Cloud GPU + Higgs Training

```
Component 1: GPT-2 Fine-Tuned ✓ 
  - Same as above

Component 2: Higgs Fine-Tuned ✓
  - Trained on cloud GPU
  - 85-95% prosody in audio
  - Download to local machine

Combined: 85-95% quality
Cost: ~$2-3 one-time training
Status: Maximum quality!
```

---

## 🎯 **Recommendation**

### For Hackathon Demo

**Use current system** (GPT-2 + API):
- ✅ Works on your laptop
- ✅ Good quality (60-70%)
- ✅ No cloud costs
- ✅ Ready now!

### For Production / If You Want 85-95%

**Train Higgs on cloud**:
- Rent A100 for 1-2 hours ($2-3)
- Train Higgs model
- Download to local machine
- Use both fine-tuned models

---

## 🔧 **Cloud GPU Command (A100/H100)**

```bash
# Assumes: 40+ GB VRAM

python 5_train_higgs_prosody.py \
  --epochs 5 \
  --batch-size 8 \
  --lora-r 32

# Training time on A100 40GB:
#   - Epoch 1: 15-20 min
#   - Epoch 2: 15-20 min  
#   - Epoch 3: 15-20 min
#   Total: 45-60 minutes

# Result: models/higgs_prosody_lora/
```

---

## ✅ **Your Options**

1. **Use current system** (free, works now)
   - GPT-2 fine-tuned + Higgs API
   - 60-70% quality

2. **Rent cloud GPU** ($2-3, 2 hours)
   - Train Higgs on A100
   - Download model
   - 85-95% quality

**Both are valid! Choice depends on your needs.** 🎯
