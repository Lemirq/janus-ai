# Janus AI - Quick Reference

## 🎯 Simple Usage

```powershell
# One-line response generation
python simple_demo.py -i "Their question" -p "Your point 1" "Your point 2"

# Interactive mode
python simple_demo.py --interactive
```

## 📝 Output Format

```
[THINKING PROCESS]
  - Shows AI reasoning
  
[FINAL RESPONSE]
  - Plain text
  - With prosody markers
  
[AUDIO FILE]
  - response.wav (only final response, with prosody in speech)
```

## 🎵 Prosody Markers

```
<EMPH>word          - Emphasize
<STRONG>word        - Strong emphasis
<SLOW>              - Slow down
<PAUSE_SHORT>       - Brief pause
<PAUSE_LONG>        - Long pause
<CONFIDENT>         - Confident tone
<FRIENDLY>          - Friendly tone
```

## 🔧 How It Works Now

**Without Fine-tuning**: System prompt tells model what markers mean
- Success rate: ~30-50%
- Sometimes ignored
- API errors if too complex

**Keep it simple**: 1-2 markers per sentence works best

## 🚀 Fine-tuning (For Better Results)

```powershell
# Step 1: Auto-segment audio (no manual work!)
python fine_tuning/auto_segment_audio.py

# Step 2: Train model
pip install torch
python fine_tuning/train_prosody.py
```

**After fine-tuning**: 85-95% success rate, consistent prosody

## ✅ Examples

### Pricing Question
```powershell
python simple_demo.py -i "How much does this cost?" -p "30% cheaper" "No hidden fees"
```

### Security Concern
```powershell
python simple_demo.py -i "Is this secure?" -p "Military-grade encryption" "Zero breaches"
```

### ROI Question
```powershell
python simple_demo.py -i "When will we see ROI?" -p "6 months average" "Proven results"
```

---

**Full guide**: See `USAGE_GUIDE.md`
