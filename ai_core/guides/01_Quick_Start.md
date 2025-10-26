# Janus AI - Quick Start Guide

## 🚀 Run in 30 Seconds

```powershell
# 1. Set API key
$env:BOSON_API_KEY="bai-ze8Ed2o8AC3RT9vee4WgeL384vpmTo_HnQZMzz3Xt4kdmBFB"

# 2. Generate response
python main.py -i "How much?" -p "30% savings" "No fees"

# 3. Listen to output
start output/response.wav
```

---

## 📋 Common Commands

```powershell
# Single response
python main.py -i "Question" -p "Point 1" "Point 2"

# Interactive mode
python main.py --interactive

# Custom output file
python main.py -i "Is it secure?" -p "Military encryption" -o output/secure.wav
```

---

## 📊 What You Get

```
[INPUT] Their statement
[YOUR POINTS] To emphasize
[THINKING PROCESS] How AI reasoned
[FINAL RESPONSE] Plain and with prosody
[AUDIO FILE] Saved to output/
```

---

## 🎯 Prosody Tokens (7 Total)

```
<emph>          - Emphasize word
<pause_short>   - Brief pause (0.5s)
<pause_long>    - Long pause (1.5s)
<pitch_high>    - Higher pitch
<pitch_low>     - Lower pitch
<pitch_rising>  - Rising intonation
<pitch_falling> - Falling intonation
```

---

## ✅ Verification

Your system works if:
- ✓ Response makes sense (not thinking process)
- ✓ Audio 5-20 seconds
- ✓ File 300-700 KB
- ✓ Saved to `output/` directory

---

**Next**: See `02_Fine_Tuning_Guide.md` for better quality (85%+ prosody consistency)
