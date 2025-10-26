# 🎯 Janus AI - START HERE

## Choose Your Path

### Path A: Download Pre-Trained Model ⚡ **RECOMMENDED**
```
1. Download: https://drive.google.com/file/d/18uoP8ecfsCwDTEdTJZKzwhGTulTSoTsb/view?usp=sharing
2. Create: ai_core/fine_tuning/models/ folder
3. Extract: ZIP into models/ folder
4. Use: python main.py --interactive
```
**Prosody**: 85-95% consistent (production quality)  
**Time**: 5 minutes setup  
**Read**: `../fine_tuning/DOWNLOAD_MODEL.md` (complete instructions)

### Path B: Use API Only (No Download)
```powershell
python main.py --interactive
```
**Prosody**: 30-50% consistent (testing only)  
**Time**: Instant  
**Read**: `01_Quick_Start.md` (2 min)

### Path C: Train Your Own Model 🎓
```powershell
cd fine_tuning
python 1_segment_audio.py
```
**Prosody**: 85-95% consistent (customized)  
**Time**: 1.5-2 hours on RTX 4050  
**Read**: `02_Fine_Tuning_LoRA.md` (5 min)

---

## 📚 Documentation Guide

| File | Read When | Time |
|------|-----------|------|
| **00_NAVIGATION.md** | Want guide to guides | 2 min |
| **01_Quick_Start.md** | Using now | 2 min |
| **02_Fine_Tuning_LoRA.md** | Ready to train | 5 min |
| **03_Troubleshooting.md** | Having issues | 3 min |
| **04_Complete_Reference.md** | Want everything | 20 min |

---

## ✅ What Janus Does

1. You input: Question + Your talking points
2. AI generates: Strategic persuasive response
3. Prosody added: Emphasis, pauses, pitch automatically
4. Audio created: Natural speech with prosody
5. File saved: `output/response.wav`

---

## 🎯 Quick Commands

```powershell
# Single response
python main.py -i "How much?" -p "30% savings" "No fees"

# Interactive mode
python main.py --interactive

# Fine-tune (in fine_tuning/)
python 1_segment_audio.py       # 5 min
python 2_prepare_data.py        # 2 min
python 3_train_lora.py --epochs 3  # 1.5-2 hrs
```

---

## 📊 Prosody Tokens (7 Total)

```
<emph>          5,000,000   Emphasize word
<pause_short>   5,000,001   Brief pause
<pause_long>    5,000,002   Long pause
<pitch_high>    5,000,003   Higher pitch
<pitch_low>     5,000,004   Lower pitch
<pitch_rising>  5,000,005   Rising intonation
<pitch_falling> 5,000,006   Falling intonation
```

---

## ⏱️ Time Estimates (RTX 4050)

| Task | Time |
|------|------|
| Use now | Instant |
| Segment audio | 5 min |
| Prepare data | 2 min |
| Install PyTorch | 10 min |
| Train (3 epochs) | 1.5-2 hours |

**Total for training**: ~2 hours from start to finish

---

## 🎓 Training vs No Training

### Without Training (Current)
- ✅ Works immediately
- ✅ No setup needed
- ⚠️ 30-50% prosody consistency
- ⚠️ Variable quality

### With LoRA Training
- ✅ 85-95% prosody consistency
- ✅ Consistent quality
- ⚠️ Requires 1.5-2 hours setup
- ⚠️ Needs GPU (RTX 4050 ✓)

---

## 📖 Recommended Reading Order

### Minimal (5 min)
1. This file
2. `01_Quick_Start.md`
3. Start using!

### Standard (20 min)
1. This file
2. `01_Quick_Start.md`
3. `02_Fine_Tuning_LoRA.md`
4. Decide: use now or train first

### Complete (60 min)
1. All guides in numerical order
2. `../README.md` (technical details)
3. Code comments

---

## ✅ You're Ready When

- ✓ Can run: `python main.py -i "Test" -p "Test"`
- ✓ Creates: `output/response.wav`
- ✓ File: 300-700 KB, 5-15 seconds
- ✓ Response makes sense

---

**Next**: Choose your path above ⬆️

**Lost?** Read `00_NAVIGATION.md`