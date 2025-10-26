# 🗺️ Janus AI - Documentation Navigation

**Choose your reading path below** ⬇️

---

## 🚀 Quick Decision Tree

```
What do you want to do?

├─ Use it RIGHT NOW
│  └─ Read: 01_Quick_Start.md (2 min read)
│     Then: python main.py --interactive
│
├─ TRAIN for better quality (have 2 hours)
│  └─ Read: 02_Fine_Tuning_LoRA.md (5 min read)
│     Or: ../fine_tuning/README_TRAINING.md (exact steps)
│     Then: cd fine_tuning && python 1_segment_audio.py
│
├─ UNDERSTAND the project
│  └─ Read: 00_START_HERE.md (10 min read)
│     Then: 04_Complete_Reference.md
│
└─ FIX a problem
   └─ Read: 03_Troubleshooting.md
      Or: Ask in terminal - errors are helpful now
```

---

## 📚 All Documents Explained

### Getting Started (Read First)
| File | Purpose | Time | When |
|------|---------|------|------|
| **00_START_HERE.md** | Project overview, paths | 10 min | First time |
| **01_Quick_Start.md** | Basic usage | 2 min | Using now |

### Training (Read When Ready)
| File | Purpose | Time | When |
|------|---------|------|------|
| **02_Fine_Tuning_LoRA.md** | Complete training guide | 5 min | Ready to train |
| **../fine_tuning/README_TRAINING.md** | Exact steps, time estimates | 3 min | Training today |

### Reference (Read When Needed)
| File | Purpose | Time | When |
|------|---------|------|------|
| **03_Troubleshooting.md** | Common issues | 3 min | Having problems |
| **04_Complete_Reference.md** | Everything in one doc | 20 min | Deep dive |
| **05_Project_Status.md** | What was completed | 5 min | Review |
| **06_Final_Summary.md** | All changes made | 5 min | Understanding fixes |

### Navigation
| File | Purpose |
|------|---------|
| **README.md** (this file) | Guide to guides |
| **00_README_START.md** | Comprehensive navigation |

---

## 🎯 By Use Case

### Use Case: "I want to test it now"
1. Read: **01_Quick_Start.md** (2 min)
2. Run: `python main.py --interactive`
3. If issues: **03_Troubleshooting.md**

### Use Case: "I want to fine-tune today"
1. Read: **../fine_tuning/README_TRAINING.md** (3 min)
2. Run: Step-by-step commands
3. Monitor: Terminal shows progress
4. Time: 1.5-2 hours on RTX 4050

### Use Case: "I want to understand everything"
1. Read: **00_START_HERE.md** (10 min)
2. Read: **04_Complete_Reference.md** (20 min)
3. Read: **../README.md** (technical details)

### Use Case: "Something's broken"
1. Read: **03_Troubleshooting.md**
2. Check terminal output (helpful errors)
3. Check code comments

---

## 📊 Quick Reference

### Commands
```powershell
# Use now
python main.py -i "Question" -p "Point 1" "Point 2"

# Interactive
python main.py --interactive

# Train
cd fine_tuning && python 1_segment_audio.py
```

### Files
```
Root: README.md only
Guides: 9 organized docs (you're here)
Training: 3 numbered scripts
Output: All in output/ folder
```

### Prosody
```
7 tokens: <emph>, <pause_short>, <pause_long>, 
          <pitch_high>, <pitch_low>, <pitch_rising>, <pitch_falling>
IDs: 5,000,000 - 5,000,006
```

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Read Quick Start | 2 min |
| Use system | Instant |
| Read training guide | 5 min |
| Segment audio | 5 min |
| Prepare data | 2 min |
| Train (RTX 4050) | 1.5-2 hours |

---

## ✅ Reading Order Recommendations

### Minimal (5 minutes)
1. **01_Quick_Start.md**
2. Start using!

### Standard (15 minutes)
1. **00_START_HERE.md**
2. **01_Quick_Start.md** or **02_Fine_Tuning_LoRA.md**
3. Start!

### Complete (45 minutes)
1. **00_START_HERE.md**
2. **04_Complete_Reference.md**
3. **02_Fine_Tuning_LoRA.md**
4. **../README.md**

---

**Start with 01_Quick_Start.md** if you just want to use it now! 🚀

**Or 02_Fine_Tuning_LoRA.md** if you're ready to train! 🎓
