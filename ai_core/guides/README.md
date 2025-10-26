# 📚 Janus AI - Documentation Index

**All guides organized for easy navigation**

---

## 🗺️ Navigation

| # | File | Purpose | Read When |
|---|------|---------|-----------|
| 0 | **00_NAVIGATION.md** | Guide to all guides | Lost/confused |
| 0 | **00_START_HERE.md** | Project overview | First time |
| 0 | **00_README_START.md** | Complete navigation | Reference |
| 1 | **01_Quick_Start.md** | Basic usage (30 sec) | Using now |
| 2 | **02_Fine_Tuning_LoRA.md** | Training guide | Ready to train |
| 3 | **03_Troubleshooting.md** | Problem solving | Having issues |
| 4 | **04_Complete_Reference.md** | Everything | Deep dive |
| 5 | **05_Project_Status.md** | What's complete | Review |
| 6 | **06_Final_Summary.md** | All changes | Understanding |

---

## 🚀 Quick Start

**New user?** → Read **00_START_HERE.md**

**Want to use now?** → Read **01_Quick_Start.md**, then:
```powershell
python main.py --interactive
```

**Want to train?** → Read **02_Fine_Tuning_LoRA.md**, then:
```powershell
cd fine_tuning
python 1_segment_audio.py
```

---

## 📖 By Topic

### Usage
- `01_Quick_Start.md` - Commands and examples
- `03_Troubleshooting.md` - Common issues

### Training
- `02_Fine_Tuning_LoRA.md` - Complete guide
- `../fine_tuning/README_TRAINING.md` - Exact steps

### Reference
- `04_Complete_Reference.md` - Everything in one doc
- `../README.md` - Technical architecture

---

## ⚡ Fastest Paths

**Use in 30 seconds**:
```powershell
python main.py -i "Test" -p "Test"
```

**Train in 2 hours**:
```powershell
cd fine_tuning
python 1_segment_audio.py
python 2_prepare_data.py
python 3_train_lora.py --epochs 3
```

**Understand everything**:
- Read: `04_Complete_Reference.md`

---

**Lost?** Start with `00_START_HERE.md` 🎯