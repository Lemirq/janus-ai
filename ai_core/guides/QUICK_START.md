# 🚀 Janus AI - Quick Start (3 Steps)

## Step 1: Set API Key (PowerShell)
```powershell
$env:BOSON_API_KEY="YOUR_KEY"
```

## Step 2: Verify Installation
```powershell
python -c "import openai; print('✓ Ready to go!')"
```

## Step 3: Run Demo
```powershell
python main.py
```

**Expected output:**
```
Janus AI initialized and ready for persuasion assistance
```

---

## ✅ What's Fixed

1. ❌ MySQL-python error → ✅ Skipped (not needed)
2. ❌ Export command error → ✅ Use `$env:` instead
3. ❌ Llama gated model → ✅ Use GPT-2 tokenizer

## 🔧 If You Need More

**Install additional packages:**
```powershell
pip install fastapi uvicorn websockets  # For API server
pip install pytest                       # For testing
```

**Make API key permanent:**
```powershell
# Create .env file
echo "BOSON_API_KEY=YOUR_KEY" > .env
```

## 📖 Full Documentation

- See `INSTALLATION_GUIDE.md` for detailed explanations
- See `README.md` for system architecture
- See `PROJECT_SUMMARY.md` for technical overview

---

**You're all set!** 🎉
