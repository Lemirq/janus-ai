# Janus AI - Installation & Setup Guide

## ✅ Issues Fixed

### Issue 1: MySQL-python Build Error
**Problem**: `MySQL-python` tried to compile C extensions requiring Microsoft Visual C++ 14.0

**Root Cause**: Optional dependency from `pandas` or `scikit-learn` trying to install MySQL connector

**Solution**: 
- MySQL is not needed for Janus AI (we only use audio APIs)
- Installed core dependencies without optional MySQL packages
- Used `--no-deps` flag for packages with problematic optional dependencies

### Issue 2: PowerShell Export Command
**Problem**: `export` command doesn't exist in PowerShell

**Solution**: Use PowerShell syntax instead:
```powershell
# ✅ Correct (PowerShell)
$env:BOSON_API_KEY="your-api-key"

# ❌ Wrong (Unix/Bash)
export BOSON_API_KEY="your-api-key"
```

### Issue 3: Gated Llama Model Access
**Problem**: Code tried to download `meta-llama/Llama-3.2-3B` which requires HuggingFace authentication

**Solution**: 
- Switched to GPT-2 tokenizer (publicly available, no authentication needed)
- Added fallback to simple tokenization if GPT-2 unavailable
- Since we're using Boson API (not local models), we don't need Llama locally

## 🚀 Quick Start (Windows PowerShell)

### 1. Set Up Environment

```powershell
# Navigate to project directory
cd C:\Users\prabh\Downloads\boson-ai-hackathon\ai_core

# Activate virtual environment (if you have one)
.\venv\Scripts\Activate.ps1

# Set API key for current session
$env:BOSON_API_KEY="YOUR_KEY"

# Verify it's set
echo $env:BOSON_API_KEY
```

### 2. Install Core Dependencies

```powershell
# Install minimal requirements (already done)
pip install transformers numpy openai aiohttp python-dotenv soundfile

# For GPT-2 tokenizer (downloads automatically on first run)
# No action needed - it will download when you run the code
```

### 3. Run the Application

```powershell
python main.py
```

**Expected Output:**
```
Janus AI initialized and ready for persuasion assistance
```

## 📦 What's Installed

### Core Dependencies (Working)
- ✅ `openai` - Boson AI API client
- ✅ `transformers` - For tokenization
- ✅ `numpy` - Array operations
- ✅ `aiohttp` - Async HTTP client
- ✅ `python-dotenv` - Environment variable management
- ✅ `soundfile` - Audio file handling

### Skipped Dependencies (Not needed for basic operation)
- ❌ `torch` - Heavy ML framework (not needed for API-only mode)
- ❌ `pandas` - Data processing (optional)
- ❌ `librosa` - Audio processing (optional, has complex dependencies)
- ❌ `pyaudio` - Audio I/O (requires C++ build tools)

## 🔧 Configuration Options

### Option 1: Environment Variable (Current Session)
```powershell
$env:BOSON_API_KEY="your-api-key"
```

### Option 2: Permanent Environment Variable
```powershell
[System.Environment]::SetEnvironmentVariable('BOSON_API_KEY', 'your-api-key', 'User')
# Restart PowerShell after this
```

### Option 3: .env File (Recommended)
Create `.env` file in `ai_core` directory:
```
BOSON_API_KEY=YOUR_KEY
```

Code automatically loads it with `python-dotenv`

## 📝 Testing the System

### Test 1: Basic Import Test
```powershell
python -c "from core.prosody_tokenizer import ProsodyTokenizer; print('✓ Prosody tokenizer works')"
```

### Test 2: API Connection Test
```python
# test_api.py
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("BOSON_API_KEY"),
    base_url="https://hackathon.boson.ai/v1"
)

response = client.chat.completions.create(
    model="Qwen3-32B-thinking-Hackathon",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

Run with:
```powershell
python test_api.py
```

### Test 3: Prosody Token Test
```python
# test_prosody.py
from core.prosody_tokenizer import ProsodyTokenizer

tokenizer = ProsodyTokenizer()

# Test encoding with prosody
text = "We <STRONG>guarantee <EMPH>30% savings"
tokens = tokenizer.encode_with_prosody(text)
print(f"Tokens: {tokens}")
print(f"Prosody tokens found: {[t for t in tokens if t >= 128000]}")
```

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'X'"
**Solution**: Install the missing module
```powershell
pip install X
```

### Issue: "OpenAIError: The api_key client option must be set"
**Solution**: Set the environment variable
```powershell
$env:BOSON_API_KEY="your-key"
```

### Issue: "Cannot access gated repo"
**Solution**: Already fixed! We now use GPT-2 instead of Llama

### Issue: "Microsoft Visual C++ 14.0 or greater is required"
**Solution**: Already fixed! We skip packages requiring C++ compilation

### Issue: Symlink Warning
**Solution**: Harmless warning. To fix:
- Enable Windows Developer Mode, OR
- Run PowerShell as Administrator, OR
- Ignore it (caching still works)

## 📚 Next Steps

### For Development
1. Install additional dependencies as needed:
```powershell
pip install fastapi uvicorn  # For API server
pip install pytest black      # For testing/formatting
```

2. Set up audio I/O for real-time processing
3. Connect Bluetooth earpiece
4. Implement actual audio streaming

### For Production
1. Use proper tokenizer (authenticate with HuggingFace)
2. Install PyTorch for local model inference (optional)
3. Set up proper logging and monitoring
4. Implement error handling and recovery

## 🎯 Current System Capabilities

✅ **Working Now:**
- All core modules import successfully
- API client configured
- Prosody tokenizer functional (using GPT-2)
- Configuration system working
- Demo mode runs without errors

🚧 **Needs Integration:**
- Real audio input/output
- Bluetooth earpiece connection
- Live conversation processing
- Response streaming

## 💡 Tips

1. **Keep API key secret**: Don't commit `.env` to git
2. **Use virtual environment**: Isolate dependencies
3. **Check API limits**: Monitor your Boson AI usage
4. **Start small**: Test with text before adding audio
5. **Read logs**: Check console output for warnings

---

**You're now ready to use Janus AI!** 🎉

Run `python main.py` to start the demo.
