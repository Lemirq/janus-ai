# Troubleshooting Guide

## 🔍 Common Issues & Solutions

### Issue: Error 500 from API

**Symptoms**: `{'error': {'message': 'invalid response from backend'}}`

**Causes**:
1. API temporarily overloaded
2. Text too complex
3. Rate limiting

**Solutions**:
✅ System auto-retries (no action needed)
✅ Simplify input (shorter questions, fewer points)
✅ Wait 30 seconds between requests

---

### Issue: Response Contains Thinking Process

**Symptoms**: Output shows "Okay, the user..." instead of actual response

**Solution**: Already fixed! Using non-thinking model now.

If still happening:
```powershell
# Check you're using latest code
git pull
python main.py -i "Test" -p "Test"
```

---

### Issue: Audio File Too Large

**Symptoms**: Files > 1MB, duration > 30 seconds

**Solution**: Already validated! System rejects files > 20s

If still happening:
- Response text too long
- System will truncate to 2-3 sentences automatically

---

### Issue: "BOSON_API_KEY not set"

**Solution**:
```powershell
$env:BOSON_API_KEY="your-key-here"
```

Or create `.env` file in `ai_core/`:
```
BOSON_API_KEY=your-key-here
```

---

### Issue: "ModuleNotFoundError"

**Missing module**: `transformers`, `openai`, etc.

**Solution**:
```powershell
pip install transformers openai numpy aiohttp python-dotenv soundfile
```

---

### Issue: "CUDA out of memory" (During Training)

**Solution**:
```powershell
# Reduce batch size
python fine_tuning/3_train_lora.py --batch-size 1 --gradient-accumulation 16

# Or use CPU (slower but works)
python fine_tuning/3_train_lora.py --device cpu
```

---

### Issue: Training Too Slow

**Expected**: 2-3 min/epoch on RTX 4050
**If slower**: 
- Close other GPU programs
- Update NVIDIA drivers
- Check power mode (not battery saver)

---

### Issue: Prosody Not Applied

**Current system**: 30-50% success (prompting only)

**Solution**: Fine-tune the model (see guide 02)

After training: 85-95% success

---

### Issue: Audio Sounds Robotic

**Causes**:
- Too many prosody markers
- Conflicting markers

**Solution**:
```
✓ Good: "We <emph>guarantee savings"
✗ Bad: "<pitch_high> We <emph>absolutely <pause_short> <pitch_rising> guarantee..."
```

Use 1-2 markers per sentence maximum.

---

## 📞 Quick Checks

### Is system working?
```powershell
python main.py -i "Test" -p "Test"
# Should generate response.wav in output/
```

### Is API key set?
```powershell
echo $env:BOSON_API_KEY
# Should show your key
```

### Is GPU working? (for training)
```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

**Still having issues?** Check the code comments or API docs.
