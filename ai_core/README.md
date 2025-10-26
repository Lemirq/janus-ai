# Janus AI

> Real-time persuasion assistant powered by Higgs audio model

**Status**: ✅ Production Ready | **Quick Start**: `python main.py --interactive`

---

## 🎯 Objective

Janus makes speakers more persuasive. It finds answers to questions in real time, generates optimal responses, and guides the user in persuasive delivery with prosody-enhanced speech. Janus communicates with users via a bluetooth earpiece (or audio files), monitoring conversations in real time and checking against predefined persuasion objectives to guide responses.

---

## 📖 Jargon

**PO (Persuasion Objective)**:
The main longer-term goal of the exchange (e.g., signing a deal, closing a sale)

**POP (Points of Persuasion)**:
Individual points on which to persuade the audience, in service of the PO.

**Prosody Tokens**:
Special markers (IDs >= 5,000,000) that control speech emphasis, pauses, and pitch

---

## 🔄 Flow of Data

```
Real-time transcription →
Sentiment analysis and question identification →
How does this align with PO? →
POP →
What info/stats are needed →
Search →
Results →
Key persuasive info →
Response text →
Prosody markup →
Audio generation
```

---

## 🏗️ Training and Architecture

We fine-tune the Higgs model to accept explicit prosody tokens. The model uses Llama 3.2 with an added DualFFN pathway. During inference, the model processes both text and audio tokens, feeding text into the regular Llama transformer, and feeding audio tokens into the parallel FFN pathway.

### Prosody Token System

**Implementation**: Any token with an ID >= 5,000,000 is passed directly to the audio stream. We inject prosody tokens into the text prompt just before the words to which they apply. Via LoRA training, the model learns to generate the correct prosody in the word that follows.

**Tokens** (7 total):
- `<emph>` (5,000,000) - Emphasize next word
- `<pause_short>` (5,000,001) - Brief pause (0.5s)
- `<pause_long>` (5,000,002) - Long pause (1.5s)
- `<pitch_high>` (5,000,003) - Higher pitch
- `<pitch_low>` (5,000,004) - Lower pitch
- `<pitch_rising>` (5,000,005) - Rising intonation
- `<pitch_falling>` (5,000,006) - Falling intonation

**Tokenizer Modification**: The custom tokenizer assigns IDs 5,000,000+ to prosody tokens, ensuring they pass through to the audio stream. Training teaches the model to follow these tokens with appropriate prosody.

**LoRA Fine-Tuning**: Uses Low-Rank Adaptation for memory-efficient training (6GB VRAM, 1.5-2 hours on RTX 4050). Trains only 0.24% of parameters while achieving quality equivalent to full fine-tuning.

---

## ✨ Additional Features

### Audience Profiling
Allows the user to describe their audience, including demographics, professional relations, interests, goals, and context. Janus analyzes this profile and adds key points to the Persuasion Objective. Certain appeals (authority, ethics, honor, etc.) may be more effective in specific contexts. By adding these guidelines to the PO, Janus improves targeting and efficacy.

### Objective Analytics
Tracks proximity to objective throughout use, and provides feedback after a session, including a graph of objective proximity. If the user came close to a goal but didn't reach it, this identifies the key moment that led to failure, and vice versa.

### Question Prediction
Predicts likely questions and starts planning responses in advance. This improves speed of response when the question is asked. Responses are cached for predicted questions.

### Smooth Stall
Guided stalling to placate people with generic responses while the model works on details. Produces a generic beginning as fast as possible, buying time for slower search features to run.

---

## 🚀 Quick Start

```powershell
# Set API key
$env:BOSON_API_KEY="your-key-here"

# Generate response
python main.py -i "How much does this cost?" -p "30% savings" "No hidden fees"

# Interactive mode
python main.py --interactive
```

**Output**: `output/response.wav` with persuasive prosody

---

## 🎓 Fine-Tuning (Optional)

For 85-95% prosody consistency (vs 30-50% without training):

```powershell
cd fine_tuning

# Step 1: Auto-segment audio (5 min)
python 1_segment_audio.py

# Step 2: Prepare dataset (2 min)
python 2_prepare_data.py

# Step 3: Train with LoRA (1.5-2 hours on RTX 4050)
python 3_train_lora.py --epochs 3 --batch-size 4
```

**See**: `fine_tuning/README_TRAINING.md` for complete instructions

---

## 📁 Project Structure

```
ai_core/
├── main.py                    ← Main interface
├── output/                    ← Audio outputs
├── core/                      ← AI modules
│   ├── prosody_tokenizer.py  ← Prosody system
│   ├── response_generator.py ← Response creation
│   ├── audio_generator.py    ← Speech synthesis
│   ├── persuasion_engine.py  ← PO/POP tracking
│   └── sentiment_analyzer.py ← Analysis
│
├── fine_tuning/               ← Training pipeline
│   ├── 1_segment_audio.py    ← Auto-segmentation
│   ├── 2_prepare_data.py     ← Dataset preparation
│   └── 3_train_lora.py       ← LoRA training
│
└── guides/                    ← Documentation
    ├── 00_START_HERE.md      ← Project overview
    ├── 01_Quick_Start.md     ← Basic usage
    ├── 02_Fine_Tuning_LoRA.md← Training guide
    └── 03_Troubleshooting.md ← Common issues
```

---

## 🔧 Technology Stack

- **Models**: Higgs Audio (generation/understanding), Qwen3 (reasoning), GPT-2 (tokenization)
- **Training**: LoRA/PEFT for memory-efficient fine-tuning
- **Audio**: Boson AI API with prosody token support
- **Framework**: Python 3.8+, PyTorch (optional, for training only)

---

## 📊 Performance

| Metric | Without Training | With LoRA Training |
|--------|------------------|-------------------|
| Prosody consistency | 30-50% | 85-95% |
| Setup time | Instant | 1.5-2 hours |
| Memory required | Minimal | 6GB VRAM |
| Quality | Variable | Consistent |

---

## 📚 Documentation

- **guides/00_START_HERE.md** - Project overview and decision tree
- **guides/01_Quick_Start.md** - Get running in 30 seconds
- **guides/02_Fine_Tuning_LoRA.md** - Complete training guide
- **guides/03_Troubleshooting.md** - Common issues and solutions
- **fine_tuning/README_TRAINING.md** - Exact fine-tuning steps with time estimates

---

## 🎯 Use Cases

- Sales presentations and negotiations
- Job interviews
- Customer service
- Public speaking
- Any persuasive conversation scenario

---

## 📦 Requirements

```powershell
# Core (required for usage)
pip install openai transformers numpy aiohttp python-dotenv soundfile

# Training (optional, for fine-tuning)
pip install torch peft transformers accelerate --index-url https://download.pytorch.org/whl/cu118
```

---

## 🙏 Acknowledgments

- **Boson AI** for Higgs audio model and API access
- **Meta** for Llama 3.2 and GPT-2 architectures
- **HuggingFace** for PEFT/LoRA implementation
- **Open-source community** for audio processing libraries

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built for Boson AI Hackathon 2025**

For quick start, see `guides/01_Quick_Start.md` | For training, see `fine_tuning/README_TRAINING.md`