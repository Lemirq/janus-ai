# Janus AI Project Summary

## 🎯 Project Overview

Janus AI is a real-time persuasion assistant that helps speakers be more persuasive by:
- Monitoring conversations via Bluetooth earpiece
- Analyzing sentiment and identifying questions
- Generating optimal responses aligned with persuasion objectives
- Delivering responses with prosody (emphasis, pauses, emotional tone)

## 🏗️ Architecture Components

### Core Modules

1. **Real-time Transcription** (`core/transcription.py`)
   - Uses Higgs audio understanding model
   - Processes audio chunks in real-time
   - Handles streaming and buffering

2. **Sentiment Analysis** (`core/sentiment_analyzer.py`)
   - Identifies questions, concerns, emotional states
   - Detects conversation dynamics
   - Predicts likely follow-up questions

3. **Persuasion Engine** (`core/persuasion_engine.py`)
   - Tracks Persuasion Objectives (PO) and Points of Persuasion (POPs)
   - Analyzes alignment with goals
   - Provides strategic recommendations
   - Generates session analytics

4. **Prosody Tokenizer** (`core/prosody_tokenizer.py`)
   - Custom tokenizer for prosody markers
   - Tokens with IDs >= 128,000 for audio stream injection
   - Supports emphasis, pauses, pitch, emotion markers

5. **Response Generator** (`core/response_generator.py`)
   - Creates contextually appropriate responses
   - Applies intelligent prosody markup
   - Implements persuasion tactics

6. **Audio Generator** (`core/audio_generator.py`)
   - Generates speech using Higgs model
   - Processes prosody tokens
   - Supports voice cloning and emotion

### Fine-tuning Components

1. **Dataset Preparation** (`fine_tuning/prepare_dataset.py`)
   - Creates training data with prosody annotations
   - Extracts audio features (pitch, energy, duration)
   - Aligns prosody tokens with audio regions

2. **Training Script** (`fine_tuning/train_prosody.py`)
   - Fine-tunes Higgs/Llama model for prosody
   - Custom loss functions for prosody-audio alignment
   - Extends model to handle tokens >= 128,000

## 🔑 Key Innovations

### 1. Prosody Token System
- Custom tokens (>= 128,000) injected into audio stream
- Model trained to generate corresponding prosody
- Natural emphasis and emotional delivery

### 2. Real-time Persuasion Tracking
- Continuous alignment with objectives
- Dynamic strategy adjustment
- Opportunity detection

### 3. Intelligent Stalling
- Natural filler responses while processing
- Maintains conversation flow
- Reduces perceived latency

### 4. Audience Adaptation
- Profiles audience characteristics
- Adjusts communication style
- Optimizes persuasion tactics

## 📊 Data Flow

```
Audio Input → Transcription → Sentiment Analysis → Persuasion Alignment
     ↓                                                      ↓
Prosody Tokens ← Response Generation ← Strategy ← Objective Tracking
     ↓
Audio Output ← Prosody-Enhanced Speech Generation
```

## 🚀 Usage Example

```python
# Initialize Janus AI
janus = JanusAI(config)

# Set objective
objective = PersuasionObjective(
    main_goal="Sign partnership agreement",
    key_points=["30% cost savings", "Proven ROI", "24/7 support"]
)

# Process conversation
async for audio_response in janus.process_audio_stream(audio_input):
    play_through_earpiece(audio_response)
```

## 🔧 Technical Stack

- **Models**: Higgs audio (generation/understanding), Llama 3.2, Qwen3
- **Audio**: librosa, pyaudio, wave
- **ML**: PyTorch, Transformers, OpenAI API
- **Async**: asyncio, aiohttp
- **Real-time**: WebSockets, streaming processing

## 📈 Performance Considerations

- Chunk-based processing for low latency
- Response caching for predicted questions
- GPU acceleration for inference
- Parallel processing where possible

## 🎯 Future Enhancements

1. Multi-language support
2. Visual cue integration (body language)
3. Group conversation handling
4. Advanced emotion recognition
5. Contextual memory across sessions

---

This project demonstrates cutting-edge integration of:
- Real-time audio processing
- Advanced NLP with prosody
- Strategic AI assistance
- Human-AI collaboration

Perfect for sales, negotiations, presentations, and any persuasive communication scenario.
