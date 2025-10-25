# Janus AI - Real-time Persuasion Assistant

Janus AI (Mercury) is an advanced real-time persuasion assistant that helps speakers become more persuasive by analyzing conversations, generating optimal responses, and guiding prosodic delivery through a Bluetooth earpiece.

## 🎯 Overview

Janus AI monitors conversations in real-time and provides intelligent assistance to achieve persuasion objectives. It uses the Higgs audio model with custom prosody token support to generate natural, persuasive speech with appropriate emphasis and emotional tone.

### Key Features

- **Real-time Transcription**: Converts speech to text using Higgs audio understanding
- **Sentiment Analysis**: Identifies questions, concerns, and emotional states
- **Persuasion Alignment**: Tracks progress toward persuasion objectives (PO)
- **Smart Response Generation**: Creates contextually appropriate responses with prosody markup
- **Prosody-Enhanced Audio**: Generates speech with emphasis, pauses, and emotional tones
- **Audience Profiling**: Adapts strategy based on audience characteristics
- **Session Analytics**: Tracks objective proximity and provides post-session insights

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Input    │────▶│  Transcription   │────▶│   Sentiment     │
│  (Bluetooth)    │     │  (Higgs Model)   │     │   Analysis      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Output   │◀────│  Audio Generator │◀────│   Persuasion    │
│  (Earpiece)     │     │  with Prosody    │     │   Engine        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for real-time processing)
- Boson AI API key for Higgs model access

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/janus-ai.git
cd janus-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export BOSON_API_KEY="your-api-key-here"
```

### Basic Usage

```python
import asyncio
from janus_ai.main import JanusAI, JanusConfig, PersuasionObjective

# Configure Janus
config = JanusConfig(
    api_key="your-api-key",
    enable_smooth_stall=True,
    enable_question_prediction=True
)

# Initialize system
janus = JanusAI(config)

# Set persuasion objective
objective = PersuasionObjective(
    main_goal="Close the sales deal",
    key_points=[
        "30% cost reduction",
        "Proven ROI within 6 months",
        "24/7 customer support",
        "Risk-free trial period"
    ],
    audience_triggers=["savings", "reliability", "support"]
)

await janus.set_persuasion_objective(objective)

# Set audience profile
audience = {
    "type": "corporate_buyer",
    "priorities": ["cost_efficiency", "reliability"],
    "communication_style": "analytical",
    "decision_timeline": "end_of_quarter"
}

await janus.set_audience_profile(audience)

# Process audio stream (example)
async for response in janus.process_audio_stream(audio_stream):
    # Play response through earpiece
    play_audio(response)
```

## 🎭 Prosody System

Janus AI uses custom prosody tokens (IDs >= 128,000) that are injected into the audio stream:

### Available Prosody Tokens

| Token | ID | Description |
|-------|-----|-------------|
| `<EMPH>` | 128000 | Emphasize next word |
| `<STRONG>` | 128001 | Strong emphasis |
| `<SLOW>` | 128003 | Slow down pace |
| `<PAUSE_SHORT>` | 128005 | Brief pause (0.5s) |
| `<PAUSE_LONG>` | 128006 | Long pause (1.5s) |
| `<CONFIDENT>` | 128011 | Confident tone |
| `<FRIENDLY>` | 128012 | Friendly tone |
| `<CALM>` | 128015 | Calm tone |

### Example with Prosody

```python
# Input text
"We guarantee 30% savings immediately"

# With prosody markup
"We <STRONG>guarantee <EMPH>30% savings <FAST>immediately"
```

## 🔧 Fine-tuning for Prosody

To train the model with custom prosody tokens:

1. Prepare training data:
```python
from janus_ai.fine_tuning.prepare_dataset import create_training_examples

# Create examples with prosody markup
examples = [
    ("This saves you money", "This <EMPH>saves you money"),
    ("Act now for best results", "Act <STRONG>now for best results")
]

create_training_examples(examples, audio_dir, "training_data.json")
```

2. Run fine-tuning:
```python
from janus_ai.fine_tuning.train_prosody import train_prosody_model, ProsodyTrainingConfig

config = ProsodyTrainingConfig(
    model_name="meta-llama/Llama-3.2-3B",
    output_dir="./prosody_model",
    num_train_epochs=3
)

train_prosody_model(config)
```

## 📊 Session Analytics

Get insights after each session:

```python
analytics = await janus.get_session_analytics()

print(f"Objective completion: {analytics['objective_completion']:.1%}")
print(f"Key moments: {len(analytics['key_moments'])}")
print(f"Recommendations: {analytics['recommendations']}")
```

## 🏃 Running the System

### Development Mode

```bash
# Run with mock audio input
python -m janus_ai.main --mode development

# Run with live audio
python -m janus_ai.main --mode live --audio-device "Bluetooth Headset"
```

### Production Deployment

For production use with real-time audio:

1. Set up audio streaming server:
```bash
uvicorn janus_ai.api.server:app --host 0.0.0.0 --port 8000
```

2. Connect Bluetooth device and run client:
```bash
python -m janus_ai.client --server-url http://localhost:8000
```

## 🔍 Advanced Features

### Question Prediction

Janus predicts likely questions and pre-generates responses:

```python
config = JanusConfig(enable_question_prediction=True)
# System will automatically predict and cache responses
```

### Smooth Stalling

Generates natural stalling responses while processing complex queries:

```python
config = JanusConfig(enable_smooth_stall=True)
# Automatically inserts phrases like "That's an excellent question..."
```

### Audience Profiling

Adapts persuasion strategy based on audience:

```python
audience_profile = {
    "demographics": {"age_range": "35-50", "industry": "tech"},
    "personality": {"type": "analytical", "risk_tolerance": "low"},
    "context": {"meeting_type": "initial_pitch", "duration": "30min"}
}

await janus.set_audience_profile(audience_profile)
```

## 📈 Performance Optimization

- **GPU Acceleration**: Use CUDA for faster inference
- **Batch Processing**: Process multiple audio chunks together
- **Response Caching**: Cache responses for predicted questions
- **Model Quantization**: Use INT8 quantization for faster inference

## 🛠️ Troubleshooting

### Common Issues

1. **Audio latency**: Reduce chunk size or upgrade to faster GPU
2. **Prosody not working**: Ensure prosody tokens have IDs >= 128,000
3. **API errors**: Check API key and rate limits

### Debug Mode

```python
config = JanusConfig(debug=True)
# Enables detailed logging
```

## 📚 API Reference

See [API Documentation](docs/api.md) for detailed API reference.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Boson AI for the Higgs audio model
- Meta for Llama 3.2 base model
- The open-source community for audio processing libraries

---

**Note**: This is a hackathon project demonstrating advanced concepts in real-time AI-assisted communication. Production deployment requires additional security, privacy, and performance considerations.
