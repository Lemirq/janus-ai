# Janus AI

![](janus_header.gif)

## Overview

Say goodbye to prompt engineering.

Janus is the union of AI and Human, a double-headed intelligent being with effortless access to the vast body of digital knowledge. Janus is a voice assitant that follows conversations in depth, doing research in real-time and providing relevant information on the fly. For the user, the experience is simple and intuitive: simply talk as normal, and Janus becomes a second brain, controlled by subtle gesture cues that fit with the natural flow of speech. From the second-person perspective, Janus is seamless: no explicit voice commands. No text prompts or chat interface. It communicates with users via a bluetooth earpiece, monitoring and transcribing conversations in real time, and deciding what kind of insight is helpful at a given moment. We've also built Janus to excel in high-stakes scenarios like business pitches, sales calls, and legal settings. Users can define specific objectives in advance, and Janus will target persuasive techniques toward that particular audience to achieve the user's goal. 

For more on the theoretical side, check out our paper, *Toward a Framework for Cooperative Hybrid Intelligence*, in the main folder of each repo. 

**Custom Prosody Encoding:** We defined a custom prosody notation and finetuned Higgs Audio V2 on special prosody tokens for enhanced prosody modulation.  

**Novel Human-Annotated Dataset:** We annotated our own dataset, notated with our custom prosody notation. 

**Bilingual:** Janus has fully native support for both English and French. 

**Hardware Integration:** Janus reads accelerometer and gyroscope data to track gesture commands in real time, allowing fine-grained control with no text or voice commands. 

**Direct Prosody Injection:** We leveraged Higgs' DualFFN architecture to inject prosody tokens directly into the audio stream. This allows the model to process prosody cues directly, instead of relying on the underlying Llama model, which is less specialized for this task. 

**Real-Time Web Search:** Janus searches the web for relevant information in real time, without requiring prompts from the user. This lets you focus on the conversation or task at hand, while Janus looks for information to help you get the job done. 

**RAG:** Janus uses retrieval-augmented generation (RAG) to fetch important documents quickly and accurately. Need a figure from this quarter's financial report? No problem. Just upload the file in advance, and Janus will tell you the facts when you need them. 

**Native iOS App:** We built Janus as a native app for iOS, avoiding the hassle of mobile web UIs. 
