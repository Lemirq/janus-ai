# Mercury AI

## Objective

Mercury makes speakers more persuasive. It finds answers to questions in real time, generates optimal responses, and guides the user in persuasive delivery. Mercury communicates with users via a bluetooth earpiece, monitoring conversations in real time and checking against predefined persuasion objectives to guide responses.

## Jargon

PO (Persuasion Objective):
The main longer-term goal of the exchange (eg. signing a deal)

(Points of Persuasion):
Individual points on which to persuade the audience, in service of the PO.

## Flow of Data

Real-time transcription ->
Sentiment analysis and question identification ->
How does this align with PO? ->
POP ->
what info/stats are needed ->
search ->
results ->
key persuasive info ->
response text ->
prosidy ->
audio

## Training and Architecture

We will need to finetune the Higgs model to accept explicit prosody tokens. Currently, the model uses Llama 3.2 with an added DualFFN pathway. During inference, the model processes both text and audio tokens, feeding the text into the regular Llama transformer, and feeding the audio tokens into the parallel FFN pathway. We want to modify this architecture so that the model accepts explicit prosody notation. Prosody can be fed into the model through two routes: by adding additional text tokens for processing by the transformer, or by adding additional tokens into the audio stream.
Thankfully, injecting an audio token is very straightforward. Any token with an ID greater than or equal to 128,000 will be passed directly to the audio stream. Any audio tokens that we inject will be interpreted as being part of the previous audio signal. But we want the prosody token to direct the prosody of the next word, not the previous word. The solution is to train the model to follow a given prosody token with the corresponding prosody. For example, an emphasis token in the audio stream should be followed by emphasized tokens in the flow of speech.
There are a few things we need to implement to make this work. First of all, we need a way to inject the prosody tokens at the correct time points. The simplest way is to simply insert the prosody tokens into the text prompt, just before the words to which they apply. As long as the token IDs are above the threshold, the prosody tokens will be passed directly from the text prompt to the audio stream. Via training, we can then get the model to generate the correct prosody in the word that follows.
This means we need to modify the tokenizer, so that it assigns the right IDs to the special prosody tokens. Thankfully, this basically just entails adding some values to a lookup table.

## Additional Features

**Audience profiling**
Allows the user to describe their audience, including factors like demographics, professional relations, interests, goals, and context. Mercury then analyzes this audience profile and adds key points to the Persuasion Objective. For example, certain kinds of appeals to authority, ethics, honour, etc. may be more effective in certain contexts and with specific audiences. By adding these guidelines to the PO, Mercury could improve targeting and efficacy.

**Objective analytics**
Tracks proximity to objective throughout use, and provides feedback after a session, including a graph of objective proximity. If the user came close to a goal but didn’t reach it, this should identify the key moment that led to failure, and vice versa.

**Question Prediction**
Predicting likely questions and starting to plan responses in advance. This would improve speed of response when the question is asked.

**Smooth Stall**
Guided stalling to placate people with generic responses while the model goes to work on the details. Produces a more generic beginning as fast as possible, buying time for the slower search features to run.
