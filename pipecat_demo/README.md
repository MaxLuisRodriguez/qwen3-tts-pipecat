# Pipecat Demo

Minimal Pipecat pipeline demo skeleton that demonstrates the streaming flow between STT, LLM, and TTS services.

## Overview

This demo shows how to wire together:
1. **STT** (Speech-to-Text) - converts audio to text
2. **LLM** - generates response tokens (calls `services/llm_megakernel`)
3. **TTS** (Text-to-Speech) - converts text to audio (calls `services/tts_qwen3`)
4. **Audio Output** - plays synthesized audio

Currently implemented as a mock pipeline runner that demonstrates the streaming flow. Actual Pipecat integration is marked with TODO comments.

## Setup

```bash
pip install -r requirements.txt
```

## Prerequisites

The LLM and TTS services must be running:
- LLM service: `http://localhost:8000`
- TTS service: `http://localhost:8001`

See their respective READMEs for how to start them.

## Running

```bash
python app.py
```

This will run a mock pipeline that:
1. Simulates audio input
2. Calls STT (mock)
3. Streams tokens from LLM service
4. Streams audio from TTS service
5. Logs the output (audio playback is stubbed)

## Current Status

⚠️ **Skeleton Implementation**: This is a mock pipeline runner. Actual Pipecat integration is marked with TODO comments in `app.py`.

## Integration TODO

1. Install and import Pipecat framework
2. Replace `MockPipelineRunner` with actual `PipelineRunner`
3. Add STT processor (e.g., Whisper, Deepgram)
4. Create custom LLM processor that calls our LLM service
5. Create custom TTS processor that calls our TTS service
6. Add audio output processor (e.g., PlayAudioProcessor, WebRTC output)
7. Wire everything together in a `Pipeline` object

## Architecture

```
Audio Input → STT → Text
                      ↓
                   LLM Service (streaming tokens)
                      ↓
                   TTS Service (streaming audio)
                      ↓
                   Audio Output
```

## Example Flow

1. User speaks → Audio captured
2. STT processes audio → "Hello, how are you?"
3. LLM service streams tokens → "I'm", "doing", "well", "thank", "you"
4. TTS service streams audio chunks → PCM audio data
5. Audio output plays → User hears response
