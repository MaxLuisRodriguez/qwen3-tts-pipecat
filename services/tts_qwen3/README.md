# Qwen3 TTS Service

Streaming text-to-speech service skeleton for Qwen3 TTS model.

## Overview

This service provides a REST API for streaming audio synthesis. Currently implemented as a stub that generates silence chunks, with clear TODOs for integrating the actual Qwen3 TTS model.

## Audio Format

- **Sample Rate**: 24,000 Hz
- **Format**: 16-bit PCM, mono
- **Chunk Size**: 1,600 samples (~66ms per chunk)
- **Bytes per Chunk**: 3,200 bytes

## Setup

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
python server.py
```

The server will start on `http://localhost:8001`.

## API Endpoints

### POST /synthesize

Stream audio chunks as Server-Sent Events (SSE).

**Request Body:**
```json
{
  "text": "Hello, this is a test."
}
```

**Response:** SSE stream with base64-encoded audio chunks
```
data: {"chunk_index": 0, "audio_base64": "...", "sample_rate": 24000, "chunk_size_samples": 1600}
data: {"chunk_index": 1, "audio_base64": "...", "sample_rate": 24000, "chunk_size_samples": 1600}
data: {"done": true}
```

### POST /synthesize_binary

Stream raw PCM audio chunks (binary format).

**Request Body:**
```json
{
  "text": "Hello, this is a test."
}
```

**Response:** Raw binary PCM stream (Content-Type: `audio/pcm`)

### GET /health

Health check endpoint.

### GET /spec

Get audio format specification.

## Current Status

⚠️ **Stub Implementation**: The service currently generates silence chunks. The actual Qwen3 TTS model integration is marked with TODO comments in `server.py`.

## Integration TODO

1. Integrate Qwen3 TTS model (load weights, tokenizer)
2. Implement `synthesize_audio_stub()` to call actual model inference
3. Convert model output to 16-bit PCM format
4. Stream audio chunks as they are generated
5. Add voice/style parameters to request model

## Interface Contract

**Input:**
- Text string (UTF-8)

**Output:**
- Stream of audio chunks
- Each chunk: 1,600 samples = 3,200 bytes (16-bit PCM)
- Sample rate: 24,000 Hz
- Format: Mono (single channel)
