# Qwen3 TTS Service

Streaming text-to-speech service backed by `Qwen/Qwen3-TTS`.

## Overview

This service exposes a simple HTTP API:
- text in
- chunked 24kHz 16-bit PCM out

It loads Qwen3-TTS lazily (or on startup if configured) and supports:
- binary PCM streaming (`/synthesize_binary`)
- SSE base64 chunk streaming (`/synthesize`)

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python server.py
```

Server runs on `http://localhost:8001`.

## Environment Variables

- `QWEN3_TTS_MODEL_NAME` (default: `Qwen/Qwen3-TTS`)
- `QWEN3_TTS_DEFAULT_VOICE` (default: `Cherry`)
- `QWEN3_TTS_ATTN_IMPL` (default: `sdpa`)
- `QWEN3_TTS_SAMPLE_RATE` (default: `24000`)
- `QWEN3_TTS_CHUNK_SIZE` (default: `1600`)
- `TTS_PRELOAD_MODEL` (`1` to load at startup, default `0`)

## API Endpoints

### `POST /synthesize`

SSE stream with base64-encoded PCM chunks.

Request body:

```json
{
  "text": "Hello, this is a test.",
  "voice": "Cherry",
  "max_new_tokens": 1024
}
```

Response events:
- `chunk_index`
- `audio_base64`
- `sample_rate`
- `chunk_size_samples`
- `ttfc_ms` (included on first chunk)
- final `{ "done": true }`

### `POST /synthesize_binary`

Raw PCM stream (16-bit mono @ 24kHz).

Request body matches `/synthesize`.

Response headers include:
- `Content-Type: audio/pcm; rate=24000; channels=1; width=16`
- `X-TTFC-Ms`

### `POST /load_model`

Force model load before synthesis.

### `GET /health`

Shows model load status and selected device.

### `GET /spec`

Audio format and chunk configuration.
