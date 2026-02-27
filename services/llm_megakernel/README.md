# LLM Megakernel Service

Streaming LLM service wrapper around the Qwen megakernel.

## Overview

This service provides a REST API for streaming token generation using the Qwen megakernel. It wraps the kernel code in `../../kernel/` and exposes it via FastAPI with Server-Sent Events (SSE) for streaming.

## Setup

```bash
pip install -r requirements.txt
```

Run this service in an environment that can import `../../kernel/qwen_megakernel`
(typically `kernel/.venv` created by the bootstrap script).

## Running Locally

```bash
python server.py
```

The server will start on `http://localhost:8000`.

Optional preload on startup:

```bash
export LLM_PRELOAD_WEIGHTS=1
export QWEN_MEGAKERNEL_MODEL_NAME=kernel/weights/Qwen3-0.6B
python server.py
```

## API Endpoints

### POST /generate

Stream tokens for a given prompt.

**Request Body:**
```json
{
  "prompt": "Hello, how are you?",
  "max_tokens": 100,
  "weights_path": "Qwen/Qwen3-0.6B"  // optional
}
```

**Response:** Server-Sent Events stream
```
data: {"token": "hello"}
data: {"token": "world"}
data: {"done": true}
```

### POST /load_weights

Explicitly load model weights.

**Request Body:**
```json
{
  "weights_path": "Qwen/Qwen3-0.6B"
}
```

### GET /health

Health check endpoint.

## Current Status

✅ **Kernel-backed Streaming**: The wrapper now loads real model weights and streams
incremental decoded text from the megakernel decode loop.

- `load_weights(weights_path)` loads from Hugging Face model id or local path.
- `generate_stream(prompt, max_tokens)` lazily loads weights if needed and emits
  incremental text deltas as tokens are generated.
