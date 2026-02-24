# LLM Megakernel Service

Streaming LLM service wrapper around the Qwen megakernel.

## Overview

This service provides a REST API for streaming token generation using the Qwen megakernel. It wraps the kernel code in `../../kernel/` and exposes it via FastAPI with Server-Sent Events (SSE) for streaming.

## Setup

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
python server.py
```

The server will start on `http://localhost:8000`.

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

⚠️ **Stub Implementation**: The service currently returns fake tokens. Integration with the actual kernel is marked with TODO comments in `wrapper.py`.

## Integration TODO

1. Import `qwen_megakernel` from `../../kernel/`
2. Implement `MegakernelDecoder.load_weights()` to call kernel's `load_weights()`
3. Implement `MegakernelDecoder.generate_stream()` to call kernel's decoder in a streaming loop
