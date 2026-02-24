# Qwen Megakernel Monorepo

A monorepo containing the Qwen megakernel implementation and streaming services for building voice AI applications.

## Repository Structure

```
qwen_megakernel/
├── kernel/                    # Original Qwen megakernel code (untouched)
│   ├── csrc/                  # CUDA kernel source
│   ├── qwen_megakernel/       # Python bindings and model code
│   └── requirements.txt
├── services/
│   ├── llm_megakernel/        # LLM streaming service wrapper
│   └── tts_qwen3/             # TTS streaming service skeleton
├── pipecat_demo/              # Pipecat pipeline demo skeleton
├── scripts/                   # Utility scripts
├── docker-compose.yml         # Docker orchestration
├── assets_original/           # Original assets (from GitHub clone)
├── csrc_original/             # Original CUDA source (from GitHub clone)
├── qwen_megakernel_original/  # Original Python package (from GitHub clone)
├── requirements_original.txt # Original requirements (from GitHub clone)
├── README_original.md         # Original README (from GitHub clone)
└── README.md                  # This file
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Voice AI Pipeline                        │
└─────────────────────────────────────────────────────────────┘

    Audio Input
        │
        ▼
    ┌─────────┐
    │   STT   │  (Speech-to-Text)
    └─────────┘
        │
        ▼ Text
    ┌──────────────────┐
    │  LLM Service     │  ────►  ┌──────────────┐
    │  (Megakernel)    │         │   Kernel/    │
    │  Port: 8000      │         │  CUDA Code   │
    └──────────────────┘         └──────────────┘
        │
        ▼ Tokens (streaming)
    ┌──────────────────┐
    │  TTS Service     │
    │  (Qwen3)        │
    │  Port: 8001     │
    └──────────────────┘
        │
        ▼ Audio (streaming)
    ┌─────────┐
    │ Output  │  (Speakers/WebRTC/etc.)
    └─────────┘
```

## Quick Start

### 1. Build the Kernel

The kernel code is in `kernel/`. Build it from there:

```bash
cd kernel
pip install -r requirements.txt
python -m qwen_megakernel.bench  # Test the kernel
```

**Note**: Requires CUDA 12.8+ and an RTX 5090 (or compatible GPU). The kernel is optimized for RTX 5090 (sm_120).

### 2. Run Services Locally

#### LLM Service

```bash
cd services/llm_megakernel
pip install -r requirements.txt
python server.py
```

Service runs on `http://localhost:8000`

#### TTS Service

```bash
cd services/tts_qwen3
pip install -r requirements.txt
python server.py
```

Service runs on `http://localhost:8001`

### 3. Run the Demo

```bash
cd pipecat_demo
pip install -r requirements.txt
python app.py
```

Or use the convenience script:

```bash
./scripts/run_local.sh
```

## Docker Compose

Start all services with Docker Compose:

```bash
docker-compose up
```

**Note**: GPU passthrough requires NVIDIA Container Toolkit. See `docker-compose.yml` for GPU configuration notes.

## Component Details

### `/kernel`

Original Qwen megakernel implementation. Optimized for Qwen3-0.6B on RTX 5090.

- **Performance**: ~8.4x faster than PyTorch (1036 tok/s vs 123 tok/s)
- **Architecture**: Single fused CUDA kernel for entire decode pass
- **See**: `README_original.md` for original documentation

### `/services/llm_megakernel`

Python wrapper around the kernel with FastAPI streaming server.

- **API**: REST + Server-Sent Events (SSE)
- **Status**: Stub implementation (returns fake tokens)
- **TODO**: Integrate with actual kernel code

### `/services/tts_qwen3`

Streaming TTS service skeleton.

- **Format**: 16-bit PCM, 24kHz, mono
- **Status**: Stub implementation (generates silence)
- **TODO**: Integrate Qwen3 TTS model

### `/pipecat_demo`

Minimal Pipecat pipeline demo showing the full flow.

- **Status**: Mock pipeline runner
- **TODO**: Integrate actual Pipecat framework

## Development Status

⚠️ **Current State**: This is a restructured skeleton. Most components are stubbed with clear TODOs:

- ✅ Repository structure created
- ✅ Kernel code moved (untouched)
- ✅ Service skeletons created
- ⚠️ LLM service: Stub (needs kernel integration)
- ⚠️ TTS service: Stub (needs model integration)
- ⚠️ Pipecat demo: Mock (needs framework integration)

## Requirements

- **Python**: 3.10+
- **CUDA**: 12.8+ (for kernel)
- **GPU**: RTX 5090 or compatible (for kernel)
- **Docker**: Optional (for containerized deployment)

## API Documentation

### LLM Service (`http://localhost:8000`)

- `POST /generate` - Stream tokens (SSE)
- `POST /load_weights` - Load model weights
- `GET /health` - Health check

See `services/llm_megakernel/README.md` for details.

### TTS Service (`http://localhost:8001`)

- `POST /synthesize` - Stream audio (SSE, base64)
- `POST /synthesize_binary` - Stream audio (raw PCM)
- `GET /health` - Health check
- `GET /spec` - Audio format specification

See `services/tts_qwen3/README.md` for details.

## Credits

Original megakernel by [AlpinDale](https://github.com/AlpinDale/qwen_megakernel), based on [MegaQwen](https://github.com/Infatoshi/MegaQwen) by Elliot Arledge.

## License

See original repository for license information.
