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
│   └── tts_qwen3/             # Qwen3-TTS streaming service
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

### Automated bootstrap (recommended on Linux + RTX 5090)

```bash
cp .env.qwen_megakernel.template .env.qwen_megakernel
# Edit .env.qwen_megakernel as needed (HF token, cache paths, tuning vars)
bash scripts/bootstrap_qwen_megakernel.sh
```

The bootstrap script performs host checks, creates `kernel/.venv`, installs
`torch==2.7.0` (cu128) + dependencies, validates decode optimization symbols,
downloads `Qwen/Qwen3-0.6B` weights to `kernel/weights/Qwen3-0.6B`, verifies
the published SHA256, JIT-builds the extension, and runs
`python -m qwen_megakernel.bench` (log output in `smoke_test.log`).

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

Pipecat runtime config is loaded from `.env.pipecat`.

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
- **Status**: Integrated with kernel-backed streaming generation
- **Note**: Must run in an environment with kernel dependencies and weights

### `/services/tts_qwen3`

Streaming TTS service backed by Qwen3-TTS.

- **Format**: 16-bit PCM, 24kHz, mono
- **Status**: Integrated with `Qwen/Qwen3-TTS` inference
- **API**: `/synthesize` (SSE) and `/synthesize_binary` (raw PCM)

### `/pipecat_demo`

Pipecat voice pipeline demo with Daily transport.

- **Status**: Integrated (Deepgram STT + Megakernel LLM service + local Qwen3-TTS)
- **Config**: Uses `.env.pipecat`

## Development Status

⚠️ **Current State**: This is a restructured monorepo. Some components are still stubbed with clear TODOs:

- ✅ Repository structure created
- ✅ Kernel code moved (untouched)
- ✅ Service skeletons created
- ✅ LLM service: Megakernel wrapper integrated
- ✅ TTS service: Qwen3-TTS integration complete
- ✅ Pipecat demo: Integrated (Daily + Deepgram + local Qwen3-TTS + Megakernel LLM)

## Benchmarking

With services running (`bash scripts/run_local.sh`), collect local metrics:

```bash
kernel/.venv/bin/python scripts/benchmark_stack.py
```

This prints:
- Megakernel LLM TTFT and decode tok/s from `services/llm_megakernel`
- TTS first-chunk latency (TTFC), total synthesis time, and RTF from `services/tts_qwen3`
- A simple e2e estimate (`llm_ttft + tts_first_chunk`)

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

