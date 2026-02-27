# Qwen Megakernel Monorepo

This repo contains:
- A Qwen3 megakernel decode path optimized for RTX 5090
- A FastAPI LLM service (`services/llm_megakernel`)
- A FastAPI Qwen3-TTS service (`services/tts_qwen3`)
- A Pipecat + Daily demo app (`pipecat_demo`)

## What You Asked For

This repository includes, clearly:
- Working build instructions for a **single RTX 5090**
- A short README section on:
  - architecture decisions
  - kernel modifications
  - how to run the Pipecat demo

## Architecture Decisions

1. Keep kernel logic isolated in `kernel/`, and expose it through a minimal Python wrapper.
2. Use service boundaries for orchestration:
   - `llm_megakernel` handles text generation via SSE
   - `tts_qwen3` handles speech synthesis via HTTP streaming
3. Use Pipecat as the real-time pipeline coordinator with Daily transport.
4. Prefer local TTS (`Qwen3TTSModel`) over hosted TTS to keep inference local.
5. Use environment-driven configuration so tuning and credentials can be changed without code edits.

## Kernel Modifications (Summary)

Compared to baseline/original layout, this repo includes tuned decode defaults and pipeline integration:

- Tuned default kernel macro values are now set directly in `kernel/csrc/kernel.cu`:
  - `LDG_LM_NUM_BLOCKS=1280`
  - `LDG_LM_BLOCK_SIZE=384`
  - `LDG_ATTN_BLOCKS=8`
  - `LDG_PREFETCH_QK=0`
  - `LDG_PREFETCH_THREAD_STRIDE=10`
- Matching defaults are exposed in:
  - `kernel/qwen_megakernel/build.py`
  - `.env.qwen_megakernel.template`
  - `scripts/bootstrap_qwen_megakernel.sh`
- The decode path includes:
  - persistent/direct decode kernels
  - flag-based synchronization (`kv_flag`, `attn_flag`)
  - fused LM head reduction path (`ldg_lm_head_fused`)
  - no-sync multi-step generation (`generate_nosync`)

## Build Instructions (Single RTX 5090)

These steps are the recommended path on Linux with one RTX 5090.

1. Clone the repo and enter it.
```bash
git clone <your-repo-url> qwen_megakernel
cd qwen_megakernel
```

2. Create runtime env file.
```bash
cp .env.qwen_megakernel.template .env.qwen_megakernel
```

3. Edit `.env.qwen_megakernel` if needed:
- `HF_TOKEN` (optional, for gated/private downloads)
- cache paths
- model path/id

4. Run bootstrap.
```bash
bash scripts/bootstrap_qwen_megakernel.sh
```

What bootstrap does:
- checks host prerequisites (`nvidia-smi`, `nvcc`, etc.)
- creates `kernel/.venv`
- installs CUDA-compatible Python deps
- validates optimization symbols
- resolves model weights
- JIT builds extension
- runs a kernel smoke benchmark

## Run the Pipecat Demo

1. Create/edit Pipecat runtime env:
```bash
cp .env.pipecat .env.pipecat 2>/dev/null || true
```

Required values in `.env.pipecat`:
- `DEEPGRAM_API_KEY`
- either:
  - `DAILY_ROOM_URL` + `DAILY_ROOM_TOKEN`
  - or `DAILY_API_KEY`

2. Start full local stack and demo:
```bash
bash scripts/run_local.sh
```

This starts:
- LLM service on `localhost:8000`
- TTS service on `localhost:8001`
- Pipecat app (`pipecat_demo/app.py`) with Daily transport

## Service Endpoints

LLM service (`services/llm_megakernel`):
- `POST /generate`
- `POST /load_weights`
- `GET /health`

TTS service (`services/tts_qwen3`):
- `POST /synthesize`
- `POST /synthesize_binary`
- `POST /load_model`
- `GET /health`
- `GET /spec`

## Key Paths

- `kernel/csrc/kernel.cu`
- `kernel/qwen_megakernel/build.py`
- `services/llm_megakernel/server.py`
- `services/tts_qwen3/server.py`
- `pipecat_demo/app.py`
- `scripts/bootstrap_qwen_megakernel.sh`
- `scripts/run_local.sh`

