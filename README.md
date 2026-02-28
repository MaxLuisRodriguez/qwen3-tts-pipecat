# RTX 5090 Decode Megakernel -> Qwen3-TTS on Pipecat

This repo wires AlpinDale's `qwen_megakernel` decode path into a Pipecat voice pipeline:

- `STT (Deepgram) -> LLM (Megakernel service) -> TTS (Qwen3-TTS talker + megakernel backend) -> Daily audio output`

It is designed around the take-home scope:

- Adapt megakernel for Qwen3-TTS talker decode
- Expose streaming inference services
- Integrate with Pipecat + Daily
- Measure TTFC/RTF/tokens-per-second with reproducible scripts

## Repository Layout

- `kernel/`: CUDA + Python extension for megakernel decode
- `services/llm_megakernel/`: FastAPI LLM streaming service (`/generate`)
- `services/tts_qwen3/`: FastAPI streaming TTS service (`/synthesize_binary`)
- `pipecat_demo/`: voice bot app for Daily room conversation
- `scripts/bootstrap_qwen_megakernel.sh`: environment + extension bootstrap
- `scripts/run_local.sh`: start local stack + Pipecat demo
- `scripts/benchmark_stack.py`: benchmark client used by quick scripts
- `scripts/quick_bench_once.sh`: one-shot benchmark runner
- `scripts/quick_bench_sweep.sh`: short/medium/long benchmark sweep

## Requirements

- Linux host with NVIDIA GPU (target: RTX 5090, `sm_120`)
- NVIDIA driver + CUDA toolchain available (`nvidia-smi`, `nvcc`)
- Python 3.11 (managed by bootstrap in `kernel/.venv`)
- API keys:
  - `DEEPGRAM_API_KEY`
  - `DAILY_ROOM_URL` + `DAILY_ROOM_TOKEN` or `DAILY_API_KEY`

## Build

From repo root:

```bash
cp -n .env.qwen_megakernel.template .env.qwen_megakernel
bash scripts/bootstrap_qwen_megakernel.sh
```

Notes:

- Bootstrap creates `kernel/.venv`, installs deps, resolves model weights, and JIT-builds the extension.
- Keep `QWEN_MEGAKERNEL_MODEL_NAME` set to the intended Qwen backbone in `.env.qwen_megakernel`.

## Configure Runtime

Edit `.env.pipecat`:

- `DEEPGRAM_API_KEY=...`
- Either:
  - `DAILY_ROOM_URL=...` and `DAILY_ROOM_TOKEN=...`
  - or `DAILY_API_KEY=...`

Optional tuning knobs are already documented in `.env.pipecat` (TTS cadence, token budgeting, guards).

### Voice Reliability Defaults (Current)

- Streaming remains decode-time only (no full-audio fallback path).
- Pipecat sanitizes assistant text before TTS:
  - numbers are normalized to words (`42` -> `forty two`)
  - punctuation is stripped from spoken output text
- TTS silent-prefix behavior in `services/tts_qwen3/server.py` is non-fatal by default:
  - `QWEN3_TTS_REQUIRE_NON_SILENT_PREFIX=0` (default) streams anyway if early chunks are low-energy
  - set `QWEN3_TTS_REQUIRE_NON_SILENT_PREFIX=1` to restore strict rejection

## Run and Talk with the Model

From repo root:

```bash
START_TTS_SERVICE=1 bash scripts/run_local.sh
```

When the app prints:

```text
[pipecat] Join this Daily room: <url>
```

open that URL in your browser and speak.  
Stop with `Ctrl+C` in the terminal running `run_local.sh`.

## Architecture Decisions and Kernel Integration

1. Kept service boundaries explicit:
   - LLM service: `services/llm_megakernel/server.py`
   - TTS service: `services/tts_qwen3/server.py`
2. Added talker-specific megakernel adapter:
   - `services/tts_qwen3/megakernel_talker.py`
3. Ensured TTS output is streamed as decode progresses:
   - incremental frame decode -> PCM chunk streaming, no full-utterance buffer
4. Added Pipecat-side response stabilization and chunking:
   - `pipecat_demo/app.py`

## Quick Benchmark Scripts

### 1) Single quick benchmark

```bash
bash scripts/quick_bench_once.sh
```

Behavior:

- Uses running services if healthy
- Otherwise starts local LLM/TTS services temporarily
- Runs `scripts/benchmark_stack.py` and saves JSON to `/tmp/qwen_bench_once_<timestamp>.json`

Useful env overrides:

```bash
LLM_URL=http://127.0.0.1:8000 \
TTS_URL=http://127.0.0.1:8001 \
BENCH_TTS_MAX_NEW_TOKENS=256 \
BENCH_TIMEOUT_S=600 \
bash scripts/quick_bench_once.sh
```

### 2) Multi-case sweep (short/medium/long)

```bash
bash scripts/quick_bench_sweep.sh
```

Behavior:

- Runs 3 benchmark cases with different utterance lengths
- Emits per-case JSON plus `summary.json`
- Output directory defaults to `/tmp/qwen_bench_sweep_<timestamp>/`

Override output directory:

```bash
BENCH_SWEEP_OUT_DIR=/tmp/my_bench_sweep bash scripts/quick_bench_sweep.sh
```

## Metrics and Measurement Definitions

Implemented in `scripts/benchmark_stack.py`:

- `llm.decode_tok_s = token_count / (t_end - t_first_token)`
- `tts.first_chunk_ms = (t_first_chunk - t_request_start) * 1000`
- `tts.audio_s = bytes_received / (24000 * 2)`
- `tts.rtf = tts_total_seconds / tts.audio_s`
- `e2e_estimate_ms = llm.ttft_ms + tts.first_chunk_ms`

Backend TTS also exposes `X-TTFC-Ms`, measured server-side and returned in benchmark output as `header_ttfc_ms`.

## Deliverables Mapping

- Working build/run flow on one machine: yes (bootstrap + run_local)
- Architecture/kernels documented: yes (this README + source files)
- Quick benchmark methodology and scripts: yes (`quick_bench_once.sh`, `quick_bench_sweep.sh`)
- Short engineering note: see `README_BENCHMARK_NOTES.md`

## References

- Blog: `blog.alpindale.net/posts/5090_decode_optimization/`
- Source: `github.com/AlpinDale/qwen_megakernel`
- Pipecat docs: `docs.pipecat.ai`
- Qwen3-TTS: `huggingface.co/Qwen/Qwen3-TTS`
