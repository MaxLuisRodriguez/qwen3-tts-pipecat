# RTX 5090 Decode Megakernel -> Qwen3-TTS on Pipecat

This repo adapts AlpinDale's `qwen_megakernel` decode kernel to run the Qwen3-TTS talker decoder inside a local Pipecat voice pipeline:

`Deepgram STT -> Megakernel LLM service -> Local Qwen3-TTS talker service -> Daily audio output`

The take-home prompt that guided this work is preserved in [project_instructions.md](/root/qwen3-tts-pipecat/project_instructions.md).

## What Is Working

- Local RTX 5090 megakernel LLM service on `:8000`
- Local Qwen3-TTS talker service on `:8001`
- Daily Pipecat room bot with streamed PCM audio
- Decode-time audio streaming to Pipecat without full-utterance buffering
- End-to-end voice turns with live terminal metrics

The pipeline is now stable enough for normal short voice turns. The main remaining gap is performance: TTFC and especially RTF are still above the stretch targets from the prompt.

## Key Integration Decisions

### 1. Kept the original service boundaries

- LLM service: [services/llm_megakernel/server.py](/root/qwen3-tts-pipecat/services/llm_megakernel/server.py)
- TTS service: [services/tts_qwen3/server.py](/root/qwen3-tts-pipecat/services/tts_qwen3/server.py)
- Pipecat app: [pipecat_demo/app.py](/root/qwen3-tts-pipecat/pipecat_demo/app.py)

That keeps the stack easy to benchmark and debug without changing the user-facing pipeline.

### 2. Added a talker-specific megakernel path

The original megakernel was text decode oriented. This repo adds a Qwen3-TTS talker adapter in [services/tts_qwen3/megakernel_talker.py](/root/qwen3-tts-pipecat/services/tts_qwen3/megakernel_talker.py) and new bindings in [kernel/csrc/torch_bindings.cpp](/root/qwen3-tts-pipecat/kernel/csrc/torch_bindings.cpp).

Important kernel-side changes:

- hidden-state decode entrypoint for talker continuation
- hidden-only decode path so the next codec token can be chosen from the correct normalized hidden state
- reuse of the original fused decode kernel structure instead of rewriting the architecture

### 3. Kept streaming decode-time audio

The TTS service still streams PCM while the utterance is being generated:

- no full-waveform fallback path is used in the active pipeline
- Pipecat consumes the HTTP stream incrementally
- the speech tokenizer decode is stateful/incremental within a request

### 4. Hardened short-turn stability without changing providers

The repo now includes:

- safer text normalization before TTS
- repetition/degeneration guards in the local talker path
- candidate-based speech stabilization inside the same local backend
- warmup/preload support for the first turn

No external TTS fallback was added to make the bot appear better than the local stack really is.

## Repo Layout

- [kernel/](/root/qwen3-tts-pipecat/kernel): CUDA extension and Python bindings
- [services/llm_megakernel/](/root/qwen3-tts-pipecat/services/llm_megakernel): FastAPI SSE LLM service
- [services/tts_qwen3/](/root/qwen3-tts-pipecat/services/tts_qwen3): FastAPI streaming TTS service
- [pipecat_demo/](/root/qwen3-tts-pipecat/pipecat_demo): Daily/Pipecat voice bot
- [scripts/bootstrap_qwen_megakernel.sh](/root/qwen3-tts-pipecat/scripts/bootstrap_qwen_megakernel.sh): bootstrap/runtime setup
- [scripts/run_local.sh](/root/qwen3-tts-pipecat/scripts/run_local.sh): one-command local demo
- [scripts/benchmark_stack.py](/root/qwen3-tts-pipecat/scripts/benchmark_stack.py): service-level benchmark
- [scripts/benchmark_roundtrip.py](/root/qwen3-tts-pipecat/scripts/benchmark_roundtrip.py): parses Pipecat turn metrics
- [scripts/quick_bench_once.sh](/root/qwen3-tts-pipecat/scripts/quick_bench_once.sh): one-shot local benchmark
- [scripts/quick_bench_sweep.sh](/root/qwen3-tts-pipecat/scripts/quick_bench_sweep.sh): short/medium/long benchmark sweep

## Setup

### 1. Bootstrap

```bash
cp -n .env.qwen_megakernel.template .env.qwen_megakernel
bash scripts/bootstrap_qwen_megakernel.sh
```

Bootstrap creates `kernel/.venv`, seeds `pip`, installs the validated runtime, resolves model weights, and builds the megakernel extension.

### 2. Runtime config

Create a Pipecat runtime env from the new template:

```bash
cp -n .env.pipecat.template .env.pipecat
```

Fill in:

- `DEEPGRAM_API_KEY`
- either `DAILY_API_KEY`
- or `DAILY_ROOM_URL` plus `DAILY_ROOM_TOKEN`

### 3. Run the full stack

```bash
START_TTS_SERVICE=1 bash scripts/run_local.sh
```

When the terminal prints the Daily room URL, join from a browser and speak.

## Benchmarking

### Service-level quick check

```bash
bash scripts/quick_bench_once.sh
```

### Sweep

```bash
bash scripts/quick_bench_sweep.sh
```

### End-to-end Pipecat turn parsing

```bash
bash scripts/quick_bench_roundtrip.sh
```

## Current Best-Known Measurements

These are the latest service-level numbers from this repo after the final audit, using the tuned decode cadence and benchmark streaming read size:

- Short case:
  - `header_ttfc_ms ~= 162.6`
  - `tts_rtf ~= 1.48`
- Medium case:
  - `header_ttfc_ms ~= 165.1`
  - `tts_rtf ~= 1.08`
- Long case:
  - `header_ttfc_ms ~= 117.0`
  - `tts_rtf ~= 3.92`

Sweep artifact:

- [/tmp/qwen_bench_sweep_20260323_043310/summary.json](/tmp/qwen_bench_sweep_20260323_043310/summary.json)

Interpretation:

- The local LLM side is no longer the main blocker.
- The remaining performance miss is mostly on the talker/audio side.
- TTFC is still well above the prompt target, though stable enough for a demo.
- RTF remains far above target, especially on longer utterances.

## Why The Targets Are Still Missed

The repo now has a stable and honest local stack, but the remaining bottlenecks are real:

1. Talker prefill is still expensive.
2. Speech tokenizer decode, even incrementally, is still a major steady-state cost.
3. Some short-turn reliability logic still trades latency for robustness.
4. Long-form talker generation is much less efficient than the raw Qwen3 megakernel text decode path.

This is why the repo can feel smooth in short conversations while still missing the take-home RTF target in formal measurement.

## Submission Notes

- The take-home prompt asked for one informative README: this file is intended to be that document.
- The original prompt is preserved in [project_instructions.md](/root/qwen3-tts-pipecat/project_instructions.md).
- Additional measurement notes are in [README_BENCHMARK_NOTES.md](/root/qwen3-tts-pipecat/README_BENCHMARK_NOTES.md).

## References

- AlpinDale blog: `blog.alpindale.net/posts/5090_decode_optimization/`
- `qwen_megakernel`: `github.com/AlpinDale/qwen_megakernel`
- Pipecat docs: `docs.pipecat.ai`
- Qwen3-TTS: `huggingface.co/Qwen/Qwen3-TTS`
