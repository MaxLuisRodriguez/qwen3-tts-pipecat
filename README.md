# RTX 5090 Decode Megakernel -> Qwen3-TTS on Pipecat

This repo adapts AlpinDale's `qwen_megakernel` decode kernel to run the Qwen3-TTS talker decoder inside a local Pipecat voice pipeline:

`Deepgram STT -> Megakernel LLM service -> Local Qwen3-TTS talker service -> Daily audio output`

The take-home prompt that guided this work is preserved in [project_instructions.md](/root/qwen3-tts-pipecat/project_instructions.md).

## What Is Working

- Local RTX 5090 megakernel LLM service on `:8000`
- Local Qwen3-TTS talker service on `:8001`
- Daily Pipecat room bot with streamed PCM audio
- Decode-time audio streaming to Pipecat without full-utterance buffering
- Binary TTS path now pushes chunks to Pipecat immediately as they are decoded
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
- fused `decode_hidden_fp32_head` path so talker continuation keeps the correct fp32 LM-head token selection while removing one host-side step per token
- reuse of the original fused decode kernel structure instead of rewriting the architecture

### 3. Kept streaming decode-time audio

The TTS service still streams PCM while the utterance is being generated:

- no full-waveform fallback path is used in the active pipeline
- Pipecat consumes the HTTP stream incrementally
- `/synthesize_binary` no longer front-buffers a non-silent prefix before returning the streaming response
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
kernel/.venv/bin/python scripts/benchmark_stack.py \
  --llm-url http://127.0.0.1:8000 \
  --tts-url http://127.0.0.1:8001 \
  --llm-prompt "Answer in one short sentence: what can you do?" \
  --tts-text "This is a short benchmark utterance." \
  --tts-max-new-tokens 48 \
  --tts-read-chunk-bytes 960 \
  --json
```

### End-to-end Pipecat turn parsing

```bash
kernel/.venv/bin/python scripts/benchmark_roundtrip.py \
  --log-file /path/to/run_local.log \
  --timeout-s 120
```

## Current Measurements

These are the current representative numbers for the code on `main`. They are intentionally reported as measured, not idealized:

- Megakernel text decode smoke benchmark on the RTX 5090:
  - about `1043 tok/s`
- Representative warm local talker/TTS backend run:
  - text: `the city is tehran`
  - `ttfc_ms ~= 114.6`
  - `generation_s ~= 4.686`
  - `audio_s ~= 5.120`
  - `rtf ~= 0.915`
- Representative cold short-utterance run:
  - text: `ready`
  - `ttfc_ms ~= 731.0`
  - `generation_s ~= 2.153`
  - `audio_s ~= 1.520`
  - `rtf ~= 1.416`
- Last captured live Pipecat round-trip turn:
  - `overall_ms ~= 5193.4`
  - `llm_tok_s ~= 296.7`
  - `ttfc_ms ~= 171.5`
  - `rtf ~= 0.970`
  - `audio_s ~= 4.560`

Interpretation:

- The local LLM decode path is no longer the main bottleneck.
- Warm TTS is materially better than the earlier audit numbers that were in this README before the latest fixes.
- Cold start is still much slower than warm steady-state.
- TTFC and RTF are improved enough for a smooth demo, but they still miss the take-home stretch targets.

## Why The Targets Are Still Missed

The repo now has a stable and honest local stack, but the remaining bottlenecks are real:

1. Talker prefill is still expensive.
2. Speech tokenizer decode, even incrementally, is still a major steady-state cost.
3. Some short-turn reliability logic still trades latency for robustness.
4. Long-form talker generation is much less efficient than the raw Qwen3 megakernel text decode path.

This is why the repo can feel smooth in short conversations while still missing the take-home RTF target in formal measurement.

## Deliverables Mapping

- Working repo with build instructions:
  - covered by the `Setup` section and [scripts/bootstrap_qwen_megakernel.sh](/root/qwen3-tts-pipecat/scripts/bootstrap_qwen_megakernel.sh)
- Short README with architecture decisions, kernel modifications, and how to run the demo:
  - this file
- Performance numbers:
  - reported in `Current Measurements`
- End-to-end latency and streaming confirmation:
  - reported via the live Pipecat metric lines and [scripts/benchmark_roundtrip.py](/root/qwen3-tts-pipecat/scripts/benchmark_roundtrip.py)
- Demo recording:
  - not stored in the repo; attach separately as an out-of-repo submission artifact

## Submission Notes

- The take-home prompt asked for one informative README: this file is intended to be that document.
- The original prompt is preserved in [project_instructions.md](/root/qwen3-tts-pipecat/project_instructions.md).

## References

- AlpinDale blog: `blog.alpindale.net/posts/5090_decode_optimization/`
- `qwen_megakernel`: `github.com/AlpinDale/qwen_megakernel`
- Pipecat docs: `docs.pipecat.ai`
- Qwen3-TTS: `huggingface.co/Qwen/Qwen3-TTS`
