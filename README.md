# RTX 5090 Decode Megakernel -> Qwen3-TTS on Pipecat

This repo adapts AlpinDale's `qwen_megakernel` decode kernel to run the Qwen3-TTS talker decoder inside a local Pipecat voice pipeline:

`Deepgram STT -> Megakernel LLM service -> Local Qwen3-TTS talker service -> Daily audio output`

The original take-home prompt is no longer stored in the repo; this README is the current source of run, benchmark, and limitation notes.

## What Is Working

- Local RTX 5090 megakernel LLM service on `:8000`
- Local Qwen3-TTS talker service on `:8001`
- Daily Pipecat room bot with streamed PCM audio
- Decode-time audio streaming to Pipecat without full-utterance buffering
- Binary TTS path now pushes chunks to Pipecat immediately as they are decoded
- Scheduler-level multi-request TTS serving with up to 4 scalar backend slots
- End-to-end voice turns with live terminal metrics

The pipeline is now stable for short voice interactions. 
The primary remaining limitation is performance: TTFC and RTF remain above the stretch targets.
The batch-4 path is currently scheduler-level concurrency, not a true batched CUDA megakernel.

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

No external TTS fallback was introduced; all results reflect the performance of the local stack.

## Repo Layout

- [kernel/](/root/qwen3-tts-pipecat/kernel): CUDA extension and Python bindings
- [services/llm_megakernel/](/root/qwen3-tts-pipecat/services/llm_megakernel): FastAPI SSE LLM service
- [services/tts_qwen3/](/root/qwen3-tts-pipecat/services/tts_qwen3): FastAPI streaming TTS service
- [pipecat_demo/](/root/qwen3-tts-pipecat/pipecat_demo): Daily/Pipecat voice bot
- [scripts/bootstrap_qwen_megakernel.sh](/root/qwen3-tts-pipecat/scripts/bootstrap_qwen_megakernel.sh): bootstrap/runtime setup
- [scripts/run_local.sh](/root/qwen3-tts-pipecat/scripts/run_local.sh): one-command local demo
- [scripts/benchmark_stack.py](/root/qwen3-tts-pipecat/scripts/benchmark_stack.py): service-level benchmark
- [scripts/benchmark_concurrent_tts.py](/root/qwen3-tts-pipecat/scripts/benchmark_concurrent_tts.py): jittered concurrent TTS benchmark
- [scripts/benchmark_roundtrip.py](/root/qwen3-tts-pipecat/scripts/benchmark_roundtrip.py): parses Pipecat turn metrics

## Setup

### 1. Bootstrap

```bash
cp -n .env.qwen_megakernel.template .env.qwen_megakernel
bash scripts/bootstrap_qwen_megakernel.sh
```

Bootstrap creates `kernel/.venv`, seeds `pip`, installs the validated runtime, resolves model weights, and builds the megakernel extension.
It also attempts to install `flash-attn==2.8.3` with `TORCH_CUDA_ARCH_LIST=12.0` for RTX 5090/Blackwell. Set `REQUIRE_FLASH_ATTN=1` before running bootstrap if you want setup to fail loudly when Flash Attention cannot be built.

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
  --log-path /path/to/run_local.log \
  --wait-timeout-s 120
```

### Jittered concurrent TTS check

Start the TTS service with anti-cheat cache disabling if you want a clean benchmark:

```bash
QWEN3_TTS_ANTI_CHEAT=1 QWEN3_TTS_SYNC_TIMING=0 START_TTS_SERVICE=1 bash scripts/run_local.sh
```

Then run:

```bash
kernel/.venv/bin/python scripts/benchmark_concurrent_tts.py \
  --tts-url http://127.0.0.1:8001 \
  --concurrency 4 \
  --requests 16 \
  --request-rate 4 \
  --jitter-ms 250 \
  --json-out benchmark_batch4.json \
  --csv-out benchmark_batch4.csv
```

This benchmark measures observed client-side time to first PCM chunk. It does not use generated headers for TTFC.
The JSON/CSV outputs include backend headers such as `scheduler_mode`, `kernel_path`, and the effective max-token cap.

### Deterministic parity/sanity check

```bash
kernel/.venv/bin/python scripts/check_tts_parity.py \
  --text "the city is tehran" \
  --voice vivian \
  --language english \
  --max-new-tokens 96 \
  --json
```

### Exact backend measurement command used for the README numbers

This command measures the local talker backend directly, without Daily or HTTP transport in the loop:

```bash
cd /root/qwen3-tts-pipecat
set -a
source .env.qwen_megakernel
[ -f .env.pipecat ] && source .env.pipecat || true
set +a
export PYTHONPATH="/root/qwen3-tts-pipecat/kernel"
export PATH="/root/qwen3-tts-pipecat/kernel/.venv/bin:/venv/qwen-live-chatbot/bin:$PATH"
./kernel/.venv/bin/python - <<'PY'
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from services.tts_qwen3.megakernel_talker import TalkerMegakernelBackend

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
model.model.eval()
backend = TalkerMegakernelBackend(model)

for text in ["ready", "the city is tehran"]:
    stats, audio_iter = backend.stream_audio(
        text=text,
        speaker="vivian",
        language="english",
        max_new_tokens=64,
    )
    sample_count = 0
    chunk_count = 0
    for chunk in audio_iter:
        arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
        sample_count += arr.size
        chunk_count += 1
    rtf = (stats.generation_s / stats.audio_seconds) if stats.audio_seconds else None
    print(
        {
            "text": text,
            "chunks": chunk_count,
            "sample_count": sample_count,
            "ttfc_ms": stats.ttfc_ms,
            "audio_s": stats.audio_seconds,
            "generation_s": stats.generation_s,
            "rtf": rtf,
            "stop_reason": stats.stop_reason,
        }
    )
PY
```

### Exact live Pipecat measurement flow

1. Start the stack:

```bash
START_TTS_SERVICE=1 bash scripts/run_local.sh
```

2. Join the printed Daily room URL and do one short voice turn.

3. Read the terminal lines printed by [pipecat_demo/app.py](/root/qwen3-tts-pipecat/pipecat_demo/app.py):

- `[metrics][roundtrip]`
- `[metrics][stream]`
- `[metrics][quality]`

4. Or parse a saved log file with:

```bash
kernel/.venv/bin/python scripts/benchmark_roundtrip.py \
  --log-path /path/to/run_local.log \
  --wait-timeout-s 120
```

## Current Measurements

These are the current representative numbers from the remote RTX 5090 VM on 2026-05-15. They are intentionally reported as measured, not idealized.

Runtime:

- `torch==2.7.0+cu128`
- `flash-attn==2.8.3`
- TTS attention implementation: `flash_attention_2`
- benchmark env: `QWEN3_TTS_ANTI_CHEAT=1`, `QWEN3_TTS_SYNC_TIMING=0`
- prefill optimization: `QWEN3_TTS_PREFILL_OPTIMIZED=1`, `QWEN3_TTS_PREFILL_KERNEL=0`, `QWEN3_TTS_PREWARM_BACKENDS=1`
- decode cadence: `adaptive(mid=4,late=8@24,ctx=25)`
- scheduler mode: `scheduler_scalar`
- kernel path reported by service: `qwen_megakernel_C.decode_hidden_fp32_head`

Parity/sanity check:

- command: `kernel/.venv/bin/python scripts/check_tts_parity.py --text ready --voice vivian --language english --max-new-tokens 64 --json`
- result: `ok=true`
- first-step PyTorch token: `215`
- first-step custom-kernel token: `215`
- hidden max/mean abs diff: `0.4534 / 0.0432`
- logit max/mean abs diff: `0.1520 / 0.0320`
- optimized/base audio duration delta: `0.32 s`
- optimized/base RMS relative delta: `0.0021`
- optimized/base peak relative delta: `0.3006`

Phase timing from the same short parity run with `QWEN3_TTS_SYNC_TIMING=1` and the safe prefill path:

- prefill: `219.8 ms` on the first backend request
- prompt build: `139.7 ms`
- PyTorch prefill model: `79.4 ms`
- prefill KV copy: `0.7 ms`
- subtalker/code predictor: `1250.1 ms`
- talker custom-kernel decode: `33.6 ms`
- speech tokenizer/audio decode: `287.7 ms`
- frames generated: `22`
- emitted audio: `1.76 s`

Warm same-backend prefill check:

- first request: `235.5 ms` prefill (`151.8 ms` prompt build + `83.0 ms` model)
- second request: `23.8 ms` prefill (`0.7 ms` prompt build + `22.4 ms` model)
- explanation: `QWEN3_TTS_PREFILL_OPTIMIZED=1` caches only request-independent speaker/language scaffold tensors per backend slot. It does not cache projected request text, generated tokens, or audio.
- experimental result: `QWEN3_TTS_PREFILL_KERNEL=1` reduced short-prompt prefill to about `166 ms` in the end-to-end parity run, but full utterance parity failed because generation drifted and hit `max_new_tokens`; it is therefore kept off by default.

Jittered concurrent TTS benchmark artifacts:

- [benchmark_results/concurrency_1_after_instrumentation.json](/root/qwen3-tts-pipecat/benchmark_results/concurrency_1_after_instrumentation.json)
- [benchmark_results/concurrency_2_after_instrumentation.json](/root/qwen3-tts-pipecat/benchmark_results/concurrency_2_after_instrumentation.json)
- [benchmark_results/concurrency_4_after_instrumentation.json](/root/qwen3-tts-pipecat/benchmark_results/concurrency_4_after_instrumentation.json)
- [benchmark_results/concurrency_8_after_instrumentation.json](/root/qwen3-tts-pipecat/benchmark_results/concurrency_8_after_instrumentation.json)
- [benchmark_results/prefill_opt_c1.json](/root/qwen3-tts-pipecat/benchmark_results/prefill_opt_c1.json)
- [benchmark_results/prefill_opt_c2.json](/root/qwen3-tts-pipecat/benchmark_results/prefill_opt_c2.json)
- [benchmark_results/prefill_opt_c4.json](/root/qwen3-tts-pipecat/benchmark_results/prefill_opt_c4.json)
- [benchmark_results/prefill_opt_c8.json](/root/qwen3-tts-pipecat/benchmark_results/prefill_opt_c8.json)

Previous 8-request benchmark, before the prefill/scaffold prewarm pass:

| concurrency | successful | max active observed | TTFC p90 / p99 ms | RTF p50 / p99 | max chunk gap p99 ms | GPU avg / max util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8/8 | 1 | `119.5 / 126.9` | `0.786 / 0.792` | `488.9` | `17.2% / 23.0%` |
| 2 | 8/8 | 2 | `208.2 / 227.7` | `1.426 / 1.515` | `961.5` | `20.4% / 25.0%` |
| 4 | 8/8 | 4 | `494.7 / 514.1` | `3.246 / 3.954` | `2527.3` | `18.4% / 24.0%` |
| 8 | 8/8 | 8 client streams | `12792.1 / 14616.0` | `3.988 / 7.320` | `2547.6` | `17.7% / 24.0%` |

After the prefill/scaffold prewarm pass, 32-request benchmark:

| concurrency | successful | max active observed | TTFC p90 / p99 ms | RTF p50 / p99 | max chunk gap p99 ms | GPU avg / max util |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32/32 | 1 | `111.4 / 267.8` | `0.781 / 0.892` | `593.6` | `9.0% / 23.0%` |
| 2 | 32/32 | 2 | `207.8 / 231.0` | `1.207 / 1.571` | `1036.4` | `14.8% / 25.0%` |
| 4 | 32/32 | 4 | `482.4 / 513.0` | `2.641 / 3.638` | `2625.8` | `18.6% / 26.0%` |
| 8 | 32/32 | 8 client streams | `13810.6 / 16368.3` | `5.955 / 8.678` | `2500.9` | `18.3% / 24.0%` |

Apples-to-apples 8-request concurrency-4 check after prefill optimization:

- TTFC p90/p99: `444.7 / 457.2 ms`
- RTF p50/p99: `3.141 / 3.440`
- max chunk gap p99: `2581.6 ms`
- result: p90/p99 TTFC improved, RTF improved modestly, chunk gaps remain far above target.

Interpretation:

- Single-stream TTFC is now good, but single-stream RTF still misses the `<= 0.5` target.
- Concurrency 4 reaches four active scalar backend slots and p90 TTFC is under `500 ms`, but RTF and chunk gaps miss the target.
- Concurrency 8 is not a true eight-slot server run. The server defaults to `QWEN3_TTS_MAX_ACTIVE_REQUESTS=4`, so the benchmark's eight client streams include queue wait.
- Low GPU utilization plus the phase timings point to Python/PyTorch subtalker work as the current top bottleneck, not prefill or the custom talker decode kernel.
- This is not a completed true batch-4 CUDA submission yet. It is an honest scheduler-level concurrency implementation with measurement, parity checks, and a clear bottleneck report.

## How TTFC And RTF Were Calculated

### Direct backend TTS numbers

The warm and cold local TTS numbers in `Current Measurements` were computed from the backend stats object returned by [megakernel_talker.py](/root/qwen3-tts-pipecat/services/tts_qwen3/megakernel_talker.py#L653).

- `ttfc_ms`
  - recorded when the first non-empty decoded audio chunk is yielded
  - set in [megakernel_talker.py](/root/qwen3-tts-pipecat/services/tts_qwen3/megakernel_talker.py#L872) or the final flush path at [megakernel_talker.py](/root/qwen3-tts-pipecat/services/tts_qwen3/megakernel_talker.py#L919)
- `generation_s`
  - total CUDA-event elapsed generation time
  - set in [megakernel_talker.py](/root/qwen3-tts-pipecat/services/tts_qwen3/megakernel_talker.py#L928)
- `audio_s`
  - total emitted audio duration in seconds
  - set in [megakernel_talker.py](/root/qwen3-tts-pipecat/services/tts_qwen3/megakernel_talker.py#L924)
- `rtf`
  - calculated as:

```text
rtf = generation_s / audio_s
```

### Live Pipecat turn numbers

The live round-trip numbers come from [pipecat_demo/app.py](/root/qwen3-tts-pipecat/pipecat_demo/app.py).

- `ttfc_ms`
  - measured from request start to the first received PCM body chunk
  - `/synthesize_binary` intentionally reports `X-TTFC-Ms: na` because the server cannot know TTFC before response headers are sent
  - see [app.py](/root/qwen3-tts-pipecat/pipecat_demo/app.py#L915) and [app.py](/root/qwen3-tts-pipecat/pipecat_demo/app.py#L987)
- `audio_s`
  - calculated from streamed PCM bytes:

```text
audio_s = raw_bytes / (sample_rate * 2)
```

- `rtf`
  - calculated as:

```text
rtf = tts_stream_s / audio_s
```

  - see [app.py](/root/qwen3-tts-pipecat/pipecat_demo/app.py#L983) and [app.py](/root/qwen3-tts-pipecat/pipecat_demo/app.py#L988)

Interpretation:

- The local LLM decode path is no longer the dominant bottleneck.  
- Warm-state TTS TTFC has improved significantly compared to earlier versions.
- Cold start is still much slower than warm steady-state.
- RTF and chunk gaps still miss the take-home stretch targets under concurrent load.

## Why The Targets Are Still Missed

The repo now has a stable and honest local stack, but the remaining bottlenecks are real:

1. Subtalker/code predictor execution is the largest measured short-turn cost.
2. Speech tokenizer decode, even incrementally, is still a major steady-state cost.
3. Talker prefill is improved for warm backend slots, but PyTorch prefill is still used for correctness.
4. The current multi-request path uses scheduler-level scalar backend slots; true batched CUDA decode is still future work.
5. Some short-turn reliability logic still trades latency for robustness.
6. Long-form talker generation is much less efficient than the raw Qwen3 megakernel text decode path.

This explains why the system feels smooth during short interactions while still missing the take-home RTF target under formal measurement.

### Empirical note: audio decode overlap is implemented but does not move the needle

An audio-decode-overlap path is wired into [services/tts_qwen3/megakernel_talker.py](/root/qwen3-tts-pipecat/services/tts_qwen3/megakernel_talker.py) and is opt-in via `QWEN3_TTS_AUDIO_DECODE_OVERLAP=1`. It runs the incremental tokenizer decode on a dedicated CUDA stream from a worker thread, pinning that thread to its own `IncrementalTokenizerDecoderV2` state. Output audio is bit-identical to the serial path (RMS / peak / prefix-correlation match in `check_tts_parity.py`).

Direct backend benchmark on the warm path (`vivian` voice, `english`, `max_new_tokens=64`):

| text | mode | wall_total_s | TTFC ms | max chunk gap ms | RTF | subtalker_ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ready | overlap | 1.41 | 162.9 | 248.7 | 0.80 | 1330.6 |
| ready | serial  | 1.37 |  92.8 | 244.5 | 0.78 | 1252.7 |
| the city is tehran | overlap | 1.90 | 153.9 | 271.6 | 0.79 | 1801.8 |
| the city is tehran | serial  | 1.85 |  92.4 | 302.2 | 0.77 | 1705.5 |
| performance benchmark testing | overlap | 3.52 | 157.2 | 482.5 | 0.78 | 3386.3 |
| performance benchmark testing | serial  | 3.44 |  92.4 | 475.7 | 0.77 | 3241.4 |

Subtalker dominates per-frame wall time (~60 ms / frame) while audio decode contributes only ~12 ms / frame averaged. Even with perfect overlap, the savings are bounded by audio-decode time, which is ~6% of total wall. The overlap path also pays a first-chunk delay of one extra subtalker iteration (it cannot yield the first chunk until at least one drain), which regresses TTFC by ~60-70 ms.

Conclusion: the audio decode overlap is **correct** but **not a measurable win**. The default is `QWEN3_TTS_AUDIO_DECODE_OVERLAP=0`. The real next-step optimization target is subtalker execution itself (CUDA graphs, kernel fusion, or speculative decoding).

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

## References

- AlpinDale blog: `blog.alpindale.net/posts/5090_decode_optimization/`
- `qwen_megakernel`: `github.com/AlpinDale/qwen_megakernel`
- Pipecat docs: `docs.pipecat.ai`
- Qwen3-TTS: `huggingface.co/Qwen/Qwen3-TTS`
