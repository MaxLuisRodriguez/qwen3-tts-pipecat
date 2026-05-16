# RTX 5090 Decode Megakernel -> Qwen3-TTS on Pipecat

This repo adapts AlpinDale's `qwen_megakernel` decode kernel to run the Qwen3-TTS talker decoder inside a local Pipecat voice pipeline:

`Deepgram STT -> Megakernel LLM service -> Local Qwen3-TTS talker service -> Daily audio output`

The original take-home prompt is no longer stored in the repo; this README is the current source of run, benchmark, and limitation notes.

## What Is Working

- Local RTX 5090 megakernel LLM service on `:8000`
- Local Qwen3-TTS talker service on `:8001`
- Daily Pipecat room bot with streamed PCM audio
- Decode-time audio streaming to Pipecat without full-utterance buffering
- Binary TTS path pushes chunks to Pipecat immediately as they are decoded
- Scheduler-level multi-request TTS serving with up to 4 scalar backend slots
- **Cross-slot subtalker batching** via `SubtalkerBatchService`: all four slots' subtalker forwards are routed through a single owner thread that gathers them in a `~1.5 ms` adaptive window and runs one batched forward. This takes concurrent c4 GPU util from `~18%` to `~97%` and c4 RTF p50 from `2.64` to `0.59`.
- End-to-end voice turns with live terminal metrics

The pipeline is stable for short voice interactions, hits real-time on single streams (RTF p50 `0.47`), and stays under real-time on concurrent c4 traffic (RTF p50 `0.59`).

A Python-side dispatcher for a future true batched CUDA talker megakernel (`QWEN3_TTS_TALKER_BATCH=1`) is committed as scaffolding but off by default — see `services/tts_qwen3/batched_talker_service.py` and `STILL_TO_IMPLEMENT.md` for what the CUDA-side work requires.

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

The TTS service streams PCM while the utterance is being generated:

- no full-waveform fallback path is used in the active pipeline
- Pipecat consumes the HTTP stream incrementally
- `/synthesize_binary` does not front-buffer a non-silent prefix before returning the streaming response
- the speech tokenizer decode is stateful/incremental within a request

### 4. Cross-slot subtalker batching for concurrent throughput

The concurrent c4 path used to collapse to RTF `2.64` with GPU util `18.6%`. The bottleneck was not GPU compute — it was the GIL serializing four worker threads, each making 8 subtalker forwards per frame through the same `torch.compile`d module.

`SubtalkerBatchService` ([services/tts_qwen3/subtalker_batch_service.py](/root/qwen3-tts-pipecat/services/tts_qwen3/subtalker_batch_service.py)) owns the compiled subtalker and serves all four slots from a single owner thread:

- Slots submit `(inputs_embeds, past_key_values, sampling)` to a condition-variable queue.
- The owner thread gathers submissions arriving within a `1.5 ms` adaptive window, groups them by sampling parameters + KV sequence length, concatenates per-layer K/V along the batch dim, runs one batched forward, and splits per-row outputs / updated `DynamicCache` rows back to the submitting threads.
- A persistent merged-cache pool keyed by batch size keeps `past_key_values` object identity stable across calls so dynamo doesn't recompile per call.
- The owner thread runs inside `torch.inference_mode()` so dispatch keys match the warmup tensors.
- Adaptive window suppresses the wait when the previous served batch was a singleton, so single-stream traffic doesn't pay the batching latency floor.
- Warmup driver exercises the same code paths for `B ∈ {1..4}`, `kv_len ∈ {0..7}` at server load.

Result: c4 RTF p50 `2.64 → 0.59`, GPU util avg `18.6% → 97.1%`, parity preserved. Single-stream RTF also improved (`0.78 → 0.47`) because the compile cache is now shared across slots.

### 5. Hardened short-turn stability without changing providers

The repo includes:

- safer text normalization before TTS
- repetition/degeneration guards in the local talker path
- candidate-based speech stabilization inside the same local backend
- warmup/preload support for the first turn

No external TTS fallback was introduced; all results reflect the performance of the local stack.

## Repo Layout

- [kernel/](/root/qwen3-tts-pipecat/kernel): CUDA extension and Python bindings
- [services/llm_megakernel/](/root/qwen3-tts-pipecat/services/llm_megakernel): FastAPI SSE LLM service
- [services/tts_qwen3/](/root/qwen3-tts-pipecat/services/tts_qwen3): FastAPI streaming TTS service
- [services/tts_qwen3/subtalker_batch_service.py](/root/qwen3-tts-pipecat/services/tts_qwen3/subtalker_batch_service.py): cross-slot batched subtalker dispatcher (default on)
- [services/tts_qwen3/batched_talker_service.py](/root/qwen3-tts-pipecat/services/tts_qwen3/batched_talker_service.py): Stage-2 scaffolding for batched talker megakernel (default off)
- [pipecat_demo/](/root/qwen3-tts-pipecat/pipecat_demo): Daily/Pipecat voice bot
- [scripts/bootstrap_qwen_megakernel.sh](/root/qwen3-tts-pipecat/scripts/bootstrap_qwen_megakernel.sh): bootstrap/runtime setup
- [scripts/run_local.sh](/root/qwen3-tts-pipecat/scripts/run_local.sh): one-command local demo
- [scripts/benchmark_stack.py](/root/qwen3-tts-pipecat/scripts/benchmark_stack.py): service-level benchmark
- [scripts/benchmark_concurrent_tts.py](/root/qwen3-tts-pipecat/scripts/benchmark_concurrent_tts.py): jittered concurrent TTS benchmark
- [scripts/benchmark_roundtrip.py](/root/qwen3-tts-pipecat/scripts/benchmark_roundtrip.py): parses Pipecat turn metrics
- [scripts/check_tts_parity.py](/root/qwen3-tts-pipecat/scripts/check_tts_parity.py): parity check against PyTorch reference (supports `--use-subtalker-service`)

## Setup

### 1. Bootstrap

```bash
cp -n .env.qwen_megakernel.template .env.qwen_megakernel
bash scripts/bootstrap_qwen_megakernel.sh
```

Bootstrap creates `kernel/.venv`, seeds `pip`, installs the validated runtime, resolves model weights, and builds the megakernel extension. It also attempts to install `flash-attn==2.8.3` with `TORCH_CUDA_ARCH_LIST=12.0` for RTX 5090/Blackwell. Set `REQUIRE_FLASH_ATTN=1` before running bootstrap if you want setup to fail loudly when Flash Attention cannot be built.

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

## Key Environment Variables

These all have sensible defaults; set them only if you want to override:

| variable | default | purpose |
| --- | --- | --- |
| `QWEN3_TTS_MAX_ACTIVE_REQUESTS` | `4` | scheduler-level concurrent backend slot count |
| `QWEN3_TTS_SUBTALKER_COMPILE` | `1` | `torch.compile` the subtalker (SDPA attention) at backend init |
| `QWEN3_TTS_SUBTALKER_BATCH` | `1` | enable cross-slot subtalker batching service |
| `QWEN3_TTS_SUBTALKER_BATCH_WINDOW_MS` | `1.5` | base batching window (ms) for the subtalker service |
| `QWEN3_TTS_SUBTALKER_BATCH_MAX_B` | `8` | maximum batch size the service will form |
| `QWEN3_TTS_SUBTALKER_BATCH_ADAPTIVE_WINDOW` | `1` | suppress window wait after a singleton served (helps c1) |
| `QWEN3_TTS_SUBTALKER_BATCH_WARMUP` | `1` | run B=1..4 × kv_len=0..7 warmup at service init |
| `QWEN3_TTS_PREFILL_OPTIMIZED` | `1` | cache request-independent speaker/language scaffold tensors |
| `QWEN3_TTS_PREWARM_BACKENDS` | `1` | run synthetic warm request at server load time |
| `QWEN3_TTS_ANTI_CHEAT` | `0` | disable prompt/projection caches for honest benchmarks |
| `QWEN3_TTS_SYNC_TIMING` | `0` | `cuda.synchronize` around phase timing for hot-path attribution |
| `QWEN3_TTS_AUDIO_DECODE_OVERLAP` | `0` | run incremental tokenizer decode on a worker CUDA stream |
| `QWEN3_TTS_TALKER_BATCH` | `0` | enable Stage-2 batched talker dispatcher (requires batched kernel — currently `NotImplementedError`) |
| `QWEN3_TTS_PER_SLOT_STREAM` | `0` | give each backend its own CUDA stream (needs event-based sync — scaffolding only) |
| `QWEN3_TTS_DYNAMO_CACHE_SIZE_LIMIT` | `256` | dynamo recompile cache size |
| `QWEN3_TTS_DYNAMO_ACCUM_CACHE_SIZE_LIMIT` | `1024` | dynamo accumulated cache size |

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

Start the TTS service with anti-cheat cache disabling for a clean benchmark:

```bash
QWEN3_TTS_ANTI_CHEAT=1 QWEN3_TTS_SYNC_TIMING=0 START_TTS_SERVICE=1 bash scripts/run_local.sh
```

Then run:

```bash
kernel/.venv/bin/python scripts/benchmark_concurrent_tts.py \
  --tts-url http://127.0.0.1:8001 \
  --concurrency 4 \
  --requests 32 \
  --request-rate 4 \
  --jitter-ms 250 \
  --json-out benchmark_batch4.json \
  --csv-out benchmark_batch4.csv
```

This benchmark measures observed client-side time to first PCM chunk. The JSON/CSV outputs include backend headers such as `scheduler_mode`, `kernel_path`, and the effective max-token cap.

### Deterministic parity/sanity check

```bash
kernel/.venv/bin/python scripts/check_tts_parity.py \
  --text "the city is tehran" \
  --voice vivian \
  --language english \
  --max-new-tokens 96 \
  --use-subtalker-service \
  --json
```

`--use-subtalker-service` runs the parity check against the batched-subtalker path so both code paths are covered.

### Exact backend measurement command used for single-stream numbers

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

These are the headline numbers on the remote RTX 5090 VM on 2026-05-16. They are intentionally reported as measured, not idealized.

Runtime:

- `torch==2.7.0+cu128`
- `flash-attn==2.8.3`
- TTS attention: `flash_attention_2` (talker), `sdpa` (subtalker — required for `torch.compile`)
- benchmark env: `QWEN3_TTS_ANTI_CHEAT=1`, `QWEN3_TTS_SYNC_TIMING=0`
- subtalker compile: `QWEN3_TTS_SUBTALKER_COMPILE=1`
- subtalker batching: `QWEN3_TTS_SUBTALKER_BATCH=1`
- prefill optimization: `QWEN3_TTS_PREFILL_OPTIMIZED=1`, `QWEN3_TTS_PREFILL_KERNEL=0`, `QWEN3_TTS_PREWARM_BACKENDS=1`
- decode cadence: `adaptive(mid=4, late=8@24, ctx=25)`
- scheduler mode: `scheduler_scalar`, max active requests = 4
- kernel path reported by service: `qwen_megakernel_C.decode_hidden_fp32_head`

### Single-stream warm-path numbers (direct backend)

`vivian` voice, `english`, `max_new_tokens=64`:

| text | wall_s | RTF | TTFC ms | subtalker_ms |
| --- | ---: | ---: | ---: | ---: |
| `ready` | `0.555` | **`0.289`** | `53.3` | `432.3` |
| `the city is tehran` | `0.785` | **`0.280`** | `53.5` | `628.9` |
| `performance benchmark testing` | `0.805` | **`0.279`** | `53.2` | `648.4` |

### Concurrent c4 benchmark (32 requests, 4 req/s, jitter 250ms, anti-cheat caches off)

The headline result of this submission:

| concurrency | successful | TTFC p50 / p90 / p99 ms | RTF p50 / p90 / p99 | max chunk gap p99 ms | GPU util avg / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 16/16 | `136 / 139 / 140` | `0.474 / 0.483 / 0.491` | `286` | `99.5% / 100%` |
| **4** | **32/32** | **`167 / 264 / 308`** | **`0.588 / 1.14 / 1.30`** | **`851`** | **`97.1% / 100%`** |

For comparison, pre-batching baseline numbers from the same benchmark (commit `0c55fa8`, same env, same protocol):

| concurrency | TTFC p50 / p90 / p99 ms | RTF p50 / p90 / p99 | GPU util avg |
| --- | ---: | ---: | ---: |
| 1 | `109 / 111 / 268` | `0.781 / 0.788 / 0.892` | `9.0%` |
| 4 | `267 / 482 / 513` | `2.641 / 3.242 / 3.638` | `18.6%` |

### Stage-1/Stage-2 benchmark artifacts

- [benchmark_results/stage1_c1.json](/root/qwen3-tts-pipecat/benchmark_results/stage1_c1.json)
- [benchmark_results/stage1_c4.json](/root/qwen3-tts-pipecat/benchmark_results/stage1_c4.json)
- [benchmark_results/stage2_c1.json](/root/qwen3-tts-pipecat/benchmark_results/stage2_c1.json)
- [benchmark_results/stage2_c4.json](/root/qwen3-tts-pipecat/benchmark_results/stage2_c4.json)
- [benchmark_results/prefill_opt_c1.json](/root/qwen3-tts-pipecat/benchmark_results/prefill_opt_c1.json) (pre-batching baseline)
- [benchmark_results/prefill_opt_c4.json](/root/qwen3-tts-pipecat/benchmark_results/prefill_opt_c4.json) (pre-batching baseline)

### Parity / sanity check

`scripts/check_tts_parity.py` returns `"ok": true` for the three test texts (`ready`, `the city is tehran`, `performance benchmark testing`) on both the scalar path and the `--use-subtalker-service` path. First-frame codec codes are bit-identical between paths.

Phase-level timing from a `QWEN3_TTS_SYNC_TIMING=1` parity run:

- prefill: `219.8 ms` on the first backend request (warm: `23.8 ms`)
- subtalker / code predictor: `1250.1 ms` baseline, `432.3 ms` after `torch.compile`, much lower per-row under c4 batching
- talker custom-kernel decode: `33.6 ms`
- speech tokenizer / audio decode: `287.7 ms`

## Architecture Summary

### Request lifecycle

1. Client posts to `/synthesize_binary` on `:8001`.
2. `TTSRequestScheduler` admits the request into one of `QWEN3_TTS_MAX_ACTIVE_REQUESTS` (default 4) backend slots, with a small batching window for arrival coalescing.
3. The slot's worker thread runs `TalkerMegakernelBackend.stream_audio`:
   - Build prompt (cached scaffold tensors per speaker/language).
   - Prefill via PyTorch `talker.model(inputs_embeds=..., use_cache=True)`; copy KV into the megakernel's pre-allocated cache.
   - Loop: predict next codec frame via `SubtalkerBatchService.submit(...)` (which delivers the per-row hidden + updated `DynamicCache`), then advance the talker hidden state one step via the scalar `decode_hidden_fp32_head` megakernel binding. Append the audio codec frame to a rolling buffer.
   - Incrementally decode audio chunks via `IncrementalTokenizerDecoderV2` and yield PCM as soon as the first non-empty chunk is ready.
4. `SubtalkerBatchService` runs on its own owner thread, gathers per-slot submissions, runs one batched forward through the compiled subtalker, and routes per-row results back to slots via `threading.Event` per call.

### Why this layout

- Scheduler-level scalar slot parallelism is the cheapest correct way to admit multiple concurrent requests without a true batched megakernel.
- The talker megakernel is scalar by design (single-token, hand-rolled). Each slot's per-step CUDA work is `~2 ms`.
- The dominant cost was the subtalker, which is a HF code-predictor module that benefits massively from `torch.compile` (`2.9×` single-stream) and from cross-slot batching (`5.2×` GPU util under c4).
- The talker megakernel remaining as scalar is fine because its per-call cost is already small enough to fit inside the time spent on subtalker batching. A true batched megakernel is scaffolded (`batched_talker_service.py`) but would add only `~20-30 ms` per c4 request to the wins, with substantial rewrite cost — see `STILL_TO_IMPLEMENT.md`.

## Remaining Bottlenecks

After Stage 1 + Stage 2 finalization:

1. **Audio decode** (`~12 ms/frame`) is now a larger relative fraction of per-frame cost on c4. The overlap path (`QWEN3_TTS_AUDIO_DECODE_OVERLAP=1`) is implemented and bit-identical but was measured pre-Stage-1; it should be re-evaluated on the current build.
2. **CUDA graph capture for the subtalker step** is the single highest-payoff remaining change.
3. **True batched CUDA talker megakernel** unblocks higher concurrency tiers (c8+) but is bounded payoff at c4 (~20-30 ms per request). Stage-2 scaffolding is committed; the CUDA-side rewrite has five specific changes documented in `batched_talker_service.py`.
4. **Talker prefill** is still PyTorch by default. The experimental megakernel prefill path matched the first token but drifted on full utterance parity; kept off.
5. **fp32 LM-head / argmax fusion** is small but free.
6. **Long-form talker generation** is still less efficient than the raw text decode path.

Cold-start trade-offs:
- `~8 s` startup compile warmup when the backend loads (one-time per process)
- Additional `~5-8 s` warmup for the batched subtalker service to compile B=1..4 shape variants
- KV-cache memory increases from `~4.7 GiB` (pre-batching) to `~8.0 GiB` peak at c4 (within RTX 5090's 32 GiB)

## How TTFC And RTF Were Calculated

### Direct backend TTS numbers

The warm and cold local TTS numbers in `Current Measurements` were computed from the backend stats object returned by [megakernel_talker.py](/root/qwen3-tts-pipecat/services/tts_qwen3/megakernel_talker.py#L653).

- `ttfc_ms`
  - recorded when the first non-empty decoded audio chunk is yielded
- `generation_s`
  - total CUDA-event elapsed generation time
- `audio_s`
  - total emitted audio duration in seconds
- `rtf`
  - `rtf = generation_s / audio_s`

### Concurrent benchmark numbers

The c1/c4 numbers in `Current Measurements` come from `scripts/benchmark_concurrent_tts.py`, which spawns jittered client requests and measures observed client-side time to first PCM chunk and total client-side stream time. RTF is computed from the PCM byte count: `audio_s = raw_bytes / (sample_rate * 2)`.

### Live Pipecat turn numbers

The live round-trip numbers come from [pipecat_demo/app.py](/root/qwen3-tts-pipecat/pipecat_demo/app.py).

- `ttfc_ms`
  - measured from request start to the first received PCM body chunk
  - `/synthesize_binary` intentionally reports `X-TTFC-Ms: na` because the server cannot know TTFC before response headers are sent
- `audio_s = raw_bytes / (sample_rate * 2)`
- `rtf = tts_stream_s / audio_s`

## Deliverables Mapping

- Working repo with build instructions:
  - covered by the `Setup` section and [scripts/bootstrap_qwen_megakernel.sh](/root/qwen3-tts-pipecat/scripts/bootstrap_qwen_megakernel.sh)
- Short README with architecture decisions, kernel modifications, and how to run the demo:
  - this file
- Performance numbers:
  - reported in `Current Measurements`; full write-up in `PERFORMANCE_NOTES.md`
- End-to-end latency and streaming confirmation:
  - reported via the live Pipecat metric lines and [scripts/benchmark_roundtrip.py](/root/qwen3-tts-pipecat/scripts/benchmark_roundtrip.py)
- Honest gap list:
  - `STILL_TO_IMPLEMENT.md`
- Demo recording:
  - not stored in the repo; attach separately as an out-of-repo submission artifact

## References

- AlpinDale blog: `blog.alpindale.net/posts/5090_decode_optimization/`
- `qwen_megakernel`: `github.com/AlpinDale/qwen_megakernel`
- Pipecat docs: `docs.pipecat.ai`
- Qwen3-TTS: `huggingface.co/Qwen/Qwen3-TTS`
