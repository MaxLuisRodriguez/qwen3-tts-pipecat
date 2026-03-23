# Benchmark Notes (Brief)

## Key Struggles

1. Talker integration complexity:
   - The original megakernel path is text decode oriented, while Qwen3-TTS talker requires codec-frame generation and audio decode orchestration.
2. Streaming correctness vs latency:
   - Decoding too frequently inflates compute overhead.
   - Decoding too sparsely hurts TTFC or risks audible discontinuity.
3. Model/runtime variability:
   - `flash-attn` compatibility can fail on some host stacks.
   - Local-vs-remote model resolution can introduce startup stalls.
4. Long-form response stability:
   - Repetition, cutoff, and text/audio drift needed guardrails in Pipecat/TTS settings.

## Key Solutions

1. Added talker-specific megakernel backend:
   - `services/tts_qwen3/megakernel_talker.py`
2. Implemented decode-time streaming path:
   - `services/tts_qwen3/server.py` streams PCM chunks during incremental decode, not full-utterance post-buffering.
3. Tuned adaptive cadence knobs:
   - `QWEN3_TTS_DECODE_STRIDE_MID`
   - `QWEN3_TTS_DECODE_STRIDE_LATE`
   - `QWEN3_TTS_INCREMENTAL_LEFT_CONTEXT_FRAMES`
4. Hardened response stability:
   - `pipecat_demo/app.py` text stabilization and chunking before TTS push.

## How Benchmark Accuracy Is Established

Four timing views are captured:

1. Client-observed streaming timings:
   - from `scripts/benchmark_stack.py` wall-clock at request, first chunk/token, and completion.
2. Backend-reported TTFC:
   - `X-TTFC-Ms` response header from TTS service, measured server-side.
3. Byte-derived audio duration:
   - computed as `bytes / (sample_rate * bytes_per_sample)` to avoid assumptions.
4. Pipecat round-trip turn metrics:
   - parsed from the existing `[metrics][roundtrip]`, `[metrics][stream]`, and `[metrics][quality]` lines emitted by `pipecat_demo/app.py`.

Consistency checks:

- Compare `tts.first_chunk_ms` (client) vs `tts.header_ttfc_ms` (server).
- Ensure `audio_s` aligns with expected utterance duration.
- Use sweep runs (short/medium/long) to confirm trend stability, not one-off samples.

## Exact Measurement Formulas

LLM:

- `ttft_ms = (t_first_token - t_request_start) * 1000`
- `decode_tok_s = token_count / (t_request_end - t_first_token)`

TTS:

- `first_chunk_ms = (t_first_audio_chunk - t_request_start) * 1000`
- `audio_s = bytes_received / (24000 * 2)` for 24kHz mono int16 PCM
- `rtf = (t_request_end - t_request_start) / audio_s`

Service-only additive estimate:

- `service_pipeline_estimate_ms = llm.ttft_ms + tts.first_chunk_ms`

This estimate intentionally excludes STT and Daily network/browser playback variability.

Actual end-to-end turn metrics:

- `overall_ms`, `llm_tok_s`, `ttfc_ms`, and `rtf` are emitted by the Pipecat app after a completed voice turn.
- `scripts/benchmark_roundtrip.py` waits for and parses those log lines into JSON.

## How to Reproduce Quickly

Single run:

```bash
bash scripts/quick_bench_once.sh
```

Single end-to-end turn:

```bash
bash scripts/quick_bench_roundtrip.sh
```

Sweep:

```bash
bash scripts/quick_bench_sweep.sh
```

Each script writes JSON artifacts to `/tmp` by default for auditable reruns.
