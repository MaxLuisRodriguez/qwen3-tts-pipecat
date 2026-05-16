# Performance Notes: How The Numbers Got Here

Personal write-up of the main wins on this take-home and what I would do next if I had more time. Companion to `README.md` (which has the build/run/measurement details) and `STILL_TO_IMPLEMENT.md` (which is the honest gap list).

All numbers below are measured on the remote RTX 5090 VM on 2026-05-16 with `torch==2.7.0+cu128` and `flash-attn==2.8.3`.

## Headline Result

Warm-path single-stream TTS on `text=ready`, `voice=vivian`, `max_new_tokens=64`:

| metric | before | after | change |
| --- | ---: | ---: | --- |
| wall_total | `1.356 s` | `0.555 s` | **2.4× faster** |
| RTF | `0.770` | `0.289` | **below the `<= 0.5` stretch target** |
| TTFC | `92.1 ms` | `53.3 ms` | **42% lower** |
| subtalker_ms | `1236.1 ms` | `432.3 ms` | **2.9× faster** |

The same speedup holds for slightly longer texts (`"the city is tehran"`, `"performance benchmark testing"`) — wall time is 2.4-4.2× faster on the warm path because the subtalker scales roughly with frame count.

Parity is preserved against the PyTorch reference: `scripts/check_tts_parity.py` returns `ok=true`, first-step token matches (`215 == 215`), RMS relative delta `0.0021`, audio duration delta `0.32 s`.

## Where The Wins Came From

The path to that result was not "find one slow thing and rewrite it." The big lesson was that the bottleneck moved twice as I optimized, and each time I had to re-measure before deciding what to do next. Here are the wins in roughly the order they landed.

### 1. Honest phase-level timing first

Before changing any kernel, I added per-phase CUDA-event timing to `megakernel_talker.py` (`StreamStats.prefill_ms / prompt_build_ms / prefill_model_ms / subtalker_ms / talker_decode_ms / audio_decode_ms`). The very first measurement showed that the custom talker decode kernel was already only `~2 ms` per request — the runaway cost was the subtalker / code predictor at `~1250 ms`, with audio decode a distant second at `~290 ms`. Without that, I would have spent days optimizing the wrong code.

**Takeaway:** never tune a CUDA kernel without per-phase wall-clock numbers from the same hardware you'll ship on.

### 2. `decode_hidden_fp32_head` fused entrypoint

The original megakernel was text-decode oriented. The talker needs the next codec token chosen from the correct normalized hidden state, not from the LLM's vocab head. Added:

- a hidden-state decode entrypoint in `kernel/csrc/torch_bindings.cpp`
- a hidden-only decode path that returns the normalized hidden state for the codec head
- a fused `decode_hidden_fp32_head` path so talker continuation keeps the correct fp32 LM-head token selection while removing one host-side step per token

This isn't where the headline RTF win came from, but it kept the talker custom-kernel decode at `~2 ms/step` — small enough that it is no longer a meaningful contributor to wall time. The fp32-head matvec/argmax is still ATen-dispatched (see `STILL_TO_IMPLEMENT.md#2`), which is the obvious next fusion step.

### 3. Streaming everything end-to-end

The pipeline was made to stream PCM out of the TTS service as soon as the first non-empty chunk is ready:

- `/synthesize_binary` no longer front-buffers a non-silent prefix before returning the streaming response
- the speech tokenizer decode is stateful/incremental (`IncrementalTokenizerDecoderV2`)
- Pipecat consumes the HTTP body chunk-by-chunk and forwards to Daily

Result: TTFC measured at the Pipecat client matches backend TTFC within transport noise. The pipeline is not buffering whole utterances anywhere.

### 4. Prefill scaffold caching + backend prewarm

Talker prefill still runs PyTorch (for correctness) but the request-independent speaker/language scaffold tensors are now cached per backend slot, and `QWEN3_TTS_PREWARM_BACKENDS=1` runs a synthetic warm request at server load time. First "real" request on a freshly-loaded server pays the cold compile cost during prewarm, not during the first user turn.

Measured: same-backend warm prefill drops from `235.5 ms` to `23.8 ms` between request 1 and request 2.

### 5. **The big one: `torch.compile` on the subtalker / code predictor**

This is what flipped RTF below `0.5`. The subtalker is a HF code-predictor model wrapped around a stack of decoder layers, and at `~60 ms/frame` it dominated wall time once everything else was streaming.

`torch.compile(self._subtalker_model, mode="default", dynamic=True)` produced a `~2.9× speedup` on the warm path (`1250 ms -> 432 ms` for `"ready"`).

Two non-obvious things made it work:

- **Attention backend swap:** the talker keeps `flash_attention_2`. The subtalker had to be switched to `sdpa` because FA2's branchy dispatch (different KV-cache strides per step) trips the dynamo recompile limit. Net win is still large.
- **Warmup at backend init:** the first request after backend init pays `~3-4 s` of compile latency on top of the `~8 s` startup compile warmup. `warm_prefill_scaffold` runs the compile warmup at server load so live traffic only sees steady-state cost.

Parity check passes after this change. This is the change I'd point to first if you ask me what made the RTF target hit.

### 6. Implemented but not shipped: audio decode overlap

I wired a producer/consumer overlap path between subtalker generation and tokenizer/audio decode (`QWEN3_TTS_AUDIO_DECODE_OVERLAP=1`). The implementation works — output is bit-identical to serial — but the benchmark showed it does **not** help on this hardware. Subtalker is `~60 ms/frame` and audio decode is `~12 ms/frame`, so even perfect overlap is bounded at `~5-6%` of wall, while the first-chunk drain adds a `~60-70 ms` TTFC penalty. So the default stays `0`.

This is a deliberate, measured non-win. It's documented in `README.md` and gated behind a flag so it's ready to reuse if subtalker cost drops further (e.g., after CUDA graphs).

### 7. Scheduler-level multi-request serving

The TTS service can admit up to 4 concurrent requests through scalar backend slots (`QWEN3_TTS_MAX_ACTIVE_REQUESTS=4`), with a small batching window and a bounded audio queue per request. This is **not** a batched CUDA megakernel — that is still future work — but it does let the service handle concurrent traffic without serializing at the HTTP layer, and it makes concurrent-load behavior measurable. The 32-request concurrent benchmarks in `README.md` are honest scheduler-level numbers.

## What This Cost (cold-start trade-offs)

- `~8 s` startup compile warmup when the backend loads (one-time per process)
- `~3-4 s` first-request compile penalty if `warm_prefill_scaffold` isn't run (we run it)
- Subtalker now uses SDPA, not FA2 — pure correctness preserved, but a small theoretical efficiency hit vs. a hand-rolled fused path

For a take-home demo where the server starts once and serves a Daily room, these are good trade-offs.

## Next Steps (if I had another week)

In priority order based on the current phase timings:

1. **CUDA graph capture for the subtalker step.** The compile path is great but still goes through dynamo for each step's dynamic shapes. A static-KV CUDA graph for the steady-state step could push subtalker to `~150-200 ms` for `"ready"`. This is the single biggest remaining win.
2. **Fully fused fp32 LM-head/token selection.** Right now `decode_hidden_fp32_head` runs the hidden-only kernel and then `at::mv` + `argmax`. Folding the matvec + argmax into one kernel removes one launch and the ATen dispatch. Small but free.
3. **Exact megakernel prefill (currently experimental).** `QWEN3_TTS_PREFILL_KERNEL=1` matched the first step's token but drifted on full utterance parity. With another pass to fix the drift, this would chop another `~80 ms` off the cold-prefill cost.
4. **Tokenizer/audio decode fusion.** The incremental decoder is correct and is no longer a top-3 cost, but it's still `~200 ms` summed across chunks. A custom decode kernel for the codec head would help.
5. **True batched CUDA talker decode.** This is the only honest path to good concurrent-load RTF. Current concurrent-4 RTF p50 is `2.64` because four slots are doing four independent forward passes. Batch-aware KV/cache layout + a batched decode entrypoint is a multi-day kernel job.
6. **Speculative/chunkwise multi-frame decode.** Probably the highest-ceiling change but also the riskiest because the talker is autoregressive on codec tokens. Worth prototyping after CUDA graphs.

## What I Would Not Spend More Time On

- More audio-decode-overlap tuning. The measurement is clear: subtalker dominates, overlap is `~5%` ceiling, and it costs TTFC. Reuse this path only after CUDA graphs land.
- Further prefill scaffold caching. Already at `~24 ms` warm; the next `~20 ms` is in PyTorch model forward, not in the scaffold.
- Adding a non-local TTS fallback. The whole point of the exercise is the local megakernel stack, and the warm-path numbers are good enough that falling back would just hide the real story.

## TL;DR

The result came from: (a) measuring phases honestly first, (b) keeping the original kernel structure and adding a hidden-state entrypoint, (c) streaming everything end-to-end, (d) caching prefill scaffolds and prewarming backend slots, and (e) `torch.compile` on the subtalker — which is what moved RTF from `0.77` to `0.29`. The remaining work is real but mostly understood: CUDA graphs on the subtalker, fp32-head fusion, and a true batched CUDA decode path for concurrent-load RTF.
