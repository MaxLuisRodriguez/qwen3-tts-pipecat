# Still To Implement

This document captures the major performance-critical items that are still not implemented in the current Qwen3-TTS + Pipecat integration.

As of the 2026-05-16 finalization pass, the repo:
- installs and auto-selects Flash Attention 2 when available,
- records per-phase TTS timing,
- exposes scheduler-level scalar concurrency with up to 4 active backend slots,
- prewarms static prefill scaffolds for backend slots,
- `torch.compile`s the subtalker / code predictor (SDPA attention) at backend init, and
- **batches subtalker forwards across all concurrent backend slots through a single `SubtalkerBatchService` owner thread** (default on).

Steady-state direct-backend timing on `text=ready`, `voice=vivian`, `max_new_tokens=64` (single stream):

- wall_total: `0.555 s`
- TTFC: `53.3 ms`
- subtalker_ms (compiled): `432.3 ms`
- talker custom-kernel decode: `~2 ms`
- speech tokenizer / audio decode: `~200 ms` summed across chunks
- RTF: `0.289`

Concurrent c4 numbers (32 req @ 4 req/s, anti-cheat caches off):

- RTF p50 / p90 / p99: `0.588 / 1.14 / 1.30`
- TTFC p50 / p90 / p99: `167 / 264 / 308 ms`
- GPU util avg / max: `97.1% / 100%`

Subtalker batching is the single biggest contributor to the c4 numbers. Remaining work targets the next bottlenecks (audio decode, prefill, and the long-tail of items where a true batched CUDA path would still win).

## 1. Exact megakernel talker prefill

Talker prefill still uses the Hugging Face/PyTorch model path by default for correctness. The repo caches request-independent speaker/language scaffold tensors and prewarms backend slots, but the actual request text prefill remains PyTorch.

```python
# services/tts_qwen3/megakernel_talker.py
prefill = self._talker.model(
    inputs_embeds=talker_input_embed,
    use_cache=True,
    return_dict=True,
)
```

Why it matters:
- This prefill path is still a TTFC cost, especially under load.
- The fast decode kernel only helps after prefill has completed.
- An experimental scalar megakernel prompt replay path (`QWEN3_TTS_PREFILL_KERNEL=1`) produced matching first prefill token but failed full utterance parity, so it remains off by default.
- A correct exact prefill kernel or graph-captured PyTorch prefill would still be useful.

## 2. `decode_hidden_fp32_head` is not GPU-fused

The `decode_hidden_fp32_head` path is not fully fused on GPU. It runs the hidden-state decode kernel first, then performs the LM head matvec and argmax via ATen.

```cpp
// kernel/csrc/torch_bindings.cpp
launch_ldg_decode_hidden_only_direct(...);

auto logits = at::mv(lm_head_weight_f32, normalized);
auto max_result = logits.max(0, false);
```

Why it matters:
- This is materially weaker than the AlpinDale blog's more fully fused LM-head direction.
- It adds extra framework dispatch and device work outside the custom CUDA kernel.
- A true fused fp32-head kernel would reduce overhead in the talker loop.

## 3. Subtalker decode is batched-Python, not a custom CUDA megakernel

Subtalker decode is now executed through a cross-slot batched `torch.compile`-wrapped HF code predictor (`services/tts_qwen3/subtalker_batch_service.py`). All four backend slots submit per-step forwards to a single owner thread that concatenates per-layer K/V along the batch dim, runs one batched forward, and splits the per-row outputs back. The compiled path uses SDPA attention (FA2 trips the dynamo recompile limit on shape-changing KV-cache strides).

```python
# services/tts_qwen3/megakernel_talker.py
if self._subtalker_service is not None:
    hidden, kv_cache = self._subtalker_service.submit(
        inputs_embeds=self._subtalker_projection(self._subtalker_prefill_buf),
        past_key_values=None,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
```

Why a custom CUDA path would still help:
- Cross-slot batching solved the GIL bottleneck (c4 RTF p50 `2.64 → 0.588`, GPU util `18.6% → 97.1%`). The remaining cost is GPU compute spent inside the compiled HF graph, which is well-optimized but still dispatched layer-by-layer via dynamo.
- A hand-rolled fused step (CUDA graphs with a static KV cache, or a true fused per-step kernel) could push subtalker_ms further down. At the current 97% GPU util, additional wins here improve audio quality headroom (more time for higher-fidelity sampling) rather than raw RTF.
- The first request after backend init also pays compile warmup for B=1..4 shape variants. Already mitigated by `service.warmup()` at server load.

## 4. Speech tokenizer waveform decode is still expensive

The speech tokenizer waveform decode has been made incremental, but it is still not fused or custom-kernelized.

```python
# services/tts_qwen3/megakernel_talker.py
class IncrementalTokenizerDecoderV2:
    ...
    def decode_new_frames(self, audio_codes: torch.Tensor) -> torch.Tensor:
        ...
```

Why it matters:
- After subtalker batching, audio decode is a larger fraction of remaining per-frame cost (`~12 ms/frame` averaged).
- The audio-decode-overlap path (`QWEN3_TTS_AUDIO_DECODE_OVERLAP=1`) is wired but off by default. Worth re-measuring now that subtalker per-frame cost is lower on c4.
- A custom decode kernel for the codec head would help further.

## 5. No CUDA graph capture for the TTS hot loop

The TTS generation path still relies on Python orchestration and repeated framework dispatch in the hot path. CUDA graph capture has not been added.

Relevant sources:
- `services/tts_qwen3/megakernel_talker.py`
- `services/tts_qwen3/subtalker_batch_service.py`
- `services/tts_qwen3/server.py`

Why it matters:
- The AlpinDale blog explicitly targets launch overhead.
- The batched subtalker service has reduced GIL overhead but each step still goes through dynamo specialization.
- Static-KV CUDA graphs for the steady-state subtalker step are the highest-payoff remaining single change.

## 6. No speculative or chunkwise talker decode

The system still advances frame-by-frame autoregressively.

```python
# services/tts_qwen3/megakernel_talker.py
for step in range(max_new_tokens):
    ...
```

Why it matters:
- There is no speculative, multi-frame, or chunkwise decode strategy yet.
- That limits how much TTFC and RTF can improve beyond single-step optimization.

## 7. Real overlap pipeline is wired in but does not currently help

A producer/consumer overlap path between talker generation and tokenizer/vocoder decode is wired into `services/tts_qwen3/megakernel_talker.py` behind `QWEN3_TTS_AUDIO_DECODE_OVERLAP=1`. It runs the incremental tokenizer decoder on a dedicated CUDA stream from a worker thread; output audio is bit-identical to the serial path.

Empirically, on the RTX 5090 with `flash_attention_2`, the overlap path does not improve TTFC, RTF, or max chunk gap meaningfully on the pre-Stage-1 path. Subtalker execution dominates and audio decode contributes only `~5-6%` of total wall, while the overlap pays a one-frame TTFC penalty.

Why it still matters:
- Now that subtalker per-frame cost is much lower on c4 (post Stage 1), the relative cost of audio decode has gone up. Re-running the overlap benchmark on the current build is worth doing.
- The implementation is correct and ready to reuse.

## 8. No true batched CUDA talker kernel yet (Stage 2 scaffolding committed)

The service can admit multiple TTS requests through scheduler-level scalar backend slots, and after Stage 1 the subtalker calls are batched across slots, but the **CUDA talker decode path** itself is still invoked per request.

A Python-side dispatcher (`services/tts_qwen3/batched_talker_service.py`) is committed off by default (`QWEN3_TTS_TALKER_BATCH=0`). Its `_run_batched` path raises `NotImplementedError` pending a batch-aware kernel. The file documents the five specific changes the CUDA side needs:

1. KV cache layout grows a batch dim, or the kernel accepts a pointer array indexed by `bid`.
2. The attention block fans out to `(bid, head)` pairs (`blockIdx.y` no longer maps to head alone).
3. The persistent grid-barrier counter is partitioned per `bid` (single global counter is not safe across independent decode rows).
4. The LM-head phase becomes a `B x vocab` matmul + per-row argmax instead of one `mv`.
5. New `decode_hidden_fp32_head_batched` and `decode_hidden_only_batched` torch bindings + Python wrappers.

Why it matters:
- After Stage 1, `talker_decode_ms` is `~7-10 ms` per c4 request vs `subtalker_ms ~3000 ms`. Expected payoff from a true batched megakernel is modest (`~20-30 ms` per c4 request, a few % RTF improvement).
- The architectural changes required are substantial. The scaffolding makes the wiring obvious so the next iteration's CUDA-side work is bounded.

## 9. Per-slot CUDA streams need event-based ordering

A `QWEN3_TTS_PER_SLOT_STREAM=1` knob gives each `TalkerMegakernelBackend` its own CUDA stream. It is off by default because naive enabling stalls concurrent c4 traffic: the `SubtalkerBatchService` owner thread submits forwards on a different stream from the slot's talker megakernel calls, and the cross-stream ordering is not made explicit.

To re-enable safely:
- Insert a `torch.cuda.Event` after subtalker outputs are split per row.
- Have each slot's talker megakernel `wait_event` on its row's event before reading the updated hidden.
- Drop the implicit default-stream synchronization in the slot loop.

## Priority Order

If optimizing for the next biggest wins, the likely order is:

1. **Static-KV CUDA graph capture for the subtalker step** — single biggest remaining win
2. **Re-measure audio-decode overlap on post-Stage-1 build** — possibly free win
3. **True batched CUDA talker megakernel** — bounded payoff but unblocks higher concurrency tiers
4. **Per-slot CUDA streams with event sync** — small win, scaffolding already in place
5. **Fully fused fp32 LM-head / token selection** — small but free
6. **Exact megakernel or graph-captured talker prefill** — TTFC cold-start improvement
7. **Custom-kernel speech tokenizer decode** — only after the above
8. **Speculative or chunkwise multi-frame decode** — highest ceiling, highest risk
