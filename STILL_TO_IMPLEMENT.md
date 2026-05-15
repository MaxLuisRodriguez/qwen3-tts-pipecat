# Still To Implement

This document captures the major performance-critical items that are still not implemented in the current Qwen3-TTS + Pipecat integration.

As of the 2026-05-15 validation pass, the repo installs and auto-selects Flash Attention 2 when available, records per-phase TTS timing, and exposes scheduler-level scalar concurrency. The remaining work below is still required before claiming a true batch-4 CUDA submission.

Measured short-turn timing with synchronized probes on `text=ready`, `voice=vivian`, `max_new_tokens=64`:

- prefill: `220.7 ms`
- subtalker/code predictor: `1259.6 ms`
- talker custom-kernel decode: `33.5 ms`
- speech tokenizer/audio decode: `281.9 ms`

The top measured bottleneck is currently the PyTorch subtalker/code predictor path, followed by speech tokenizer/audio decode and prefill. The custom talker decode kernel is being called, but it is no longer the dominant measured cost on this short prompt.

## 1. Megakernel talker prefill

Talker prefill is still done through the Hugging Face model path rather than a specialized megakernel prefill path.

```python
# services/tts_qwen3/megakernel_talker.py
prefill = self._talker.model(
    inputs_embeds=talker_input_embed,
    use_cache=True,
    return_dict=True,
)
```

Source: `services/tts_qwen3/megakernel_talker.py:711`

Why it matters:
- This prefill path is still a large TTFC cost.
- The fast decode kernel only helps after prefill has completed.
- A dedicated prefill path would likely be one of the highest-impact next optimizations.

## 2. `decode_hidden_fp32_head` is not GPU-fused

The `decode_hidden_fp32_head` path is not fully fused on GPU. It runs the hidden-state decode kernel first, then performs the LM head matvec and argmax via ATen.

```cpp
// kernel/csrc/torch_bindings.cpp
launch_ldg_decode_hidden_only_direct(...);

auto logits = at::mv(lm_head_weight_f32, normalized);
auto max_result = logits.max(0, false);
```

Source: `kernel/csrc/torch_bindings.cpp:162`, `kernel/csrc/torch_bindings.cpp:175`, `kernel/csrc/torch_bindings.cpp:176`

Why it matters:
- This is materially weaker than the blog's more fully fused LM-head direction.
- It adds extra framework dispatch and device work outside the custom CUDA kernel.
- A true fused fp32-head kernel would reduce overhead in the talker loop.

## 3. Subtalker decode is still mostly PyTorch execution

Subtalker decode no longer uses generic HF `generate()`, but it is still primarily executed through PyTorch model calls rather than a custom CUDA megakernel path.

```python
# services/tts_qwen3/megakernel_talker.py
outputs = self._subtalker_model(
    input_ids=None,
    inputs_embeds=self._subtalker_projection(self._subtalker_prefill_buf),
    past_key_values=None,
    use_cache=True,
    ...
)
```

Source: `services/tts_qwen3/megakernel_talker.py:610`

Why it matters:
- This remains part of the steady-state generation cost.
- The fixed-shape loop is better than HF `generate()`, but it is not yet kernel-specialized.

## 4. Speech tokenizer waveform decode is still expensive

The speech tokenizer waveform decode has been made incremental, but it is still not fused or custom-kernelized.

```python
# services/tts_qwen3/megakernel_talker.py
class IncrementalTokenizerDecoderV2:
    ...
    def decode_new_frames(self, audio_codes: torch.Tensor) -> torch.Tensor:
        ...
```

Source: `services/tts_qwen3/megakernel_talker.py:25`

Why it matters:
- This is still a major runtime component.
- Incremental decode helps TTFC and avoids full re-decode, but it does not eliminate framework overhead.

## 5. No CUDA graph capture for the TTS hot loop

The TTS generation path still relies on Python orchestration and repeated framework dispatch in the hot path. CUDA graph capture has not been added.

Relevant sources:
- `services/tts_qwen3/megakernel_talker.py`
- `services/tts_qwen3/server.py`

Why it matters:
- The AlpinDale blog explicitly targets launch overhead.
- CUDA graphs could reduce per-step CPU and launch overhead in the talker path.

## 6. No speculative or chunkwise talker decode

The system still advances frame-by-frame autoregressively.

```python
# services/tts_qwen3/megakernel_talker.py
for step in range(max_new_tokens):
    ...
```

Source: `services/tts_qwen3/megakernel_talker.py:749`

Why it matters:
- There is no speculative, multi-frame, or chunkwise decode strategy yet.
- That limits how much TTFC and RTF can improve beyond single-step optimization.

## 7. No true overlap pipeline between talker generation and audio decode

There is cadence control for when audio decode runs, but not a real producer/consumer multi-stream overlap design between talker generation and tokenizer/vocoder decode.

```python
# services/tts_qwen3/megakernel_talker.py
if adaptive_decode_cadence:
    ...
if should_decode:
    ...
```

Source: `services/tts_qwen3/megakernel_talker.py:838`, `services/tts_qwen3/megakernel_talker.py:848`

Why it matters:
- Cadence tuning helps, but it is not the same as overlapping independent work on separate streams.
- A true overlap pipeline could improve steady-state RTF.

## 8. No true batched CUDA talker kernel yet

The service can admit multiple TTS requests through scheduler-level scalar backend slots, but the CUDA talker decode path itself is still invoked per request. This is useful for measurement and request lifecycle work, but it should not be reported as true batched megakernel execution.

Why it matters:
- Batch-4 serving can now be benchmarked honestly at the scheduler/API level.
- The kernel still needs batch-aware KV/cache layout and batched decode entrypoints before it is a true batch-4 CUDA implementation.

## Priority Order

If optimizing for the next biggest wins, the likely order is:

1. Subtalker/code predictor kernel specialization or graph capture
2. Better overlap between talker generation and tokenizer/audio decode
3. Speech tokenizer custom-kernel acceleration
4. Megakernel or otherwise specialized talker prefill
5. Fully fused fp32 LM-head/token selection path
6. CUDA-graph or similar launch-overhead reduction for the TTS hot loop
7. True batched CUDA talker decode with batch-aware KV/cache layout
8. Speculative or chunkwise multi-frame decode
