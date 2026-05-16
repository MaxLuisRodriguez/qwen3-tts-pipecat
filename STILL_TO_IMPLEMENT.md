# Still To Implement

This document captures the major performance-critical items that are still not implemented in the current Qwen3-TTS + Pipecat integration.

As of the 2026-05-16 validation pass, the repo installs and auto-selects Flash Attention 2 when available, records per-phase TTS timing, exposes scheduler-level scalar concurrency, prewarms static prefill scaffolds for backend slots, and `torch.compile`s the subtalker / code predictor (SDPA attention) at backend init. The remaining work below is still required before claiming a true batch-4 CUDA submission.

Measured warm-state direct-backend timing on `text=ready`, `voice=vivian`, `max_new_tokens=64` (compile on):

- wall_total: `0.555 s`
- TTFC: `53.3 ms`
- subtalker_ms (compiled): `432.3 ms` (was `1250.1 ms` pre-compile, ~2.9× speedup)
- talker custom-kernel decode: `~2 ms`
- speech tokenizer / audio decode: still meaningful (~`200 ms` summed across chunks)
- RTF: `0.289`

Subtalker is no longer the runaway bottleneck on the warm path. Remaining work targets concurrent-load behaviour (true batched CUDA decode) and the audio-decode / prefill paths.

## 1. Exact megakernel talker prefill

Talker prefill still uses the Hugging Face/PyTorch model path by default for correctness. The repo now caches request-independent speaker/language scaffold tensors and prewarms backend slots, but the actual request text prefill remains PyTorch.

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

Source: `kernel/csrc/torch_bindings.cpp:162`, `kernel/csrc/torch_bindings.cpp:175`, `kernel/csrc/torch_bindings.cpp:176`

Why it matters:
- This is materially weaker than the blog's more fully fused LM-head direction.
- It adds extra framework dispatch and device work outside the custom CUDA kernel.
- A true fused fp32-head kernel would reduce overhead in the talker loop.

## 3. Subtalker decode is `torch.compile`d, not a custom CUDA megakernel

Subtalker decode is now executed through a `torch.compile`d wrapper around the HF code-predictor model (`QWEN3_TTS_SUBTALKER_COMPILE=1`, default on). The compiled path uses SDPA attention (FA2 trips the dynamo recompile limit on shape-changing KV cache strides).

```python
# services/tts_qwen3/megakernel_talker.py
self._subtalker_model_compiled = torch.compile(
    self._subtalker_model, mode="default", dynamic=True
)
outputs = self._subtalker_model_compiled(
    input_ids=None,
    inputs_embeds=self._subtalker_projection(self._subtalker_prefill_buf),
    past_key_values=None,
    use_cache=True,
    ...
)
```

Why a custom CUDA path would still help:
- Compile gives `~2.9×` speedup but the subtalker still runs ~`30%` of warm-path wall time (`~432 ms` on `"ready"`).
- A true kernel-specialized path (CUDA graphs with a static KV cache, or a hand-rolled fused step) could push subtalker below `~150 ms`.
- The first request after backend init also pays a `~3-4 s` extra compile cost; a hand-rolled path avoids that.

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

## 7. Real overlap pipeline is wired in but does not currently help

A producer/consumer overlap path between talker generation and tokenizer/vocoder decode is wired into `services/tts_qwen3/megakernel_talker.py` behind `QWEN3_TTS_AUDIO_DECODE_OVERLAP=1`. It runs the incremental tokenizer decoder on a dedicated CUDA stream from a worker thread; output audio is bit-identical to the serial path.

Empirically, on the RTX 5090 with `flash_attention_2`, the overlap path does not improve TTFC, RTF, or max chunk gap meaningfully. Subtalker execution at ~60 ms / frame dominates wall time; audio decode contributes only ~12 ms / frame averaged, capping overlap gains at roughly 5-6% of total wall. The overlap path also adds a one-frame TTFC penalty because the first chunk cannot be yielded until the next iteration's drain.

Why it still matters:
- The overlap implementation is now ready to be reused if subtalker cost drops materially (e.g., after CUDA graph capture or speculative decode).
- The benchmark numbers in the README show the empirical ceiling and where new effort should go.

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
4. Exact megakernel or graph-captured talker prefill
5. Fully fused fp32 LM-head/token selection path
6. CUDA-graph or similar launch-overhead reduction for the TTS hot loop
7. True batched CUDA talker decode with batch-aware KV/cache layout
8. Speculative or chunkwise multi-frame decode
