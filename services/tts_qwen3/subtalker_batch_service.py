"""
Cross-slot batching service for the Qwen3-TTS subtalker (code predictor).

Background
----------
The 4-slot concurrent decode pipeline has each slot independently calling the
shared, ``torch.compile``-d subtalker once per code group (8 calls per frame).
All 4 worker threads serialize on the Python GIL while submitting these
forwards, so concurrent c4 throughput collapses to ~18% GPU utilization even
though the GPU has 4-5x headroom.

This service collapses all per-slot subtalker calls onto a single owner
thread, gathers them in a short batching window, runs one batched forward
across the active rows, and routes per-row outputs (and updated KV caches)
back to the submitting threads via per-call condition variables.

The service owns the compile wrapper, so the dynamo compile cache is shared
across slots. Without this, every backend's first call hits a cold compile
queue serially.

Design constraints
------------------
- Subtalker call shape is identical across slots within a frame group step
  (single-token decode after a 2-token prefill). The per-row KV cache lengths
  do match within a step because each slot independently runs the same
  sequence of 8 forwards per frame.
- HF ``DynamicCache`` is ``[B, H, S, D]`` per layer (see
  ``cache_utils.DynamicLayer``). We can build a fresh ``DynamicCache`` by
  assigning the concatenated K/V directly onto each layer.
- We only batch among rows whose sampling parameters are identical AND
  whose past-KV sequence lengths match exactly. In the steady state this
  ranges over (a) the 2-token prefill across the frame and (b) each
  subsequent 1-token continuation; ~all callers use greedy decoding so they
  fall into the same group.
- A "scalar" fallback runs the original per-row subtalker forward in the
  service thread for rows that can't be grouped together this iteration.
  This preserves parity for any edge case.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer


_DEFAULT_WINDOW_MS = float(os.getenv("QWEN3_TTS_SUBTALKER_BATCH_WINDOW_MS", "1.5"))
_DEFAULT_MAX_BATCH = int(os.getenv("QWEN3_TTS_SUBTALKER_BATCH_MAX_B", "8"))
# When the last served batch was a singleton, suppress the next batching
# wait. This avoids the ~window_ms latency tax on c1/single-slot traffic
# while still allowing lockstep concurrent traffic to coalesce naturally.
_ADAPTIVE_WINDOW = os.getenv("QWEN3_TTS_SUBTALKER_BATCH_ADAPTIVE_WINDOW", "1") == "1"
_DEFAULT_WARMUP_ENABLED = os.getenv("QWEN3_TTS_SUBTALKER_BATCH_WARMUP", "1") == "1"
_SERVICE_ENABLED = os.getenv("QWEN3_TTS_SUBTALKER_BATCH", "1") == "1"
_COMPILE_ENABLED = (
    os.getenv("QWEN3_TTS_SUBTALKER_COMPILE", "1") == "1"
    and torch.cuda.is_available()
)


def service_enabled() -> bool:
    return _SERVICE_ENABLED


@dataclass
class _SamplingKey:
    do_sample: bool
    top_p: float
    top_k: int
    temperature: float

    def __hash__(self) -> int:
        return hash((self.do_sample, round(self.top_p, 4), self.top_k, round(self.temperature, 4)))

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        if not isinstance(other, _SamplingKey):
            return False
        return (
            self.do_sample == other.do_sample
            and self.top_p == other.top_p
            and self.top_k == other.top_k
            and self.temperature == other.temperature
        )


@dataclass
class _PendingCall:
    """One slot's pending subtalker forward."""

    # Inputs (already projected through small_to_mtp_projection by caller).
    inputs_embeds: torch.Tensor  # [1, q, hidden_predictor]
    past_key_values: Optional[DynamicCache]
    sampling: _SamplingKey

    # Results (filled by service thread).
    last_hidden: Optional[torch.Tensor] = None  # [1, hidden_predictor]
    out_past_key_values: Optional[DynamicCache] = None
    error: Optional[BaseException] = None

    # Synchronization.
    done: threading.Event = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.done is None:
            self.done = threading.Event()


class SubtalkerBatchService:
    """Owner-thread batching wrapper around the subtalker forward."""

    def __init__(
        self,
        subtalker_model,
        *,
        window_ms: float = _DEFAULT_WINDOW_MS,
        max_batch: int = _DEFAULT_MAX_BATCH,
        compile_enabled: bool = _COMPILE_ENABLED,
    ):
        self._subtalker_model = subtalker_model
        self._window_s = max(0.0, float(window_ms) / 1000.0)
        self._max_batch = max(1, int(max_batch))
        self._compile_enabled = bool(compile_enabled)
        self._num_layers = int(subtalker_model.config.num_hidden_layers)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: list[_PendingCall] = []
        self._shutdown = False

        # Move the compile wrapper into the service so the dynamo cache is
        # shared across slots (each backend used to torch.compile its own,
        # which forced redundant warmups and contended the compile lock).
        #
        # We also bump the dynamo recompile limit because the cache call site
        # creates a fresh DynamicCache per call (so each invocation is a new
        # object identity for the past_key_values argument). Without the
        # bump, dynamo hits its default cap (8) within seconds of c4 traffic
        # and falls back to eager mode for the rest of the workload, which
        # destroys steady-state throughput. 256 is enough headroom for the
        # shape grid we actually traverse during decode.
        try:
            import torch._dynamo as _dynamo

            _dynamo.config.cache_size_limit = max(
                int(getattr(_dynamo.config, "cache_size_limit", 8)),
                int(os.getenv("QWEN3_TTS_DYNAMO_CACHE_SIZE_LIMIT", "256")),
            )
            # accumulated_cache_size_limit covers the per-instance code object
            # cache; without this dynamo also bails out on the same workload.
            _dynamo.config.accumulated_cache_size_limit = max(
                int(getattr(_dynamo.config, "accumulated_cache_size_limit", 64)),
                int(os.getenv("QWEN3_TTS_DYNAMO_ACCUM_CACHE_SIZE_LIMIT", "1024")),
            )
        except Exception:
            pass

        self._compiled_model = subtalker_model
        if self._compile_enabled:
            try:
                subtalker_model.config._attn_implementation = "sdpa"
                for layer in subtalker_model.layers:
                    setattr(layer.self_attn, "_attn_implementation", "sdpa")
                self._compiled_model = torch.compile(
                    subtalker_model,
                    mode="default",
                    dynamic=True,
                )
            except Exception:
                self._compile_enabled = False
                self._compiled_model = subtalker_model

        self._owner = threading.Thread(
            target=self._owner_loop,
            name="qwen3-tts-subtalker-batch",
            daemon=True,
        )
        self._owner.start()

        self._stats_lock = threading.Lock()
        self._stat_total_calls = 0
        self._stat_total_batches = 0
        self._stat_batched_calls = 0  # number of calls served as part of B>=2 batches
        self._stat_max_b = 0
        # Tracks if the last served batch had ≥2 rows. When False and adaptive
        # window is on, the next iteration won't wait for additional arrivals.
        self._last_batch_was_multi = False

        # Persistent merged-cache pool, keyed by batch size. dynamo specializes
        # `torch.compile`d code by `past_key_values` object identity (see the
        # ___check_obj_id guard). Constructing a fresh DynamicCache per batched
        # call invalidates that guard and forces a recompile per call, which
        # destroys steady-state throughput. By reusing one persistent
        # DynamicCache per batch size, we keep the cache object identity
        # stable across calls so dynamo only compiles each (B, q_len) once.
        self._merged_cache_pool: dict[int, DynamicCache] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return {
                "calls": self._stat_total_calls,
                "batches": self._stat_total_batches,
                "batched_calls": self._stat_batched_calls,
                "max_batch_observed": self._stat_max_b,
            }

    def shutdown(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()
        self._owner.join(timeout=5.0)

    def warmup(
        self,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        *,
        batch_sizes: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        """Drive synthetic forwards so the compiled batched path is hot.

        We run a full subtalker frame for each batch size, going through the
        same `_concat_caches`/`_split_cache` pool paths the live traffic uses,
        so the merged-cache pool's DynamicCache identity is the same one
        dynamo sees later. Without this, dynamo guards on a different
        ``past_key_values`` object identity and recompiles each batch on
        first contact with real traffic (10+ second steady-state cliff).

        We exercise q_len=2 (prefill) + the full 7 continuation steps (q_len=1
        with kv_len growing from 2 to 8) per batch size, which is the exact
        shape progression one frame of subtalker decode produces.
        """
        if not _DEFAULT_WARMUP_ENABLED:
            return
        num_subtalker_steps = 7  # num_code_groups - 1 for Qwen3-TTS 12Hz CV
        try:
            with torch.inference_mode():
                for batch_size in batch_sizes:
                    # 2-token prefill via the same pool path as live traffic.
                    prefill_embeds = torch.zeros(
                        (batch_size, 2, hidden_size), device=device, dtype=dtype
                    )
                    out = self._compiled_model(
                        input_ids=None,
                        inputs_embeds=prefill_embeds,
                        past_key_values=None,
                        use_cache=True,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                    per_row_caches = self._split_cache(out.past_key_values, batch_size)

                    # 6 continuation steps with kv_len growing from 2 to 7,
                    # each going through _concat_caches so the merged-cache
                    # pool identity is the one dynamo sees.
                    for _ in range(num_subtalker_steps - 1):
                        cont_embed = torch.zeros(
                            (batch_size, 1, hidden_size), device=device, dtype=dtype
                        )
                        merged = self._concat_caches(per_row_caches)
                        out = self._compiled_model(
                            input_ids=None,
                            inputs_embeds=cont_embed,
                            past_key_values=merged,
                            use_cache=True,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                        per_row_caches = self._split_cache(out.past_key_values, batch_size)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception:
            # Warmup must not block startup.
            pass

    def submit(
        self,
        *,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[DynamicCache],
        do_sample: bool,
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Submit a single-slot subtalker forward; block until result is ready.

        The caller passes inputs_embeds *after* small_to_mtp_projection has
        been applied (same as the existing per-backend code does), so this
        service can be model-agnostic about that projection.
        """
        call = _PendingCall(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            sampling=_SamplingKey(do_sample, float(top_p), int(top_k), float(temperature)),
        )
        with self._cond:
            self._pending.append(call)
            self._cond.notify()
        call.done.wait()
        if call.error is not None:
            raise call.error
        assert call.last_hidden is not None and call.out_past_key_values is not None
        return call.last_hidden, call.out_past_key_values

    # ------------------------------------------------------------------
    # Owner thread
    # ------------------------------------------------------------------

    def _owner_loop(self) -> None:
        # Run the entire owner loop in inference_mode so all tensors created
        # here (concatenated inputs_embeds, concatenated K/V) carry the same
        # dispatch keys as live caller tensors. Without this, dynamo recompiles
        # because owner-thread cat() produces ADInplaceOrView tensors while
        # warmup-thread inference_mode produces pure inference tensors.
        with torch.inference_mode():
            self._owner_loop_inner()

    def _owner_loop_inner(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._shutdown:
                    self._cond.wait()
                if self._shutdown and not self._pending:
                    return
                # Open a small batching window. Pick up extra arrivals so a
                # second slot's submission within the window can ride along.
                #
                # When _ADAPTIVE_WINDOW is on and the last served batch was a
                # singleton, skip the wait entirely on this iteration. The
                # next batched-traffic burst will refresh the multi flag on
                # its first ≥2 batch.
                if _ADAPTIVE_WINDOW and not self._last_batch_was_multi:
                    effective_window_s = 0.0
                else:
                    effective_window_s = self._window_s
                if effective_window_s > 0.0 and len(self._pending) < self._max_batch:
                    window_deadline = time.perf_counter() + effective_window_s
                    while (
                        len(self._pending) < self._max_batch
                        and time.perf_counter() < window_deadline
                    ):
                        remaining = window_deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self._cond.wait(timeout=remaining)
                batch = self._pending
                self._pending = []
            try:
                self._serve_batch(batch)
            except BaseException as exc:  # pragma: no cover - defensive
                for call in batch:
                    if not call.done.is_set():
                        call.error = exc
                        call.done.set()

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _serve_batch(self, batch: list[_PendingCall]) -> None:
        # Group by sampling params, query length, and per-row past KV length.
        # A batch row is compatible only if (sampling, q_len, kv_len) match.
        groups: dict[tuple, list[_PendingCall]] = {}
        for call in batch:
            q_len = int(call.inputs_embeds.shape[1])
            kv_len = (
                0 if call.past_key_values is None
                else int(call.past_key_values.get_seq_length())
            )
            key = (call.sampling, q_len, kv_len)
            groups.setdefault(key, []).append(call)

        max_group_size = 0
        for key, calls in groups.items():
            try:
                if len(calls) == 1:
                    self._serve_scalar(calls[0])
                else:
                    self._serve_grouped(calls)
                max_group_size = max(max_group_size, len(calls))
            except BaseException as exc:  # pragma: no cover - defensive
                for call in calls:
                    if not call.done.is_set():
                        call.error = exc
                        call.done.set()

        with self._stats_lock:
            self._stat_total_calls += len(batch)
        # Update adaptive-window hint.
        self._last_batch_was_multi = max_group_size >= 2

    def _serve_scalar(self, call: _PendingCall) -> None:
        # Single-row path. Route through the same pooled DynamicCache as the
        # B=1 entry of the merged-cache pool, so dynamo sees a stable
        # past_key_values object identity across all singleton calls
        # (otherwise each call presents a fresh DynamicCache from the caller
        # and triggers a recompile).
        with torch.inference_mode():
            if call.past_key_values is None:
                past_kv = None
            else:
                past_kv = self._concat_caches([call.past_key_values])
            outputs = self._compiled_model(
                input_ids=None,
                inputs_embeds=call.inputs_embeds,
                past_key_values=past_kv,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
        call.last_hidden = outputs.last_hidden_state[:, -1, :]
        # Always split back to a per-row identity so the *caller's* next
        # invocation goes through _concat_caches again rather than reusing
        # our pool object as if it were a private cache.
        call.out_past_key_values = self._split_cache(outputs.past_key_values, 1)[0]
        with self._stats_lock:
            self._stat_total_batches += 1
            self._stat_max_b = max(self._stat_max_b, 1)
        call.done.set()

    def _run_compiled(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[DynamicCache],
    ):
        return self._compiled_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        ).past_key_values

    def _serve_grouped(self, calls: list[_PendingCall]) -> None:
        # All calls in this group share sampling, q_len, and kv_len.
        batch_size = len(calls)
        # Build batched inputs_embeds.
        inputs_embeds = torch.cat([c.inputs_embeds for c in calls], dim=0)

        # Build batched past_key_values.
        batched_pkv = self._concat_caches([c.past_key_values for c in calls])

        with torch.inference_mode():
            outputs = self._compiled_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                past_key_values=batched_pkv,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )

        batched_last_hidden = outputs.last_hidden_state[:, -1, :]
        out_pkv = outputs.past_key_values

        # Split back per row.
        split_caches = self._split_cache(out_pkv, batch_size)
        for idx, call in enumerate(calls):
            call.last_hidden = batched_last_hidden[idx : idx + 1]
            call.out_past_key_values = split_caches[idx]

        with self._stats_lock:
            self._stat_total_batches += 1
            self._stat_batched_calls += batch_size
            self._stat_max_b = max(self._stat_max_b, batch_size)
        for call in calls:
            call.done.set()

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _concat_caches(self, caches: list[Optional[DynamicCache]]) -> Optional[DynamicCache]:
        if all(c is None for c in caches):
            return None
        # All non-None caches share kv_len thanks to the group key.
        non_none = [c for c in caches if c is not None]
        if len(non_none) != len(caches):
            raise RuntimeError(
                "SubtalkerBatchService grouped a mix of None and non-None caches"
            )
        batch_size = len(caches)
        merged = self._merged_cache_pool.get(batch_size)
        if merged is None:
            merged = DynamicCache()
            # Pre-populate placeholder layers so we can mutate-in-place from
            # the first call onward without growing the layers list.
            for _ in range(self._num_layers):
                layer = DynamicLayer()
                merged.layers.append(layer)
            self._merged_cache_pool[batch_size] = merged
        for layer_idx in range(self._num_layers):
            keys = torch.cat([c.layers[layer_idx].keys for c in caches], dim=0).contiguous()
            values = torch.cat([c.layers[layer_idx].values for c in caches], dim=0).contiguous()
            layer = merged.layers[layer_idx]
            if not layer.is_initialized:
                layer.lazy_initialization(keys)
            layer.keys = keys
            layer.values = values
        return merged

    def _split_cache(self, cache: DynamicCache, batch_size: int) -> list[DynamicCache]:
        outputs: list[DynamicCache] = [DynamicCache() for _ in range(batch_size)]
        for layer_idx in range(self._num_layers):
            k = cache.layers[layer_idx].keys
            v = cache.layers[layer_idx].values
            for row in range(batch_size):
                row_k = k[row : row + 1].contiguous()
                row_v = v[row : row + 1].contiguous()
                row_layer = DynamicLayer()
                row_layer.lazy_initialization(row_k)
                row_layer.keys = row_k
                row_layer.values = row_v
                outputs[row].layers.append(row_layer)
        return outputs
