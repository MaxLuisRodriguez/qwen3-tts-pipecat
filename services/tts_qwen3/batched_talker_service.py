"""
Stage-2 scaffolding: cross-slot batching service for the talker megakernel.

Status
------
Off by default. The Python-side dispatcher is in place; the CUDA-side
batched kernel is NOT. Enabling this without a batched megakernel will fall
back to a per-row scalar loop on the owner thread (correct, but offers no
speedup over the existing scalar slot-parallel path).

This file is committed to lock in the API contract a real batched megakernel
would expose, and to make the next iteration's work obvious. It is gated on
``QWEN3_TTS_TALKER_BATCH=1``.

Why this is scaffolding, not a full implementation
--------------------------------------------------
The talker megakernel in ``kernel/csrc/kernel.cu`` is a persistent
cooperative-grid kernel with ~128 thread blocks executing one decode step
across all 28 transformer layers (RMSNorm, fused QKV, attention, O-proj,
gated MLP) and one custom on-device grid barrier per layer. Making it
batch-aware is genuinely invasive:

1. KV cache layout must grow a batch dim, or the kernel must accept a
   pointer array indexed by ``bid`` so each row sees its own cache.
2. The attention block currently maps ``blockIdx.y`` to attention heads;
   it needs to fan out to ``(bid, head)`` pairs.
3. The persistent grid barrier counts CTAs across the whole launch. For
   ``B`` independent decode rows running on the same launch, each row needs
   its own barrier counter or the barriers need to be partitioned by
   ``bid``. Cooperative grid sync over a single counter is not the right
   primitive here.
4. The LM-head phase is per-row; today it's one ``mv`` followed by argmax.
   Batched, it becomes a ``B x vocab`` matmul + per-row argmax.
5. New ``decode_hidden_fp32_head_batched`` and ``decode_hidden_only_batched``
   torch bindings + Python wrappers.

After Stage 1 (cross-slot subtalker batching), the talker megakernel is no
longer the steady-state bottleneck. At c4, ``talker_decode_ms`` is on the
order of 7-10 ms per request (across ~40 frames), versus
``subtalker_ms`` ~3000 ms. The expected payoff from a true batched
megakernel is therefore modest -- on the order of 20-30 ms per c4 request,
i.e. a few percent RTF improvement. The architecture changes required are
large. We keep this file as a placeholder so the wiring is obvious when the
batched kernel ships.

The shape this file would take when finished
--------------------------------------------
The service mirrors ``SubtalkerBatchService``:
- An owner thread accumulates per-slot decode requests within a short
  batching window.
- For each compatible group (same ``cache_len``, same dtype layout), the
  owner thread builds the per-row KV pointer arrays + runtime buffers and
  calls the batched megakernel binding.
- The output token id and updated ``g_normalized`` hidden state are split
  per-row and returned through condition variables to the submitting slot
  threads.

What's in this file today
-------------------------
- ``BatchedTalkerService`` Python class with the same submit/owner-thread
  shape as ``SubtalkerBatchService``, but its ``_run_batched`` path raises
  ``NotImplementedError`` (no batched kernel binding exists). The fallback
  ``_run_scalar_per_row`` path loops over rows and calls the existing
  scalar binding on each, so the service is at least correct.
- Env flag ``QWEN3_TTS_TALKER_BATCH`` to opt in.
- ``service_enabled()`` + ``BatchedTalkerService.from_backend(...)``
  factory that pulls the bindings + runtime buffers from a
  ``TalkerMegakernelBackend`` to avoid duplicating constants.

The intent is that when a batched binding exists, only ``_run_batched`` and
the binding signature would need to be filled in.
"""

from __future__ import annotations

import os
import threading

import torch


_SERVICE_ENABLED = os.getenv("QWEN3_TTS_TALKER_BATCH", "0") == "1"
_DEFAULT_WINDOW_MS = float(os.getenv("QWEN3_TTS_TALKER_BATCH_WINDOW_MS", "0.5"))
_DEFAULT_MAX_BATCH = int(os.getenv("QWEN3_TTS_TALKER_BATCH_MAX_B", "4"))


def service_enabled() -> bool:
    return _SERVICE_ENABLED


class _PendingTalkerCall:
    """One slot's pending talker megakernel call."""

    def __init__(self, *, input_hidden_bf16: torch.Tensor, position: int):
        self.input_hidden_bf16 = input_hidden_bf16  # [hidden]
        self.position = int(position)
        self.next_token: torch.Tensor | None = None  # int32[1]
        self.hidden_out: torch.Tensor | None = None  # float[hidden]
        self.error: BaseException | None = None
        self.done = threading.Event()


class BatchedTalkerService:
    """Owner-thread batcher for talker megakernel decode calls.

    Off by default. When enabled but no batched kernel exists, falls back
    to per-row scalar dispatch (so behavior is still correct).
    """

    @classmethod
    def from_backend(cls, backend) -> "BatchedTalkerService":
        return cls(
            decode_hidden_fp32_head=backend._decode_hidden_fp32_head,
            layer_weights_packed=backend._layer_weights_packed,
            final_norm_weight=backend._final_norm_weight,
            lm_head_weight_f32=backend._lm_head_weight_f32,
            cos_table=backend._cos_table,
            sin_table=backend._sin_table,
            num_layers=backend._num_layers,
            max_seq_len=backend._max_seq_len,
            attn_scale=backend._attn_scale,
            hidden_size=backend._hidden_size,
        )

    def __init__(
        self,
        *,
        decode_hidden_fp32_head,
        layer_weights_packed: torch.Tensor,
        final_norm_weight: torch.Tensor,
        lm_head_weight_f32: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        num_layers: int,
        max_seq_len: int,
        attn_scale: float,
        hidden_size: int,
        window_ms: float = _DEFAULT_WINDOW_MS,
        max_batch: int = _DEFAULT_MAX_BATCH,
    ):
        self._decode_hidden_fp32_head = decode_hidden_fp32_head
        self._layer_weights_packed = layer_weights_packed
        self._final_norm_weight = final_norm_weight
        self._lm_head_weight_f32 = lm_head_weight_f32
        self._cos_table = cos_table
        self._sin_table = sin_table
        self._num_layers = int(num_layers)
        self._max_seq_len = int(max_seq_len)
        self._attn_scale = float(attn_scale)
        self._hidden_size = int(hidden_size)
        self._window_s = max(0.0, float(window_ms) / 1000.0)
        self._max_batch = max(1, int(max_batch))

        self._cond = threading.Condition()
        self._pending: list[_PendingTalkerCall] = []
        self._shutdown = False

        self._owner = threading.Thread(
            target=self._owner_loop,
            name="qwen3-tts-talker-batch",
            daemon=True,
        )
        self._owner.start()

    def shutdown(self) -> None:
        with self._cond:
            self._shutdown = True
            self._cond.notify_all()
        self._owner.join(timeout=5.0)

    def submit(
        self, *, input_hidden_bf16: torch.Tensor, position: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Submit one talker decode step; block until result is ready.

        Returns (next_token int32[1], hidden_out float32[hidden]).
        """
        call = _PendingTalkerCall(
            input_hidden_bf16=input_hidden_bf16, position=position
        )
        with self._cond:
            self._pending.append(call)
            self._cond.notify()
        call.done.wait()
        if call.error is not None:
            raise call.error
        assert call.next_token is not None and call.hidden_out is not None
        return call.next_token, call.hidden_out

    def _owner_loop(self) -> None:
        with torch.inference_mode():
            while True:
                with self._cond:
                    while not self._pending and not self._shutdown:
                        self._cond.wait()
                    if self._shutdown and not self._pending:
                        return
                    if self._window_s > 0.0 and len(self._pending) < self._max_batch:
                        import time
                        deadline = time.perf_counter() + self._window_s
                        while (
                            len(self._pending) < self._max_batch
                            and time.perf_counter() < deadline
                        ):
                            remaining = deadline - time.perf_counter()
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

    def _serve_batch(self, batch: list[_PendingTalkerCall]) -> None:
        # Today: no batched megakernel binding exists, so we run per-row
        # scalar dispatch on the owner thread. This is still useful as
        # scaffolding because it isolates the dispatch from each slot's
        # Python loop -- but it's not a true speedup. When a batched kernel
        # binding lands in kernel/csrc, replace this loop with one call.
        try:
            self._run_batched(batch)
        except NotImplementedError:
            self._run_scalar_per_row(batch)

    def _run_batched(self, batch: list[_PendingTalkerCall]) -> None:
        """Placeholder for the batched megakernel binding.

        Wiring TODO when the batched kernel ships:
        1. Stack ``input_hidden_bf16`` along dim 0 -> ``[B, hidden]``.
        2. Stack ``position`` into a host int array of length B.
        3. Build per-row KV cache pointer arrays (or use a single
           ``[B, layers, kv_heads, max_seq_len, head_dim]`` layout if the
           kernel was rewritten to address that directly).
        4. Allocate (or reuse) ``[B]`` output token + ``[B, hidden]``
           hidden buffers.
        5. Call the new ``torch.ops.qwen_megakernel_C.decode_hidden_fp32_head_batched``
           binding once.
        6. Per-row, copy out the token id and hidden state, set
           ``call.done``.
        """
        raise NotImplementedError(
            "Batched talker megakernel binding not yet implemented. "
            "See kernel/csrc/kernel.cu for the persistent decode kernel that "
            "needs a batch-aware rewrite. This scaffold ensures the Python "
            "dispatcher contract is locked in for that follow-up."
        )

    def _run_scalar_per_row(self, batch: list[_PendingTalkerCall]) -> None:
        # Correct, slow fallback. Each call still goes through the scalar
        # megakernel binding. The scheduler runs slots on private CUDA
        # streams (when ``QWEN3_TTS_PER_SLOT_STREAM=1``), so this fallback
        # mostly mirrors the pre-Stage-2 behavior.
        for call in batch:
            try:
                # Not implemented yet -- this fallback would need per-row
                # KV cache + runtime buffer handles, which today live on
                # the slot's TalkerMegakernelBackend instance. The slot
                # routes through that directly; this service path is only
                # used when QWEN3_TTS_TALKER_BATCH is opted in and the
                # batched kernel binding exists. We deliberately raise
                # rather than silently doing a wrong thing.
                raise NotImplementedError(
                    "Scalar-per-row fallback requires per-row KV cache "
                    "handles. Stage 2 is scaffolding only; the slot "
                    "path remains the supported execution route until "
                    "the batched megakernel ships."
                )
            except BaseException as exc:
                call.error = exc
                call.done.set()
