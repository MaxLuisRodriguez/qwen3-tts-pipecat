"""Streaming TTS server backed by Qwen3-TTS with talker megakernel decode."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Iterator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

_KERNEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../kernel"))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from megakernel_talker import TalkerMegakernelBackend
from subtalker_batch_service import SubtalkerBatchService, service_enabled as _subtalker_batch_enabled

app = FastAPI(title="Qwen3 TTS Service")
LOGGER = logging.getLogger(__name__)
_LOG_LEVEL = os.getenv("QWEN3_TTS_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

SAMPLE_RATE = int(os.getenv("QWEN3_TTS_SAMPLE_RATE", "24000"))
CHUNK_SIZE = int(os.getenv("QWEN3_TTS_CHUNK_SIZE", "480"))
SAMPLE_WIDTH = 2
BYTES_PER_CHUNK = CHUNK_SIZE * SAMPLE_WIDTH
DECODE_STRIDE = int(os.getenv("QWEN3_TTS_DECODE_STRIDE", "1"))
_QUEUE_SENTINEL = object()


def _decode_stride_header_value() -> str:
    if os.getenv("QWEN3_TTS_ADAPTIVE_DECODE_CADENCE", "1") == "1":
        mid = max(1, int(os.getenv("QWEN3_TTS_DECODE_STRIDE_MID", "4")))
        late = max(1, int(os.getenv("QWEN3_TTS_DECODE_STRIDE_LATE", "8")))
        late_start = int(os.getenv("QWEN3_TTS_DECODE_STRIDE_LATE_START_FRAME", "24"))
        left_context = max(0, int(os.getenv("QWEN3_TTS_INCREMENTAL_LEFT_CONTEXT_FRAMES", "12")))
        return f"adaptive(mid={mid},late={late}@{late_start},ctx={left_context})"
    return str(DECODE_STRIDE)


def _default_attn_implementation() -> str:
    configured = os.getenv("QWEN3_TTS_ATTN_IMPL")
    if configured:
        return configured
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


class TTSRequest(BaseModel):
    """Request model for text-to-speech."""

    text: str = Field(min_length=1, description="Input text to synthesize.")
    voice: str | None = Field(
        default=None, description="Optional speaker label (e.g. Cherry)."
    )
    max_new_tokens: int = Field(
        default=1024, ge=32, le=8192, description="Max decode tokens for TTS generation."
    )


@dataclass
class ScheduledTTSRequest:
    """A queued TTS request with its own streaming output queue."""

    request_id: str
    text: str
    speaker: str
    language: str
    max_new_tokens: int
    stats: object
    output_queue: "queue.Queue[object]"
    cancelled: threading.Event = field(default_factory=threading.Event)
    submitted_at: float = field(default_factory=time.perf_counter)
    admitted_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    first_audio_at: float | None = None
    error: BaseException | None = None
    batch_admit_size: int = 1


class ScheduledAudioIterator:
    """Blocking iterator over a scheduled request's audio chunks."""

    def __init__(self, state: ScheduledTTSRequest):
        self._state = state
        self._closed = False

    def __iter__(self) -> "ScheduledAudioIterator":
        return self

    def __next__(self) -> np.ndarray:
        if self._closed:
            raise StopIteration
        item = self._state.output_queue.get()
        if item is _QUEUE_SENTINEL:
            self._closed = True
            raise StopIteration
        if isinstance(item, BaseException):
            self._closed = True
            raise item
        return item

    def close(self) -> None:
        self._closed = True
        self._state.cancelled.set()


class TTSRequestScheduler:
    """
    Scheduler-level batch admission for Qwen3-TTS requests.

    This intentionally does not claim true CUDA kernel batching. Each admitted
    request runs on an independent scalar backend slot until the CUDA path is
    made batch-aware.
    """

    def __init__(self, engine: "Qwen3TTSEngine"):
        self._engine = engine
        self._max_active = max(1, int(os.getenv("QWEN3_TTS_MAX_ACTIVE_REQUESTS", "4")))
        self._batch_window_s = max(
            0.0, float(os.getenv("QWEN3_TTS_BATCH_WINDOW_MS", "6.0")) / 1000.0
        )
        self._max_prefill_wait_s = max(
            0.0, float(os.getenv("QWEN3_TTS_MAX_PREFILL_WAIT_MS", "100.0")) / 1000.0
        )
        self._queue_maxsize = max(1, int(os.getenv("QWEN3_TTS_AUDIO_QUEUE_MAX_CHUNKS", "16")))
        self._pending: "queue.Queue[ScheduledTTSRequest]" = queue.Queue()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_active,
            thread_name_prefix="qwen3-tts-slot",
        )
        self._lock = threading.Lock()
        self._active = 0
        self._shutdown = threading.Event()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="qwen3-tts-scheduler",
            daemon=True,
        )
        self._dispatcher.start()

    @property
    def mode(self) -> str:
        return "scheduler_scalar"

    @property
    def max_active(self) -> int:
        return self._max_active

    def submit(
        self,
        *,
        text: str,
        speaker: str,
        language: str,
        max_new_tokens: int,
    ) -> tuple[object, Iterator[np.ndarray]]:
        from megakernel_talker import StreamStats

        state = ScheduledTTSRequest(
            request_id=uuid.uuid4().hex,
            text=text,
            speaker=speaker,
            language=language,
            max_new_tokens=max_new_tokens,
            stats=StreamStats(),
            output_queue=queue.Queue(maxsize=self._queue_maxsize),
        )
        self._pending.put(state)
        return state.stats, ScheduledAudioIterator(state)

    def _available_slots(self) -> int:
        with self._lock:
            return max(0, self._max_active - self._active)

    def _mark_active(self, count: int) -> None:
        with self._lock:
            self._active += count

    def _mark_finished(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def _dispatch_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                first = self._pending.get(timeout=0.05)
            except queue.Empty:
                continue

            batch = [first]
            deadline = min(
                first.submitted_at + self._max_prefill_wait_s,
                time.perf_counter() + self._batch_window_s,
            )
            while len(batch) < self._max_active:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._pending.get(timeout=remaining))
                except queue.Empty:
                    break

            for state in batch:
                while not self._shutdown.is_set() and self._available_slots() <= 0:
                    time.sleep(0.002)
                state.admitted_at = time.perf_counter()
                state.batch_admit_size = len(batch)
                self._mark_active(1)
                self._executor.submit(self._run_state, state)

    def _put_or_cancel(self, state: ScheduledTTSRequest, item: object) -> bool:
        while not state.cancelled.is_set():
            try:
                state.output_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run_state(self, state: ScheduledTTSRequest) -> None:
        backend = None
        raw_iter: Iterator[np.ndarray] | None = None
        try:
            state.started_at = time.perf_counter()
            backend = self._engine.checkout_backend()
            raw_stats, raw_iter = backend.stream_audio(
                text=state.text,
                speaker=state.speaker,
                language=state.language,
                max_new_tokens=state.max_new_tokens,
                decode_stride=DECODE_STRIDE,
            )
            for audio in raw_iter:
                if state.cancelled.is_set():
                    break
                if audio.size > 0 and state.first_audio_at is None:
                    state.first_audio_at = time.perf_counter()
                    raw_stats.ttfc_ms = (state.first_audio_at - state.submitted_at) * 1000.0
                    setattr(state.stats, "ttfc_ms", raw_stats.ttfc_ms)
                    setattr(state.stats, "scheduler_mode", self.mode)
                    setattr(state.stats, "batch_admit_size", state.batch_admit_size)
                if not self._put_or_cancel(state, audio):
                    break
            for attr in (
                "ttfc_ms",
                "generation_s",
                "audio_seconds",
                "frames_generated",
                "stop_reason",
                "prefill_ms",
                "prompt_build_ms",
                "prefill_model_ms",
                "prefill_cache_ms",
                "prefill_mode",
                "subtalker_ms",
                "talker_decode_ms",
                "audio_decode_ms",
                "subtalker_calls",
                "talker_decode_calls",
                "audio_decode_calls",
                "audio_chunks",
                "first_decode_ms",
                "kernel_path",
                "timing_mode",
                "audio_decode_overlap",
                "audio_decode_wait_ms",
                "subtalker_compile",
            ):
                if hasattr(raw_stats, attr):
                    setattr(state.stats, attr, getattr(raw_stats, attr))
            setattr(state.stats, "scheduler_mode", self.mode)
            setattr(state.stats, "batch_admit_size", state.batch_admit_size)
            setattr(state.stats, "queue_wait_ms", ((state.started_at or state.submitted_at) - state.submitted_at) * 1000.0)
            LOGGER.info(
                "TTS request completed id=%s mode=%s batch_admit=%s ttfc_ms=%s "
                "generation_s=%.3f audio_s=%.3f frames=%s chunks=%s "
                "prefill_mode=%s prefill_ms=%.2f prompt_ms=%.2f prefill_model_ms=%.2f "
                "prefill_cache_ms=%.2f subtalker_ms=%.2f talker_decode_ms=%.2f audio_decode_ms=%.2f "
                "audio_decode_overlap=%s subtalker_compile=%s",
                state.request_id,
                self.mode,
                state.batch_admit_size,
                getattr(state.stats, "ttfc_ms", None),
                float(getattr(state.stats, "generation_s", 0.0) or 0.0),
                float(getattr(state.stats, "audio_seconds", 0.0) or 0.0),
                getattr(state.stats, "frames_generated", None),
                getattr(state.stats, "audio_chunks", None),
                getattr(state.stats, "prefill_mode", "unknown"),
                float(getattr(state.stats, "prefill_ms", 0.0) or 0.0),
                float(getattr(state.stats, "prompt_build_ms", 0.0) or 0.0),
                float(getattr(state.stats, "prefill_model_ms", 0.0) or 0.0),
                float(getattr(state.stats, "prefill_cache_ms", 0.0) or 0.0),
                float(getattr(state.stats, "subtalker_ms", 0.0) or 0.0),
                float(getattr(state.stats, "talker_decode_ms", 0.0) or 0.0),
                float(getattr(state.stats, "audio_decode_ms", 0.0) or 0.0),
                bool(getattr(state.stats, "audio_decode_overlap", False)),
                bool(getattr(state.stats, "subtalker_compile", False)),
            )
        except BaseException as exc:
            state.error = exc
            self._put_or_cancel(state, exc)
        finally:
            state.finished_at = time.perf_counter()
            if raw_iter is not None:
                close = getattr(raw_iter, "close", None)
                if callable(close):
                    close()
            if backend is not None:
                self._engine.return_backend(backend)
            self._put_or_cancel(state, _QUEUE_SENTINEL)
            self._mark_finished()


class Qwen3TTSEngine:
    """Lazy-loaded Qwen3-TTS + talker megakernel backend."""

    def __init__(self):
        self.model_name = os.getenv(
            "QWEN3_TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )
        self.default_voice = os.getenv("QWEN3_TTS_DEFAULT_VOICE", "vivian")
        self.default_language = os.getenv("QWEN3_TTS_LANGUAGE", "english").strip().lower()
        self.attn_implementation = _default_attn_implementation()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._load_lock = threading.Lock()
        self._backend_lock = threading.Lock()
        self._backend_pool: "queue.LifoQueue[TalkerMegakernelBackend]" = queue.LifoQueue()
        self._backend_count = 0
        self._prewarm_backends = os.getenv("QWEN3_TTS_PREWARM_BACKENDS", "1") == "1"
        self._generate_lock = threading.Lock()
        self._generate_lock_timeout_s = float(
            os.getenv("QWEN3_TTS_GENERATE_LOCK_TIMEOUT_S", "180")
        )
        # Rebuilding backend per request avoids stale internal state carrying
        # across turns, which can otherwise cause first-turn-only audio.
        self._rebuild_backend_per_request = os.getenv(
            "QWEN3_TTS_REBUILD_BACKEND_PER_REQUEST", "0"
        ) == "1"
        self._model = None
        self._backend: TalkerMegakernelBackend | None = None
        self._subtalker_service: SubtalkerBatchService | None = None
        self._scheduler = TTSRequestScheduler(self)

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._backend_count > 0

    def load(self):
        if self.loaded:
            return
        with self._load_lock:
            if self.loaded:
                return
            self._model = Qwen3TTSModel.from_pretrained(
                self.model_name,
                device_map="cuda:0" if self._device.type == "cuda" else "cpu",
                dtype=self._dtype,
                attn_implementation=self.attn_implementation,
            )
            self._model.model.eval()
            if self._device.type != "cuda":
                self._model.model.to(self._device)

            # Build one shared SubtalkerBatchService that all backend slots
            # route their subtalker forwards through. This is the Stage 1
            # batching optimization: per-slot Python orchestration is
            # decoupled from GPU forwards, and the compile cache is shared.
            if _subtalker_batch_enabled():
                subtalker = self._model.model.talker.code_predictor.model
                self._subtalker_service = SubtalkerBatchService(subtalker)
                talker_hidden = int(self._model.model.talker.config.hidden_size)
                subtalker_hidden = int(subtalker.config.hidden_size)
                # The service expects inputs already projected through
                # small_to_mtp_projection (which maps talker_hidden ->
                # subtalker_hidden), so warmup tensors use subtalker_hidden.
                self._subtalker_service.warmup(
                    hidden_size=subtalker_hidden,
                    device=self._model.model.talker.device,
                    dtype=self._model.model.talker.dtype,
                    batch_sizes=(1, 2, 4),
                )
                LOGGER.info(
                    "Subtalker batching service initialized "
                    "(compile=%s talker_hidden=%d subtalker_hidden=%d)",
                    self._subtalker_service.compile_enabled,
                    talker_hidden,
                    subtalker_hidden,
                )

            target_backends = self._scheduler.max_active if self._prewarm_backends else 1
            for idx in range(target_backends):
                backend = TalkerMegakernelBackend(
                    self._model,
                    subtalker_service=self._subtalker_service,
                )
                backend.warm_prefill_scaffold(self.default_voice, self.default_language)
                if idx == 0:
                    self._backend = backend
                self._backend_pool.put(backend)
                self._backend_count += 1
            LOGGER.info(
                "Loaded Qwen3-TTS with %d backend slot(s), prewarm_backends=%s, prefill_optimized=%s, subtalker_batch=%s",
                self._backend_count,
                self._prewarm_backends,
                os.getenv("QWEN3_TTS_PREFILL_OPTIMIZED", "1") == "1",
                self._subtalker_service is not None,
            )

    def checkout_backend(self) -> TalkerMegakernelBackend:
        self.load()
        if self._model is None:
            raise RuntimeError("Qwen3-TTS model is not loaded.")
        if self._rebuild_backend_per_request:
            return TalkerMegakernelBackend(self._model, subtalker_service=self._subtalker_service)
        try:
            return self._backend_pool.get_nowait()
        except queue.Empty:
            pass
        with self._backend_lock:
            if self._backend_count < self._scheduler.max_active:
                self._backend_count += 1
                try:
                    backend = TalkerMegakernelBackend(
                        self._model, subtalker_service=self._subtalker_service
                    )
                    backend.warm_prefill_scaffold(self.default_voice, self.default_language)
                    return backend
                except Exception:
                    self._backend_count -= 1
                    raise
        return self._backend_pool.get()

    def return_backend(self, backend: TalkerMegakernelBackend) -> None:
        if self._rebuild_backend_per_request:
            return
        self._backend_pool.put(backend)

    def stream_synthesize(
        self, text: str, voice: str | None, max_new_tokens: int
    ) -> tuple[object, Iterator[np.ndarray], int]:
        self.load()
        speaker = (voice or self.default_voice).strip() or self.default_voice
        effective_max_new_tokens = _estimate_max_new_tokens(text, max_new_tokens)
        stats, audio_iter = self._scheduler.submit(
            text=text,
            speaker=speaker,
            language=self.default_language,
            max_new_tokens=effective_max_new_tokens,
        )
        return stats, audio_iter, effective_max_new_tokens

    def synthesize_fallback(
        self, text: str, voice: str | None, max_new_tokens: int
    ) -> tuple[np.ndarray, int]:
        """
        Reliability fallback: non-megakernel Qwen3-TTS waveform generation.
        Used only when the talker streaming path returns no audio for a request.
        """
        self.load()
        if self._model is None:
            raise RuntimeError("Qwen3-TTS model is not loaded.")

        speaker = (voice or self.default_voice).strip() or self.default_voice
        effective_max_new_tokens = _estimate_max_new_tokens(text, max_new_tokens)
        acquired = self._generate_lock.acquire(timeout=self._generate_lock_timeout_s)
        if not acquired:
            LOGGER.warning(
                "TTS fallback lock wait exceeded %.1fs; waiting for release.",
                self._generate_lock_timeout_s,
            )
            self._generate_lock.acquire()
        try:
            with torch.inference_mode():
                wavs, _sr = self._model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    language=self.default_language,
                    max_new_tokens=effective_max_new_tokens,
                )
        finally:
            self._generate_lock.release()

        if not wavs:
            raise RuntimeError("Fallback TTS produced no waveform.")
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Fallback TTS produced empty waveform.")
        return audio, effective_max_new_tokens


engine = Qwen3TTSEngine()


def _warmup_tts_engine() -> None:
    if os.getenv("TTS_WARMUP_ON_STARTUP", "0") != "1":
        return
    warmup_text = os.getenv("TTS_WARMUP_TEXT", "I am here.").strip()
    if not warmup_text:
        return

    audio_iter: Iterator[np.ndarray] | None = None
    try:
        stats, audio_iter, _ = engine.stream_synthesize(
            text=warmup_text,
            voice=engine.default_voice,
            max_new_tokens=128,
        )
        first_audio = next(audio_iter, None)
        LOGGER.info(
            "TTS warmup completed: text=%r ttfc_ms=%s first_audio_samples=%s",
            warmup_text,
            getattr(stats, "ttfc_ms", None),
            0 if first_audio is None else int(first_audio.size),
        )
    except Exception:
        LOGGER.exception("TTS warmup failed")
    finally:
        if audio_iter is not None:
            _close_iter_safely(audio_iter)


def _iter_pcm_chunks_from_audio_stream(audio_iter: Iterator[np.ndarray]) -> Iterator[bytes]:
    tail = np.empty((0,), dtype=np.int16)
    try:
        for audio in audio_iter:
            if audio.size == 0:
                continue
            clipped = np.clip(audio, -1.0, 1.0)
            pcm = (clipped * 32767.0).astype(np.int16, copy=False)
            if tail.size:
                pcm = np.concatenate((tail, pcm))
                tail = np.empty((0,), dtype=np.int16)
            start = 0
            while start + CHUNK_SIZE <= pcm.shape[0]:
                yield pcm[start : start + CHUNK_SIZE].tobytes()
                start += CHUNK_SIZE
            if start < pcm.shape[0]:
                tail = pcm[start:]
        if tail.size:
            yield tail.tobytes()
    finally:
        close = getattr(audio_iter, "close", None)
        if callable(close):
            close()


def _merged_audio_iter(
    first_audio: np.ndarray,
    audio_iter: Iterator[np.ndarray],
) -> Iterator[np.ndarray]:
    try:
        yield first_audio
        yield from audio_iter
    finally:
        close = getattr(audio_iter, "close", None)
        if callable(close):
            close()


def _close_iter_safely(audio_iter: Iterator[np.ndarray]) -> None:
    close = getattr(audio_iter, "close", None)
    if callable(close):
        close()


def _audio_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32, copy=False)))))


def _is_effectively_silent(audio: np.ndarray, rms_threshold: float) -> bool:
    if audio.size == 0:
        return True
    rms = _audio_rms(audio)
    peak = float(np.max(np.abs(audio)))
    return rms < rms_threshold and peak < (rms_threshold * 3.0)


def _peek_audio_prefix(
    audio_iter: Iterator[np.ndarray],
    *,
    max_chunks: int,
    max_prefix_samples: int,
    rms_threshold: float,
) -> tuple[list[np.ndarray], bool]:
    prefix: list[np.ndarray] = []
    has_non_silent = False
    total_samples = 0
    min_chunks = max(1, max_chunks)
    hard_max_samples = max(0, max_prefix_samples)
    while True:
        if len(prefix) >= min_chunks and (has_non_silent or (hard_max_samples > 0 and total_samples >= hard_max_samples)):
            break
        if hard_max_samples == 0 and len(prefix) >= min_chunks:
            break
        chunk = next(audio_iter, None)
        if chunk is None:
            break
        if chunk.size == 0:
            continue
        prefix.append(chunk)
        total_samples += int(chunk.shape[0])
        if not _is_effectively_silent(chunk, rms_threshold):
            has_non_silent = True
    return prefix, has_non_silent


def _trim_leading_silence(audio: np.ndarray, rms_threshold: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    amplitude_threshold = max(float(rms_threshold) * 2.5, 0.0025)
    voiced = np.flatnonzero(np.abs(audio.astype(np.float32, copy=False)) >= amplitude_threshold)
    if voiced.size == 0:
        return audio
    start = int(voiced[0])
    if start <= 0:
        return audio
    return audio[start:].copy()


def _prepend_audio_iter(
    prefix: list[np.ndarray],
    audio_iter: Iterator[np.ndarray],
) -> Iterator[np.ndarray]:
    try:
        for chunk in prefix:
            yield chunk
        yield from audio_iter
    finally:
        close = getattr(audio_iter, "close", None)
        if callable(close):
            close()


def _speech_stabilization_candidates(text: str) -> list[str]:
    normalized = " ".join(text.split()).strip()
    if not normalized:
        return [text]

    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = " ".join(candidate.split()).strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if os.getenv("QWEN3_TTS_ENABLE_SPEECH_STABILIZATION", "1") != "1":
        add(normalized)
        return candidates

    lowered = normalized.lower()
    risky_short_copular, risky_yes_no_confirmation = _speech_text_risk_flags(lowered)

    if risky_short_copular or risky_yes_no_confirmation:
        prioritized: list[str] = []
        if risky_yes_no_confirmation:
            prioritized.append(f"well {normalized}")
            prioritized.append(f"okay {normalized}")
            prioritized.append(normalized)
            prioritized.append(f"here is my reply {normalized}")
        elif lowered.startswith(("that is ", "this is ")):
            prioritized.append(f"here is my reply {normalized}")
            prioritized.append(normalized)
            prioritized.append(f"okay {normalized}")
            prioritized.append(f"well {normalized}")
        else:
            prioritized.append(f"okay {normalized}")
            prioritized.append(f"well {normalized}")
            prioritized.append(normalized)
            prioritized.append(f"here is my reply {normalized}")
        for candidate in prioritized:
            add(candidate)
        return candidates

    add(normalized)
    if not lowered.startswith("okay "):
        add(f"okay {normalized}")
    if not lowered.startswith("well "):
        add(f"well {normalized}")
    if not lowered.startswith("here is my reply "):
        add(f"here is my reply {normalized}")
    return candidates


def _speech_text_risk_flags(lowered_text: str) -> tuple[bool, bool]:
    tokens = lowered_text.split()
    risky_short_copular = (
        len(tokens) <= 5
        and len(tokens) >= 3
        and tokens[0] in {"it", "that", "this", "you", "he", "she", "we", "they"}
        and tokens[1] in {"is", "are", "re"}
    )
    risky_yes_no_confirmation = (
        len(tokens) <= 6
        and len(tokens) >= 4
        and tokens[0] in {"yes", "no"}
        and tokens[1] in {"it", "that", "this", "you", "he", "she", "we", "they"}
        and tokens[2] in {"is", "are", "re"}
    )
    return risky_short_copular, risky_yes_no_confirmation


def _prefers_strict_prefix_probe(text: str) -> bool:
    lowered = " ".join(text.split()).strip().lower()
    if not lowered:
        return False
    risky_short_copular, risky_yes_no_confirmation = _speech_text_risk_flags(lowered)
    if risky_short_copular or risky_yes_no_confirmation:
        return True
    token_count = len(lowered.split())
    return token_count <= 4


def _estimate_max_new_tokens(text: str, requested: int) -> int:
    if os.getenv("QWEN3_TTS_DYNAMIC_MAX_NEW_TOKENS", "1") != "1":
        return requested

    text = text.strip()
    char_count = len(text)
    punctuation_count = sum(text.count(ch) for ch in ".!?;,")
    sentence_boundary_count = sum(text.count(ch) for ch in ".!?")
    clause_separator_count = text.count(",") + text.count(";") + text.count(":")
    token_base = int(os.getenv("QWEN3_TTS_TOKEN_BASE", "32"))
    tokens_per_char = float(os.getenv("QWEN3_TTS_TOKENS_PER_CHAR", "1.25"))
    punctuation_bonus = int(os.getenv("QWEN3_TTS_PUNCT_BONUS", "2"))
    min_dynamic = int(os.getenv("QWEN3_TTS_MIN_DYNAMIC_MAX_NEW_TOKENS", "128"))
    medium_text_char_threshold = int(
        os.getenv("QWEN3_TTS_MEDIUM_TEXT_CHAR_THRESHOLD", "40")
    )
    short_text_char_threshold = int(
        os.getenv("QWEN3_TTS_SHORT_TEXT_CHAR_THRESHOLD", "48")
    )
    min_dynamic_short = int(os.getenv("QWEN3_TTS_MIN_DYNAMIC_MAX_NEW_TOKENS_SHORT", "28"))
    tiny_text_char_threshold = int(
        os.getenv("QWEN3_TTS_TINY_TEXT_CHAR_THRESHOLD", "12")
    )
    min_dynamic_tiny = int(os.getenv("QWEN3_TTS_MIN_DYNAMIC_MAX_NEW_TOKENS_TINY", "28"))
    hard_min = int(os.getenv("QWEN3_TTS_HARD_MIN_EFFECTIVE_MAX_NEW_TOKENS", "16"))
    sentence_min = int(os.getenv("QWEN3_TTS_SENTENCE_MIN_EFFECTIVE_MAX_NEW_TOKENS", "160"))
    clause_min = int(os.getenv("QWEN3_TTS_CLAUSE_MIN_EFFECTIVE_MAX_NEW_TOKENS", "64"))

    estimated = token_base + int(char_count * tokens_per_char) + punctuation_count * punctuation_bonus
    min_floor = min_dynamic
    if char_count <= short_text_char_threshold:
        min_floor = min(min_floor, min_dynamic_short)
    if char_count <= tiny_text_char_threshold:
        min_floor = min(min_floor, min_dynamic_tiny)
    if sentence_boundary_count > 0 and char_count >= medium_text_char_threshold:
        min_floor = max(min_floor, sentence_min)
    if clause_separator_count > 0 or sentence_boundary_count > 1:
        # Clause-heavy text needs a larger decode budget to avoid truncation.
        min_floor = max(min_floor, clause_min)
    min_floor = max(hard_min, min_floor)

    # Preserve the caller's requested decode budget and only bump upward when
    # the local heuristic says more tokens are needed. The caller already
    # applies its own text-aware estimate, so shrinking here can truncate
    # phrases before speech ever reaches a voiced region.
    return max(requested, min_floor, estimated)


@app.on_event("startup")
async def startup_event():
    if os.getenv("TTS_PRELOAD_MODEL", "0") == "1":
        engine.load()
        _warmup_tts_engine()


@app.post("/load_model")
async def load_model():
    try:
        engine.load()
        return {"status": "success", "model_name": engine.model_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/synthesize")
async def synthesize_stream(request: TTSRequest):
    """Stream audio chunks as SSE events with base64 PCM."""
    audio_iter: Iterator[np.ndarray] | None = None
    try:
        stats, audio_iter, effective_max_new_tokens = engine.stream_synthesize(
            text=request.text, voice=request.voice, max_new_tokens=request.max_new_tokens
        )
        first_audio = next(audio_iter, None)
    except Exception as exc:
        if audio_iter is not None:
            _close_iter_safely(audio_iter)
        LOGGER.exception("TTS /synthesize primary stream failed")
        # Fallback intentionally disabled so output is always decode-time stream.
        # try:
        #     fallback_audio, effective_max_new_tokens = engine.synthesize_fallback(
        #         request.text,
        #         request.voice,
        #         request.max_new_tokens,
        #     )
        #     first_audio = fallback_audio
        #     audio_iter = iter(())
        #     stats = type("Stats", (), {"ttfc_ms": (time.perf_counter() - request_started) * 1000.0})()
        # except Exception as fallback_exc:
        #     raise HTTPException(
        #         status_code=500,
        #         detail=f"Primary stream failed: {exc}; fallback failed: {fallback_exc}",
        #     ) from fallback_exc
        raise HTTPException(
            status_code=500,
            detail=f"Primary decode-time stream failed (fallback disabled): {exc}",
        ) from exc

    if first_audio is None or first_audio.size == 0:
        _close_iter_safely(audio_iter)
        # Fallback intentionally disabled so output is always decode-time stream.
        # try:
        #     fallback_audio, effective_max_new_tokens = engine.synthesize_fallback(
        #         request.text,
        #         request.voice,
        #         request.max_new_tokens,
        #     )
        #     first_audio = fallback_audio
        #     audio_iter = iter(())
        #     stats = type("Stats", (), {"ttfc_ms": (time.perf_counter() - request_started) * 1000.0})()
        # except Exception as fallback_exc:
        #     raise HTTPException(
        #         status_code=500,
        #         detail=f"Model returned no audio and fallback failed: {fallback_exc}",
        #     ) from fallback_exc
        raise HTTPException(
            status_code=500,
            detail="Model returned no audio from decode-time stream (fallback disabled).",
        )

    def audio_generator():
        chunk_idx = 0
        for chunk in _iter_pcm_chunks_from_audio_stream(
            _merged_audio_iter(first_audio, audio_iter)
        ):
            event_data = {
                "chunk_index": chunk_idx,
                "audio_base64": base64.b64encode(chunk).decode("utf-8"),
                "sample_rate": SAMPLE_RATE,
                "chunk_size_samples": CHUNK_SIZE,
                "ttfc_ms": stats.ttfc_ms if chunk_idx == 0 else None,
                "rtf": None,
                "max_new_tokens_effective": effective_max_new_tokens,
                "streaming_mode": "decode_time_codec_stream",
            }
            yield f"data: {json.dumps(event_data)}\n\n"
            chunk_idx += 1
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        audio_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/synthesize_binary")
async def synthesize_stream_binary(request: TTSRequest):
    """Stream raw PCM audio chunks (16-bit mono @ 24kHz)."""
    request_started = time.perf_counter()
    audio_iter: Iterator[np.ndarray] | None = None
    stats = None
    effective_max_new_tokens = request.max_new_tokens
    last_error: str | None = None
    candidate_texts = _speech_stabilization_candidates(request.text)
    selected_text = candidate_texts[0] if candidate_texts else request.text
    selected_candidate_index = 0
    text_stabilized = selected_text != request.text
    max_attempts = max(1, int(os.getenv("QWEN3_TTS_PRIMARY_STREAM_MAX_ATTEMPTS", "2")))
    retry_token_bump = int(os.getenv("QWEN3_TTS_PRIMARY_STREAM_RETRY_TOKEN_BUMP", "24"))
    for attempt in range(max_attempts):
        attempt_max_new_tokens = min(8192, request.max_new_tokens + attempt * retry_token_bump)
        try:
            stats, audio_iter, effective_max_new_tokens = engine.stream_synthesize(
                text=selected_text,
                voice=request.voice,
                max_new_tokens=attempt_max_new_tokens,
            )
            break
        except Exception as exc:
            if audio_iter is not None:
                _close_iter_safely(audio_iter)
                audio_iter = None
            last_error = f"Primary decode-time stream failed: {exc}"
            if attempt < max_attempts - 1:
                continue
            LOGGER.exception("TTS /synthesize_binary primary stream failed")

    if audio_iter is None or stats is None:
        raise HTTPException(
            status_code=500,
            detail=last_error or "Model returned no usable audio from decode-time stream.",
        )

    headers = {
        "Content-Type": "audio/pcm; rate=24000; channels=1; width=16",
        "Cache-Control": "no-cache",
        # TTFC for a streaming response is only known after the first body
        # chunk is emitted, so the client/benchmark must observe it directly.
        "X-TTFC-Ms": "na",
        "X-TTFC-Source": "client_observed_first_pcm_chunk",
        "X-RTF": "na",
        "X-Streaming-Mode": "decode_time_codec_stream",
        "X-Scheduler-Mode": engine._scheduler.mode,
        "X-Kernel-Path": "qwen_megakernel_C.decode_hidden_fp32_head",
        "X-Max-Active-Requests": str(engine._scheduler.max_active),
        "X-Max-New-Tokens-Effective": str(effective_max_new_tokens),
        "X-Decode-Stride": _decode_stride_header_value(),
        "X-Stop-Reason": "streaming",
        "X-Text-Stabilized": "1" if text_stabilized else "0",
    }
    if text_stabilized:
        LOGGER.info(
            "TTS speech stabilization selected candidate %d/%d: %r -> %r",
            selected_candidate_index + 1,
            len(candidate_texts),
            request.text,
            selected_text,
        )
    return StreamingResponse(
        _iter_pcm_chunks_from_audio_stream(audio_iter),
        media_type="audio/pcm",
        headers=headers,
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "sample_rate": SAMPLE_RATE,
        "chunk_size_samples": CHUNK_SIZE,
        "implementation": "qwen3-tts-megakernel-talker",
        "model_loaded": engine.loaded,
        "model_name": engine.model_name,
        "device": str(engine._device),
        "attn_implementation": engine.attn_implementation,
        "scheduler_mode": engine._scheduler.mode,
        "max_active_requests": engine._scheduler.max_active,
        "prefill_optimized": os.getenv("QWEN3_TTS_PREFILL_OPTIMIZED", "1") == "1",
        "prefill_kernel": os.getenv("QWEN3_TTS_PREFILL_KERNEL", "0") == "1",
        "prefill_mode": "pytorch_static_scaffold_by_default",
        "prewarm_backends": engine._prewarm_backends,
        "backend_slots_loaded": engine._backend_count,
    }


@app.get("/spec")
async def spec():
    return {
        "sample_rate": SAMPLE_RATE,
        "chunk_size_samples": CHUNK_SIZE,
        "sample_width_bytes": SAMPLE_WIDTH,
        "bytes_per_chunk": BYTES_PER_CHUNK,
        "format": "16-bit PCM, mono",
        "status": "talker megakernel + incremental codec decode",
        "scheduler_mode": engine._scheduler.mode,
        "max_active_requests": engine._scheduler.max_active,
        "kernel_path": "qwen_megakernel_C.decode_hidden_fp32_head",
        "batching_note": "scheduler-level scalar backend slots; CUDA talker decode is not yet a true batched kernel",
        "default_voice": engine.default_voice,
        "attn_implementation": engine.attn_implementation,
        "prefill_optimized": os.getenv("QWEN3_TTS_PREFILL_OPTIMIZED", "1") == "1",
        "prefill_kernel": os.getenv("QWEN3_TTS_PREFILL_KERNEL", "0") == "1",
        "prefill_mode": "pytorch_static_scaffold_by_default",
        "prefill_kernel_max_seq_len": int(os.getenv("QWEN3_TTS_PREFILL_KERNEL_MAX_SEQ_LEN", "96")),
        "prefill_graph": os.getenv("QWEN3_TTS_PREFILL_GRAPH", "0") == "1",
        "prewarm_backends": engine._prewarm_backends,
        "backend_slots_loaded": engine._backend_count,
        "decode_stride": _decode_stride_header_value(),
        "incremental_left_context_frames": int(
            os.getenv("QWEN3_TTS_INCREMENTAL_LEFT_CONTEXT_FRAMES", "25")
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
