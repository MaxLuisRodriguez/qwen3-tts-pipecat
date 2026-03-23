"""Streaming TTS server backed by Qwen3-TTS with talker megakernel decode."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
import time
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

app = FastAPI(title="Qwen3 TTS Service")
LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = int(os.getenv("QWEN3_TTS_SAMPLE_RATE", "24000"))
CHUNK_SIZE = int(os.getenv("QWEN3_TTS_CHUNK_SIZE", "480"))
SAMPLE_WIDTH = 2
BYTES_PER_CHUNK = CHUNK_SIZE * SAMPLE_WIDTH
DECODE_STRIDE = int(os.getenv("QWEN3_TTS_DECODE_STRIDE", "1"))


def _decode_stride_header_value() -> str:
    if os.getenv("QWEN3_TTS_ADAPTIVE_DECODE_CADENCE", "1") == "1":
        mid = max(1, int(os.getenv("QWEN3_TTS_DECODE_STRIDE_MID", "12")))
        late = max(1, int(os.getenv("QWEN3_TTS_DECODE_STRIDE_LATE", "24")))
        late_start = int(os.getenv("QWEN3_TTS_DECODE_STRIDE_LATE_START_FRAME", "24"))
        left_context = max(0, int(os.getenv("QWEN3_TTS_INCREMENTAL_LEFT_CONTEXT_FRAMES", "12")))
        return f"adaptive(mid={mid},late={late}@{late_start},ctx={left_context})"
    return str(DECODE_STRIDE)


class TTSRequest(BaseModel):
    """Request model for text-to-speech."""

    text: str = Field(min_length=1, description="Input text to synthesize.")
    voice: str | None = Field(
        default=None, description="Optional speaker label (e.g. Cherry)."
    )
    max_new_tokens: int = Field(
        default=1024, ge=32, le=8192, description="Max decode tokens for TTS generation."
    )


class Qwen3TTSEngine:
    """Lazy-loaded Qwen3-TTS + talker megakernel backend."""

    def __init__(self):
        self.model_name = os.getenv(
            "QWEN3_TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
        )
        self.default_voice = os.getenv("QWEN3_TTS_DEFAULT_VOICE", "vivian")
        self.default_language = os.getenv("QWEN3_TTS_LANGUAGE", "english").strip().lower()
        self.attn_implementation = os.getenv("QWEN3_TTS_ATTN_IMPL", "sdpa")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._load_lock = threading.Lock()
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

    @property
    def loaded(self) -> bool:
        return self._model is not None and self._backend is not None

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
            self._backend = TalkerMegakernelBackend(self._model)

    def stream_synthesize(
        self, text: str, voice: str | None, max_new_tokens: int
    ) -> tuple[object, Iterator[np.ndarray], int]:
        self.load()
        if self._backend is None:
            raise RuntimeError("Megakernel talker backend failed to initialize.")
        if self._rebuild_backend_per_request:
            # Keep model weights resident; just rebuild decoding runtime state.
            self._backend = TalkerMegakernelBackend(self._model)
        speaker = (voice or self.default_voice).strip() or self.default_voice
        effective_max_new_tokens = _estimate_max_new_tokens(text, max_new_tokens)
        acquired = self._generate_lock.acquire(timeout=self._generate_lock_timeout_s)
        if not acquired:
            # Queue rather than hard-failing under overlap.
            LOGGER.warning(
                "TTS generation lock wait exceeded %.1fs; waiting for release.",
                self._generate_lock_timeout_s,
            )
            self._generate_lock.acquire()
        try:
            stats, raw_iter = self._backend.stream_audio(
                text=text,
                speaker=speaker,
                language=self.default_language,
                max_new_tokens=effective_max_new_tokens,
                decode_stride=DECODE_STRIDE,
            )
        except Exception:
            self._generate_lock.release()
            raise

        def guarded_iter() -> Iterator[np.ndarray]:
            try:
                with torch.no_grad():
                    for audio in raw_iter:
                        yield audio
            finally:
                self._generate_lock.release()

        return stats, guarded_iter(), effective_max_new_tokens

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
    first_audio: np.ndarray | None = None
    stats = None
    effective_max_new_tokens = request.max_new_tokens
    last_error: str | None = None
    selected_text = request.text
    max_attempts = max(1, int(os.getenv("QWEN3_TTS_PRIMARY_STREAM_MAX_ATTEMPTS", "2")))
    retry_token_bump = int(os.getenv("QWEN3_TTS_PRIMARY_STREAM_RETRY_TOKEN_BUMP", "24"))
    non_silent_peek_chunks = int(os.getenv("QWEN3_TTS_NON_SILENT_PEEK_CHUNKS", "1"))
    strict_non_silent_peek_chunks = int(
        os.getenv("QWEN3_TTS_STRICT_NON_SILENT_PEEK_CHUNKS", "12")
    )
    max_prefix_silence_s = float(os.getenv("QWEN3_TTS_MAX_PREFIX_SILENCE_S", "0.8"))
    strict_max_prefix_silence_s = float(
        os.getenv("QWEN3_TTS_STRICT_MAX_PREFIX_SILENCE_S", "1.6")
    )
    non_silent_rms = float(os.getenv("QWEN3_TTS_NON_SILENT_RMS", "0.0015"))
    require_non_silent_prefix = os.getenv("QWEN3_TTS_REQUIRE_NON_SILENT_PREFIX", "0") == "1"
    candidate_texts = _speech_stabilization_candidates(request.text)
    selected_candidate_index = 0

    for candidate_index, candidate_text in enumerate(candidate_texts):
        for attempt in range(max_attempts):
            attempt_max_new_tokens = min(8192, request.max_new_tokens + attempt * retry_token_bump)
            try:
                stats, audio_iter, effective_max_new_tokens = engine.stream_synthesize(
                    text=candidate_text,
                    voice=request.voice,
                    max_new_tokens=attempt_max_new_tokens,
                )
                peek_chunks = non_silent_peek_chunks
                max_prefix_samples = int(max_prefix_silence_s * SAMPLE_RATE)
                if _prefers_strict_prefix_probe(candidate_text) or candidate_index > 0:
                    peek_chunks = max(peek_chunks, strict_non_silent_peek_chunks)
                    max_prefix_samples = max(
                        max_prefix_samples,
                        int(strict_max_prefix_silence_s * SAMPLE_RATE),
                    )

                if peek_chunks <= 0:
                    first_audio = next(audio_iter, None)
                    if first_audio is None or first_audio.size == 0:
                        _close_iter_safely(audio_iter)
                        audio_iter = None
                        last_error = "Model returned no audio chunks from decode-time stream."
                        continue
                    selected_text = candidate_text
                    selected_candidate_index = candidate_index
                    break

                prefix, has_non_silent = _peek_audio_prefix(
                    audio_iter,
                    max_chunks=peek_chunks,
                    max_prefix_samples=max_prefix_samples,
                    rms_threshold=non_silent_rms,
                )
                if not prefix:
                    _close_iter_safely(audio_iter)
                    audio_iter = None
                    last_error = "Model returned no audio chunks from decode-time stream."
                    continue

                if has_non_silent:
                    while len(prefix) > 1 and _is_effectively_silent(prefix[0], non_silent_rms):
                        prefix.pop(0)
                    first_audio = _trim_leading_silence(prefix[0], non_silent_rms)
                    if first_audio.size == 0 and len(prefix) > 1:
                        prefix.pop(0)
                        first_audio = prefix[0]
                    audio_iter = _prepend_audio_iter(prefix[1:], audio_iter)
                    selected_text = candidate_text
                    selected_candidate_index = candidate_index
                    break

                LOGGER.warning(
                    "Leading decode-time audio remained silent for the initial prefix "
                    "(candidate %d/%d, attempt %d/%d).",
                    candidate_index + 1,
                    len(candidate_texts),
                    attempt + 1,
                    max_attempts,
                )
                _close_iter_safely(audio_iter)
                audio_iter = None
                last_error = "Leading decode-time audio chunks were effectively silent."
                continue
            except Exception as exc:
                if audio_iter is not None:
                    _close_iter_safely(audio_iter)
                    audio_iter = None
                last_error = f"Primary decode-time stream failed: {exc}"
                if attempt < max_attempts - 1:
                    continue
                LOGGER.exception("TTS /synthesize_binary primary stream failed")
        if first_audio is not None and audio_iter is not None and stats is not None:
            break

    if (
        (first_audio is None or audio_iter is None or stats is None)
        and not require_non_silent_prefix
        and candidate_texts
    ):
        # Final fallback within the same local decode-time streaming backend:
        # if all stabilization candidates stayed silent, preserve previous
        # behavior and stream the original request immediately.
        candidate_text = candidate_texts[0]
        for attempt in range(max_attempts):
            attempt_max_new_tokens = min(8192, request.max_new_tokens + attempt * retry_token_bump)
            try:
                stats, audio_iter, effective_max_new_tokens = engine.stream_synthesize(
                    text=candidate_text,
                    voice=request.voice,
                    max_new_tokens=attempt_max_new_tokens,
                )
                peek_chunks = non_silent_peek_chunks
                max_prefix_samples = int(max_prefix_silence_s * SAMPLE_RATE)
                if _prefers_strict_prefix_probe(candidate_text):
                    peek_chunks = max(peek_chunks, strict_non_silent_peek_chunks)
                    max_prefix_samples = max(
                        max_prefix_samples,
                        int(strict_max_prefix_silence_s * SAMPLE_RATE),
                    )

                if peek_chunks <= 0:
                    first_audio = next(audio_iter, None)
                else:
                    prefix, _ = _peek_audio_prefix(
                        audio_iter,
                        max_chunks=peek_chunks,
                        max_prefix_samples=max_prefix_samples,
                        rms_threshold=non_silent_rms,
                    )
                    if prefix:
                        first_audio = _trim_leading_silence(prefix[0], non_silent_rms)
                        if first_audio.size == 0 and len(prefix) > 1:
                            prefix.pop(0)
                            first_audio = prefix[0]
                        audio_iter = _prepend_audio_iter(prefix[1:], audio_iter)
                if first_audio is not None and first_audio.size > 0:
                    selected_text = candidate_text
                    selected_candidate_index = 0
                    LOGGER.info(
                        "Streaming decode-time audio despite silent prefix after exhausting stabilization candidates."
                    )
                    break
                _close_iter_safely(audio_iter)
                audio_iter = None
            except Exception:
                if audio_iter is not None:
                    _close_iter_safely(audio_iter)
                    audio_iter = None

    if first_audio is None or audio_iter is None or stats is None:
        raise HTTPException(
            status_code=500,
            detail=last_error or "Model returned no usable audio from decode-time stream.",
        )

    ttfc_ms = stats.ttfc_ms
    if ttfc_ms is None:
        ttfc_ms = (time.perf_counter() - request_started) * 1000.0

    headers = {
        "Content-Type": "audio/pcm; rate=24000; channels=1; width=16",
        "Cache-Control": "no-cache",
        "X-TTFC-Ms": f"{ttfc_ms:.2f}",
        "X-RTF": "na",
        "X-Streaming-Mode": "decode_time_codec_stream",
        "X-Max-New-Tokens-Effective": str(effective_max_new_tokens),
        "X-Decode-Stride": _decode_stride_header_value(),
        "X-Stop-Reason": getattr(stats, "stop_reason", "unknown"),
        "X-Text-Stabilized": "1" if selected_candidate_index > 0 else "0",
    }
    if selected_candidate_index > 0:
        LOGGER.info(
            "TTS speech stabilization selected candidate %d/%d: %r -> %r",
            selected_candidate_index + 1,
            len(candidate_texts),
            request.text,
            selected_text,
        )
    return StreamingResponse(
        _iter_pcm_chunks_from_audio_stream(_merged_audio_iter(first_audio, audio_iter)),
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
        "default_voice": engine.default_voice,
        "decode_stride": _decode_stride_header_value(),
        "incremental_left_context_frames": int(
            os.getenv("QWEN3_TTS_INCREMENTAL_LEFT_CONTEXT_FRAMES", "25")
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
