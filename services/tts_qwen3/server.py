"""Streaming TTS server backed by Qwen/Qwen3-TTS."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from typing import Iterator

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoProcessor

try:
    from transformers import Qwen3TTSForConditionalGeneration
except ImportError:
    Qwen3TTSForConditionalGeneration = None

app = FastAPI(title="Qwen3 TTS Service")

SAMPLE_RATE = int(os.getenv("QWEN3_TTS_SAMPLE_RATE", "24000"))
CHUNK_SIZE = int(os.getenv("QWEN3_TTS_CHUNK_SIZE", "1600"))
SAMPLE_WIDTH = 2
BYTES_PER_CHUNK = CHUNK_SIZE * SAMPLE_WIDTH


class TTSRequest(BaseModel):
    """Request model for text-to-speech."""

    text: str = Field(min_length=1, description="Input text to synthesize.")
    voice: str | None = Field(
        default=None, description="Optional speaker label (e.g. Cherry)."
    )
    max_new_tokens: int = Field(
        default=1024, ge=64, le=8192, description="Max decode tokens for TTS generation."
    )


class Qwen3TTSEngine:
    """Lazy-loaded Qwen3-TTS inference wrapper."""

    def __init__(self):
        self.model_name = os.getenv("QWEN3_TTS_MODEL_NAME", "Qwen/Qwen3-TTS")
        self.default_voice = os.getenv("QWEN3_TTS_DEFAULT_VOICE", "Cherry")
        self.attn_implementation = os.getenv("QWEN3_TTS_ATTN_IMPL", "sdpa")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._processor = None
        self._model = None

    @property
    def loaded(self) -> bool:
        return self._processor is not None and self._model is not None

    def _build_prompt(self, text: str, voice: str | None) -> str:
        # Qwen3-TTS model card pattern: "<speaker>: <chat_template_text>".
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Convert the text to speech:{text}"}],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        speaker = (voice or self.default_voice).strip()
        return f"{speaker}: {chat_text}" if speaker else chat_text

    def load(self):
        if self.loaded:
            return
        if Qwen3TTSForConditionalGeneration is None:
            raise RuntimeError(
                "Your transformers build does not expose Qwen3TTSForConditionalGeneration. "
                "Upgrade transformers to a version that includes Qwen3-TTS."
            )

        with self._load_lock:
            if self.loaded:
                return
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = Qwen3TTSForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self._dtype,
                device_map="auto" if self._device.type == "cuda" else None,
                attn_implementation=self.attn_implementation,
            )
            self._model.eval()
            if self._device.type != "cuda":
                self._model.to(self._device)

    def synthesize(self, text: str, voice: str | None, max_new_tokens: int) -> np.ndarray:
        self.load()
        prompt = self._build_prompt(text=text, voice=voice)

        with self._generate_lock:
            inputs = self._processor(text=[prompt], padding=True, return_tensors="pt")
            inputs = inputs.to("cuda" if self._device.type == "cuda" else self._device)

            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=True,
                    max_new_tokens=max_new_tokens,
                )

        audio = generated.reshape(-1).detach().float().cpu().numpy()
        if audio.size == 0:
            raise RuntimeError("Model returned empty audio.")
        return audio


engine = Qwen3TTSEngine()


def _audio_to_pcm16(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _iter_pcm_chunks(pcm_bytes: bytes) -> Iterator[bytes]:
    for start in range(0, len(pcm_bytes), BYTES_PER_CHUNK):
        yield pcm_bytes[start : start + BYTES_PER_CHUNK]


@app.on_event("startup")
async def startup_event():
    if os.getenv("TTS_PRELOAD_MODEL", "0") == "1":
        engine.load()


@app.post("/load_model")
async def load_model():
    try:
        engine.load()
        return {"status": "success", "model_name": engine.model_name}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/synthesize")
async def synthesize_stream(request: TTSRequest):
    """
    Stream audio chunks as Server-Sent Events (base64 PCM chunks).
    """
    started = time.perf_counter()
    try:
        audio = engine.synthesize(
            text=request.text, voice=request.voice, max_new_tokens=request.max_new_tokens
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ttfc_ms = (time.perf_counter() - started) * 1000.0
    pcm = _audio_to_pcm16(audio)

    def audio_generator():
        chunk_idx = 0
        for chunk in _iter_pcm_chunks(pcm):
            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
            event_data = {
                "chunk_index": chunk_idx,
                "audio_base64": chunk_b64,
                "sample_rate": SAMPLE_RATE,
                "chunk_size_samples": CHUNK_SIZE,
                "ttfc_ms": ttfc_ms if chunk_idx == 0 else None,
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
    """
    Stream raw PCM audio chunks (16-bit mono @ 24kHz).
    """
    started = time.perf_counter()
    try:
        audio = engine.synthesize(
            text=request.text, voice=request.voice, max_new_tokens=request.max_new_tokens
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ttfc_ms = (time.perf_counter() - started) * 1000.0
    pcm = _audio_to_pcm16(audio)

    def audio_generator():
        for chunk in _iter_pcm_chunks(pcm):
            yield chunk

    headers = {
        "Content-Type": "audio/pcm; rate=24000; channels=1; width=16",
        "Cache-Control": "no-cache",
        "X-TTFC-Ms": f"{ttfc_ms:.2f}",
    }
    return StreamingResponse(audio_generator(), media_type="audio/pcm", headers=headers)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "sample_rate": SAMPLE_RATE,
        "chunk_size_samples": CHUNK_SIZE,
        "implementation": "qwen3-tts",
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
        "status": "qwen3-tts inference",
        "default_voice": engine.default_voice,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
