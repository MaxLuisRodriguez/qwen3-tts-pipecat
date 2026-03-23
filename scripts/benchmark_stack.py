#!/usr/bin/env python3
"""Service-level benchmark helper for local Megakernel + Qwen3-TTS services."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass

import requests


LLM_URL_DEFAULT = "http://localhost:8000"
TTS_URL_DEFAULT = "http://localhost:8001"
PCM_BYTES_PER_SECOND = 24000 * 2  # 24kHz mono int16


@dataclass
class LLMMetrics:
    tokens: int
    ttft_ms: float
    total_ms: float
    decode_tok_s: float


@dataclass
class TTSMetrics:
    bytes: int
    audio_s: float
    total_ms: float
    first_chunk_ms: float
    header_ttfc_ms: float | None
    rtf: float


def _safe_div(numer: float, denom: float) -> float:
    if denom <= 0:
        return math.nan
    return numer / denom


def benchmark_llm(base_url: str, prompt: str, max_tokens: int, timeout_s: int) -> LLMMetrics:
    start = time.perf_counter()
    first_token_time = None
    token_count = 0

    with requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={"prompt": prompt, "max_tokens": max_tokens},
        stream=True,
        timeout=timeout_s,
    ) as response:
        response.raise_for_status()
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            payload = json.loads(raw[6:])
            if "token" in payload:
                token_count += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter()
            if payload.get("done"):
                break

    end = time.perf_counter()
    if first_token_time is None:
        ttft_ms = math.nan
        decode_tok_s = math.nan
    else:
        ttft_ms = (first_token_time - start) * 1000.0
        decode_tok_s = _safe_div(token_count, end - first_token_time)

    return LLMMetrics(
        tokens=token_count,
        ttft_ms=ttft_ms,
        total_ms=(end - start) * 1000.0,
        decode_tok_s=decode_tok_s,
    )


def benchmark_tts(base_url: str, text: str, max_new_tokens: int, timeout_s: int) -> TTSMetrics:
    start = time.perf_counter()
    first_chunk_time = None
    bytes_total = 0
    header_ttfc = None

    with requests.post(
        f"{base_url.rstrip('/')}/synthesize_binary",
        json={"text": text, "max_new_tokens": max_new_tokens},
        stream=True,
        timeout=timeout_s,
    ) as response:
        response.raise_for_status()
        if "x-ttfc-ms" in response.headers:
            try:
                header_ttfc = float(response.headers["x-ttfc-ms"])
            except ValueError:
                header_ttfc = None
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            bytes_total += len(chunk)

    end = time.perf_counter()
    audio_s = bytes_total / PCM_BYTES_PER_SECOND
    first_chunk_ms = math.nan if first_chunk_time is None else (first_chunk_time - start) * 1000.0
    total_s = end - start
    rtf = _safe_div(total_s, audio_s)

    return TTSMetrics(
        bytes=bytes_total,
        audio_s=audio_s,
        total_ms=total_s * 1000.0,
        first_chunk_ms=first_chunk_ms,
        header_ttfc_ms=header_ttfc,
        rtf=rtf,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-url", default=LLM_URL_DEFAULT)
    parser.add_argument("--tts-url", default=TTS_URL_DEFAULT)
    parser.add_argument("--llm-prompt", default="Give a short 1-sentence response about your capabilities.")
    parser.add_argument("--llm-max-tokens", type=int, default=128)
    parser.add_argument(
        "--tts-text",
        default="This is a short benchmark utterance for measuring local Qwen three TTS latency.",
    )
    parser.add_argument("--tts-max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = benchmark_llm(args.llm_url, args.llm_prompt, args.llm_max_tokens, args.timeout_s)
    tts = benchmark_tts(args.tts_url, args.tts_text, args.tts_max_new_tokens, args.timeout_s)

    # Service-only additive estimate for "LLM first token -> TTS first chunk".
    # This excludes STT, Daily transport, and browser playback. Use
    # scripts/benchmark_roundtrip.py for end-to-end Pipecat turn metrics.
    service_pipeline_estimate_ms = llm.ttft_ms + tts.first_chunk_ms

    result = {
        "llm": asdict(llm),
        "tts": asdict(tts),
        "service_pipeline_estimate_ms": service_pipeline_estimate_ms,
        "e2e_estimate_ms": service_pipeline_estimate_ms,
        "estimate_scope": "service_only_excludes_stt_daily_browser",
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("LLM")
    print(f"  tokens: {llm.tokens}")
    print(f"  ttft_ms: {llm.ttft_ms:.2f}")
    print(f"  total_ms: {llm.total_ms:.2f}")
    print(f"  decode_tok_s: {llm.decode_tok_s:.2f}")
    print("TTS")
    print(f"  bytes: {tts.bytes}")
    print(f"  audio_s: {tts.audio_s:.3f}")
    print(f"  first_chunk_ms: {tts.first_chunk_ms:.2f}")
    print(f"  header_ttfc_ms: {tts.header_ttfc_ms if tts.header_ttfc_ms is not None else 'n/a'}")
    print(f"  total_ms: {tts.total_ms:.2f}")
    print(f"  rtf: {tts.rtf:.4f}")
    print("Service Estimate")
    print(f"  estimate_ms (llm_ttft + tts_first_chunk): {service_pipeline_estimate_ms:.2f}")


if __name__ == "__main__":
    main()
