#!/usr/bin/env python3
"""Deterministic sanity/parity check for the Qwen3-TTS talker path."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KERNEL_DIR = os.path.join(ROOT_DIR, "kernel")
if KERNEL_DIR not in sys.path:
    sys.path.insert(0, KERNEL_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from qwen_tts import Qwen3TTSModel  # noqa: E402
from services.tts_qwen3.megakernel_talker import TalkerMegakernelBackend  # noqa: E402
from services.tts_qwen3.subtalker_batch_service import SubtalkerBatchService  # noqa: E402


@dataclass
class AudioSummary:
    samples: int
    audio_s: float
    rms: float
    peak: float


def _summarize(audio: np.ndarray, sample_rate: int) -> AudioSummary:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return AudioSummary(samples=0, audio_s=0.0, rms=0.0, peak=0.0)
    return AudioSummary(
        samples=int(arr.size),
        audio_s=float(arr.size / sample_rate),
        rms=float(np.sqrt(np.mean(np.square(arr)))),
        peak=float(np.max(np.abs(arr))),
    )


def _relative_delta(left: float, right: float) -> float:
    denom = max(abs(left), abs(right), 1e-9)
    return abs(left - right) / denom


def _prefix_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    n = min(int(left.size), int(right.size))
    if n < 1024:
        return None
    a = np.asarray(left[:n], dtype=np.float32)
    b = np.asarray(right[:n], dtype=np.float32)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-9:
        return None
    return float(np.dot(a, b) / denom)


def _default_attn_implementation() -> str:
    configured = os.getenv("QWEN3_TTS_ATTN_IMPL")
    if configured:
        return configured
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("QWEN3_TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"))
    parser.add_argument("--text", default="the city is tehran")
    parser.add_argument("--voice", default=os.getenv("QWEN3_TTS_DEFAULT_VOICE", "vivian"))
    parser.add_argument("--language", default=os.getenv("QWEN3_TTS_LANGUAGE", "english"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--duration-tolerance-s", type=float, default=1.0)
    parser.add_argument("--rms-relative-tolerance", type=float, default=0.6)
    parser.add_argument("--peak-relative-tolerance", type=float, default=0.8)
    parser.add_argument(
        "--min-prefix-correlation",
        type=float,
        default=-1.0,
        help="Optional waveform prefix correlation floor. Disabled by default because codec timing can shift.",
    )
    parser.add_argument(
        "--use-subtalker-service",
        action="store_true",
        help="Route subtalker forwards through SubtalkerBatchService to test the batched path.",
    )
    parser.add_argument("--skip-official", action="store_true", help="Only check optimized backend construction and streaming.")
    parser.add_argument("--skip-kernel-parity", action="store_true", help="Skip first-step custom-kernel vs PyTorch parity.")
    parser.add_argument(
        "--allow-kernel-token-mismatch",
        action="store_true",
        help="Report but do not fail when the first custom-kernel token differs from PyTorch.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=_default_attn_implementation(),
    )
    model.model.eval()

    subtalker_service = None
    if args.use_subtalker_service:
        subtalker_model = model.model.talker.code_predictor.model
        subtalker_service = SubtalkerBatchService(subtalker_model)
        subtalker_service.warmup(
            hidden_size=int(subtalker_model.config.hidden_size),
            device=model.model.talker.device,
            dtype=model.model.talker.dtype,
        )
    backend = TalkerMegakernelBackend(model, subtalker_service=subtalker_service)
    stats, audio_iter = backend.stream_audio(
        text=args.text,
        speaker=args.voice,
        language=args.language,
        max_new_tokens=args.max_new_tokens,
    )
    optimized_chunks = [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in audio_iter if chunk.size > 0]
    optimized_audio = np.concatenate(optimized_chunks) if optimized_chunks else np.empty((0,), dtype=np.float32)
    sample_rate = int(os.getenv("QWEN3_TTS_SAMPLE_RATE", "24000"))
    optimized = _summarize(optimized_audio, sample_rate)

    result = {
        "text": args.text,
        "voice": args.voice,
        "language": args.language,
        "seed": args.seed,
        "optimized": asdict(optimized),
        "optimized_stats": {
            "ttfc_ms": stats.ttfc_ms,
            "generation_s": stats.generation_s,
            "audio_seconds": stats.audio_seconds,
            "frames_generated": stats.frames_generated,
            "stop_reason": stats.stop_reason,
            "prefill_ms": stats.prefill_ms,
            "prompt_build_ms": stats.prompt_build_ms,
            "prefill_model_ms": stats.prefill_model_ms,
            "prefill_cache_ms": stats.prefill_cache_ms,
            "prefill_mode": stats.prefill_mode,
            "subtalker_ms": stats.subtalker_ms,
            "talker_decode_ms": stats.talker_decode_ms,
            "audio_decode_ms": stats.audio_decode_ms,
            "subtalker_calls": stats.subtalker_calls,
            "talker_decode_calls": stats.talker_decode_calls,
            "audio_decode_calls": stats.audio_decode_calls,
            "audio_chunks": stats.audio_chunks,
            "first_decode_ms": stats.first_decode_ms,
            "kernel_path": stats.kernel_path,
            "timing_mode": stats.timing_mode,
            "audio_decode_overlap": stats.audio_decode_overlap,
            "audio_decode_wait_ms": stats.audio_decode_wait_ms,
            "subtalker_compile": stats.subtalker_compile,
        },
        "kernel_first_step_parity": None,
        "official": None,
        "audio_comparison": None,
        "checks": {
            "optimized_has_audio": optimized.samples > 0,
            "duration_within_tolerance": None,
            "rms_within_tolerance": None,
            "peak_within_tolerance": None,
            "prefix_correlation_within_tolerance": None,
            "kernel_prefill_token_match": None,
            "kernel_first_step_token_match": None,
        },
    }

    if not args.skip_kernel_parity:
        parity = backend.debug_first_step_parity(
            args.text,
            args.voice,
            language=args.language,
        )
        result["kernel_first_step_parity"] = parity
        if parity.get("kernel_prefill_token_match") is not None:
            result["checks"]["kernel_prefill_token_match"] = bool(parity.get("kernel_prefill_token_match"))
        result["checks"]["kernel_first_step_token_match"] = bool(parity.get("token_match"))

    if not args.skip_official:
        with torch.inference_mode():
            wavs, sr = model.generate_custom_voice(
                text=args.text,
                speaker=args.voice,
                language=args.language,
                max_new_tokens=args.max_new_tokens,
            )
        official_audio = np.asarray(wavs[0] if wavs else [], dtype=np.float32).reshape(-1)
        official = _summarize(official_audio, int(sr or sample_rate))
        duration_delta = abs(optimized.audio_s - official.audio_s)
        rms_relative_delta = _relative_delta(optimized.rms, official.rms)
        peak_relative_delta = _relative_delta(optimized.peak, official.peak)
        prefix_corr = _prefix_correlation(optimized_audio, official_audio)
        result["official"] = asdict(official)
        result["checks"]["duration_within_tolerance"] = duration_delta <= args.duration_tolerance_s
        result["checks"]["rms_within_tolerance"] = rms_relative_delta <= args.rms_relative_tolerance
        result["checks"]["peak_within_tolerance"] = peak_relative_delta <= args.peak_relative_tolerance
        result["checks"]["prefix_correlation_within_tolerance"] = (
            True if args.min_prefix_correlation < 0.0
            else prefix_corr is not None and prefix_corr >= args.min_prefix_correlation
        )
        result["duration_delta_s"] = duration_delta
        result["audio_comparison"] = {
            "duration_delta_s": duration_delta,
            "rms_relative_delta": rms_relative_delta,
            "peak_relative_delta": peak_relative_delta,
            "prefix_correlation": prefix_corr,
        }

    ok = bool(result["checks"]["optimized_has_audio"])
    for key, value in result["checks"].items():
        if value is None:
            continue
        if key == "kernel_first_step_token_match" and args.allow_kernel_token_mismatch:
            continue
        ok = ok and bool(value)
    result["ok"] = ok

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
