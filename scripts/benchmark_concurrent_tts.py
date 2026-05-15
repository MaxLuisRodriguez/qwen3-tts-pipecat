#!/usr/bin/env python3
"""Jittered concurrent benchmark for the streaming Qwen3-TTS service."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

import requests


PCM_BYTES_PER_SECOND = 24000 * 2
DEFAULT_TEXTS = [
    "Sure, I can help with that.",
    "The meeting starts in about ten minutes.",
    "I found three options that look useful.",
    "Let me summarize the important part.",
    "The weather should be clear this afternoon.",
    "That answer is correct for the example.",
    "Please try the shorter version first.",
    "I can check the benchmark after this run.",
    "The next step is to measure latency.",
    "We should keep the response concise.",
]


@dataclass
class RequestMetrics:
    request_id: int
    scheduled_delay_s: float
    text: str
    status_code: int | None
    error: str | None
    bytes: int
    audio_s: float
    total_ms: float
    ttfc_ms: float | None
    rtf: float | None
    chunk_count: int
    max_chunk_gap_ms: float
    streaming_mode: str
    scheduler_mode: str
    kernel_path: str
    max_new_tokens_effective: str
    active_at_start: int


class ActiveCounter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0

    def enter(self) -> int:
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
            return self._active

    def exit(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)


class GPUPoller:
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                ).strip()
                first = raw.splitlines()[0]
                util_s, mem_s = [part.strip() for part in first.split(",", 1)]
                self.samples.append(
                    {
                        "t": time.perf_counter(),
                        "gpu_util_pct": float(util_s),
                        "memory_used_mib": float(mem_s),
                    }
                )
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def _safe_div(numer: float, denom: float) -> float | None:
    if denom <= 0:
        return None
    return numer / denom


def _percentile(values: list[float], percentile: float) -> float | None:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * percentile
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return clean[lo]
    frac = rank - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def _run_request(
    *,
    request_id: int,
    base_url: str,
    text: str,
    max_new_tokens: int,
    read_chunk_bytes: int,
    timeout_s: float,
    scheduled_delay_s: float,
    active: ActiveCounter,
) -> RequestMetrics:
    if scheduled_delay_s > 0:
        time.sleep(scheduled_delay_s)

    active_at_start = active.enter()
    started = time.perf_counter()
    first_chunk_at: float | None = None
    last_chunk_at: float | None = None
    max_chunk_gap_s = 0.0
    bytes_total = 0
    chunk_count = 0
    status_code: int | None = None
    streaming_mode = "unknown"
    scheduler_mode = "unknown"
    kernel_path = "unknown"
    max_new_tokens_effective = str(max_new_tokens)
    error: str | None = None

    try:
        with requests.post(
            f"{base_url.rstrip('/')}/synthesize_binary",
            json={"text": text, "max_new_tokens": max_new_tokens},
            stream=True,
            timeout=timeout_s,
        ) as response:
            status_code = response.status_code
            streaming_mode = response.headers.get("X-Streaming-Mode", "unknown")
            scheduler_mode = response.headers.get("X-Scheduler-Mode", "unknown")
            kernel_path = response.headers.get("X-Kernel-Path", "unknown")
            max_new_tokens_effective = response.headers.get(
                "X-Max-New-Tokens-Effective",
                str(max_new_tokens),
            )
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=read_chunk_bytes):
                if not chunk:
                    continue
                now = time.perf_counter()
                if first_chunk_at is None:
                    first_chunk_at = now
                if last_chunk_at is not None:
                    max_chunk_gap_s = max(max_chunk_gap_s, now - last_chunk_at)
                last_chunk_at = now
                bytes_total += len(chunk)
                chunk_count += 1
    except Exception as exc:
        error = str(exc)
    finally:
        active.exit()

    total_s = max(time.perf_counter() - started, 0.0)
    audio_s = bytes_total / PCM_BYTES_PER_SECOND
    ttfc_ms = None if first_chunk_at is None else (first_chunk_at - started) * 1000.0
    rtf = _safe_div(total_s, audio_s)
    return RequestMetrics(
        request_id=request_id,
        scheduled_delay_s=scheduled_delay_s,
        text=text,
        status_code=status_code,
        error=error,
        bytes=bytes_total,
        audio_s=audio_s,
        total_ms=total_s * 1000.0,
        ttfc_ms=ttfc_ms,
        rtf=rtf,
        chunk_count=chunk_count,
        max_chunk_gap_ms=max_chunk_gap_s * 1000.0,
        streaming_mode=streaming_mode,
        scheduler_mode=scheduler_mode,
        kernel_path=kernel_path,
        max_new_tokens_effective=max_new_tokens_effective,
        active_at_start=active_at_start,
    )


def _build_schedule(total_requests: int, request_rate: float, jitter_ms: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    interval_s = 1.0 / max(request_rate, 1e-9)
    delays: list[float] = []
    current = 0.0
    for idx in range(total_requests):
        if idx > 0:
            jitter_s = rng.uniform(-jitter_ms, jitter_ms) / 1000.0
            current = max(0.0, current + interval_s + jitter_s)
        delays.append(current)
    return delays


def _summarize(results: list[RequestMetrics], gpu_samples: list[dict[str, float]], max_active: int) -> dict[str, Any]:
    ok = [result for result in results if result.error is None and result.bytes > 0]
    ttfc = [float(result.ttfc_ms) for result in ok if result.ttfc_ms is not None]
    total = [float(result.total_ms) for result in ok]
    rtf = [float(result.rtf) for result in ok if result.rtf is not None]
    max_gap = [float(result.max_chunk_gap_ms) for result in ok]
    gpu_util = [sample["gpu_util_pct"] for sample in gpu_samples]
    gpu_mem = [sample["memory_used_mib"] for sample in gpu_samples]
    return {
        "requests": len(results),
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "max_active_observed": max_active,
        "ttfc_ms": {
            "p50": _percentile(ttfc, 0.50),
            "p90": _percentile(ttfc, 0.90),
            "p99": _percentile(ttfc, 0.99),
        },
        "total_ms": {
            "p50": _percentile(total, 0.50),
            "p90": _percentile(total, 0.90),
            "p99": _percentile(total, 0.99),
        },
        "rtf": {
            "p50": _percentile(rtf, 0.50),
            "p90": _percentile(rtf, 0.90),
            "p99": _percentile(rtf, 0.99),
        },
        "max_chunk_gap_ms": {
            "p50": _percentile(max_gap, 0.50),
            "p90": _percentile(max_gap, 0.90),
            "p99": _percentile(max_gap, 0.99),
        },
        "gpu": {
            "samples": len(gpu_samples),
            "util_avg_pct": (sum(gpu_util) / len(gpu_util)) if gpu_util else None,
            "util_max_pct": max(gpu_util) if gpu_util else None,
            "memory_max_mib": max(gpu_mem) if gpu_mem else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts-url", default="http://127.0.0.1:8001")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=4.0)
    parser.add_argument("--jitter-ms", type=float, default=250.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--read-chunk-bytes", type=int, default=960)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-out")
    parser.add_argument("--csv-out")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schedule = _build_schedule(args.requests, args.request_rate, args.jitter_ms, args.seed)
    texts = [DEFAULT_TEXTS[idx % len(DEFAULT_TEXTS)] for idx in range(args.requests)]
    active = ActiveCounter()
    gpu = GPUPoller()
    started = time.perf_counter()
    gpu.start()
    results: list[RequestMetrics] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [
                pool.submit(
                    _run_request,
                    request_id=idx,
                    base_url=args.tts_url,
                    text=texts[idx],
                    max_new_tokens=args.max_new_tokens,
                    read_chunk_bytes=args.read_chunk_bytes,
                    timeout_s=args.timeout_s,
                    scheduled_delay_s=schedule[idx],
                    active=active,
                )
                for idx in range(args.requests)
            ]
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        gpu.stop()

    results.sort(key=lambda item: item.request_id)
    summary = _summarize(results, gpu.samples, active.max_active)
    output = {
        "config": {
            "tts_url": args.tts_url,
            "concurrency": args.concurrency,
            "requests": args.requests,
            "request_rate": args.request_rate,
            "jitter_ms": args.jitter_ms,
            "max_new_tokens": args.max_new_tokens,
            "read_chunk_bytes": args.read_chunk_bytes,
            "seed": args.seed,
            "anti_cheat_note": "Start the TTS service with QWEN3_TTS_ANTI_CHEAT=1 to disable prompt/projection caches.",
        },
        "wall_s": time.perf_counter() - started,
        "summary": summary,
        "results": [asdict(result) for result in results],
        "gpu_samples": gpu.samples,
    }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
    if args.csv_out:
        with open(args.csv_out, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()) if results else [])
            if results:
                writer.writeheader()
                for result in results:
                    writer.writerow(asdict(result))

    if args.json:
        print(json.dumps(output, indent=2))
        return

    print("Concurrent TTS benchmark")
    print(f"  requests: {summary['requests']} successful={summary['successful']} failed={summary['failed']}")
    print(f"  max_active_observed: {summary['max_active_observed']}")
    print(f"  ttfc_ms p50/p90/p99: {summary['ttfc_ms']['p50']} / {summary['ttfc_ms']['p90']} / {summary['ttfc_ms']['p99']}")
    print(f"  rtf p50/p90/p99: {summary['rtf']['p50']} / {summary['rtf']['p90']} / {summary['rtf']['p99']}")
    print(f"  total_ms p50/p90/p99: {summary['total_ms']['p50']} / {summary['total_ms']['p90']} / {summary['total_ms']['p99']}")
    print(f"  gpu avg/max util: {summary['gpu']['util_avg_pct']} / {summary['gpu']['util_max_pct']}")


if __name__ == "__main__":
    main()
