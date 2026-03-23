#!/usr/bin/env python3
"""Parse end-to-end Pipecat turn metrics from the app log."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any


METRIC_RE = re.compile(r"^\[metrics\]\[(?P<kind>[a-z_]+)\]\s*(?P<body>.*)$")


def _coerce_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in {"n/a", "na", "none"}:
        return None
    if lowered == "yes":
        return True
    if lowered == "no":
        return False
    try:
        if any(ch in raw for ch in (".", "e", "E")):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _parse_fields(body: str) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for token in body.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = _coerce_value(value)
    return fields


def _parse_complete_turn(lines: list[str]) -> dict[str, Any] | None:
    turns: list[dict[str, Any]] = []
    for raw_line in lines:
        match = METRIC_RE.match(raw_line.strip())
        if not match:
            continue
        kind = match.group("kind")
        body = match.group("body")
        if kind == "roundtrip":
            turns.append(
                {
                    "roundtrip": _parse_fields(body),
                    "stream": None,
                    "quality": None,
                    "warnings": [],
                }
            )
            continue
        if not turns:
            continue
        if kind == "warn":
            turns[-1]["warnings"].append(body)
            continue
        if kind in {"stream", "quality"}:
            turns[-1][kind] = _parse_fields(body)

    for turn in reversed(turns):
        if turn.get("roundtrip") and turn.get("stream") and turn.get("quality"):
            return turn
    return None


def _wait_for_complete_turn(log_path: str, start_at_end: bool, timeout_s: float) -> dict[str, Any] | None:
    cursor = os.path.getsize(log_path) if start_at_end and os.path.exists(log_path) else 0
    pending = ""
    parsed_lines: list[str] = []
    deadline = time.time() + timeout_s

    while time.time() <= deadline:
        if not os.path.exists(log_path):
            time.sleep(0.25)
            continue

        with open(log_path, "rb") as handle:
            handle.seek(cursor)
            chunk = handle.read()

        if chunk:
            cursor += len(chunk)
            pending += chunk.decode("utf-8", errors="replace")
            lines = pending.splitlines()
            if pending and not pending.endswith(("\n", "\r")):
                pending = lines.pop()
            else:
                pending = ""
            parsed_lines.extend(lines)
            turn = _parse_complete_turn(parsed_lines)
            if turn is not None:
                return turn

        time.sleep(0.25)
    return None


def _build_result(turn: dict[str, Any]) -> dict[str, Any]:
    roundtrip = dict(turn["roundtrip"])
    stream = dict(turn["stream"])
    quality = dict(turn["quality"])
    ttfc_ms = roundtrip.get("ttfc_ms")
    rtf = roundtrip.get("rtf")
    return {
        "roundtrip": roundtrip,
        "stream": stream,
        "quality": quality,
        "warnings": list(turn.get("warnings", [])),
        "targets": {
            "ttfc_lt_60ms": ttfc_ms is not None and float(ttfc_ms) < 60.0,
            "rtf_lt_0.15": rtf is not None and float(rtf) < 0.15,
            "frame_by_frame": bool(stream.get("frame_by_frame")),
            "audio_ok": bool(quality.get("audio_ok")),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True, help="Path to the run_local / Pipecat app log.")
    parser.add_argument(
        "--wait-timeout-s",
        type=float,
        default=0.0,
        help="Wait for a complete new turn for up to this many seconds.",
    )
    parser.add_argument(
        "--start-at-end",
        action="store_true",
        help="When waiting, ignore existing log content and only watch for newly appended metrics.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.wait_timeout_s > 0:
        turn = _wait_for_complete_turn(args.log_path, args.start_at_end, args.wait_timeout_s)
    else:
        if not os.path.exists(args.log_path):
            raise SystemExit(f"Log file not found: {args.log_path}")
        with open(args.log_path, "r", encoding="utf-8", errors="replace") as handle:
            turn = _parse_complete_turn(handle.readlines())

    if turn is None:
        raise SystemExit("No complete Pipecat metric turn found in the provided log.")

    result = _build_result(turn)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    roundtrip = result["roundtrip"]
    stream = result["stream"]
    quality = result["quality"]

    print("Roundtrip")
    print(f"  overall_ms: {roundtrip.get('overall_ms')}")
    print(f"  llm_tok_s: {roundtrip.get('llm_tok_s')}")
    print(f"  ttfc_ms: {roundtrip.get('ttfc_ms')}")
    print(f"  rtf: {roundtrip.get('rtf')}")
    print("Stream")
    print(f"  mode: {stream.get('mode')}")
    print(f"  frame_by_frame: {stream.get('frame_by_frame')}")
    print(f"  chunk_count: {stream.get('chunk_count')}")
    print("Quality")
    print(f"  audio_ok: {quality.get('audio_ok')}")
    print(f"  dropped_frames_suspected: {quality.get('dropped_frames_suspected')}")
    if result["warnings"]:
        print("Warnings")
        for warning in result["warnings"]:
            print(f"  {warning}")


if __name__ == "__main__":
    main()
