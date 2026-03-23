#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

KERNEL_VENV="$REPO_ROOT/kernel/.venv"
PYTHON_BIN="$KERNEL_VENV/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing kernel virtualenv at $KERNEL_VENV"
  echo "Run: bash scripts/bootstrap_qwen_megakernel.sh"
  exit 1
fi

if [ -f "$REPO_ROOT/.env.qwen_megakernel" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env.qwen_megakernel"
  set +a
fi
if [ -f "$REPO_ROOT/.env.pipecat" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env.pipecat"
  set +a
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
RUN_LOCAL_LOG="${RUN_LOCAL_LOG:-/tmp/qwen_roundtrip_${timestamp}.log}"
BENCH_JSON_OUT="${BENCH_JSON_OUT:-/tmp/qwen_roundtrip_${timestamp}.json}"
ROUNDTRIP_WAIT_S="${ROUNDTRIP_WAIT_S:-240}"
START_STACK="${START_STACK:-1}"
KEEP_STACK_RUNNING="${KEEP_STACK_RUNNING:-0}"

RUN_LOCAL_PID=""

cleanup() {
  if [ -n "$RUN_LOCAL_PID" ] && [ "$KEEP_STACK_RUNNING" != "1" ]; then
    kill "$RUN_LOCAL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_room_url() {
  local timeout_secs="${1:-180}"
  local i=0
  while [ "$i" -lt "$timeout_secs" ]; do
    if [ -f "$RUN_LOCAL_LOG" ]; then
      local room_url
      room_url="$(grep -m1 -o 'https://[^[:space:]]*daily.co/[^[:space:]]*' "$RUN_LOCAL_LOG" || true)"
      if [ -n "$room_url" ]; then
        printf '%s\n' "$room_url"
        return 0
      fi
    fi
    if [ -n "$RUN_LOCAL_PID" ] && ! kill -0 "$RUN_LOCAL_PID" 2>/dev/null; then
      echo "run_local.sh exited before the Daily room URL appeared."
      return 1
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "Timed out waiting for the Daily room URL in $RUN_LOCAL_LOG"
  return 1
}

if [ "$START_STACK" = "1" ]; then
  : >"$RUN_LOCAL_LOG"
  bash scripts/run_local.sh >"$RUN_LOCAL_LOG" 2>&1 &
  RUN_LOCAL_PID="$!"
  echo "Started run_local.sh (pid: $RUN_LOCAL_PID)"
else
  if [ ! -f "$RUN_LOCAL_LOG" ]; then
    echo "RUN_LOCAL_LOG does not exist: $RUN_LOCAL_LOG"
    exit 1
  fi
fi

room_url="$(wait_for_room_url 240)"
echo "Daily room URL: $room_url"
echo "Complete one short voice turn in the browser, then this script will capture the emitted Pipecat metrics."

"$PYTHON_BIN" scripts/benchmark_roundtrip.py \
  --log-path "$RUN_LOCAL_LOG" \
  --start-at-end \
  --wait-timeout-s "$ROUNDTRIP_WAIT_S" \
  --json | tee "$BENCH_JSON_OUT"

echo
echo "Saved roundtrip benchmark JSON: $BENCH_JSON_OUT"
echo "Captured from log: $RUN_LOCAL_LOG"
