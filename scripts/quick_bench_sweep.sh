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
export PATH="$KERNEL_VENV/bin:$PATH"
export PYTHONUNBUFFERED=1

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

# Match run_local.sh behavior so benchmark startup uses the same model/runtime
# resolution as the main demo path.
: "${LLM_PRELOAD_WEIGHTS:=1}"
export LLM_PRELOAD_WEIGHTS
if [ -n "${QWEN_MEGAKERNEL_MODEL_NAME:-}" ]; then
  if [[ "$QWEN_MEGAKERNEL_MODEL_NAME" != /* ]] && [[ ! "$QWEN_MEGAKERNEL_MODEL_NAME" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
    QWEN_MEGAKERNEL_MODEL_NAME="$REPO_ROOT/${QWEN_MEGAKERNEL_MODEL_NAME#./}"
    export QWEN_MEGAKERNEL_MODEL_NAME
  fi
fi

LLM_URL="${LLM_URL:-http://127.0.0.1:8000}"
TTS_URL="${TTS_URL:-http://127.0.0.1:8001}"
BENCH_TIMEOUT_S="${BENCH_TIMEOUT_S:-600}"
BENCH_LLM_MAX_TOKENS="${BENCH_LLM_MAX_TOKENS:-128}"
BENCH_TTS_READ_CHUNK_BYTES="${BENCH_TTS_READ_CHUNK_BYTES:-960}"
START_SERVICES="${START_SERVICES:-1}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${BENCH_SWEEP_OUT_DIR:-/tmp/qwen_bench_sweep_${timestamp}}"
mkdir -p "$OUT_DIR"

LLM_PID=""
TTS_PID=""

cleanup() {
  if [ -n "$LLM_PID" ]; then
    kill "$LLM_PID" 2>/dev/null || true
  fi
  if [ -n "$TTS_PID" ]; then
    kill "$TTS_PID" 2>/dev/null || true
  fi
}

health_ok() {
  local url="$1"
  curl -fsS -m 2 "${url%/}/health" >/dev/null 2>&1
}

wait_for_health() {
  local name="$1"
  local url="$2"
  local timeout_secs="$3"
  local i=0
  while [ "$i" -lt "$timeout_secs" ]; do
    if health_ok "$url"; then
      return 0
    fi
    sleep 1
    i=$((i + 1))
  done
  echo "${name} did not become healthy at ${url%/}/health within ${timeout_secs}s"
  return 1
}

if ! health_ok "$LLM_URL" || ! health_ok "$TTS_URL"; then
  if [ "$START_SERVICES" != "1" ]; then
    echo "Services are not healthy and START_SERVICES=0."
    echo "LLM: ${LLM_URL%/}/health"
    echo "TTS: ${TTS_URL%/}/health"
    exit 1
  fi

  trap cleanup EXIT INT TERM

  if ! health_ok "$LLM_URL"; then
    echo "Starting LLM service for benchmark sweep..."
    (cd services/llm_megakernel && "$PYTHON_BIN" server.py >/tmp/quick_bench_sweep_llm.log 2>&1) &
    LLM_PID="$!"
  fi
  if ! health_ok "$TTS_URL"; then
    echo "Starting TTS service for benchmark sweep..."
    (cd services/tts_qwen3 && "$PYTHON_BIN" server.py >/tmp/quick_bench_sweep_tts.log 2>&1) &
    TTS_PID="$!"
  fi

  wait_for_health "LLM service" "$LLM_URL" 180
  wait_for_health "TTS service" "$TTS_URL" 240
fi

declare -a CASE_NAMES=("short" "medium" "long")
declare -a CASE_TTS_TOKENS=("96" "160" "256")
declare -a CASE_TTS_TEXTS=(
  "Hello. This is a short benchmark case."
  "This is a medium benchmark case with two short sentences. It is used to evaluate streaming behavior under moderate duration."
  "This is a long benchmark case intended to stress sustained synthesis over a longer utterance. It should highlight time to first chunk, stream real time factor, and long response stability."
)

for i in "${!CASE_NAMES[@]}"; do
  name="${CASE_NAMES[$i]}"
  tts_tokens="${CASE_TTS_TOKENS[$i]}"
  tts_text="${CASE_TTS_TEXTS[$i]}"
  out_json="$OUT_DIR/${name}.json"

  echo "Running case: $name"
  "$PYTHON_BIN" scripts/benchmark_stack.py \
    --llm-url "${LLM_URL%/}" \
    --tts-url "${TTS_URL%/}" \
    --llm-prompt "Answer in one short sentence: what can you do?" \
    --llm-max-tokens "$BENCH_LLM_MAX_TOKENS" \
    --tts-text "$tts_text" \
    --tts-max-new-tokens "$tts_tokens" \
    --tts-read-chunk-bytes "$BENCH_TTS_READ_CHUNK_BYTES" \
    --timeout-s "$BENCH_TIMEOUT_S" \
    --json > "$out_json"
done

summary_json="$OUT_DIR/summary.json"
"$PYTHON_BIN" - "$OUT_DIR" "$summary_json" <<'PY'
import json
import pathlib
import statistics
import sys

out_dir = pathlib.Path(sys.argv[1])
summary_path = pathlib.Path(sys.argv[2])
rows = []
for fp in sorted(out_dir.glob("*.json")):
    if fp.name == "summary.json":
        continue
    data = json.loads(fp.read_text())
    rows.append(
        {
            "case": fp.stem,
            "llm_decode_tok_s": data["llm"]["decode_tok_s"],
            "tts_first_chunk_ms": data["tts"]["first_chunk_ms"],
            "tts_header_ttfc_ms": data["tts"]["header_ttfc_ms"],
            "tts_rtf": data["tts"]["rtf"],
            "e2e_estimate_ms": data["e2e_estimate_ms"],
        }
    )

summary = {
    "cases": rows,
    "median_tts_first_chunk_ms": statistics.median(r["tts_first_chunk_ms"] for r in rows),
    "median_tts_rtf": statistics.median(r["tts_rtf"] for r in rows),
}
summary_path.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
PY

echo
echo "Saved benchmark sweep directory: $OUT_DIR"
echo "Per-case files: short.json, medium.json, long.json"
echo "Summary: $summary_json"
