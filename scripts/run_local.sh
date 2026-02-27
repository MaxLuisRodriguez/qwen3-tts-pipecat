#!/bin/bash
# Convenience script to run all services and demo locally

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Qwen Megakernel Services...${NC}"

# Get the repo root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Always run with the kernel virtualenv so qwen_megakernel imports work.
KERNEL_VENV="$REPO_ROOT/kernel/.venv"
PYTHON_BIN="$KERNEL_VENV/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    echo -e "${YELLOW}Missing kernel virtualenv at $KERNEL_VENV${NC}"
    echo -e "${YELLOW}Run: bash scripts/bootstrap_qwen_megakernel.sh${NC}"
    exit 1
fi
export PATH="$KERNEL_VENV/bin:$PATH"
echo -e "${GREEN}Using Python: $PYTHON_BIN${NC}"

# Load shared runtime config if present.
if [ -f "$REPO_ROOT/.env.qwen_megakernel" ]; then
    # shellcheck disable=SC1091
    set -a
    source "$REPO_ROOT/.env.qwen_megakernel"
    set +a
fi

# Load Pipecat runtime config if present.
if [ -f "$REPO_ROOT/.env.pipecat" ]; then
    # shellcheck disable=SC1091
    set -a
    source "$REPO_ROOT/.env.pipecat"
    set +a
fi

# Preload LLM weights on startup unless explicitly disabled.
: "${LLM_PRELOAD_WEIGHTS:=1}"
export LLM_PRELOAD_WEIGHTS

# Validate Pipecat provider credentials before starting processes.
if [ -z "${DEEPGRAM_API_KEY:-}" ]; then
    echo -e "${YELLOW}Missing DEEPGRAM_API_KEY. Set it in .env.pipecat.${NC}"
    exit 1
fi
if { [ -z "${DAILY_ROOM_URL:-}" ] || [ -z "${DAILY_ROOM_TOKEN:-}" ]; } && [ -z "${DAILY_API_KEY:-}" ]; then
    echo -e "${YELLOW}Missing Daily credentials. Set DAILY_API_KEY or DAILY_ROOM_URL+DAILY_ROOM_TOKEN in .env.pipecat.${NC}"
    exit 1
fi

# Normalize local model path relative to repo root.
if [ -n "${QWEN_MEGAKERNEL_MODEL_NAME:-}" ]; then
    if [[ "$QWEN_MEGAKERNEL_MODEL_NAME" != /* ]] && [[ ! "$QWEN_MEGAKERNEL_MODEL_NAME" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
        QWEN_MEGAKERNEL_MODEL_NAME="$REPO_ROOT/${QWEN_MEGAKERNEL_MODEL_NAME#./}"
        export QWEN_MEGAKERNEL_MODEL_NAME
    fi
fi

# Start local Qwen TTS by default (used by pipecat_demo/app.py).
: "${START_TTS_SERVICE:=1}"

# Check if services are already running
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        echo -e "${YELLOW}Port $1 is already in use. Skipping service on that port.${NC}"
        return 1
    fi
    return 0
}

# Start LLM service in background
if check_port 8000; then
    echo -e "${GREEN}Starting LLM service on port 8000...${NC}"
    cd services/llm_megakernel
    "$PYTHON_BIN" server.py &
    LLM_PID=$!
    cd "$REPO_ROOT"
    echo "LLM service PID: $LLM_PID"
else
    LLM_PID=""
fi

# Wait a bit for service to start
sleep 2

# Start TTS service in background (optional)
if [ "$START_TTS_SERVICE" = "1" ] && check_port 8001; then
    echo -e "${GREEN}Starting TTS service on port 8001...${NC}"
    cd services/tts_qwen3
    "$PYTHON_BIN" server.py &
    TTS_PID=$!
    cd "$REPO_ROOT"
    echo "TTS service PID: $TTS_PID"
else
    TTS_PID=""
fi

# Wait for services to be ready
echo -e "${GREEN}Waiting for services to start...${NC}"
sleep 3

# Check if services are responding
if [ -n "$LLM_PID" ]; then
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}LLM service is ready${NC}"
    else
        echo -e "${YELLOW}LLM service may not be ready yet${NC}"
    fi
fi

if [ -n "$TTS_PID" ]; then
    if curl -s http://localhost:8001/health > /dev/null; then
        echo -e "${GREEN}TTS service is ready${NC}"
    else
        echo -e "${YELLOW}TTS service may not be ready yet${NC}"
    fi
fi

# Run the demo
echo -e "${GREEN}Running Pipecat demo...${NC}"
cd pipecat_demo
"$PYTHON_BIN" app.py

# Cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [ -n "$LLM_PID" ]; then
        kill $LLM_PID 2>/dev/null || true
    fi
    if [ -n "$TTS_PID" ]; then
        kill $TTS_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}Done${NC}"
}

trap cleanup EXIT INT TERM

# Wait for demo to finish
wait
