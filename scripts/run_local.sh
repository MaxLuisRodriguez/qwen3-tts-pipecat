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
    python server.py &
    LLM_PID=$!
    cd "$REPO_ROOT"
    echo "LLM service PID: $LLM_PID"
else
    LLM_PID=""
fi

# Wait a bit for service to start
sleep 2

# Start TTS service in background
if check_port 8001; then
    echo -e "${GREEN}Starting TTS service on port 8001...${NC}"
    cd services/tts_qwen3
    python server.py &
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
python app.py

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
