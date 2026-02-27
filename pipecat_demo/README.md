# Pipecat Demo

Pipecat voice pipeline using:
- **Deepgram STT**
- **Megakernel LLM service** (`services/llm_megakernel`)
- **Local Qwen3-TTS service** (`services/tts_qwen3`)
- **Daily transport** (WebRTC room link)

## Setup

Install dependencies in the same Python environment used by the kernel:

```bash
pip install -r requirements.txt
```

## Environment

Configure credentials in repo root:

```bash
.env.pipecat
```

Required:
- `DEEPGRAM_API_KEY`
- Daily credentials:
  - either `DAILY_ROOM_URL` + `DAILY_ROOM_TOKEN`
  - or `DAILY_API_KEY` (app auto-creates room and token)

Optional:
- `QWEN3_TTS_MODEL_NAME`
- `QWEN3_TTS_DEFAULT_VOICE`
- `PIPECAT_SYSTEM_PROMPT`
- `PIPECAT_MAX_TOKENS`
- `LLM_SERVICE_URL`

## Running

Start the LLM service and Pipecat app together:

```bash
bash scripts/run_local.sh
```

Or run manually:

```bash
# terminal 1
cd services/llm_megakernel
python server.py

# terminal 2
cd services/tts_qwen3
python server.py

# terminal 3
cd pipecat_demo
python app.py
```

When `app.py` starts it prints a Daily room URL. Open it in browser, allow mic/audio, and talk.

## Pipeline

```
Daily audio input
  -> Deepgram STT
  -> User-turn aggregator
  -> HTTP call to local megakernel LLM service
  -> TTSSpeakFrame
  -> Local Qwen3-TTS HTTP service
  -> Daily audio output
```
