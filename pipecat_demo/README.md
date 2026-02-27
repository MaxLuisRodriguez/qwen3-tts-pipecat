# Pipecat Demo

Pipecat voice pipeline using:
- **Deepgram STT**
- **Megakernel LLM service** (`services/llm_megakernel`)
- **Cartesia TTS**
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
- `CARTESIA_API_KEY`
- Daily credentials:
  - either `DAILY_ROOM_URL` + `DAILY_ROOM_TOKEN`
  - or `DAILY_API_KEY` (app auto-creates room and token)

Optional:
- `CARTESIA_VOICE_ID`
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
  -> Cartesia TTS
  -> Daily audio output
```
