"""Pipecat voice app: Deepgram STT + Megakernel LLM service + local Qwen TTS."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator

import aiohttp
from dotenv import load_dotenv

from pipecat.frames.frames import EndFrame, ErrorFrame, Frame, TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.tts_service import TTSService
from pipecat.transports.daily.transport import DailyParams, DailyTransport
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from pipecat.services.deepgram import DeepgramSTTService
except ImportError:
    from pipecat.services.deepgram.stt import DeepgramSTTService


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT_DIR, ".env.pipecat"))
load_dotenv(os.path.join(ROOT_DIR, ".env.qwen_megakernel"), override=False)


@dataclass
class DailySession:
    room_url: str
    token: str


class MegakernelLLMClient:
    """Minimal client for services/llm_megakernel SSE endpoint."""

    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        self._base_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8000").rstrip("/")
        self._max_tokens = int(os.getenv("PIPECAT_MAX_TOKENS", "120"))
        self._system_prompt = os.getenv(
            "PIPECAT_SYSTEM_PROMPT",
            "You are a concise, helpful voice assistant. Keep responses short and clear.",
        )
        self._history: list[tuple[str, str]] = []
        self._max_history_turns = int(os.getenv("PIPECAT_MAX_HISTORY_TURNS", "6"))
        self._response_first_line_only = os.getenv("PIPECAT_RESPONSE_FIRST_LINE_ONLY", "1") == "1"
        self._response_first_sentence_only = os.getenv("PIPECAT_RESPONSE_FIRST_SENTENCE_ONLY", "1") == "1"
        self._response_max_chars = int(os.getenv("PIPECAT_RESPONSE_MAX_CHARS", "120"))

    def _build_prompt(self, user_text: str) -> str:
        turns = self._history[-self._max_history_turns :]
        lines = [f"System: {self._system_prompt}"]
        for role, content in turns:
            lines.append(f"{role.capitalize()}: {content}")
        lines.append(f"User: {user_text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    async def _stream_response_chunks(self, prompt: str):
        payload = {"prompt": prompt, "max_tokens": self._max_tokens}
        async with self._session.post(
            f"{self._base_url}/generate",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=aiohttp.ClientTimeout(total=180),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LLM service returned HTTP {resp.status}")

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data: "):
                    continue

                body = line[6:]
                if not body:
                    continue

                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    continue

                if data.get("done"):
                    break
                if "error" in data:
                    raise RuntimeError(f"LLM service error: {data['error']}")
                if "token" in data:
                    yield data["token"]

    async def generate(self, user_text: str) -> str:
        prompt = self._build_prompt(user_text)
        parts: list[str] = []
        async for chunk in self._stream_response_chunks(prompt):
            parts.append(chunk)

        response = "".join(parts).strip()
        # Keep voice output short and stable for lower TTS latency.
        response = response.replace("\r", "")
        if self._response_first_line_only:
            first_line = response.splitlines()[0].strip() if response else ""
            if first_line:
                response = first_line
        if self._response_first_sentence_only and response:
            for i, ch in enumerate(response):
                if ch in ".?!":
                    response = response[: i + 1].strip()
                    break
        response = response[: self._response_max_chars].strip()
        if not response:
            response = "I did not catch that. Could you repeat it?"

        self._history.append(("user", user_text))
        self._history.append(("assistant", response))
        return response


class LocalQwenTTSService(TTSService):
    """TTS service backed by services/tts_qwen3/server.py."""

    def __init__(
        self,
        *,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str,
        voice: str | None = None,
        max_new_tokens: int = 256,
        in_sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=in_sample_rate, **kwargs)
        self._session = aiohttp_session
        self._base_url = base_url.rstrip("/")
        self._voice = voice
        self._max_new_tokens = max_new_tokens
        self._in_sample_rate = in_sample_rate
        self._dynamic_max_new_tokens = os.getenv("PIPECAT_TTS_DYNAMIC_MAX_NEW_TOKENS", "1") == "1"
        self._tts_token_base = int(os.getenv("PIPECAT_TTS_TOKEN_BASE", "32"))
        self._tts_tokens_per_char = float(os.getenv("PIPECAT_TTS_TOKENS_PER_CHAR", "0.75"))
        self._tts_punct_bonus = int(os.getenv("PIPECAT_TTS_PUNCT_BONUS", "2"))
        self._tts_min_new_tokens = int(os.getenv("PIPECAT_TTS_MIN_NEW_TOKENS", "80"))

    def can_generate_metrics(self) -> bool:
        return True

    def _estimate_max_new_tokens(self, text: str) -> int:
        if not self._dynamic_max_new_tokens:
            return self._max_new_tokens

        stripped = text.strip()
        punct = sum(stripped.count(ch) for ch in ".!?;,")
        estimate = self._tts_token_base + int(len(stripped) * self._tts_tokens_per_char)
        estimate += punct * self._tts_punct_bonus
        return max(self._tts_min_new_tokens, min(self._max_new_tokens, estimate))

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        effective_max_new_tokens = self._estimate_max_new_tokens(text)
        payload = {
            "text": text,
            "voice": self._voice,
            "max_new_tokens": effective_max_new_tokens,
        }
        headers = {"Content-Type": "application/json"}

        try:
            await self.start_ttfb_metrics()
            async with self._session.post(
                f"{self._base_url}/synthesize_binary",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=240),
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(
                        error=f"Qwen TTS HTTP {response.status} from {self._base_url}/synthesize_binary: {error}"
                    )
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame(context_id=context_id)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=self._in_sample_rate,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as exc:
            yield ErrorFrame(error=f"Qwen TTS request failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)


async def _create_daily_session(api_key: str) -> DailySession:
    """Create a temporary Daily room + meeting token via Daily REST API."""
    exp = int(time.time()) + int(os.getenv("DAILY_ROOM_TTL_SECONDS", "3600"))
    room_payload = {"properties": {"exp": exp, "eject_at_room_exp": True}}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/rooms",
            json=room_payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as room_resp:
            if room_resp.status not in (200, 201):
                body = await room_resp.text()
                raise RuntimeError(f"Daily room creation failed ({room_resp.status}): {body}")
            room_info = await room_resp.json()

        room_name = room_info["name"]
        token_payload = {"properties": {"room_name": room_name, "is_owner": True}}
        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            json=token_payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as token_resp:
            if token_resp.status not in (200, 201):
                body = await token_resp.text()
                raise RuntimeError(f"Daily token creation failed ({token_resp.status}): {body}")
            token_info = await token_resp.json()

    return DailySession(room_url=room_info["url"], token=token_info["token"])


async def _resolve_daily_session() -> DailySession:
    room_url = os.getenv("DAILY_ROOM_URL")
    room_token = os.getenv("DAILY_ROOM_TOKEN")
    if room_url and room_token:
        return DailySession(room_url=room_url, token=room_token)

    api_key = os.getenv("DAILY_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Daily credentials. Set DAILY_ROOM_URL+DAILY_ROOM_TOKEN, "
            "or set DAILY_API_KEY for automatic room creation."
        )

    return await _create_daily_session(api_key)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


async def main():
    deepgram_key = _require_env("DEEPGRAM_API_KEY")
    tts_service_url = os.getenv("TTS_SERVICE_URL", "http://localhost:8001")
    qwen_tts_voice = os.getenv("QWEN3_TTS_VOICE", os.getenv("QWEN3_TTS_DEFAULT_VOICE", "vivian"))
    qwen_tts_sample_rate = int(os.getenv("QWEN3_TTS_SAMPLE_RATE", "24000"))
    qwen_tts_max_new_tokens = int(os.getenv("QWEN3_TTS_MAX_NEW_TOKENS", "256"))

    daily_session = await _resolve_daily_session()
    print(f"[pipecat] Join this Daily room: {daily_session.room_url}")

    async with aiohttp.ClientSession() as http_session:
        llm_client = MegakernelLLMClient(http_session)

        transport = DailyTransport(
            daily_session.room_url,
            daily_session.token,
            "Qwen Megakernel Voice Bot",
            DailyParams(audio_in_enabled=True, audio_out_enabled=True),
        )

        stt = DeepgramSTTService(api_key=deepgram_key)
        tts = LocalQwenTTSService(
            aiohttp_session=http_session,
            base_url=tts_service_url,
            voice=qwen_tts_voice,
            in_sample_rate=qwen_tts_sample_rate,
            max_new_tokens=qwen_tts_max_new_tokens,
        )

        context = LLMContext(
            [
                {
                    "role": "system",
                    "content": os.getenv(
                        "PIPECAT_SYSTEM_PROMPT",
                        "You are a concise, helpful voice assistant.",
                    ),
                }
            ]
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_aggregator,
                tts,
                transport.output(),
                assistant_aggregator,
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        )

        generation_lock = asyncio.Lock()

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(_, __, message):
            user_text = (message.content or "").strip()
            if not user_text:
                return
            print(f"[user] {user_text}")

            async with generation_lock:
                try:
                    reply = await llm_client.generate(user_text)
                except Exception as exc:
                    reply = f"I hit an upstream LLM error: {exc}"
                print(f"[assistant] {reply}")
                await task.queue_frame(TTSSpeakFrame(reply))

        @assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def on_assistant_turn_stopped(_, message):
            if message and message.content:
                print(f"[assistant_spoken] {message.content}")

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(_, participant):
            print(f"[pipecat] Participant joined: {participant.get('id')}")
            welcome = os.getenv(
                "PIPECAT_WELCOME_MESSAGE",
                "Hi, I am online and ready. Ask me anything.",
            )
            await task.queue_frame(TTSSpeakFrame(welcome))

        @transport.event_handler("on_participant_left")
        async def on_participant_left(_, participant, reason):
            print(f"[pipecat] Participant left: {participant.get('id')} ({reason})")
            await task.queue_frame(EndFrame())

        @transport.event_handler("on_call_state_updated")
        async def on_call_state_updated(_, state):
            if state == "left":
                await task.queue_frame(EndFrame())

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())
