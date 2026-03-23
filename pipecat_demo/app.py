"""Pipecat voice app: Deepgram STT + Megakernel LLM service + local Qwen TTS."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import Any, AsyncGenerator

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


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


class MegakernelLLMClient:
    """Minimal client for services/llm_megakernel SSE endpoint."""

    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        self._base_url = os.getenv("LLM_SERVICE_URL", "http://localhost:8000").rstrip("/")
        self._max_tokens = int(os.getenv("PIPECAT_MAX_TOKENS", "256"))
        self._system_prompt = os.getenv(
            "PIPECAT_SYSTEM_PROMPT",
            "You are a concise, helpful voice assistant. Keep responses short and clear.",
        )
        self._history: list[tuple[str, str]] = []
        self._pending_turn: tuple[str, str] | None = None
        self._service_done_metrics: dict[str, Any] = {}
        self._last_llm_metrics: dict[str, Any] = {
            "token_steps": 0,
            "emitted_chunks": 0,
            "decode_s": 0.0,
            "tok_per_s": 0.0,
            "client_s": 0.0,
        }
        self._last_stream_was_partial = False
        self._llm_total_timeout_s = float(os.getenv("PIPECAT_LLM_TOTAL_TIMEOUT_S", "120"))
        self._llm_first_token_timeout_s = float(
            os.getenv("PIPECAT_LLM_FIRST_TOKEN_TIMEOUT_S", "30")
        )
        self._llm_stream_idle_timeout_s = float(
            os.getenv("PIPECAT_LLM_STREAM_IDLE_TIMEOUT_S", "8")
        )
        self._llm_allow_partial_on_idle_timeout = (
            os.getenv("PIPECAT_LLM_ALLOW_PARTIAL_ON_IDLE_TIMEOUT", "1") == "1"
        )
        self._llm_max_attempts = max(1, int(os.getenv("PIPECAT_LLM_MAX_ATTEMPTS", "2")))
        self._llm_retry_backoff_s = float(os.getenv("PIPECAT_LLM_RETRY_BACKOFF_S", "0.35"))
        self._max_history_turns = int(os.getenv("PIPECAT_MAX_HISTORY_TURNS", "8"))
        self._max_assistant_history_turns = max(
            0, int(os.getenv("PIPECAT_MAX_ASSISTANT_HISTORY_TURNS", "0"))
        )
        self._response_first_line_only = os.getenv("PIPECAT_RESPONSE_FIRST_LINE_ONLY", "0") == "1"
        self._response_first_sentence_only = (
            os.getenv("PIPECAT_RESPONSE_FIRST_SENTENCE_ONLY", "1") == "1"
        )
        self._response_max_chars = int(os.getenv("PIPECAT_RESPONSE_MAX_CHARS", "120"))
        self._response_clause_soft_limit = int(
            os.getenv("PIPECAT_RESPONSE_CLAUSE_SOFT_LIMIT", "80")
        )
        self._max_repeat_sentence_run = int(os.getenv("PIPECAT_MAX_REPEAT_SENTENCE_RUN", "3"))

    def _build_prompt(self, user_text: str) -> str:
        return self._build_prompt_with_options(user_text)

    def _build_prompt_with_options(
        self,
        user_text: str,
        *,
        drop_history: bool = False,
        extra_instruction: str | None = None,
    ) -> str:
        turns = [] if drop_history else self._history[-self._max_history_turns :]
        if self._max_assistant_history_turns >= 0:
            kept_assistants = 0
            filtered_reversed: list[tuple[str, str]] = []
            for role, content in reversed(turns):
                if role == "assistant":
                    if kept_assistants >= self._max_assistant_history_turns:
                        continue
                    kept_assistants += 1
                filtered_reversed.append((role, content))
            turns = list(reversed(filtered_reversed))
        lines = [
            f"System: {self._system_prompt}",
            "System: Respond directly in one short sentence. Do not narrate the transcript or your reasoning.",
            "System: Prefer a simple subject-verb sentence, such as 'It is blue.' or 'My favorite color is blue.', over noun-first phrasing.",
        ]
        if extra_instruction:
            lines.append(f"System: {extra_instruction}")
        last_assistant = next(
            (content for role, content in reversed(turns) if role == "assistant" and content.strip()),
            None,
        )
        if last_assistant:
            lines.append("System: Do not repeat your previous assistant answer verbatim.")
        for role, content in turns:
            lines.append(f"{role.capitalize()}: {content}")
        lines.append(f"User: {user_text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    async def _stream_response_chunks(
        self,
        prompt: str,
        *,
        first_token_timeout_s: float | None = None,
        stream_idle_timeout_s: float | None = None,
    ):
        self._last_stream_was_partial = False
        first_token_timeout = (
            self._llm_first_token_timeout_s if first_token_timeout_s is None else first_token_timeout_s
        )
        stream_idle_timeout = (
            self._llm_stream_idle_timeout_s if stream_idle_timeout_s is None else stream_idle_timeout_s
        )
        payload = {"prompt": prompt, "max_tokens": self._max_tokens}
        async with self._session.post(
            f"{self._base_url}/generate",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=aiohttp.ClientTimeout(total=self._llm_total_timeout_s),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"LLM service returned HTTP {resp.status}")

            stream_iter = resp.content.__aiter__()
            saw_token = False
            while True:
                timeout_s = (
                    first_token_timeout
                    if not saw_token
                    else stream_idle_timeout
                )
                try:
                    raw_line = await asyncio.wait_for(
                        stream_iter.__anext__(),
                        timeout=timeout_s,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    if saw_token and self._llm_allow_partial_on_idle_timeout:
                        self._last_stream_was_partial = True
                        break
                    phase = "first token" if not saw_token else "stream"
                    raise RuntimeError(
                        f"LLM {phase} idle timeout ({timeout_s:.1f}s)"
                    ) from exc
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
                    done_metrics = data.get("metrics")
                    if isinstance(done_metrics, dict):
                        self._service_done_metrics = done_metrics
                    break
                if "error" in data:
                    raise RuntimeError(f"LLM service error: {data['error']}")
                if "token" in data:
                    saw_token = True
                    yield data["token"]

    @staticmethod
    def _ends_with_sentence_punctuation(text: str) -> bool:
        return text.rstrip().endswith((".", "!", "?"))

    @staticmethod
    def _normalize_response(text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _is_low_quality_assistant_response(self, text: str) -> bool:
        normalized = self._normalize_response(text)
        if not normalized:
            return True
        bad_prefixes = (
            "i don't remember",
            "i do not remember",
            "i don't see",
            "i do not see",
            "i am not sure",
        )
        if normalized.startswith(bad_prefixes):
            return True
        words = normalized.split()
        if len(words) >= 3:
            run = 1
            best_run = 1
            for idx in range(1, len(words)):
                if words[idx] == words[idx - 1]:
                    run += 1
                    best_run = max(best_run, run)
                else:
                    run = 1
            if best_run >= 3:
                return True
            for ngram_size in (2, 3):
                if len(words) >= ngram_size * 3:
                    pattern = words[-ngram_size:]
                    repeats = 1
                    while len(words) >= (repeats + 1) * ngram_size:
                        start = len(words) - (repeats + 1) * ngram_size
                        end = start + ngram_size
                        if words[start:end] != pattern:
                            break
                        repeats += 1
                    if repeats >= 3:
                        return True
        if len(words) >= 8:
            tail = " ".join(words[-4:])
            if normalized.count(tail) >= 2:
                return True
        return False

    def _postprocess_response(self, text: str, stop_markers: tuple[str, ...]) -> str:
        response = text
        stop_pos = min(
            (idx for idx in (response.find(m) for m in stop_markers) if idx != -1),
            default=-1,
        )
        if stop_pos != -1:
            response = response[:stop_pos]
        if self._response_first_line_only:
            first_line = response.splitlines()[0].strip() if response else ""
            if first_line:
                response = first_line
        if self._response_first_sentence_only and response:
            for i, ch in enumerate(response):
                if ch in ".?!":
                    response = response[: i + 1].strip()
                    break
        if (
            self._response_clause_soft_limit > 0
            and len(response) > self._response_clause_soft_limit
        ):
            for sep in (",", ";", ":"):
                sep_pos = response.find(sep)
                if 24 <= sep_pos <= self._response_clause_soft_limit:
                    response = response[:sep_pos].rstrip(" \t\r\n,;:-")
                    break
        if self._response_max_chars > 0 and len(response) > self._response_max_chars:
            if not self._response_first_sentence_only:
                response = response[: self._response_max_chars]
            elif self._ends_with_sentence_punctuation(response[: self._response_max_chars]):
                response = response[: self._response_max_chars]
        return response.rstrip()

    async def generate(self, user_text: str) -> str:
        last_exc: Exception | None = None
        parts: list[str] = []
        emitted_chunks = 0
        stop_markers = ("\nUser:", "\nAssistant:", "\nSystem:")
        started = time.perf_counter()
        response = ""
        recovery_mode = False
        for attempt in range(self._llm_max_attempts):
            prompt = self._build_prompt_with_options(
                user_text,
                drop_history=recovery_mode,
                extra_instruction=(
                    "Answer the current user request naturally in 2 to 8 words. "
                    "Do not repeat any word or phrase."
                    if recovery_mode
                    else None
                ),
            )
            parts = []
            emitted_chunks = 0
            self._service_done_metrics = {}
            try:
                first_timeout = self._llm_first_token_timeout_s * (1.0 + 0.5 * attempt)
                stream_idle_timeout = self._llm_stream_idle_timeout_s * (1.0 + 0.5 * attempt)
                async for chunk in self._stream_response_chunks(
                    prompt,
                    first_token_timeout_s=first_timeout,
                    stream_idle_timeout_s=stream_idle_timeout,
                ):
                    parts.append(chunk)
                    emitted_chunks += 1
                candidate = self._postprocess_response("".join(parts), stop_markers)
                if (
                    emitted_chunks > 0
                    and self._last_stream_was_partial
                    and not self._ends_with_sentence_punctuation(candidate)
                    and attempt < self._llm_max_attempts - 1
                ):
                    last_exc = RuntimeError(
                        "LLM stream ended mid-sentence; retrying for a complete sentence."
                    )
                    await asyncio.sleep(self._llm_retry_backoff_s * (attempt + 1))
                    continue
                response = candidate
                if self._is_low_quality_assistant_response(response):
                    recovery_mode = True
                    last_exc = RuntimeError(
                        "LLM produced a repetitive or low-quality response; retrying."
                    )
                    if attempt < self._llm_max_attempts - 1:
                        await asyncio.sleep(self._llm_retry_backoff_s * (attempt + 1))
                        continue
                if emitted_chunks > 0 or attempt == self._llm_max_attempts - 1:
                    break
            except Exception as exc:
                last_exc = exc
                recovery_mode = True
                if attempt == self._llm_max_attempts - 1:
                    raise
                await asyncio.sleep(self._llm_retry_backoff_s * (attempt + 1))
                continue
        if emitted_chunks == 0 and last_exc is not None:
            raise last_exc

        client_s = max(time.perf_counter() - started, 1e-9)

        service_metrics = self._service_done_metrics
        token_steps = int(service_metrics.get("token_steps", emitted_chunks))
        decode_s = float(service_metrics.get("decode_s", client_s))
        tok_per_s = float(service_metrics.get("tok_per_s", _safe_div(float(token_steps), decode_s)))
        self._last_llm_metrics = {
            "token_steps": token_steps,
            "emitted_chunks": int(service_metrics.get("emitted_chunks", emitted_chunks)),
            "decode_s": decode_s,
            "tok_per_s": tok_per_s,
            "client_s": float(client_s),
        }
        if not response:
            response = self._postprocess_response("".join(parts), stop_markers)
        if (
            self._response_first_sentence_only
            and response
            and not self._ends_with_sentence_punctuation(response)
        ):
            response = response.rstrip(" \t\r\n,;:-")
            if response and not self._ends_with_sentence_punctuation(response):
                response = f"{response}."
        return response.rstrip()

    def stage_turn(self, user_text: str, assistant_text: str) -> None:
        self._pending_turn = (user_text, assistant_text)

    def commit_staged_turn(self, spoken_text: str | None = None) -> None:
        if not self._pending_turn:
            return
        user_text, assistant_text = self._pending_turn
        final_assistant = (spoken_text or assistant_text).strip() or assistant_text
        normalized_final = self._normalize_response(final_assistant)
        assistant_history = [
            self._normalize_response(content)
            for role, content in self._history
            if role == "assistant"
        ]
        last_assistant = assistant_history[-1] if assistant_history else None
        if last_assistant and (
            last_assistant == normalized_final
        ):
            # Avoid reinforcing same-response loops in subsequent prompts.
            self._history.append(("user", user_text))
            self._pending_turn = None
            return
        if self._is_low_quality_assistant_response(final_assistant):
            # Keep user context but drop low-quality assistant replies to avoid
            # progressive degradation over turns.
            self._history.append(("user", user_text))
            self._pending_turn = None
            return
        if assistant_history[-3:].count(normalized_final) >= 2:
            # Hard reset of assistant context when repeated loops begin.
            user_only = [(role, content) for role, content in self._history if role == "user"]
            self._history = user_only[-2:]
            self._history.append(("user", user_text))
            self._pending_turn = None
            return
        self._history.append(("user", user_text))
        self._history.append(("assistant", final_assistant))
        self._pending_turn = None

    def drop_staged_turn(self) -> None:
        self._pending_turn = None

    def get_last_metrics(self) -> dict[str, Any]:
        return dict(self._last_llm_metrics)


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
        self._tts_tokens_per_char = float(os.getenv("PIPECAT_TTS_TOKENS_PER_CHAR", "1.0"))
        self._tts_punct_bonus = int(os.getenv("PIPECAT_TTS_PUNCT_BONUS", "2"))
        self._tts_min_new_tokens = int(os.getenv("PIPECAT_TTS_MIN_NEW_TOKENS", "80"))
        self._tts_medium_text_char_threshold = int(
            os.getenv("PIPECAT_TTS_MEDIUM_TEXT_CHAR_THRESHOLD", "40")
        )
        self._tts_short_text_char_threshold = int(
            os.getenv("PIPECAT_TTS_SHORT_TEXT_CHAR_THRESHOLD", "48")
        )
        self._tts_min_new_tokens_short = int(
            os.getenv("PIPECAT_TTS_MIN_NEW_TOKENS_SHORT", "28")
        )
        self._tts_tiny_text_char_threshold = int(
            os.getenv("PIPECAT_TTS_TINY_TEXT_CHAR_THRESHOLD", "12")
        )
        self._tts_min_new_tokens_tiny = int(
            os.getenv("PIPECAT_TTS_MIN_NEW_TOKENS_TINY", "16")
        )
        self._tts_api_min_new_tokens = int(
            os.getenv("PIPECAT_TTS_API_MIN_NEW_TOKENS", "64")
        )
        self._tts_sentence_min_new_tokens = int(
            os.getenv("PIPECAT_TTS_SENTENCE_MIN_NEW_TOKENS", "160")
        )
        self._tts_normalize_numbers = os.getenv("PIPECAT_TTS_NORMALIZE_NUMBERS", "1") == "1"
        self._tts_max_normalize_number = int(
            os.getenv("PIPECAT_TTS_MAX_NORMALIZE_NUMBER", "9999")
        )
        self._tts_strip_punctuation = os.getenv("PIPECAT_TTS_STRIP_PUNCTUATION", "1") == "1"
        self._tts_lowercase_text = os.getenv("PIPECAT_TTS_LOWERCASE_TEXT", "1") == "1"
        self._last_tts_had_audio = False
        self._last_tts_metrics: dict[str, Any] = {}
        self._current_turn_context: dict[str, Any] = {}
        self._on_tts_started = None
        self._on_tts_stopped = None

    def can_generate_metrics(self) -> bool:
        return True

    def _estimate_max_new_tokens(self, text: str) -> int:
        if not self._dynamic_max_new_tokens:
            return self._max_new_tokens

        stripped = text.strip()
        char_count = len(stripped)
        punct = sum(stripped.count(ch) for ch in ".!?;,")
        sentence_boundary_count = sum(stripped.count(ch) for ch in ".!?")
        estimate = self._tts_token_base + int(char_count * self._tts_tokens_per_char)
        estimate += punct * self._tts_punct_bonus

        min_floor = self._tts_min_new_tokens
        if char_count <= self._tts_short_text_char_threshold:
            min_floor = min(min_floor, self._tts_min_new_tokens_short)
        if char_count <= self._tts_tiny_text_char_threshold:
            min_floor = min(min_floor, self._tts_min_new_tokens_tiny)
        if sentence_boundary_count > 0 and char_count >= self._tts_medium_text_char_threshold:
            min_floor = max(min_floor, self._tts_sentence_min_new_tokens)
        min_floor = max(self._tts_api_min_new_tokens, min_floor)

        return max(self._tts_api_min_new_tokens, max(min_floor, min(self._max_new_tokens, estimate)))

    @staticmethod
    def _int_to_words(value: int) -> str:
        ones = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        scales = [
            (1_000_000_000, "billion"),
            (1_000_000, "million"),
            (1_000, "thousand"),
        ]

        if value < 20:
            return ones[value]
        if value < 100:
            t, r = divmod(value, 10)
            return tens[t] if r == 0 else f"{tens[t]} {ones[r]}"
        if value < 1000:
            h, r = divmod(value, 100)
            return f"{ones[h]} hundred" if r == 0 else f"{ones[h]} hundred {LocalQwenTTSService._int_to_words(r)}"
        for scale_value, scale_name in scales:
            if value >= scale_value:
                left, right = divmod(value, scale_value)
                left_words = LocalQwenTTSService._int_to_words(left)
                if right == 0:
                    return f"{left_words} {scale_name}"
                return f"{left_words} {scale_name} {LocalQwenTTSService._int_to_words(right)}"
        return str(value)

    @staticmethod
    def _digits_to_words(token: str) -> str:
        digit_words = {
            "0": "zero",
            "1": "one",
            "2": "two",
            "3": "three",
            "4": "four",
            "5": "five",
            "6": "six",
            "7": "seven",
            "8": "eight",
            "9": "nine",
        }
        return " ".join(digit_words[ch] for ch in token if ch in digit_words)

    def _number_match_to_words(self, match: re.Match[str]) -> str:
        token = match.group(0)
        cleaned = token.replace(",", "")
        negative = cleaned.startswith("-")
        if negative:
            cleaned = cleaned[1:]
        if not cleaned:
            return token

        if "." in cleaned:
            whole, frac = cleaned.split(".", 1)
            whole_val = int(whole) if whole else 0
            whole_words = self._int_to_words(abs(whole_val))
            frac_words = self._digits_to_words(frac)
            number_words = f"{whole_words} point {frac_words}".strip()
        else:
            try:
                value = int(cleaned)
            except ValueError:
                return token
            abs_value = abs(value)
            if abs_value <= self._tts_max_normalize_number:
                number_words = self._int_to_words(abs_value)
            else:
                number_words = self._digits_to_words(cleaned)

        if negative:
            return f"minus {number_words}"
        return number_words

    def _sanitize_tts_text(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return cleaned
        cleaned = cleaned.replace("\u2019", "'")
        contraction_rewrites = (
            (r"\bain't\b", "is not"),
            (r"\baren't\b", "are not"),
            (r"\bcan't\b", "cannot"),
            (r"\bcouldn't\b", "could not"),
            (r"\bdidn't\b", "did not"),
            (r"\bdoesn't\b", "does not"),
            (r"\bdon't\b", "do not"),
            (r"\bhaven't\b", "have not"),
            (r"\bhe's\b", "he is"),
            (r"\bi'm\b", "i am"),
            (r"\bisn't\b", "is not"),
            (r"\bit's\b", "it is"),
            (r"\blet's\b", "let us"),
            (r"\bshe's\b", "she is"),
            (r"\bthat's\b", "that is"),
            (r"\bthere's\b", "there is"),
            (r"\bthey're\b", "they are"),
            (r"\bwe're\b", "we are"),
            (r"\bwhat's\b", "what is"),
            (r"\bwho's\b", "who is"),
            (r"\bwon't\b", "will not"),
            (r"\bwouldn't\b", "would not"),
            (r"\byou're\b", "you are"),
            (r"\b([A-Za-z]+)'ll\b", r"\1 will"),
            (r"\b([A-Za-z]+)'ve\b", r"\1 have"),
            (r"\b([A-Za-z]+)'d\b", r"\1 would"),
            (r"\b([A-Za-z]+)n't\b", r"\1 not"),
        )
        for pattern, replacement in contraction_rewrites:
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        if self._tts_normalize_numbers:
            cleaned = re.sub(
                r"(?<!\w)-?\d[\d,]*(?:\.\d+)?(?!\w)",
                self._number_match_to_words,
                cleaned,
            )
        if self._tts_strip_punctuation:
            cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if self._tts_lowercase_text:
            cleaned = cleaned.lower()
        # Some short arithmetic-result phrasings are unstable in the local
        # Qwen3-TTS talker path (e.g. "three plus three is six"), while a
        # semantically equivalent answer form is reliable. Rewrite only this
        # narrow pattern for speech without changing the upstream LLM/output
        # pipeline.
        match = re.fullmatch(
            r"(?P<lhs>.+?)\s+(?:is|equals)\s+(?P<rhs>[a-z0-9][a-z0-9\s-]*)",
            cleaned,
        )
        if match:
            lhs = match.group("lhs").strip()
            rhs = match.group("rhs").strip()
            arithmetic_markers = (
                " plus ",
                " minus ",
                " times ",
                " multiplied by ",
                " divided by ",
                " over ",
            )
            if any(marker in f" {lhs} " for marker in arithmetic_markers):
                cleaned = f"the answer is {rhs}"
        # Short compliment/acknowledgment replies are another fragile shape in
        # the local talker path. Rewrite only the unstable compliment forms to
        # a semantically equivalent acknowledgment that speaks reliably.
        compliment_match = re.fullmatch(
            r"you\s+(?:are|re)\s+(?:(?:very|so|really)\s+)?(?P<adj>[a-z][a-z\s-]*)",
            cleaned,
        )
        if compliment_match:
            compliment_adj = compliment_match.group("adj").strip()
            fragile_compliments = {
                "beautiful",
                "lovely",
                "kind",
                "very kind",
                "so kind",
                "really kind",
                "sweet",
                "nice",
                "wonderful",
                "amazing",
                "great",
                "awesome",
                "pretty",
                "handsome",
                "cute",
                "brilliant",
            }
            if compliment_adj in fragile_compliments:
                cleaned = "thank you that is kind of you"
        if cleaned == "that is kind of you":
            cleaned = "thank you that is kind of you"
        return cleaned

    @staticmethod
    def _parse_float_header(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _pcm16_is_effectively_silent(audio_bytes: bytes, rms_threshold: float) -> bool:
        if not audio_bytes:
            return True
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)
        if pcm.size == 0:
            return True
        pcm_f32 = pcm.astype(np.float32, copy=False) / 32767.0
        rms = float(np.sqrt(np.mean(np.square(pcm_f32))))
        peak = float(np.max(np.abs(pcm_f32)))
        return rms < rms_threshold and peak < (rms_threshold * 3.0)

    def get_last_metrics(self) -> dict[str, Any]:
        return dict(self._last_tts_metrics)

    def set_turn_state_callbacks(self, *, on_started=None, on_stopped=None) -> None:
        self._on_tts_started = on_started
        self._on_tts_stopped = on_stopped

    def set_turn_context(
        self,
        *,
        user_text: str,
        assistant_text: str,
        llm_metrics: dict[str, Any],
        turn_started_at: float,
    ) -> None:
        self._current_turn_context = {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "llm_metrics": dict(llm_metrics),
            "turn_started_at": turn_started_at,
        }

    @staticmethod
    def _fmt_float(value: Any, digits: int = 3) -> str:
        if value is None:
            return "n/a"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "n/a"

    def _emit_turn_metrics(self) -> None:
        turn_ctx = dict(self._current_turn_context)
        tts_metrics = self.get_last_metrics()
        llm_metrics = turn_ctx.get("llm_metrics", {})

        turn_started_at = turn_ctx.get("turn_started_at")
        overall_latency_ms: float | None = None
        if isinstance(turn_started_at, float):
            overall_latency_ms = (time.perf_counter() - turn_started_at) * 1000.0

        llm_tok_per_s = llm_metrics.get("tok_per_s")
        llm_token_steps = llm_metrics.get("token_steps")
        llm_decode_s = llm_metrics.get("decode_s")
        ttfc_ms = tts_metrics.get("ttfc_ms")
        rtf = tts_metrics.get("rtf")
        audio_s = tts_metrics.get("audio_s")
        user_chars = len(str(turn_ctx.get("user_text", "")))
        assistant_chars = len(str(turn_ctx.get("assistant_text", "")))
        streaming_mode = tts_metrics.get("streaming_mode", "unknown")
        chunk_count = int(tts_metrics.get("chunk_count", 0) or 0)
        frame_by_frame = chunk_count > 1
        max_chunk_gap_ms = tts_metrics.get("max_chunk_gap_ms")
        audio_quality_ok = bool(tts_metrics.get("audio_quality_ok", False))
        dropped_frames_suspected = bool(tts_metrics.get("dropped_frames_suspected", False))
        ttfc_target_ms = float(os.getenv("PIPECAT_TARGET_TTFC_MS", "50"))
        rtf_target = float(os.getenv("PIPECAT_TARGET_RTF", "0.1"))

        print(
            "[metrics][roundtrip] "
            f"overall_ms={self._fmt_float(overall_latency_ms, 1)} "
            f"llm_tok_s={self._fmt_float(llm_tok_per_s, 1)} "
            f"llm_token_steps={llm_token_steps if llm_token_steps is not None else 'n/a'} "
            f"llm_decode_s={self._fmt_float(llm_decode_s, 3)} "
            f"ttfc_ms={self._fmt_float(ttfc_ms, 1)} "
            f"rtf={self._fmt_float(rtf, 3)} "
            f"audio_s={self._fmt_float(audio_s, 3)} "
            f"stt_chars={user_chars} "
            f"assistant_chars={assistant_chars}"
        )
        print(
            "[metrics][stream] "
            f"mode={streaming_mode} "
            f"frame_by_frame={'yes' if frame_by_frame else 'no'} "
            f"chunk_count={chunk_count} "
            f"decode_stride={tts_metrics.get('decode_stride', 'unknown')} "
            f"max_new_tokens_effective={tts_metrics.get('max_new_tokens_effective', 'unknown')} "
            f"stop_reason={tts_metrics.get('stop_reason', 'unknown')}"
        )
        print(
            "[metrics][quality] "
            f"audio_ok={'yes' if audio_quality_ok else 'no'} "
            f"dropped_frames_suspected={'yes' if dropped_frames_suspected else 'no'} "
            f"max_chunk_gap_ms={self._fmt_float(max_chunk_gap_ms, 2)} "
            f"error={tts_metrics.get('error') or 'none'}"
        )
        if ttfc_ms is not None and float(ttfc_ms) > ttfc_target_ms:
            print(
                f"[metrics][warn] TTFC target miss: {self._fmt_float(ttfc_ms, 1)}ms > "
                f"{self._fmt_float(ttfc_target_ms, 1)}ms"
            )
        if rtf is not None and float(rtf) > rtf_target:
            print(
                f"[metrics][warn] RTF target miss: {self._fmt_float(rtf, 3)} > "
                f"{self._fmt_float(rtf_target, 3)}"
            )
        if not frame_by_frame:
            print("[metrics][warn] Streaming may be buffered (chunk_count <= 1).")
        if not audio_quality_ok:
            print("[metrics][warn] Audio quality check failed (glitch/drop suspected).")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        headers = {"Content-Type": "application/json"}
        cleaned_text = self._sanitize_tts_text(text)
        started_frame_sent = False
        audio_frames_emitted = 0
        last_error: str | None = None
        self._last_tts_had_audio = False
        self._last_tts_metrics = {
            "had_audio": False,
            "ttfc_ms": None,
            "audio_s": 0.0,
            "rtf": None,
            "tts_stream_s": 0.0,
            "chunk_count": 0,
            "max_chunk_gap_ms": 0.0,
            "streaming_mode": "unknown",
            "decode_stride": "unknown",
            "max_new_tokens_effective": "unknown",
            "stop_reason": "unknown",
            "audio_quality_ok": False,
            "dropped_frames_suspected": False,
            "error": None,
        }
        try:
            await self.start_ttfb_metrics()
            text_len = len(cleaned_text)
            max_http_attempts = max(1, int(os.getenv("PIPECAT_TTS_HTTP_MAX_ATTEMPTS", "2")))
            for attempt in range(max_http_attempts):
                effective_max_new_tokens = self._estimate_max_new_tokens(cleaned_text)
                payload = {
                    "text": cleaned_text,
                    "voice": self._voice,
                    "max_new_tokens": effective_max_new_tokens,
                }
                # Scale timeout budgets with text complexity to avoid dropping longer answers.
                base_first_chunk_timeout_s = float(os.getenv("PIPECAT_TTS_FIRST_CHUNK_TIMEOUT_S", "8.0"))
                base_stream_idle_timeout_s = float(os.getenv("PIPECAT_TTS_STREAM_IDLE_TIMEOUT_S", "16.0"))
                adaptive_first_chunk_timeout_s = min(30.0, 2.0 + (text_len * 0.03))
                adaptive_stream_idle_timeout_s = min(45.0, 6.0 + (text_len * 0.04))
                first_chunk_timeout_s = max(base_first_chunk_timeout_s, adaptive_first_chunk_timeout_s)
                stream_idle_timeout_s = max(base_stream_idle_timeout_s, adaptive_stream_idle_timeout_s)
                attempt_started = time.perf_counter()
                raw_bytes = 0
                raw_chunk_count = 0
                first_chunk_at: float | None = None
                last_chunk_at: float | None = None
                max_chunk_gap_s = 0.0
                ttfc_header_ms: float | None = None
                streaming_mode = "unknown"
                decode_stride = "unknown"
                max_new_tokens_effective = str(effective_max_new_tokens)
                stop_reason = "unknown"
                timed_out_mid_stream = False
                read_chunk_bytes = int(
                    os.getenv(
                        "PIPECAT_TTS_HTTP_READ_CHUNK_BYTES",
                        "960",
                    )
                )
                read_chunk_bytes = max(512, read_chunk_bytes - (read_chunk_bytes % 2))
                strip_leading_silence = (
                    os.getenv("PIPECAT_TTS_STRIP_LEADING_SILENCE", "0") == "1"
                )
                leading_silence_rms = float(
                    os.getenv("PIPECAT_TTS_LEADING_SILENCE_RMS", "0.0015")
                )
                max_leading_silence_bytes = int(
                    os.getenv(
                        "PIPECAT_TTS_MAX_LEADING_SILENCE_BYTES",
                        str(int(self._in_sample_rate * 2 * 2.0)),
                    )
                )
                try:
                    async with self._session.post(
                        f"{self._base_url}/synthesize_binary",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=240),
                    ) as response:
                        if response.status != 200:
                            error = await response.text()
                            last_error = (
                                f"Qwen TTS HTTP {response.status} from "
                                f"{self._base_url}/synthesize_binary: {error}"
                            )
                            continue

                        ttfc_header_ms = self._parse_float_header(response.headers.get("X-TTFC-Ms"))
                        streaming_mode = response.headers.get("X-Streaming-Mode", "unknown")
                        decode_stride = response.headers.get("X-Decode-Stride", "unknown")
                        max_new_tokens_effective = response.headers.get(
                            "X-Max-New-Tokens-Effective",
                            str(effective_max_new_tokens),
                        )
                        stop_reason = response.headers.get("X-Stop-Reason", "unknown")

                        async def counted_chunks():
                            nonlocal raw_bytes, raw_chunk_count, first_chunk_at, last_chunk_at, max_chunk_gap_s
                            nonlocal timed_out_mid_stream
                            chunk_iter = response.content.iter_chunked(read_chunk_bytes).__aiter__()
                            skipped_leading_silence_bytes = 0
                            emitted_audio = False
                            while True:
                                timeout_s = first_chunk_timeout_s if raw_chunk_count == 0 else stream_idle_timeout_s
                                try:
                                    chunk = await asyncio.wait_for(chunk_iter.__anext__(), timeout=timeout_s)
                                except StopAsyncIteration:
                                    break
                                except asyncio.TimeoutError as exc:
                                    if raw_chunk_count == 0:
                                        raise RuntimeError(
                                            f"TTS stream stalled waiting for first chunk ({timeout_s:.1f}s)."
                                        ) from exc
                                    # Treat a late idle timeout as stream end, not hard failure.
                                    timed_out_mid_stream = True
                                    break
                                if not chunk:
                                    continue
                                if strip_leading_silence and not emitted_audio:
                                    if (
                                        skipped_leading_silence_bytes < max_leading_silence_bytes
                                        and self._pcm16_is_effectively_silent(chunk, leading_silence_rms)
                                    ):
                                        skipped_leading_silence_bytes += len(chunk)
                                        continue
                                    emitted_audio = True
                                now = time.perf_counter()
                                if first_chunk_at is None:
                                    first_chunk_at = now
                                if last_chunk_at is not None:
                                    max_chunk_gap_s = max(max_chunk_gap_s, now - last_chunk_at)
                                last_chunk_at = now
                                raw_bytes += len(chunk)
                                raw_chunk_count += 1
                                yield chunk

                        await self.start_tts_usage_metrics(cleaned_text)
                        async for frame in self._stream_audio_frames_from_iterator(
                            counted_chunks(),
                            in_sample_rate=self._in_sample_rate,
                            context_id=context_id,
                        ):
                            if not started_frame_sent:
                                if self._on_tts_started:
                                    try:
                                        self._on_tts_started()
                                    except Exception:
                                        pass
                                yield TTSStartedFrame(context_id=context_id)
                                started_frame_sent = True
                            await self.stop_ttfb_metrics()
                            audio_frames_emitted += 1
                            yield frame

                        tts_stream_s = max(time.perf_counter() - attempt_started, 0.0)
                        audio_s = raw_bytes / float(max(1, self._in_sample_rate * 2))
                        ttfc_observed_ms = (
                            (first_chunk_at - attempt_started) * 1000.0 if first_chunk_at is not None else None
                        )
                        ttfc_ms = ttfc_header_ms if ttfc_header_ms is not None else ttfc_observed_ms
                        rtf = _safe_div(tts_stream_s, audio_s) if audio_s > 0 else None
                        nominal_chunk_s = _safe_div(float(self.chunk_size), float(self._in_sample_rate))
                        max_gap_tol_mult = float(os.getenv("PIPECAT_TTS_MAX_GAP_TOL_MULT", "8.0"))
                        dropped_frames_suspected = (
                            raw_chunk_count > 1 and max_chunk_gap_s > (nominal_chunk_s * max_gap_tol_mult)
                        )
                        audio_quality_ok = (
                            audio_frames_emitted > 0
                            and raw_bytes > 0
                            and (raw_bytes % 2 == 0)
                            and not dropped_frames_suspected
                        )
                        self._last_tts_metrics = {
                            "had_audio": audio_frames_emitted > 0 and raw_bytes > 0,
                            "ttfc_ms": ttfc_ms,
                            "audio_s": float(audio_s),
                            "rtf": float(rtf) if rtf is not None else None,
                            "tts_stream_s": float(tts_stream_s),
                            "chunk_count": int(raw_chunk_count),
                            "max_chunk_gap_ms": float(max_chunk_gap_s * 1000.0),
                            "streaming_mode": streaming_mode,
                            "decode_stride": decode_stride,
                            "max_new_tokens_effective": max_new_tokens_effective,
                            "stop_reason": stop_reason,
                            "audio_quality_ok": bool(audio_quality_ok),
                            "dropped_frames_suspected": bool(dropped_frames_suspected),
                            "error": (
                                f"idle-timeout after partial stream ({stream_idle_timeout_s:.1f}s)"
                                if timed_out_mid_stream
                                else None
                            ),
                        }
                except Exception as exc:
                    last_error = f"Qwen TTS request failed: {exc}"
                    self._last_tts_metrics["error"] = last_error
                    if attempt < max_http_attempts - 1:
                        await asyncio.sleep(0.15 * (attempt + 1))
                        continue

                if audio_frames_emitted > 0:
                    self._last_tts_had_audio = True
                    break

                last_error = last_error or "Qwen TTS returned no audio frames."
                self._last_tts_metrics["error"] = last_error
                if attempt < max_http_attempts - 1:
                    await asyncio.sleep(0.15 * (attempt + 1))
            if audio_frames_emitted == 0:
                yield ErrorFrame(error=last_error or "Qwen TTS returned no audio frames.")
        except Exception as exc:
            yield ErrorFrame(error=f"Qwen TTS request failed: {exc}")
        finally:
            await self.stop_ttfb_metrics()
            try:
                self._emit_turn_metrics()
            except Exception as metrics_exc:
                print(f"[metrics][error] failed to emit turn metrics: {metrics_exc}")
            self._current_turn_context = {}
            if self._on_tts_stopped:
                try:
                    self._on_tts_stopped()
                except Exception:
                    pass
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


async def _create_daily_token_for_room(api_key: str, room_name: str) -> str:
    """Create a meeting token for an existing Daily room."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    token_payload = {"properties": {"room_name": room_name, "is_owner": True}}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.daily.co/v1/meeting-tokens",
            json=token_payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=20),
        ) as token_resp:
            if token_resp.status not in (200, 201):
                body = await token_resp.text()
                raise RuntimeError(
                    f"Daily token creation for room '{room_name}' failed "
                    f"({token_resp.status}): {body}"
                )
            token_info = await token_resp.json()

    return token_info["token"]


async def _resolve_daily_session() -> DailySession:
    room_url = os.getenv("DAILY_ROOM_URL")
    room_token = os.getenv("DAILY_ROOM_TOKEN")
    if room_url and room_token:
        return DailySession(room_url=room_url, token=room_token)

    api_key = os.getenv("DAILY_API_KEY")
    if room_url and api_key:
        room_name = os.getenv("DAILY_ROOM_NAME")
        if not room_name:
            room_name = urlparse(room_url).path.strip("/")
        if not room_name:
            raise RuntimeError(
                "Could not infer Daily room name from DAILY_ROOM_URL. "
                "Set DAILY_ROOM_NAME explicitly."
            )
        room_token = await _create_daily_token_for_room(api_key, room_name)
        return DailySession(room_url=room_url, token=room_token)

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
    qwen_tts_max_new_tokens = int(os.getenv("QWEN3_TTS_MAX_NEW_TOKENS", "512"))
    allow_interruptions = os.getenv("PIPECAT_ALLOW_INTERRUPTIONS", "0") == "1"
    stt_end_of_speech_wait_s = float(os.getenv("PIPECAT_STT_END_OF_SPEECH_WAIT_S", "0.8"))

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
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
                allow_interruptions=allow_interruptions,
            ),
        )

        generation_lock = asyncio.Lock()
        assistant_turn_done = asyncio.Event()
        assistant_turn_done.set()
        last_user_turn_marker = 0.0

        def _on_tts_started():
            assistant_turn_done.clear()

        def _on_tts_stopped():
            tts_metrics = tts.get_last_metrics()
            if tts_metrics.get("had_audio") and not tts_metrics.get("error"):
                llm_client.commit_staged_turn()
            else:
                llm_client.drop_staged_turn()
            assistant_turn_done.set()

        tts.set_turn_state_callbacks(on_started=_on_tts_started, on_stopped=_on_tts_stopped)

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def on_user_turn_stopped(_, __, message):
            nonlocal last_user_turn_marker
            if not assistant_turn_done.is_set():
                return
            user_text = (message.content or "").strip()
            if not user_text:
                return
            marker = time.monotonic()
            last_user_turn_marker = marker
            await asyncio.sleep(stt_end_of_speech_wait_s)
            if marker != last_user_turn_marker:
                return
            print(f"[user] {user_text}")
            turn_started_at = time.perf_counter()

            async with generation_lock:
                if not assistant_turn_done.is_set():
                    return
                assistant_turn_done.clear()
                try:
                    reply = await llm_client.generate(user_text)
                except Exception as exc:
                    print(f"[assistant][error] upstream llm error: {exc}")
                    assistant_turn_done.set()
                    return
                reply = tts._sanitize_tts_text(reply)
                if not reply:
                    assistant_turn_done.set()
                    return
                llm_metrics = llm_client.get_last_metrics()
                print(f"[assistant] {reply}")
                llm_client.stage_turn(user_text, reply)
                tts.set_turn_context(
                    user_text=user_text,
                    assistant_text=reply,
                    llm_metrics=llm_metrics,
                    turn_started_at=turn_started_at,
                )
                try:
                    await task.queue_frame(TTSSpeakFrame(reply))
                except Exception:
                    llm_client.drop_staged_turn()
                    assistant_turn_done.set()
                    raise

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
            welcome = tts._sanitize_tts_text(welcome)
            if not welcome:
                welcome = "Hi I am online and ready Ask me anything"
            assistant_turn_done.clear()
            try:
                await task.queue_frame(TTSSpeakFrame(welcome))
            except Exception:
                assistant_turn_done.set()
                raise

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
