"""Python wrapper for streaming generation with qwen_megakernel."""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from typing import Iterator

_KERNEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../kernel"))
if _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)

_DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class MegakernelDecoder:
    """Wrapper around the Qwen megakernel for streaming token generation."""

    def __init__(self):
        self._decoder = None
        self._tokenizer = None
        self._weights_loaded = False
        self._loaded_model_name: str | None = None
        self._lock = threading.Lock()
        self._last_generation_metrics: dict[str, float | int] = {}
        self._same_token_run_limit = max(
            3, int(os.getenv("LLM_SAME_TOKEN_RUN_LIMIT", "4"))
        )
        self._ngram_repeat_limit = max(
            3, int(os.getenv("LLM_NGRAM_REPEAT_LIMIT", "3"))
        )
        self._max_ngram_size = max(
            1, int(os.getenv("LLM_MAX_REPEAT_NGRAM_SIZE", "4"))
        )

    @property
    def loaded_model_name(self) -> str | None:
        return self._loaded_model_name

    @property
    def last_generation_metrics(self) -> dict[str, float | int]:
        return dict(self._last_generation_metrics)

    def _resolve_model_name(self, weights_path: str | None) -> str:
        if weights_path:
            return weights_path
        return os.getenv("QWEN_MEGAKERNEL_MODEL_NAME", _DEFAULT_MODEL)

    def load_weights(self, weights_path: str | None = None):
        """
        Load model weights and initialize a Decoder.

        Args:
            weights_path: Hugging Face model id or local model directory path.
        """
        model_name = self._resolve_model_name(weights_path)
        with self._lock:
            if self._weights_loaded and self._loaded_model_name == model_name:
                return

            from qwen_megakernel.model import Decoder, load_weights

            weights, tokenizer = load_weights(model_name=model_name, verbose=False)
            self._decoder = Decoder(
                weights=weights,
                tokenizer=tokenizer,
                model_name=model_name,
                verbose=False,
            )
            self._tokenizer = tokenizer
            self._weights_loaded = True
            self._loaded_model_name = model_name

    @staticmethod
    def _parse_chat_prompt(prompt: str) -> list[dict[str, str]] | None:
        if "<|im_start|>" in prompt:
            return None

        role_map = {
            "System": "system",
            "User": "user",
            "Assistant": "assistant",
        }
        messages: list[dict[str, str]] = []
        current_role: str | None = None
        current_lines: list[str] = []

        for raw_line in prompt.splitlines():
            match = re.match(r"^(System|User|Assistant):\s?(.*)$", raw_line)
            if match:
                if current_role is not None:
                    content = "\n".join(current_lines).strip()
                    messages.append({"role": current_role, "content": content})
                current_role = role_map[match.group(1)]
                current_lines = [match.group(2)]
                continue
            if current_role is not None:
                current_lines.append(raw_line)

        if current_role is not None:
            content = "\n".join(current_lines).strip()
            messages.append({"role": current_role, "content": content})

        while messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
            messages.pop()

        if not messages or not any(message["role"] == "user" for message in messages):
            return None
        return messages

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized.")

        messages = self._parse_chat_prompt(prompt)
        if messages and hasattr(self._tokenizer, "apply_chat_template"):
            prompt_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            return list(prompt_ids)

        return self._tokenizer.encode(prompt, add_special_tokens=True)

    def _detect_repetition_loop(self, generated_ids: list[int]) -> tuple[int, str] | None:
        if len(generated_ids) >= self._same_token_run_limit:
            tail = generated_ids[-self._same_token_run_limit :]
            if len(set(tail)) == 1:
                return 1, f"same_token_run:{self._same_token_run_limit}"

        max_ngram = min(self._max_ngram_size, len(generated_ids) // self._ngram_repeat_limit)
        for ngram_size in range(2, max_ngram + 1):
            pattern = generated_ids[-ngram_size:]
            repeats = 1
            while len(generated_ids) >= (repeats + 1) * ngram_size:
                start = len(generated_ids) - (repeats + 1) * ngram_size
                end = start + ngram_size
                if generated_ids[start:end] != pattern:
                    break
                repeats += 1
            if repeats >= self._ngram_repeat_limit:
                return ngram_size, f"ngram_repeat:{ngram_size}x{repeats}"

        return None

    def generate_stream(self, prompt: str, max_tokens: int = 100) -> Iterator[str]:
        """
        Generate response text chunks in a streaming fashion.

        Args:
            prompt: Input text prompt.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Incremental text deltas as the model generates tokens.
        """
        if max_tokens <= 0:
            self._last_generation_metrics = {
                "token_steps": 0,
                "emitted_chunks": 0,
                "decode_s": 0.0,
                "tok_per_s": 0.0,
            }
            return

        if not self._weights_loaded:
            self.load_weights()

        with self._lock:
            if self._decoder is None or self._tokenizer is None:
                raise RuntimeError("Decoder is not initialized. Call load_weights() first.")

            prompt_ids = self._tokenize_prompt(prompt)
            if not prompt_ids:
                raise RuntimeError("Prompt tokenization returned no tokens.")

            self._decoder.reset()
            for tid in prompt_ids[:-1]:
                self._decoder.step(tid)

            next_input = prompt_ids[-1]
            eos_id = self._tokenizer.eos_token_id
            token_steps = 0
            emitted_chunks = 0
            started = time.perf_counter()
            generated_ids: list[int] = []
            decoded_so_far = ""
            termination_reason = "max_tokens"

            for _ in range(max_tokens):
                token_id = self._decoder.step(next_input)
                next_input = token_id
                if eos_id is not None and token_id == eos_id:
                    termination_reason = "eos"
                    break

                token_steps += 1
                generated_ids.append(token_id)
                repetition = self._detect_repetition_loop(generated_ids)
                if repetition is not None:
                    trim_tokens, termination_reason = repetition
                    if trim_tokens > 0:
                        del generated_ids[-trim_tokens:]
                    break

                decoded_now = self._tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                # Decode incrementally from the cumulative token stream so
                # streamed deltas match the final decoded text.
                if decoded_now.startswith(decoded_so_far):
                    delta = decoded_now[len(decoded_so_far) :]
                else:
                    # Rare tokenizer normalization edge case; fall back to
                    # emitting the newly available decoded suffix.
                    common = 0
                    max_common = min(len(decoded_so_far), len(decoded_now))
                    while common < max_common and decoded_so_far[common] == decoded_now[common]:
                        common += 1
                    delta = decoded_now[common:]
                decoded_so_far = decoded_now
                if delta:
                    emitted_chunks += 1
                    yield delta

            decode_s = max(time.perf_counter() - started, 1e-9)
            self._last_generation_metrics = {
                "token_steps": int(token_steps),
                "emitted_chunks": int(emitted_chunks),
                "decode_s": float(decode_s),
                "tok_per_s": float(token_steps / decode_s),
                "termination_reason": termination_reason,
            }
