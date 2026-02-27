"""Python wrapper for streaming generation with qwen_megakernel."""

from __future__ import annotations

import os
import sys
import threading
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

    @property
    def loaded_model_name(self) -> str | None:
        return self._loaded_model_name

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
            return

        if not self._weights_loaded:
            self.load_weights()

        with self._lock:
            if self._decoder is None or self._tokenizer is None:
                raise RuntimeError("Decoder is not initialized. Call load_weights() first.")

            prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
            if not prompt_ids:
                raise RuntimeError("Prompt tokenization returned no tokens.")

            self._decoder.reset()
            for tid in prompt_ids[:-1]:
                self._decoder.step(tid)

            next_input = prompt_ids[-1]
            eos_id = self._tokenizer.eos_token_id
            generated_ids: list[int] = []
            emitted_text = ""

            for _ in range(max_tokens):
                token_id = self._decoder.step(next_input)
                next_input = token_id
                if eos_id is not None and token_id == eos_id:
                    break

                generated_ids.append(token_id)
                decoded_text = self._tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                if decoded_text.startswith(emitted_text):
                    delta = decoded_text[len(emitted_text) :]
                else:
                    delta = decoded_text
                emitted_text = decoded_text
                if delta:
                    yield delta
