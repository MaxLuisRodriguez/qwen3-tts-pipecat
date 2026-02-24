"""Python wrapper for the Qwen megakernel decoder."""

from typing import Iterator
import sys
import os

# Add kernel directory to path to import the megakernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../kernel"))


class MegakernelDecoder:
    """Wrapper around the Qwen megakernel for streaming token generation."""
    
    def __init__(self):
        """Initialize the decoder (weights not loaded yet)."""
        self._decoder = None
        self._tokenizer = None
        self._weights_loaded = False
    
    def load_weights(self, weights_path: str):
        """
        Load model weights from the specified path.
        
        Args:
            weights_path: Path to model weights (can be HuggingFace model name or local path)
        
        TODO: Integrate with kernel/qwen_megakernel/model.py load_weights()
        TODO: Store decoder and tokenizer instances for use in generate_stream
        """
        # TODO: Import and call kernel/qwen_megakernel/model.py::load_weights()
        # from qwen_megakernel.model import load_weights, Decoder
        # weights, tokenizer = load_weights(weights_path)
        # self._decoder = Decoder(weights)
        # self._tokenizer = tokenizer
        # self._weights_loaded = True
        
        print(f"[STUB] Would load weights from: {weights_path}")
        self._weights_loaded = True
    
    def generate_stream(self, prompt: str, max_tokens: int = 100) -> Iterator[str]:
        """
        Generate tokens in a streaming fashion.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
        
        Yields:
            Token strings as they are generated
        
        TODO: Replace stub with actual kernel calls:
        1. Tokenize prompt using self._tokenizer
        2. Call self._decoder.generate() or similar in a loop
        3. Yield tokens as they are produced
        """
        if not self._weights_loaded:
            raise RuntimeError("Weights must be loaded before generation. Call load_weights() first.")
        
        # STUB: Yield fake tokens for now
        fake_tokens = ["hello", "world", "this", "is", "a", "test", "stream"]
        for i, token in enumerate(fake_tokens):
            if i >= max_tokens:
                break
            yield token
        
        # TODO: Real implementation:
        # tokens = self._tokenizer.encode(prompt)
        # for token_id in self._decoder.generate(tokens, max_tokens=max_tokens):
        #     token_str = self._tokenizer.decode([token_id])
        #     yield token_str
