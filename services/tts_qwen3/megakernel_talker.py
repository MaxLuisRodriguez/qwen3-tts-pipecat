"""Talker-specific megakernel backend with incremental codec/audio streaming."""

from __future__ import annotations

import contextlib
import math
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class StreamStats:
    ttfc_ms: float | None = None
    generation_s: float = 0.0
    audio_seconds: float = 0.0
    frames_generated: int = 0
    stop_reason: str = "unknown"
    prefill_ms: float = 0.0
    prompt_build_ms: float = 0.0
    prefill_model_ms: float = 0.0
    prefill_cache_ms: float = 0.0
    prefill_mode: str = "pytorch"
    subtalker_ms: float = 0.0
    talker_decode_ms: float = 0.0
    audio_decode_ms: float = 0.0
    subtalker_calls: int = 0
    talker_decode_calls: int = 0
    audio_decode_calls: int = 0
    audio_chunks: int = 0
    first_decode_ms: float | None = None
    kernel_path: str = "qwen_megakernel_C.decode_hidden_fp32_head"
    timing_mode: str = "wall_async"
    audio_decode_overlap: bool = False
    audio_decode_wait_ms: float = 0.0


class IncrementalTokenizerDecoderV2:
    """Stateful incremental decoder for Qwen3-TTS 12 Hz tokenizer output."""

    def __init__(self, speech_tokenizer):
        decoder = getattr(getattr(speech_tokenizer, "model", None), "decoder", None)
        if decoder is None:
            raise ValueError("Speech tokenizer decoder is unavailable for incremental decode.")
        self._decoder = decoder
        self._past_key_values = None
        self._conv_cache: dict[str, torch.Tensor] = {}

    @staticmethod
    def _class_name(module) -> str:
        return module.__class__.__name__

    def _run_causal_conv(self, module, hidden: torch.Tensor, key: str) -> torch.Tensor:
        if getattr(module, "stride", 1) != 1:
            raise ValueError(f"Incremental decode only supports stride=1 causal convs, got {module.stride}")
        pad = int(getattr(module, "padding", 0))
        if pad <= 0:
            return module.conv(hidden).contiguous()

        cached = self._conv_cache.get(key)
        if cached is None:
            cached = hidden.new_zeros(hidden.shape[0], hidden.shape[1], 0)
        work = torch.cat((cached, hidden), dim=-1)
        work = F.pad(work, (pad, 0), mode="constant", value=0)
        out = module.conv(work).contiguous()
        self._conv_cache[key] = torch.cat((cached, hidden), dim=-1)[..., -pad:].contiguous()
        return out[..., -hidden.shape[-1] :]

    def _run_transpose_conv(self, module, hidden: torch.Tensor) -> torch.Tensor:
        hidden = module.conv(hidden)
        right_pad = int(getattr(module, "right_pad", 0))
        if right_pad > 0:
            hidden = hidden[..., : hidden.shape[-1] - right_pad]
        return hidden.contiguous()

    def _run_convnext_block(self, module, hidden: torch.Tensor, key: str) -> torch.Tensor:
        residual = hidden
        hidden = self._run_causal_conv(module.dwconv, hidden, f"{key}.dwconv")
        hidden = hidden.permute(0, 2, 1)
        hidden = module.norm(hidden)
        hidden = module.pwconv1(hidden)
        hidden = module.act(hidden)
        hidden = module.pwconv2(hidden)
        hidden = module.gamma * hidden
        hidden = hidden.permute(0, 2, 1)
        return (residual + hidden).contiguous()

    def _run_residual_unit(self, module, hidden: torch.Tensor, key: str) -> torch.Tensor:
        residual = hidden
        hidden = module.act1(hidden)
        hidden = self._run_causal_conv(module.conv1, hidden, f"{key}.conv1")
        hidden = module.act2(hidden)
        hidden = self._run_causal_conv(module.conv2, hidden, f"{key}.conv2")
        return (hidden + residual).contiguous()

    def _run_decoder_block(self, module, hidden: torch.Tensor, key: str) -> torch.Tensor:
        for idx, block in enumerate(module.block):
            block_key = f"{key}.block{idx}"
            block_name = self._class_name(block)
            if block_name == "SnakeBeta":
                hidden = block(hidden)
            elif block_name == "Qwen3TTSTokenizerV2CausalTransConvNet":
                hidden = self._run_transpose_conv(block, hidden)
            elif block_name == "Qwen3TTSTokenizerV2DecoderDecoderResidualUnit":
                hidden = self._run_residual_unit(block, hidden, block_key)
            else:
                raise ValueError(f"Unsupported decoder block module: {block_name}")
        return hidden

    def decode_new_frames(self, audio_codes: torch.Tensor) -> torch.Tensor:
        if audio_codes.numel() == 0:
            return torch.empty((0,), device=audio_codes.device, dtype=torch.float32)

        if audio_codes.dim() != 2:
            raise ValueError(f"Expected audio codes shape [frames, groups], got {tuple(audio_codes.shape)}")

        codes = audio_codes.transpose(0, 1).unsqueeze(0).contiguous()
        hidden = self._decoder.quantizer.decode(codes)
        hidden = self._run_causal_conv(self._decoder.pre_conv, hidden, "pre_conv").transpose(1, 2)

        transformer_out = self._decoder.pre_transformer(
            inputs_embeds=hidden,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
        self._past_key_values = transformer_out.past_key_values
        hidden = transformer_out.last_hidden_state.permute(0, 2, 1).contiguous()

        for idx, blocks in enumerate(self._decoder.upsample):
            for block_idx, block in enumerate(blocks):
                block_name = self._class_name(block)
                key = f"upsample.{idx}.{block_idx}"
                if block_name == "Qwen3TTSTokenizerV2CausalTransConvNet":
                    hidden = self._run_transpose_conv(block, hidden)
                elif block_name == "Qwen3TTSTokenizerV2ConvNeXtBlock":
                    hidden = self._run_convnext_block(block, hidden, key)
                else:
                    raise ValueError(f"Unsupported upsample module: {block_name}")

        wav = hidden
        for idx, block in enumerate(self._decoder.decoder):
            block_name = self._class_name(block)
            key = f"decoder.{idx}"
            if block_name == "Qwen3TTSTokenizerV2CausalConvNet":
                wav = self._run_causal_conv(block, wav, key)
            elif block_name == "Qwen3TTSTokenizerV2DecoderDecoderBlock":
                wav = self._run_decoder_block(block, wav, key)
            elif block_name == "SnakeBeta":
                wav = block(wav)
            else:
                raise ValueError(f"Unsupported decoder module: {block_name}")

        return wav.clamp(min=-1, max=1).squeeze(0).squeeze(0).to(torch.float32).contiguous()


class TalkerMegakernelBackend:
    """Runs Qwen3-TTS talker decode with megakernel and streams decode-time audio."""

    def __init__(self, qwen_tts_model):
        # Trigger extension build/load.
        import qwen_megakernel  # noqa: F401

        self._decode_from_hidden = torch.ops.qwen_megakernel_C.decode_from_hidden
        self._decode_hidden_only = torch.ops.qwen_megakernel_C.decode_hidden_only
        self._decode_hidden_fp32_head = torch.ops.qwen_megakernel_C.decode_hidden_fp32_head
        self._qwen = qwen_tts_model
        self._model = qwen_tts_model.model
        self._talker = self._model.talker
        self._subtalker = self._talker.code_predictor
        self._subtalker_model = self._subtalker.model
        self._subtalker_input_embeddings = self._subtalker.get_input_embeddings()
        self._subtalker_output_heads = self._subtalker.lm_head
        self._subtalker_projection = self._subtalker.small_to_mtp_projection
        self._device = self._talker.device
        self._dtype = self._talker.dtype
        self._speech_tokenizer = self._model.speech_tokenizer
        self._head_dim = int(
            getattr(
                self._talker.config,
                "head_dim",
                self._talker.config.hidden_size // self._talker.config.num_attention_heads,
            )
        )
        self._attn_scale = 1.0 / math.sqrt(self._head_dim)

        self._num_layers = int(self._talker.config.num_hidden_layers)
        self._num_code_groups = int(self._talker.config.num_code_groups)
        self._num_subtalker_steps = self._num_code_groups - 1
        self._num_kv_heads = int(self._talker.config.num_key_value_heads)
        self._hidden_size = int(self._talker.config.hidden_size)
        self._intermediate = int(self._talker.config.intermediate_size)
        self._max_seq_len = int(os.getenv("QWEN3_TTS_MAX_SEQ_LEN", "4096"))
        self._sample_rate = self._resolve_output_sample_rate()
        self._decode_upsample_rate = self._resolve_decode_upsample_rate()
        self._assert_kernel_compatible()
        # Match upstream Qwen3-TTS CustomVoice generation defaults. This only
        # changes how text is conditioned into the talker; audio is still
        # streamed incrementally once decode begins.
        self._non_streaming_text_mode = os.getenv("QWEN3_TTS_NON_STREAMING_TEXT_MODE", "1") == "1"
        self._prompt_scaffold_cache: dict[tuple[str, str, bool], dict[str, torch.Tensor]] = {}
        self._projected_text_cache: dict[str, torch.Tensor] = {}
        self._projected_text_cache_limit = int(
            os.getenv("QWEN3_TTS_PROJECTED_TEXT_CACHE_SIZE", "64")
        )
        self._prefill_optimized = os.getenv("QWEN3_TTS_PREFILL_OPTIMIZED", "1") == "1"
        self._prefill_kernel = os.getenv("QWEN3_TTS_PREFILL_KERNEL", "0") == "1"
        self._prefill_graph = os.getenv("QWEN3_TTS_PREFILL_GRAPH", "0") == "1"
        self._prefill_kernel_max_seq_len = int(
            os.getenv("QWEN3_TTS_PREFILL_KERNEL_MAX_SEQ_LEN", "96")
        )

        self._layer_weights_packed = self._pack_layer_weights()
        self._final_norm_weight = self._talker.model.norm.weight.contiguous()
        self._lm_head_weight = self._talker.codec_head.weight.contiguous()
        self._lm_head_weight_f32 = self._lm_head_weight.float().contiguous()
        self._cos_table, self._sin_table = self._build_rope_tables()

        self._alloc_runtime_buffers()

    @staticmethod
    def _anti_cheat_mode() -> bool:
        return os.getenv("QWEN3_TTS_ANTI_CHEAT", "0") == "1"

    def _assert_kernel_compatible(self) -> None:
        """Fail early if the talker shape no longer matches the scalar kernel."""
        expected = {
            "num_hidden_layers": 28,
            "hidden_size": 1024,
            "intermediate_size": 3072,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
        actual = {
            "num_hidden_layers": self._num_layers,
            "hidden_size": self._hidden_size,
            "intermediate_size": self._intermediate,
            "num_attention_heads": int(self._talker.config.num_attention_heads),
            "num_key_value_heads": self._num_kv_heads,
            "head_dim": self._head_dim,
        }
        mismatches = {
            key: (actual[key], expected[key])
            for key in expected
            if actual[key] != expected[key]
        }
        if mismatches:
            detail = ", ".join(
                f"{key}=actual:{actual_value}/expected:{expected_value}"
                for key, (actual_value, expected_value) in mismatches.items()
            )
            raise ValueError(
                "Qwen3-TTS talker config is incompatible with the current "
                f"qwen_megakernel scalar decode constants: {detail}"
            )
        if self._device.type != "cuda":
            raise ValueError("qwen_megakernel talker backend requires CUDA.")
        if self._dtype != torch.bfloat16:
            raise ValueError(f"qwen_megakernel talker backend requires bf16 weights, got {self._dtype}.")

    def _build_rope_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        rotary = getattr(self._talker.model, "rotary_emb", None)
        if rotary is not None:
            position_ids = torch.arange(self._max_seq_len, device=self._device, dtype=torch.long)
            position_ids = position_ids.view(1, 1, -1).expand(3, 1, -1)
            dummy = torch.zeros(
                (1, self._max_seq_len, self._talker.config.num_attention_heads, self._head_dim),
                device=self._device,
                dtype=self._dtype,
            )
            cos, sin = rotary(dummy, position_ids)
            rope_scaling = getattr(self._talker.config, "rope_scaling", None) or {}
            mrope_section = rope_scaling.get("mrope_section")
            interleaved = bool(rope_scaling.get("interleaved", False))

            if mrope_section:
                half_dim = self._head_dim // 2
                if interleaved:
                    modality_num = len(mrope_section)

                    def _merge_interleaved(raw: torch.Tensor) -> torch.Tensor:
                        raw = raw[:, 0, :, :half_dim]
                        merged = raw[0].clone()
                        for section_idx, section_width in enumerate(mrope_section[1:], start=1):
                            beg_idx = section_idx
                            end_idx = section_width * modality_num
                            merged[:, beg_idx:end_idx:modality_num] = raw[section_idx, :, beg_idx:end_idx:modality_num]
                        return torch.cat((merged, merged), dim=-1)

                    cos_eff = _merge_interleaved(cos)
                    sin_eff = _merge_interleaved(sin)
                else:
                    sections = list(mrope_section) * 2

                    def _merge_sections(raw: torch.Tensor) -> torch.Tensor:
                        raw = raw[:, 0]
                        chunks = raw.split(sections, dim=-1)
                        merged = torch.cat([chunk[idx % 3] for idx, chunk in enumerate(chunks)], dim=-1)
                        return merged

                    cos_eff = _merge_sections(cos)
                    sin_eff = _merge_sections(sin)

                return (
                    cos_eff.to(torch.bfloat16).contiguous(),
                    sin_eff.to(torch.bfloat16).contiguous(),
                )

            return (
                cos[0, 0].to(torch.bfloat16).contiguous(),
                sin[0, 0].to(torch.bfloat16).contiguous(),
            )

        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self._head_dim, 2, dtype=torch.float32, device=self._device) / self._head_dim)
        )
        positions = torch.arange(self._max_seq_len, dtype=torch.float32, device=self._device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = torch.cos(emb).to(torch.bfloat16).contiguous()
        sin = torch.sin(emb).to(torch.bfloat16).contiguous()
        return cos, sin

    def _pack_layer_weights(self) -> torch.Tensor:
        ptr_size = 8
        n_ptrs = 11
        blob = bytearray(self._num_layers * n_ptrs * ptr_size)
        for i in range(self._num_layers):
            layer = self._talker.model.layers[i]
            tensors = [
                layer.input_layernorm.weight.contiguous(),
                layer.self_attn.q_proj.weight.contiguous(),
                layer.self_attn.k_proj.weight.contiguous(),
                layer.self_attn.v_proj.weight.contiguous(),
                layer.self_attn.q_norm.weight.contiguous(),
                layer.self_attn.k_norm.weight.contiguous(),
                layer.self_attn.o_proj.weight.contiguous(),
                layer.post_attention_layernorm.weight.contiguous(),
                layer.mlp.gate_proj.weight.contiguous(),
                layer.mlp.up_proj.weight.contiguous(),
                layer.mlp.down_proj.weight.contiguous(),
            ]
            for j, tensor in enumerate(tensors):
                struct.pack_into("Q", blob, (i * n_ptrs + j) * ptr_size, tensor.data_ptr())
        return torch.frombuffer(blob, dtype=torch.uint8).to(self._device)

    def _alloc_runtime_buffers(self) -> None:
        bf16 = dict(dtype=torch.bfloat16, device=self._device)
        f32 = dict(dtype=torch.float32, device=self._device)

        self._k_cache = torch.zeros(
            self._num_layers, self._num_kv_heads, self._max_seq_len, self._head_dim, **bf16
        )
        self._v_cache = torch.zeros_like(self._k_cache)
        self._hidden = torch.empty(self._hidden_size, **bf16)
        self._act = torch.empty(self._hidden_size, **f32)
        self._res = torch.empty(self._hidden_size, **f32)
        self._q = torch.empty(self._talker.config.num_attention_heads * self._head_dim, **f32)
        self._k = torch.empty(self._num_kv_heads * self._head_dim, **f32)
        self._v = torch.empty(self._num_kv_heads * self._head_dim, **f32)
        self._attn_out = torch.empty(self._talker.config.num_attention_heads * self._head_dim, **f32)
        self._mlp_inter = torch.empty(self._intermediate, **f32)
        self._norm_out = torch.empty(self._hidden_size, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device=self._device)
        self._out_token = torch.empty(1, dtype=torch.int32, device=self._device)
        self._token_buf = torch.empty(1, dtype=torch.long, device=self._device)
        self._past_hidden_buf = torch.empty((1, 1, self._hidden_size), **bf16)
        self._subtalker_prefill_buf = torch.empty((1, 2, self._hidden_size), **bf16)

    def _reset_runtime(self) -> None:
        self._k_cache.zero_()
        self._v_cache.zero_()

    def _get_prompt_scaffold(
        self, speaker: str, language: str, non_streaming_text_mode: bool
    ) -> dict[str, torch.Tensor]:
        key = (speaker.lower(), language.lower(), bool(non_streaming_text_mode))
        if self._prefill_optimized or not self._anti_cheat_mode():
            cached = self._prompt_scaffold_cache.get(key)
            if cached is not None:
                return cached

        processor = self._qwen.processor
        scaffold_text = "<|im_start|>assistant\nx<|im_end|>\n<|im_start|>assistant\n"
        input_ids = processor(text=scaffold_text, return_tensors="pt", padding=True)["input_ids"].to(
            self._device
        )

        talker_cfg = self._model.config.talker_config
        if speaker.lower() not in talker_cfg.spk_id:
            raise ValueError(f"Unknown speaker: {speaker}")
        spk_id = talker_cfg.spk_id[speaker.lower()]
        spk_ids = torch.as_tensor(spk_id, device=self._device, dtype=input_ids.dtype).view(1, -1)
        speaker_embed = self._talker.get_input_embeddings()(spk_ids).sum(dim=1, keepdim=True)

        language_id = None
        if language.lower() != "auto":
            language_id = talker_cfg.codec_language_id[language.lower()]
        elif talker_cfg.spk_is_dialect[speaker.lower()] is not False:
            dialect = talker_cfg.spk_is_dialect[speaker.lower()]
            language_id = talker_cfg.codec_language_id[dialect]

        tts_ids = torch.tensor(
            [[self._model.config.tts_bos_token_id, self._model.config.tts_eos_token_id, self._model.config.tts_pad_token_id]],
            device=self._device,
            dtype=input_ids.dtype,
        )
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._talker.text_projection(
            self._talker.get_text_embeddings()(tts_ids)
        ).chunk(3, dim=1)

        if language_id is None:
            codec_prefill = [[talker_cfg.codec_nothink_id, talker_cfg.codec_think_bos_id, talker_cfg.codec_think_eos_id]]
        else:
            codec_prefill = [[talker_cfg.codec_think_id, talker_cfg.codec_think_bos_id, language_id, talker_cfg.codec_think_eos_id]]
        codec_e0 = self._talker.get_input_embeddings()(
            torch.tensor(codec_prefill, device=self._device, dtype=input_ids.dtype)
        )
        codec_e1 = self._talker.get_input_embeddings()(
            torch.tensor([[talker_cfg.codec_pad_id, talker_cfg.codec_bos_id]], device=self._device, dtype=input_ids.dtype)
        )
        codec_embed = torch.cat([codec_e0, speaker_embed, codec_e1], dim=1)

        role_embed = self._talker.text_projection(self._talker.get_text_embeddings()(input_ids[:, :3]))
        body_embed = torch.cat((tts_pad_embed.expand(-1, codec_embed.shape[1] - 2, -1), tts_bos_embed), dim=1)
        body_embed = body_embed + codec_embed[:, :-1]
        talker_prefix = torch.cat((role_embed, body_embed), dim=1).contiguous()

        cached = {
            "talker_prefix": talker_prefix,
            "tts_eos_embed": tts_eos_embed.contiguous(),
            "tts_pad_embed": tts_pad_embed.contiguous(),
            "codec_embed_tail": codec_embed[:, -1:].contiguous(),
        }
        if self._prefill_optimized or not self._anti_cheat_mode():
            self._prompt_scaffold_cache[key] = cached
        return cached

    def warm_prefill_scaffold(self, speaker: str, language: str = "auto") -> None:
        """Populate request-independent prompt scaffold tensors for a backend slot."""
        if not self._prefill_optimized:
            return
        with torch.inference_mode():
            self._get_prompt_scaffold(speaker, language, self._non_streaming_text_mode)

    def _get_projected_text_hidden(self, text: str) -> torch.Tensor:
        if not self._anti_cheat_mode():
            cached = self._projected_text_cache.get(text)
            if cached is not None:
                return cached

        processor = self._qwen.processor
        wrapped = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = processor(text=wrapped, return_tensors="pt", padding=True)["input_ids"].to(self._device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        projected = self._talker.text_projection(
            self._talker.get_text_embeddings()(input_ids[:, 3:-5])
        ).contiguous()

        if self._projected_text_cache_limit > 0 and not self._anti_cheat_mode():
            if len(self._projected_text_cache) >= self._projected_text_cache_limit:
                oldest_key = next(iter(self._projected_text_cache))
                self._projected_text_cache.pop(oldest_key, None)
            self._projected_text_cache[text] = projected
        return projected

    def _build_custom_voice_prompt(
        self, text: str, speaker: str, language: str = "auto"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        non_streaming_text_mode = self._non_streaming_text_mode
        talker_cfg = self._model.config.talker_config
        scaffold = self._get_prompt_scaffold(speaker, language, non_streaming_text_mode)
        talker_input_embed = scaffold["talker_prefix"]
        tts_eos_embed = scaffold["tts_eos_embed"]
        tts_pad_embed = scaffold["tts_pad_embed"]
        codec_embed_tail = scaffold["codec_embed_tail"]
        projected_text_hidden = self._get_projected_text_hidden(text)

        if non_streaming_text_mode:
            codec_pad_ids = torch.tensor(
                [[talker_cfg.codec_pad_id] * (projected_text_hidden.shape[1] + 1)],
                device=self._device,
                dtype=torch.long,
            )
            codec_bos_ids = torch.tensor(
                [[talker_cfg.codec_bos_id]],
                device=self._device,
                dtype=torch.long,
            )
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    torch.cat((projected_text_hidden, tts_eos_embed), dim=1)
                    + self._talker.get_input_embeddings()(codec_pad_ids),
                    tts_pad_embed + self._talker.get_input_embeddings()(codec_bos_ids),
                ],
                dim=1,
            )
            trailing_text_hidden = tts_pad_embed
        else:
            first_text = projected_text_hidden[:, :1] + codec_embed_tail
            talker_input_embed = torch.cat([talker_input_embed, first_text], dim=1)
            trailing_text_hidden = torch.cat(
                (projected_text_hidden[:, 1:], tts_eos_embed),
                dim=1,
            )
        return talker_input_embed.contiguous(), trailing_text_hidden.contiguous(), tts_pad_embed.contiguous()

    def _make_incremental_audio_decoder(self):
        if os.getenv("QWEN3_TTS_USE_INCREMENTAL_TOKENIZER_DECODER", "1") != "1":
            return None
        try:
            return IncrementalTokenizerDecoderV2(self._speech_tokenizer)
        except Exception:
            return None

    def _copy_prefill_cache(self, past_key_values, seq_len: int) -> None:
        # DynamicCache iterates as tuples of (k, v) per layer.
        for layer_idx, (k, v) in enumerate(past_key_values):
            self._k_cache[layer_idx, :, :seq_len, :].copy_(k[0, :, :seq_len, :].to(torch.bfloat16))
            self._v_cache[layer_idx, :, :seq_len, :].copy_(v[0, :, :seq_len, :].to(torch.bfloat16))

    def _resolve_output_sample_rate(self) -> int:
        getter = getattr(self._speech_tokenizer, "get_output_sample_rate", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                pass
        return int(os.getenv("QWEN3_TTS_SAMPLE_RATE", "24000"))

    def _resolve_decode_upsample_rate(self) -> int:
        getter = getattr(self._speech_tokenizer, "get_decode_upsample_rate", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                pass
        return int(os.getenv("QWEN3_TTS_DECODE_UPSAMPLE_RATE", "1920"))

    def _decode_audio_codes(
        self, audio_codes: torch.Tensor, *, left_context_frames: int | None = None
    ) -> tuple[np.ndarray, int]:
        """
        Decode codec frames to waveform.

        Qwen3-TTS variants differ on expected payload key (`audio_codes` vs `codes`).
        Try both and return the first non-empty decode.
        """
        errors: list[Exception] = []
        decoder = getattr(self._speech_tokenizer, "decoder", None)
        if decoder is not None and hasattr(decoder, "chunked_decode"):
            try:
                batched_codes = torch.clamp(audio_codes, min=0).unsqueeze(0)
                wav = decoder.chunked_decode(
                    batched_codes.transpose(1, 2),
                    left_context_size=max(0, int(left_context_frames or 0)),
                ).squeeze(1)
                audio_len = int((audio_codes[..., 0] > -1).sum().item() * self._decode_upsample_rate)
                direct_wav = np.asarray(
                    wav[0, :audio_len].detach().float().cpu().numpy(),
                    dtype=np.float32,
                ).reshape(-1)
                if direct_wav.size > 0:
                    return direct_wav, self._sample_rate
            except Exception as exc:  # pragma: no cover - backend-dependent behavior
                errors.append(exc)

        for key in ("audio_codes", "codes"):
            try:
                wavs, sr = self._speech_tokenizer.decode([{key: audio_codes}])
            except Exception as exc:  # pragma: no cover - backend-dependent behavior
                errors.append(exc)
                continue
            if not wavs:
                continue
            wav = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
            if wav.size > 0:
                return wav, int(sr)
        if errors:
            raise RuntimeError(f"speech_tokenizer.decode failed: {errors[-1]}")
        return np.empty((0,), dtype=np.float32), self._sample_rate

    def _decode_incremental_suffix(
        self,
        frames: list[torch.Tensor],
        *,
        decoded_frames: int,
        left_context_frames: int,
    ) -> tuple[np.ndarray, int, int]:
        total_frames = len(frames)
        if total_frames <= decoded_frames:
            return np.empty((0,), dtype=np.float32), self._sample_rate, decoded_frames

        decode_start = max(0, decoded_frames - max(0, left_context_frames))
        audio_codes = torch.stack(frames[decode_start:], dim=0)
        wav, sr = self._decode_audio_codes(
            audio_codes,
            left_context_frames=left_context_frames,
        )

        overlap_frames = decoded_frames - decode_start
        overlap_samples = overlap_frames * self._decode_upsample_rate
        if overlap_samples >= wav.shape[0]:
            return np.empty((0,), dtype=np.float32), sr, total_frames
        return wav[overlap_samples:], sr, total_frames

    @staticmethod
    def _text_is_short_utterance(text: str) -> bool:
        stripped = text.strip()
        return len(stripped) <= 48 and sum(stripped.count(ch) for ch in ".!?") <= 1

    def _argmax_next_codec_token(self) -> torch.Tensor:
        logits = torch.mv(self._lm_head_weight_f32, self._norm_out)
        return torch.argmax(logits, dim=0, keepdim=True).to(torch.long)

    @staticmethod
    def _sync_timing_enabled() -> bool:
        return os.getenv("QWEN3_TTS_SYNC_TIMING", "0") == "1"

    def _kernel_prefill_enabled_for(self, seq_len: int) -> bool:
        return (
            self._prefill_kernel
            and seq_len > 0
            and seq_len <= self._prefill_kernel_max_seq_len
        )

    @staticmethod
    def _phase_start(sync_timing: bool) -> float:
        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @staticmethod
    def _phase_ms(started_at: float, sync_timing: bool) -> float:
        if sync_timing and torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - started_at) * 1000.0

    def _run_megakernel_prefill(self, talker_input_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay short prompt embeddings through the scalar megakernel decode path.

        This is a real prefill path: it fills the same KV cache used by
        continuation decode and returns the first codec token from the final
        prompt hidden state. It does not cache request outputs or audio.
        """
        seq_len = int(talker_input_embed.shape[1])
        if seq_len <= 0:
            raise ValueError("Cannot run megakernel prefill for an empty prompt.")
        if seq_len > self._max_seq_len:
            raise ValueError(f"Prompt length {seq_len} exceeds max_seq_len {self._max_seq_len}.")

        for position in range(seq_len):
            input_hidden = talker_input_embed[0, position].to(torch.bfloat16).contiguous()
            if position == seq_len - 1:
                self._decode_hidden_fp32_head(
                    self._out_token,
                    input_hidden,
                    self._layer_weights_packed,
                    self._final_norm_weight,
                    self._lm_head_weight_f32,
                    self._cos_table,
                    self._sin_table,
                    self._k_cache,
                    self._v_cache,
                    self._hidden,
                    self._act,
                    self._res,
                    self._q,
                    self._k,
                    self._v,
                    self._attn_out,
                    self._mlp_inter,
                    self._norm_out,
                    self._num_layers,
                    position,
                    self._max_seq_len,
                    self._attn_scale,
                )
            else:
                self._decode_hidden_only(
                    input_hidden,
                    self._layer_weights_packed,
                    self._final_norm_weight,
                    self._cos_table,
                    self._sin_table,
                    self._k_cache,
                    self._v_cache,
                    self._hidden,
                    self._act,
                    self._res,
                    self._q,
                    self._k,
                    self._v,
                    self._attn_out,
                    self._mlp_inter,
                    self._norm_out,
                    self._num_layers,
                    position,
                    self._max_seq_len,
                    self._attn_scale,
                )

        self._token_buf[0] = self._out_token[0]
        self._past_hidden_buf.copy_(self._norm_out.view(1, 1, -1))
        return self._token_buf, self._past_hidden_buf

    def _run_pytorch_prefill(
        self,
        talker_input_embed: torch.Tensor,
        *,
        sync_timing: bool,
        stats: StreamStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        started = self._phase_start(sync_timing)
        prefill = self._talker.model(
            inputs_embeds=talker_input_embed,
            use_cache=True,
            return_dict=True,
        )
        elapsed = self._phase_ms(started, sync_timing)
        if stats is not None:
            stats.prefill_model_ms += elapsed

        next_first = torch.argmax(self._talker.codec_head(prefill.last_hidden_state[:, -1, :]), dim=-1).to(
            torch.long
        )
        seq_len = int(talker_input_embed.shape[1])
        copy_started = self._phase_start(sync_timing)
        self._copy_prefill_cache(prefill.past_key_values, seq_len)
        copy_elapsed = self._phase_ms(copy_started, sync_timing)
        if stats is not None:
            stats.prefill_cache_ms += copy_elapsed
            stats.prefill_mode = "pytorch_static_scaffold" if self._prefill_optimized else "pytorch"
        return next_first, prefill.last_hidden_state[:, -1:, :]

    def _run_prefill(
        self,
        talker_input_embed: torch.Tensor,
        *,
        sync_timing: bool,
        stats: StreamStats,
    ) -> tuple[torch.Tensor, int, torch.Tensor]:
        seq_len = int(talker_input_embed.shape[1])
        if self._kernel_prefill_enabled_for(seq_len):
            started = self._phase_start(sync_timing)
            next_first, past_hidden = self._run_megakernel_prefill(talker_input_embed)
            elapsed = self._phase_ms(started, sync_timing)
            stats.prefill_model_ms += elapsed
            stats.prefill_mode = "megakernel_scalar"
            return next_first, seq_len, past_hidden

        next_first, past_hidden = self._run_pytorch_prefill(
            talker_input_embed,
            sync_timing=sync_timing,
            stats=stats,
        )
        return next_first, seq_len, past_hidden

    def debug_first_step_parity(
        self,
        text: str,
        speaker: str,
        *,
        language: str = "auto",
    ) -> dict[str, object]:
        """Compare one talker continuation step against the PyTorch model path."""
        with torch.inference_mode():
            self._reset_runtime()
            talker_input_embed, trailing_text_hidden, tts_pad_embed = self._build_custom_voice_prompt(
                text,
                speaker,
                language=language,
            )
            prefill = self._talker.model(
                inputs_embeds=talker_input_embed,
                use_cache=True,
                return_dict=True,
            )
            seq_len = int(talker_input_embed.shape[1])
            next_first = torch.argmax(self._talker.codec_head(prefill.last_hidden_state[:, -1, :]), dim=-1).to(
                torch.long
            )
            first_hidden = self._talker.get_input_embeddings()(next_first.unsqueeze(0))
            past_hidden = prefill.last_hidden_state[:, -1:, :]
            frame_codes, codec_sum = self._predict_subtalker_frame(
                first_token=next_first,
                first_hidden=first_hidden,
                past_hidden=past_hidden,
                do_sample=False,
                top_p=1.0,
                top_k=0,
                temperature=0.0,
            )
            input_embed = codec_sum
            if trailing_text_hidden.shape[1] > 0:
                input_embed = input_embed + trailing_text_hidden[:, 0].unsqueeze(1)
            else:
                input_embed = input_embed + tts_pad_embed

            baseline = self._talker.model(
                inputs_embeds=input_embed,
                past_key_values=prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            baseline_hidden = baseline.last_hidden_state[:, -1, :].detach().float()
            baseline_logits = torch.mv(self._lm_head_weight_f32, baseline_hidden[0])
            baseline_token = int(torch.argmax(baseline_logits).item())

            kernel_prefill_token = None
            kernel_prefill_hidden_max_diff = None
            kernel_prefill_hidden_mean_diff = None
            kernel_prefill_token_match = None
            if self._kernel_prefill_enabled_for(seq_len):
                self._reset_runtime()
                kernel_prefill_first, kernel_prefill_hidden = self._run_megakernel_prefill(talker_input_embed)
                torch.cuda.synchronize()
                kernel_prefill_token = int(kernel_prefill_first.view(-1)[0].item())
                prefill_hidden_diff = (
                    prefill.last_hidden_state[:, -1:, :].detach().float()
                    - kernel_prefill_hidden.detach().float()
                ).abs()
                kernel_prefill_hidden_max_diff = float(prefill_hidden_diff.max().item())
                kernel_prefill_hidden_mean_diff = float(prefill_hidden_diff.mean().item())
                kernel_prefill_token_match = kernel_prefill_token == int(next_first.view(-1)[0].item())

            self._copy_prefill_cache(prefill.past_key_values, seq_len)
            self._decode_hidden_fp32_head(
                self._out_token,
                input_embed[0, 0].to(torch.bfloat16).contiguous(),
                self._layer_weights_packed,
                self._final_norm_weight,
                self._lm_head_weight_f32,
                self._cos_table,
                self._sin_table,
                self._k_cache,
                self._v_cache,
                self._hidden,
                self._act,
                self._res,
                self._q,
                self._k,
                self._v,
                self._attn_out,
                self._mlp_inter,
                self._norm_out,
                self._num_layers,
                seq_len,
                self._max_seq_len,
                self._attn_scale,
            )
            torch.cuda.synchronize()
            custom_token = int(self._out_token.item())
            custom_hidden = self._norm_out.detach().float().view(1, -1)
            hidden_diff = (baseline_hidden - custom_hidden).abs()
            custom_logits = torch.mv(self._lm_head_weight_f32, custom_hidden[0])
            logit_diff = (baseline_logits - custom_logits).abs()

        return {
            "kernel_path": "qwen_megakernel_C.decode_hidden_fp32_head",
            "first_prefill_token": int(next_first.view(-1)[0].item()),
            "kernel_prefill_token": kernel_prefill_token,
            "kernel_prefill_token_match": kernel_prefill_token_match,
            "kernel_prefill_hidden_max_abs_diff": kernel_prefill_hidden_max_diff,
            "kernel_prefill_hidden_mean_abs_diff": kernel_prefill_hidden_mean_diff,
            "first_frame_codes": [int(v) for v in frame_codes.detach().cpu().tolist()],
            "baseline_next_token": baseline_token,
            "custom_next_token": custom_token,
            "token_match": baseline_token == custom_token,
            "hidden_max_abs_diff": float(hidden_diff.max().item()),
            "hidden_mean_abs_diff": float(hidden_diff.mean().item()),
            "logit_max_abs_diff": float(logit_diff.max().item()),
            "logit_mean_abs_diff": float(logit_diff.mean().item()),
        }

    @staticmethod
    def _sample_subtalker_token(
        logits: torch.Tensor,
        *,
        do_sample: bool,
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> torch.Tensor:
        if logits.dim() != 2 or logits.shape[0] != 1:
            raise ValueError(f"Expected logits shape [1, vocab], got {tuple(logits.shape)}")

        flat_logits = logits[0]
        if not do_sample or temperature <= 0.0:
            return torch.argmax(flat_logits, dim=-1, keepdim=True)

        work_logits = flat_logits.float()
        if temperature != 1.0:
            work_logits = work_logits / max(temperature, 1e-5)

        candidate_indices = torch.arange(work_logits.shape[0], device=work_logits.device)
        if 0 < top_k < work_logits.shape[0]:
            work_logits, candidate_indices = torch.topk(work_logits, k=top_k)

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_order = torch.sort(work_logits, descending=True)
            sorted_indices = candidate_indices[sorted_order]
            probs = torch.softmax(sorted_logits, dim=-1)
            keep_mask = probs.cumsum(dim=-1) <= top_p
            keep_mask[0] = True
            filtered_logits = sorted_logits.masked_fill(~keep_mask, float("-inf"))
            filtered_probs = torch.softmax(filtered_logits, dim=-1)
            sampled_sorted = torch.multinomial(filtered_probs, num_samples=1)
            return sorted_indices[sampled_sorted]

        probs = torch.softmax(work_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return candidate_indices[sampled]

    def _predict_subtalker_frame(
        self,
        *,
        first_token: torch.Tensor,
        first_hidden: torch.Tensor,
        past_hidden: torch.Tensor,
        do_sample: bool,
        top_p: float,
        top_k: int,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Custom fixed-shape decode for the code predictor.

        This replaces the generic HF `generate()` loop with a cache-reusing
        path specialized to Qwen3-TTS's fixed `num_code_groups - 1` decode.
        """
        frame_codes = torch.empty(self._num_code_groups, dtype=torch.long, device=self._device)
        frame_codes[0] = first_token.view(-1)[0]
        codec_sum = first_hidden.clone()

        self._subtalker_prefill_buf[:, :1].copy_(past_hidden)
        self._subtalker_prefill_buf[:, 1:2].copy_(first_hidden)
        outputs = self._subtalker_model(
            input_ids=None,
            inputs_embeds=self._subtalker_projection(self._subtalker_prefill_buf),
            past_key_values=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        kv_cache = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1, :]

        for group_idx in range(self._num_subtalker_steps):
            logits = self._subtalker_output_heads[group_idx](hidden)
            next_token = self._sample_subtalker_token(
                logits,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
            )
            frame_codes[group_idx + 1] = next_token[0]

            next_embed = self._subtalker_input_embeddings[group_idx](next_token.view(1, 1))
            codec_sum.add_(next_embed)

            if group_idx + 1 >= self._num_subtalker_steps:
                break

            outputs = self._subtalker_model(
                input_ids=None,
                inputs_embeds=self._subtalker_projection(next_embed),
                past_key_values=kv_cache,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            kv_cache = outputs.past_key_values
            hidden = outputs.last_hidden_state[:, -1, :]

        return frame_codes, codec_sum

    def stream_audio(
        self,
        text: str,
        speaker: str,
        max_new_tokens: int,
        *,
        language: str = "auto",
        decode_stride: int = 4,
    ) -> tuple[StreamStats, Iterator[np.ndarray]]:
        stats = StreamStats()

        def _run() -> Iterator[np.ndarray]:
            effective_decode_stride = max(1, int(decode_stride))
            first_chunk_frames = max(1, int(os.getenv("QWEN3_TTS_FIRST_CHUNK_FRAMES", "1")))
            adaptive_decode_cadence = os.getenv("QWEN3_TTS_ADAPTIVE_DECODE_CADENCE", "1") == "1"
            decode_stride_mid = max(1, int(os.getenv("QWEN3_TTS_DECODE_STRIDE_MID", "4")))
            decode_stride_late = max(1, int(os.getenv("QWEN3_TTS_DECODE_STRIDE_LATE", "8")))
            decode_stride_late_start_frame = int(
                os.getenv("QWEN3_TTS_DECODE_STRIDE_LATE_START_FRAME", "24")
            )
            incremental_left_context_frames = max(
                0, int(os.getenv("QWEN3_TTS_INCREMENTAL_LEFT_CONTEXT_FRAMES", "12"))
            )
            silence_early_stop = os.getenv("QWEN3_TTS_SILENCE_EARLY_STOP", "0") == "1"
            silence_rms = float(os.getenv("QWEN3_TTS_SILENCE_RMS", "0.003"))
            silence_tail_s = float(os.getenv("QWEN3_TTS_SILENCE_TAIL_S", "0.9"))
            min_frames_before_silence_stop = int(
                os.getenv("QWEN3_TTS_MIN_FRAMES_BEFORE_SILENCE_STOP", "48")
            )
            min_eos_steps = int(os.getenv("QWEN3_TTS_MIN_EOS_STEPS", "6"))
            min_eos_steps_short = int(os.getenv("QWEN3_TTS_MIN_EOS_STEPS_SHORT", "2"))
            min_frames_before_repeat_stop = int(
                os.getenv("QWEN3_TTS_MIN_FRAMES_BEFORE_REPEAT_STOP", "32")
            )
            min_frames_before_repeat_stop_short = int(
                os.getenv("QWEN3_TTS_MIN_FRAMES_BEFORE_REPEAT_STOP_SHORT", "20")
            )
            repeat_first_token_run_limit = int(
                os.getenv("QWEN3_TTS_REPEAT_FIRST_TOKEN_RUN_LIMIT", "48")
            )
            repeat_frame_run_limit = int(os.getenv("QWEN3_TTS_REPEAT_FRAME_RUN_LIMIT", "12"))
            do_sample = os.getenv("QWEN3_TTS_SUBTALKER_DO_SAMPLE", "0") == "1"
            top_p = float(os.getenv("QWEN3_TTS_SUBTALKER_TOP_P", "0.92"))
            top_k = int(os.getenv("QWEN3_TTS_SUBTALKER_TOP_K", "40"))
            temperature = float(os.getenv("QWEN3_TTS_SUBTALKER_TEMPERATURE", "0.8"))
            incremental_audio_decoder = self._make_incremental_audio_decoder()
            sync_timing = self._sync_timing_enabled()
            stats.timing_mode = "wall_sync" if sync_timing else "wall_async"
            audio_overlap_enabled = (
                incremental_audio_decoder is not None
                and not silence_early_stop
                and os.getenv("QWEN3_TTS_AUDIO_DECODE_OVERLAP", "0") == "1"
                and torch.cuda.is_available()
            )
            stats.audio_decode_overlap = bool(audio_overlap_enabled)

            audio_input_q: "queue.Queue[object]" | None = None
            audio_output_q: "queue.Queue[object]" | None = None
            audio_worker: threading.Thread | None = None
            worker_error: list[BaseException] = []
            if audio_overlap_enabled:
                audio_stream = torch.cuda.Stream(device=self._device)
                audio_input_q = queue.Queue()
                audio_output_q = queue.Queue()
                decoder_ref = incremental_audio_decoder
                device_ref = self._device

                def _audio_worker_loop():
                    try:
                        with torch.inference_mode():
                            while True:
                                payload = audio_input_q.get()
                                if payload is None:
                                    break
                                frames_tensor, ready_event, submit_perf, snapshot_frame_count = payload
                                audio_stream.wait_event(ready_event)
                                with torch.cuda.stream(audio_stream):
                                    delta_t = decoder_ref.decode_new_frames(frames_tensor)
                                audio_stream.synchronize()
                                delta = np.asarray(
                                    delta_t.detach().cpu().numpy(),
                                    dtype=np.float32,
                                ).reshape(-1).copy()
                                audio_output_q.put(
                                    (delta, submit_perf, time.perf_counter(), snapshot_frame_count)
                                )
                    except BaseException as exc:  # pragma: no cover - propagated to main thread
                        worker_error.append(exc)
                    finally:
                        audio_output_q.put(None)

                audio_worker = threading.Thread(
                    target=_audio_worker_loop,
                    name="qwen3-tts-audio-decode",
                    daemon=True,
                )
                audio_worker.start()

            wall_start = time.perf_counter()
            worker_shutdown_done = False

            def _shutdown_audio_worker():
                nonlocal worker_shutdown_done
                if worker_shutdown_done or not audio_overlap_enabled:
                    return
                worker_shutdown_done = True
                try:
                    audio_input_q.put_nowait(None)
                except Exception:
                    audio_input_q.put(None)
                # Drain any pending outputs so the worker can exit cleanly.
                while True:
                    try:
                        item = audio_output_q.get(timeout=5.0)
                    except Exception:
                        break
                    if item is None:
                        break
                if audio_worker is not None:
                    audio_worker.join(timeout=5.0)

            @contextlib.contextmanager
            def _audio_worker_cleanup_ctx():
                try:
                    yield
                finally:
                    _shutdown_audio_worker()

            with _audio_worker_cleanup_ctx(), torch.inference_mode():
                self._reset_runtime()
                started = torch.cuda.Event(enable_timing=True)
                ended = torch.cuda.Event(enable_timing=True)
                started.record()

                prompt_started = self._phase_start(sync_timing)
                talker_input_embed, trailing_text_hidden, tts_pad_embed = self._build_custom_voice_prompt(
                    text,
                    speaker,
                    language=language,
                )
                stats.prompt_build_ms += self._phase_ms(prompt_started, sync_timing)

                next_first, seq_len, past_hidden = self._run_prefill(
                    talker_input_embed,
                    sync_timing=sync_timing,
                    stats=stats,
                )
                stats.prefill_ms = stats.prompt_build_ms + stats.prefill_model_ms + stats.prefill_cache_ms
                generation_step = 0
                talker_cfg = self._model.config.talker_config
                eos_id = int(talker_cfg.codec_eos_token_id)
                bos_id = int(talker_cfg.codec_bos_id)
                is_short_utterance = self._text_is_short_utterance(text)
                effective_min_eos_steps = min_eos_steps_short if is_short_utterance else min_eos_steps
                effective_min_frames_before_repeat_stop = (
                    min_frames_before_repeat_stop_short
                    if is_short_utterance
                    else min_frames_before_repeat_stop
                )

                emitted_samples = 0
                decoded_frames = 0
                trailing_silence_samples = 0
                with torch.inference_mode(False):
                    frame_buffer = torch.empty(
                        (max_new_tokens, self._num_code_groups),
                        dtype=torch.long,
                        device=self._device,
                    )
                frame_count = 0
                prev_first_token: int | None = None
                repeated_first_token_run = 0
                repeated_frame_run = 0
                stats.stop_reason = "max_new_tokens"
                for step in range(max_new_tokens):
                    next_token = int(next_first.item())
                    if next_token == eos_id:
                        if step < effective_min_eos_steps:
                            # Avoid pathological immediate-eos streams that produce no audio.
                            next_token = bos_id
                        else:
                            stats.stop_reason = "eos"
                            break

                    self._token_buf[0] = next_token
                    next_first = self._token_buf
                    first_hidden = self._talker.get_input_embeddings()(next_first.unsqueeze(0))
                    subtalker_started = self._phase_start(sync_timing)
                    frame_codes, codec_sum = self._predict_subtalker_frame(
                        first_token=next_first,
                        first_hidden=first_hidden,
                        past_hidden=past_hidden,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                    )
                    stats.subtalker_ms += self._phase_ms(subtalker_started, sync_timing)
                    stats.subtalker_calls += 1

                    if prev_first_token is not None and next_token == prev_first_token:
                        repeated_first_token_run += 1
                    else:
                        repeated_first_token_run = 1
                    prev_first_token = next_token

                    if frame_count > 0 and torch.equal(frame_codes, frame_buffer[frame_count - 1]):
                        repeated_frame_run += 1
                    else:
                        repeated_frame_run = 1

                    guard_repeat = (
                        frame_count >= effective_min_frames_before_repeat_stop
                        and generation_step >= trailing_text_hidden.shape[1]
                    )
                    if guard_repeat:
                        if repeated_frame_run >= repeat_frame_run_limit:
                            stats.stop_reason = "repeat_frame_loop"
                            break
                        if repeated_first_token_run >= repeat_first_token_run_limit:
                            stats.stop_reason = "repeat_token_loop"
                            break

                    frame_buffer[frame_count].copy_(frame_codes.detach())
                    frame_count += 1

                    input_embed = codec_sum
                    if generation_step < trailing_text_hidden.shape[1]:
                        input_embed = input_embed + trailing_text_hidden[:, generation_step].unsqueeze(1)
                    else:
                        input_embed = input_embed + tts_pad_embed

                    decode_started = self._phase_start(sync_timing)
                    self._decode_hidden_fp32_head(
                        self._out_token,
                        input_embed[0, 0].to(torch.bfloat16).contiguous(),
                        self._layer_weights_packed,
                        self._final_norm_weight,
                        self._lm_head_weight_f32,
                        self._cos_table,
                        self._sin_table,
                        self._k_cache,
                        self._v_cache,
                        self._hidden,
                        self._act,
                        self._res,
                        self._q,
                        self._k,
                        self._v,
                        self._attn_out,
                        self._mlp_inter,
                        self._norm_out,
                        self._num_layers,
                        seq_len + step,
                        self._max_seq_len,
                        self._attn_scale,
                    )
                    stats.talker_decode_ms += self._phase_ms(decode_started, sync_timing)
                    stats.talker_decode_calls += 1
                    if stats.first_decode_ms is None:
                        ended.record()
                        torch.cuda.synchronize()
                        stats.first_decode_ms = float(started.elapsed_time(ended))

                    self._token_buf.copy_(self._out_token)
                    next_first = self._token_buf
                    self._past_hidden_buf.copy_(self._norm_out.view(1, 1, -1))
                    past_hidden = self._past_hidden_buf
                    generation_step += 1

                    should_decode = False
                    n_frames = frame_count
                    current_decode_stride = effective_decode_stride
                    if adaptive_decode_cadence:
                        if n_frames >= decode_stride_late_start_frame:
                            current_decode_stride = decode_stride_late
                        elif n_frames > first_chunk_frames:
                            current_decode_stride = decode_stride_mid
                    if n_frames <= first_chunk_frames:
                        should_decode = True
                    elif ((n_frames - first_chunk_frames) % current_decode_stride) == 0:
                        should_decode = True

                    if should_decode and audio_overlap_enabled:
                        new_frame_count = frame_count - decoded_frames
                        if new_frame_count > 0:
                            frames_for_decode = frame_buffer[decoded_frames:frame_count].clone()
                            ready_event = torch.cuda.Event()
                            ready_event.record()
                            decoded_frames = frame_count
                            stats.audio_decode_calls += 1
                            audio_input_q.put(
                                (
                                    frames_for_decode,
                                    ready_event,
                                    time.perf_counter(),
                                    frame_count,
                                )
                            )
                    if audio_overlap_enabled:
                        # Drain any ready chunks without blocking on every iteration.
                        while True:
                            try:
                                item = audio_output_q.get_nowait()
                            except queue.Empty:
                                break
                            if item is None:
                                if worker_error:
                                    raise worker_error[0]
                                break
                            delta, submit_perf, done_perf, snapshot_frame_count = item
                            stats.audio_decode_ms += (done_perf - submit_perf) * 1000.0
                            if delta.size > 0:
                                emitted_samples += delta.shape[0]
                                stats.audio_chunks += 1
                                stats.frames_generated = max(
                                    stats.frames_generated, snapshot_frame_count
                                )
                                if stats.ttfc_ms is None:
                                    stats.ttfc_ms = (time.perf_counter() - wall_start) * 1000.0
                                yield delta
                    elif should_decode:
                            audio_decode_started = self._phase_start(sync_timing)
                            if incremental_audio_decoder is not None:
                                new_frame_count = frame_count - decoded_frames
                                if new_frame_count > 0:
                                    new_audio_codes = frame_buffer[decoded_frames:frame_count]
                                    delta_t = incremental_audio_decoder.decode_new_frames(new_audio_codes)
                                    delta = np.asarray(
                                        delta_t.detach().cpu().numpy(),
                                        dtype=np.float32,
                                    ).reshape(-1).copy()
                                    sr = self._sample_rate
                                    decoded_frames = frame_count
                                else:
                                    delta = np.empty((0,), dtype=np.float32)
                                    sr = self._sample_rate
                            else:
                                delta, sr, decoded_frames = self._decode_incremental_suffix(
                                    [frame_buffer[idx] for idx in range(frame_count)],
                                    decoded_frames=decoded_frames,
                                    left_context_frames=incremental_left_context_frames,
                                )
                            stats.audio_decode_ms += self._phase_ms(audio_decode_started, sync_timing)
                            stats.audio_decode_calls += 1
                            if delta.size > 0:
                                emitted_samples += delta.shape[0]
                                stats.frames_generated = frame_count
                                stats.audio_chunks += 1
                                if stats.ttfc_ms is None:
                                    ended.record()
                                    torch.cuda.synchronize()
                                    stats.ttfc_ms = float(started.elapsed_time(ended))
                                yield delta

                                if silence_early_stop and n_frames >= min_frames_before_silence_stop:
                                    silent = np.abs(delta) <= silence_rms
                                    silent_tail = 0
                                    for sample_is_silent in silent[::-1]:
                                        if sample_is_silent:
                                            silent_tail += 1
                                        else:
                                            break
                                    if silent_tail == delta.shape[0]:
                                        trailing_silence_samples += silent_tail
                                    else:
                                        trailing_silence_samples = silent_tail

                                    if trailing_silence_samples >= int(silence_tail_s * sr):
                                        stats.stop_reason = "silence_tail"
                                        break

                if audio_overlap_enabled:
                    # Submit any final undelivered frames to the worker before joining.
                    if frame_count > decoded_frames:
                        frames_for_decode = frame_buffer[decoded_frames:frame_count].clone()
                        ready_event = torch.cuda.Event()
                        ready_event.record()
                        stats.audio_decode_calls += 1
                        audio_input_q.put(
                            (
                                frames_for_decode,
                                ready_event,
                                time.perf_counter(),
                                frame_count,
                            )
                        )
                        decoded_frames = frame_count
                    audio_input_q.put(None)
                    wait_started = time.perf_counter()
                    while True:
                        item = audio_output_q.get()
                        if item is None:
                            break
                        delta, submit_perf, done_perf, snapshot_frame_count = item
                        stats.audio_decode_ms += (done_perf - submit_perf) * 1000.0
                        if delta.size > 0:
                            emitted_samples += delta.shape[0]
                            stats.audio_chunks += 1
                            stats.frames_generated = max(
                                stats.frames_generated, snapshot_frame_count
                            )
                            if stats.ttfc_ms is None:
                                stats.ttfc_ms = (time.perf_counter() - wall_start) * 1000.0
                            yield delta
                    stats.audio_decode_wait_ms = (time.perf_counter() - wait_started) * 1000.0
                    if worker_error:
                        raise worker_error[0]
                    audio_worker.join()
                    worker_shutdown_done = True
                    stats.frames_generated = max(stats.frames_generated, frame_count)
                    stats.audio_seconds = float(emitted_samples) / float(self._sample_rate)
                    stats.generation_s = time.perf_counter() - wall_start
                else:
                    if frame_count > 0:
                        audio_decode_started = self._phase_start(sync_timing)
                        if incremental_audio_decoder is not None:
                            new_frame_count = frame_count - decoded_frames
                            if new_frame_count > 0:
                                new_audio_codes = frame_buffer[decoded_frames:frame_count]
                                delta_t = incremental_audio_decoder.decode_new_frames(new_audio_codes)
                                delta = np.asarray(
                                    delta_t.detach().cpu().numpy(),
                                    dtype=np.float32,
                                ).reshape(-1).copy()
                                sr = self._sample_rate
                                decoded_frames = frame_count
                            else:
                                delta = np.empty((0,), dtype=np.float32)
                                sr = self._sample_rate
                        else:
                            delta, sr, decoded_frames = self._decode_incremental_suffix(
                                [frame_buffer[idx] for idx in range(frame_count)],
                                decoded_frames=decoded_frames,
                                left_context_frames=incremental_left_context_frames,
                            )
                        stats.audio_decode_ms += self._phase_ms(audio_decode_started, sync_timing)
                        stats.audio_decode_calls += 1
                        stats.frames_generated = frame_count
                        if delta.size > 0:
                            emitted_samples += delta.shape[0]
                            stats.audio_chunks += 1
                            if stats.ttfc_ms is None:
                                ended.record()
                                torch.cuda.synchronize()
                                stats.ttfc_ms = float(started.elapsed_time(ended))
                            yield delta
                        stats.audio_seconds = float(emitted_samples) / float(sr)

                    ended.record()
                    torch.cuda.synchronize()
                    stats.generation_s = float(started.elapsed_time(ended)) / 1000.0

        return stats, _run()
