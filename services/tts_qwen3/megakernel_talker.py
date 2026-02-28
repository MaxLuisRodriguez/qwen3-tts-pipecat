"""Talker-specific megakernel backend with incremental codec/audio streaming."""

from __future__ import annotations

import math
import os
import struct
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


@dataclass
class StreamStats:
    ttfc_ms: float | None = None
    generation_s: float = 0.0
    audio_seconds: float = 0.0
    frames_generated: int = 0


class TalkerMegakernelBackend:
    """Runs Qwen3-TTS talker decode with megakernel and streams decode-time audio."""

    def __init__(self, qwen_tts_model):
        # Trigger extension build/load.
        import qwen_megakernel  # noqa: F401

        self._decode_from_hidden = torch.ops.qwen_megakernel_C.decode_from_hidden
        self._qwen = qwen_tts_model
        self._model = qwen_tts_model.model
        self._talker = self._model.talker
        self._device = self._talker.device
        self._dtype = self._talker.dtype
        self._head_dim = int(
            getattr(
                self._talker.config,
                "head_dim",
                self._talker.config.hidden_size // self._talker.config.num_attention_heads,
            )
        )
        self._attn_scale = 1.0 / math.sqrt(self._head_dim)

        self._num_layers = int(self._talker.config.num_hidden_layers)
        self._num_kv_heads = int(self._talker.config.num_key_value_heads)
        self._hidden_size = int(self._talker.config.hidden_size)
        self._intermediate = int(self._talker.config.intermediate_size)
        self._max_seq_len = int(os.getenv("QWEN3_TTS_MAX_SEQ_LEN", "4096"))

        self._layer_weights_packed = self._pack_layer_weights()
        self._final_norm_weight = self._talker.model.norm.weight.contiguous()
        self._lm_head_weight = self._talker.codec_head.weight.contiguous()
        self._cos_table, self._sin_table = self._build_rope_tables()

        self._alloc_runtime_buffers()

    def _build_rope_tables(self) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, self._head_dim, 2, dtype=torch.float32, device=self._device) / self._head_dim)
        )
        positions = torch.arange(self._max_seq_len, dtype=torch.float32, device=self._device)
        freqs = torch.outer(positions, inv_freq)
        cos = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
        sin = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).contiguous()
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

    def _reset_runtime(self) -> None:
        self._k_cache.zero_()
        self._v_cache.zero_()

    def _build_custom_voice_prompt(
        self, text: str, speaker: str, language: str = "auto"
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processor = self._qwen.processor
        text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = processor(text=text, return_tensors="pt", padding=True)["input_ids"].to(self._device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        talker_cfg = self._model.config.talker_config
        if speaker.lower() not in talker_cfg.spk_id:
            raise ValueError(f"Unknown speaker: {speaker}")
        spk_id = talker_cfg.spk_id[speaker.lower()]
        # Some checkpoints store speaker IDs as multi-token sequences.
        # Collapse to a single hidden vector to keep embedding width == hidden_size.
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
        talker_input_embed = torch.cat((role_embed, body_embed), dim=1)

        first_text = self._talker.text_projection(self._talker.get_text_embeddings()(input_ids[:, 3:4])) + codec_embed[:, -1:]
        talker_input_embed = torch.cat([talker_input_embed, first_text], dim=1)
        trailing_text_hidden = torch.cat(
            (self._talker.text_projection(self._talker.get_text_embeddings()(input_ids[:, 4:-5])), tts_eos_embed),
            dim=1,
        )
        return talker_input_embed, trailing_text_hidden, tts_pad_embed

    def _copy_prefill_cache(self, past_key_values, seq_len: int) -> None:
        # DynamicCache iterates as tuples of (k, v) per layer.
        for layer_idx, (k, v) in enumerate(past_key_values):
            self._k_cache[layer_idx, :, :seq_len, :].copy_(k[0, :, :seq_len, :].to(torch.bfloat16))
            self._v_cache[layer_idx, :, :seq_len, :].copy_(v[0, :, :seq_len, :].to(torch.bfloat16))

    def _decode_audio_codes(self, audio_codes: torch.Tensor) -> tuple[np.ndarray, int]:
        """
        Decode codec frames to waveform.

        Qwen3-TTS variants differ on expected payload key (`audio_codes` vs `codes`).
        Try both and return the first non-empty decode.
        """
        errors: list[Exception] = []
        for key in ("audio_codes", "codes"):
            try:
                wavs, sr = self._model.speech_tokenizer.decode([{key: audio_codes}])
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
        return np.empty((0,), dtype=np.float32), int(os.getenv("QWEN3_TTS_SAMPLE_RATE", "24000"))

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
            silence_early_stop = os.getenv("QWEN3_TTS_SILENCE_EARLY_STOP", "0") == "1"
            silence_rms = float(os.getenv("QWEN3_TTS_SILENCE_RMS", "0.003"))
            silence_tail_s = float(os.getenv("QWEN3_TTS_SILENCE_TAIL_S", "0.9"))
            min_frames_before_silence_stop = int(
                os.getenv("QWEN3_TTS_MIN_FRAMES_BEFORE_SILENCE_STOP", "48")
            )
            min_eos_steps = int(os.getenv("QWEN3_TTS_MIN_EOS_STEPS", "64"))
            do_sample = os.getenv("QWEN3_TTS_SUBTALKER_DO_SAMPLE", "0") == "1"
            top_p = float(os.getenv("QWEN3_TTS_SUBTALKER_TOP_P", "0.92"))
            top_k = int(os.getenv("QWEN3_TTS_SUBTALKER_TOP_K", "40"))
            temperature = float(os.getenv("QWEN3_TTS_SUBTALKER_TEMPERATURE", "0.8"))

            self._reset_runtime()
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()

            talker_input_embed, trailing_text_hidden, tts_pad_embed = self._build_custom_voice_prompt(
                text,
                speaker,
                language=language,
            )
            attn_mask = torch.ones(
                (talker_input_embed.shape[0], talker_input_embed.shape[1]), dtype=torch.long, device=self._device
            )
            prefill = self._talker(
                inputs_embeds=talker_input_embed,
                attention_mask=attn_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

            next_first = torch.argmax(prefill.logits[:, -1, :], dim=-1).to(torch.long)
            seq_len = int(talker_input_embed.shape[1])
            self._copy_prefill_cache(prefill.past_key_values, seq_len)
            past_hidden = prefill.past_hidden
            generation_step = int(prefill.generation_step)
            talker_cfg = self._model.config.talker_config
            eos_id = int(talker_cfg.codec_eos_token_id)
            bos_id = int(talker_cfg.codec_bos_id)

            emitted_samples = 0
            trailing_silence_samples = 0
            frames: list[torch.Tensor] = []
            for step in range(max_new_tokens):
                next_token = int(next_first.item())
                if next_token == eos_id:
                    if step < min_eos_steps:
                        # Avoid pathological immediate-eos streams that produce no audio.
                        next_token = bos_id
                    else:
                        break

                next_first = torch.tensor([next_token], device=self._device, dtype=torch.long)
                first_hidden = self._talker.get_input_embeddings()(next_first.unsqueeze(0))
                predictor = self._talker.code_predictor.generate(
                    inputs_embeds=torch.cat((past_hidden, first_hidden), dim=1),
                    max_new_tokens=self._talker.config.num_code_groups - 1,
                    do_sample=do_sample,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    output_hidden_states=False,
                    return_dict_in_generate=True,
                )
                frame_codes = torch.cat((next_first.unsqueeze(0), predictor.sequences), dim=-1)[0]
                frames.append(frame_codes)

                codec_hiddens = torch.cat(
                    [first_hidden]
                    + [
                        self._talker.code_predictor.get_input_embeddings()[i](
                            predictor.sequences[..., i : i + 1]
                        )
                        for i in range(self._talker.config.num_code_groups - 1)
                    ],
                    dim=1,
                )
                input_embed = codec_hiddens.sum(1, keepdim=True)
                if generation_step < trailing_text_hidden.shape[1]:
                    input_embed = input_embed + trailing_text_hidden[:, generation_step].unsqueeze(1)
                else:
                    input_embed = input_embed + tts_pad_embed

                self._decode_from_hidden(
                    self._out_token,
                    input_embed[0, 0].to(torch.bfloat16).contiguous(),
                    self._layer_weights_packed,
                    self._final_norm_weight,
                    self._lm_head_weight,
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
                    self._bmax_vals,
                    self._bmax_idxs,
                    self._num_layers,
                    seq_len + step,
                    self._max_seq_len,
                    self._attn_scale,
                )

                next_first = self._out_token.to(torch.long)
                past_hidden = self._norm_out.view(1, 1, -1).to(self._dtype)
                generation_step += 1

                should_decode = False
                n_frames = len(frames)
                if n_frames <= first_chunk_frames:
                    should_decode = True
                elif ((n_frames - first_chunk_frames) % effective_decode_stride) == 0:
                    should_decode = True

                if should_decode:
                    audio_codes = torch.stack(frames, dim=0)
                    wav, sr = self._decode_audio_codes(audio_codes)
                    if wav.shape[0] > emitted_samples:
                        delta = wav[emitted_samples:]
                        emitted_samples = wav.shape[0]
                        stats.frames_generated = len(frames)
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
                                break

            if frames:
                audio_codes = torch.stack(frames, dim=0)
                wav, sr = self._decode_audio_codes(audio_codes)
                if wav.shape[0] > emitted_samples:
                    delta = wav[emitted_samples:]
                    emitted_samples = wav.shape[0]
                    if stats.ttfc_ms is None:
                        ended.record()
                        torch.cuda.synchronize()
                        stats.ttfc_ms = float(started.elapsed_time(ended))
                    yield delta
                stats.audio_seconds = float(wav.shape[0]) / float(sr)

            ended.record()
            torch.cuda.synchronize()
            stats.generation_s = float(started.elapsed_time(ended)) / 1000.0

        return stats, _run()
