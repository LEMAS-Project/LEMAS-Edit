
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from lemas_tts.api import TTS


def build_tokens_from_text(tts: TTS, text: str) -> List[List[str]]:
    """
    Convert raw text into token sequence(s) consistent with the multilingual
    LEMAS-TTS training pipeline.

    We reuse the same frontend logic as in `TTS.infer`:
      - frontend.dtype == "phone" -> TextNorm.text2phn -> split on '|'
      - frontend.dtype == "char"  -> TextNorm.text2norm -> language tag + chars
      - frontend is None          -> simple character sequence as fallback.
    """
    text_proc = text.strip()
    if not text_proc.endswith((".", "。", "!", "？", "?", "！")):
        text_proc = text_proc + "."

    if getattr(tts, "frontend", None) is None:
        tokens = list(text_proc)
        return [tokens]

    dtype = getattr(tts.frontend, "dtype", "phone")

    if dtype == "phone":
        phones = tts.frontend.text2phn(text_proc + " ")
        phones = phones.replace("(cmn)", "(zh)")
        tokens = [tok for tok in phones.split("|") if tok]
        return [tokens]

    if dtype == "char":
        lang, norm = tts.frontend.text2norm(text_proc + " ")
        lang_tag = f"({lang.replace('cmn', 'zh')})"
        tokens = [lang_tag] + list(norm)
        return [tokens]

    # Fallback: character-level
    tokens = list(text_proc)
    return [tokens]


def gen_wav_multilingual(
    tts: TTS,
    segment_audio: torch.Tensor,
    sr: int,
    target_text: str,
    parts_to_edit: List[Tuple[float, float]],
    speed: float = 1.0,
    nfe_step: int = 64,
    cfg_strength: float = 5.0,
    sway_sampling_coef: float = 3.0,
    ref_ratio: float = 1.0,
    no_ref_audio: bool = False,
    use_acc_grl: bool = False,
    use_prosody_encoder_flag: bool = False,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Core editing routine:
      - build an edit mask over the mel frames;
      - run CFM.sample with that mask and the new text;
      - decode mel to waveform via the vocoder.
    """
    device = tts.device
    model = tts.ema_model
    vocoder = tts.vocoder

    mel_spec = getattr(model, "mel_spec", None)
    if mel_spec is None:
        raise RuntimeError("CFM model has no attached MelSpec; check your checkpoint.")

    target_sr = int(mel_spec.target_sample_rate)
    hop_length = int(mel_spec.hop_length)
    target_rms = 0.1

    if segment_audio.dim() == 1:
        audio = segment_audio.unsqueeze(0)
    else:
        audio = segment_audio

    # RMS normalization
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)

    audio = audio.to(device)

    total_frames = audio.shape[-1] // hop_length
    # Start from "keep everything", then carve out spans to re-generate.
    edit_mask = torch.ones(1, total_frames + 1, dtype=torch.bool, device=device)

    # Clamp speed and interpret it as: >1 → faster (shorter edited span),
    # <1 → slower (longer edited span).
    speed_safe = max(float(speed), 1e-3)

    for (start, end) in parts_to_edit:
        # small safety margin around the region to edit
        start_sec = max(start - 0.05, 0.0)
        end_sec = min(end + 0.05, audio.shape[-1] / target_sr)

        start_frame = int(round(start_sec * target_sr / hop_length))
        end_frame = int(round(end_sec * target_sr / hop_length))
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))

        orig_len = end_frame - start_frame
        # 根据 speed 扩展 / 收缩编辑区域长度：scaled_len 越大，模型可用的编辑时长越长
        scaled_len = max(1, int(round(orig_len / speed_safe)))

        # 将编辑区域锚定在原始起点，向右扩展或收缩，避免“吃掉”起始处未选中的词
        new_start = start_frame
        new_end = min(total_frames, new_start + scaled_len)

        edit_mask[:, new_start:new_end] = False

    duration = total_frames

    # Text tokens using multilingual frontend
    final_text_list = build_tokens_from_text(tts, target_text)

    # For multilingual models trained with `separate_langs=True`, we need to
    # post-process the phone sequence so that each non-punctuation token is
    # prefixed with its language id, consistent with training and the main API.
    if hasattr(tts, "process_phone_list") and len(final_text_list) > 0:
        final_text_list = [tts.process_phone_list(final_text_list[0])]
    print("final_text_list:", final_text_list)

    with torch.inference_mode():
        generated, _ = model.sample(
            cond=audio,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
            edit_mask=edit_mask,
            use_acc_grl=use_acc_grl,
            use_prosody_encoder=use_prosody_encoder_flag,
            ref_ratio=ref_ratio,
            no_ref_audio=no_ref_audio,
        )

    generated = generated.to(torch.float32)
    generated_mel = generated.permute(0, 2, 1)  # [B, C, T_mel]

    mel_for_vocoder = generated_mel.to(device)
    if tts.mel_spec_type == "vocos":
        wav_out = vocoder.decode(mel_for_vocoder)
    elif tts.mel_spec_type == "bigvgan":
        wav_out = vocoder(mel_for_vocoder)
    else:
        raise ValueError(f"Unsupported vocoder type: {tts.mel_spec_type}")

    if rms < target_rms:
        wav_out = wav_out * rms / target_rms

    return wav_out.squeeze(0), generated_mel
