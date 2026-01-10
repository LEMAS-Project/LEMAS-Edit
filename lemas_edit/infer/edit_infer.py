"""
Neural codec based speech editing helpers for LEMAS-Edit.

This module wraps the minimal pieces we need from the codec infilling
inference code so that the Gradio app can call into a clean API:

  - load_edit_codec_model(...)
  - edit_codec_inference_one_sample(...)
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torchaudio

from lemas_edit.infer.tokenizer import (
    AudioSR,
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    txt2phone,
)


@dataclass
class EditCodecHandle:
    model: Any
    config: Any
    phn2num: Dict[str, int]
    text_tokenizer: Dict[str, Tuple[str, TextTokenizer]]
    audio_tokenizer: AudioTokenizer
    audiosr_model: AudioSR
    device: torch.device


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_edit_codec_model(
    model_path: str,
    models_root: str,
    device: str | None = None,
) -> EditCodecHandle:
    """
    Load the neural codec model + tokenizers for editing, mirroring the
    original codec-based backend but returning a typed handle instead of
    using global state.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    models_root = os.path.abspath(models_root)

    # Encodec checkpoint for the audio tokenizer
    encodec_fn = os.path.join(models_root, "encodec_4cb2048_giga.th")

    # Use the vendored codec model under lemas_edit.model
    from lemas_edit.model import model as _edit_model

    ckpt = torch.load(model_path, map_location="cpu")
    model = _edit_model.LemasEditCodec(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    phn2num = ckpt["phn2num"]

    config = model.args
    model.to(device)
    model.eval()
    logging.info("Loaded edit codec model from %s", model_path)

    langs = {
        "en": "en-us",
        "zh": "cmn",
        "it": "it",
        "es": "es",
        "pt": "pt-br",
        "fr": "fr-fr",
        "de": "de",
    }
    text_tokenizer: Dict[str, Tuple[str, TextTokenizer]] = {}
    for k, v in langs.items():
        tokenizer = TextTokenizer(language=v, backend="espeak")
        lang = "cmn" if k == "zh" else k
        text_tokenizer[k] = (lang, tokenizer)

    audio_tokenizer = AudioTokenizer(signature=encodec_fn)
    audiosr_model = AudioSR(model_path=os.path.join(models_root, "dac_SR_8codes_2048_hop960_speech.pth"))

    return EditCodecHandle(
        model=model,
        config=config,
        phn2num=phn2num,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        audiosr_model=audiosr_model,
        device=torch.device(device),
    )


@torch.no_grad()
def edit_codec_inference_one_sample(
    handle: EditCodecHandle,
    audio,
    raw_transcript,
    target_transcript,
    mask_interval: torch.Tensor,
    decode_config: Dict[str, Any],
) -> Tuple[torch.Tensor, int]:
    """
    Port of the original `inference_one_sample` for codec infilling,
    adapted to use the EditCodecHandle structure and numpy/torch inputs.
    """
    model = handle.model
    model_args = handle.config
    phn2num = handle.phn2num
    text_tokenizer = handle.text_tokenizer
    audio_tokenizer = handle.audio_tokenizer
    device = handle.device

    # raw_transcript: [lang, text]
    lang, raw_text = raw_transcript
    raw_text = txt2phone(
        text_tokenizer[lang][1],
        raw_text.strip().replace(".", ",").replace("。", ","),
    ).split("|")
    raw_text = [f"({text_tokenizer[lang][0]})"] + raw_text

    target_text: List[str] = []
    for text_info in target_transcript:
        txt = re.sub(
            r"\s+",
            " ",
            text_info[1].strip().replace(".", ",").replace("。", ","),
        )
        txt = txt2phone(text_tokenizer[text_info[0]][1], txt).split("|")
        txt = [f"({text_tokenizer[text_info[0]][0]})"] + txt
        target_text += txt

    logging.info(
        "raw_text: %s\n target_transcript: %s\n target_text: %s",
        raw_transcript,
        target_transcript,
        target_text,
    )

    raw_text_tokens = [phn2num[phn] for phn in raw_text if phn in phn2num]
    text_tokens = [phn2num[phn] for phn in target_text if phn in phn2num]

    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

    # Tokenize audio with the neural codec
    encoded_frames = tokenize_audio(audio_tokenizer, audio)
    original_audio = encoded_frames[0][0].transpose(2, 1)  # [1, T, K]
    assert (
        original_audio.ndim == 3
        and original_audio.shape[0] == 1
        and original_audio.shape[2] == model_args.n_codebooks
    ), original_audio.shape

    avg_speed = encoded_frames[0][0].shape[-1] / len(raw_text_tokens)
    logging.info(
        "avg_speed=%.4f max_rate=%d codec_frames=%d src_text_tokens=%d tar_text_tokens=%s",
        float(avg_speed),
        round(avg_speed) + 1,
        encoded_frames[0][0].shape[-1],
        len(raw_text_tokens),
        text_tokens.shape,
    )

    n = 0
    RE_GEN = True
    repetition_penalty = decode_config["repetition_penalty"]
    logging.info("before mask_interval=%s repetition_penalty: %s", list(mask_interval), repetition_penalty)

    codec_sr = decode_config.get("codec_sr", 50)

    while RE_GEN and n < 3:
        stime = time.time()
        n += 1
        encoded_frames, num_gen, RE_GEN = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            original_audio[..., : model_args.n_codebooks].to(device),  # [1, T, K]
            mask_interval=mask_interval.unsqueeze(0).to(device),
            top_k=decode_config["top_k"],
            top_p=decode_config["top_p"],
            temperature=decode_config["temperature"],
            stop_repetition=decode_config["stop_repetition"],
            kvcache=decode_config["kvcache"],
            silence_tokens=decode_config["silence_tokens"],
            max_rate=round(avg_speed) * 1.5,
            repetition_penalty=repetition_penalty,
        )  # output is [1, K, T]
        gen_time = time.time() - stime
        logging.info("after mask_interval=%s", list(mask_interval))
        if encoded_frames.shape[-1] / text_tokens.shape[-1] < avg_speed * 0.5:
            RE_GEN = True
        logging.info(
            "Round %d RE_GEN:%s. Inference time %.3f sec. gen_length: %.3f sec. RTF: %.3f",
            n,
            RE_GEN,
            gen_time,
            int(num_gen[0]) / codec_sr,
            int(num_gen[0]) / codec_sr / gen_time,
        )
        if RE_GEN:
            mask_interval[0][0] = max(0, mask_interval[0][0] - n)
            mask_interval[0][1] = min(original_audio.shape[1], mask_interval[0][1] + n)
            repetition_penalty += 0.1

    if isinstance(encoded_frames, tuple):
        encoded_frames = encoded_frames[0]
    logging.info(
        "Audio total length: %.3f sec.",
        encoded_frames.shape[-1] / codec_sr,
    )

    # Decode generated codec frames back to waveform at codec_audio_sr
    generated_sample = audio_tokenizer.decode([(encoded_frames, None)])

    return generated_sample, int(num_gen[0])


def audio_upsampling(audio: torch.Tensor, audiosr_model: AudioSR, src_sr: int, tar_sr: int) -> torch.Tensor:
    """
    Upsample generated codec audio back to target sampling rate.
    """
    if src_sr == tar_sr:
        return audio
    resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=tar_sr)
    return resampler(audio.unsqueeze(0)).squeeze(0)
