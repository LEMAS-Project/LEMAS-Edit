"""
Autoregressive codec speech editing demo CLI for LEMAS-Edit HF.

This script runs the autoregressive edit model on the LEMAS-Edit demo set,
using the same Azure alignment JSONs as the CFM-based
`lemas_tts/scripts/speech_edit_multilingual.py`.

For each (audio, json) pair it:
  - reads the full utterance and the edit span from the JSON,
  - builds `raw_text` / `target_text` from `display_text` + `modified_text`,
  - constructs an edit mask over codec frames,
  - calls `edit_codec_inference_one_sample` from `lemas_edit.infer`,
  - upsamples and saves the edited waveform.

Example (from repo root):
    bash lemas_edit/scripts/inference_lemas_editing.sh
or equivalently:
    python -m lemas_edit.scripts.inference_lemas_editing \\
        --wav_dir pretrained_models/demos/lemas_edit_test/vocals \\
        --align_dir pretrained_models/demos/lemas_edit_test/align \\
        --save_dir pretrained_models/demos/lemas_edit_test/Autoregressive_Edit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from lemas_edit.infer.edit_infer import (
    EditCodecHandle,
    load_edit_codec_model,
    edit_codec_inference_one_sample,
    audio_upsampling,
)


LOGGER = logging.getLogger("lemas_edit_cli")


def seed_everything(seed: int) -> None:
    if seed != -1:
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_wav_mono(path: str, target_sr: int | None = None) -> Tuple[torch.Tensor, int]:
    """Load an audio file, convert to mono, optionally resample."""
    wav, sr = torchaudio.load(path)
    if wav.dim() > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if target_sr is not None and sr != target_sr:
        wav = torchaudio.functional.resample(wav.squeeze(0), sr, target_sr).unsqueeze(0)
        sr = target_sr
    wav = torch.clip(wav, -0.999, 0.999)
    return wav.squeeze(0), sr


def lang_from_original_language(orig: str, fallback: str = "en") -> str:
    """
    Map Azure `original_language` like 'en-US' or 'zh-CN' to a short code
    used by the autoregressive pipeline ('en', 'zh', ...).
    """
    if not orig:
        return fallback
    code = orig.split("-")[0].lower()
    return code


def collect_pairs(
    wav: str | None,
    wav_dir: str,
    align_dir: str,
    save_dir: str,
) -> List[Tuple[str, str, str]]:
    """
    Build a list of (wav_path, json_path, save_path) triples.
    If `wav` is provided, only that file is used; otherwise, all .wav/.mp3 in wav_dir.
    """
    pairs: List[Tuple[str, str, str]] = []

    if wav is not None:
        wav_paths = [wav]
    else:
        wav_paths = [
            os.path.join(wav_dir, f)
            for f in os.listdir(wav_dir)
            if f.lower().endswith(".wav") or f.lower().endswith(".mp3")
        ]
        wav_paths.sort()

    for wp in wav_paths:
        base = os.path.splitext(os.path.basename(wp))[0]
        jp = os.path.join(align_dir, base + ".json")
        sp = os.path.join(save_dir, base + ".wav")
        pairs.append((wp, jp, sp))

    return pairs


def run_edit_for_pair(
    handle: EditCodecHandle,
    wav_path: str,
    json_path: str,
    save_path: str,
    *,
    top_p: float,
    temperature: float,
    stop_repetition: int,
    repetition_penalty: float,
    left_margin: float,
    right_margin: float,
    kvcache: int,
) -> None:
    """
    Run AR codec editing for a single (wav, json) pair.

    We follow the same JSON convention as `speech_edit_multilingual.py`:
      - `interval`: utterance span [start_sec, end_sec]
      - `modified_index`: [start_word_idx, end_word_idx] in `words`
      - `modified_text`: [orig_phrase, new_phrase]
      - `display_text`: full original sentence
      - `original_language`: e.g. 'en-US', 'zh-CN'
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Sampling rates from codec backend
    codec_audio_sr = int(handle.audio_tokenizer.sample_rate)
    codec_sr = 50  # EnCodec 4x2048 -> ~50 frames per second

    # Load audio at original SR for output, and at codec SR for editing
    raw_audio, raw_sr = load_wav_mono(wav_path, target_sr=None)
    audio_codec, _ = load_wav_mono(wav_path, target_sr=codec_audio_sr)
    audio_dur = audio_codec.shape[-1] / codec_audio_sr

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Language + texts
    orig_lang = data.get("original_language", "")
    lang = lang_from_original_language(orig_lang, fallback="en")
    if lang not in {"de", "fr", "pt", "es", "it", "en", "zh"}:
        LOGGER.warning("Skipping %s (unsupported language: %s)", wav_path, lang)
        return

    raw_text: str = data["display_text"]
    orig_phrase, new_phrase = data["modified_text"]
    target_text = raw_text.replace(orig_phrase, new_phrase)

    LOGGER.info(
        "\n[EDIT] %s\n  lang         : %s\n  display_text : %s\n  modified_text: %r -> %r\n  target_text  : %s",
        os.path.basename(wav_path),
        lang,
        raw_text,
        orig_phrase,
        new_phrase,
        target_text,
    )

    # Utterance-level interval [sec]
    utt_start_sec, utt_end_sec = data["interval"]

    # Determine edit span from modified_index
    start_idx, end_idx = data["modified_index"]
    words = data["words"]
    start_idx = max(0, start_idx)
    end_idx = min(len(words), end_idx)
    if not words or start_idx >= end_idx:
        LOGGER.warning("Empty or invalid modified_index for %s", wav_path)
        return

    word_start_sec = float(words[start_idx]["interval"][0])
    word_end_sec = float(words[end_idx - 1]["interval"][1])

    # Slightly expand around edited phrase (similar spirit to CFM script)
    edit_start = max(0.0, word_start_sec - 0.1)
    edit_end = min(word_end_sec + 0.1, utt_end_sec + 0.1)

    # Clamp to audio duration
    edit_start = max(0.0, min(edit_start, audio_dur))
    edit_end = max(edit_start, min(edit_end, audio_dur))

    # Build mask interval in codec frames, with extra safety margins
    span_start = max(edit_start - float(left_margin), 1.0 / codec_sr)
    span_end = min(edit_end + float(right_margin), audio_dur)
    mask_start = round(span_start * codec_sr)
    mask_end = round(span_end * codec_sr)

    mask_interval = torch.LongTensor([[mask_start, mask_end]])
    LOGGER.info(
        "  edit_span    : [%.3f, %.3f] sec -> mask_interval=[%d, %d] frames",
        edit_start,
        edit_end,
        mask_start,
        mask_end,
    )

    raw_transcript = [lang, raw_text]
    target_transcript = [[lang, target_text]]

    decode_config = {
        "top_k": -1,
        "top_p": float(top_p),
        "temperature": float(temperature),
        "stop_repetition": int(stop_repetition),
        "kvcache": int(kvcache),
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": [1388, 1898, 131],
        "repetition_penalty": float(repetition_penalty),
    }

    LOGGER.info("  running codec edit inference...")
    generated_sample, num_gen = edit_codec_inference_one_sample(
        handle,
        audio_codec,
        raw_transcript,
        target_transcript,
        mask_interval,
        decode_config,
    )
    LOGGER.info(
        "  generated codec frames: %d (%.3f sec @ %d Hz)",
        int(num_gen),
        int(num_gen) / codec_sr,
        codec_audio_sr,
    )

    gen_audio = generated_sample.cpu().squeeze()
    edited = audio_upsampling(gen_audio, handle.audiosr_model, codec_audio_sr, raw_sr)
    edited = torch.clip(edited, -0.999, 0.999)

    torchaudio.save(save_path, edited.unsqueeze(0), raw_sr, format="wav")
    LOGGER.info("  saved: %s", save_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autoregressive codec-based speech editing demo (LEMAS-Edit HF)."
    )

    # Data paths (mirroring F5-TTS / VoiceCraft demo layout)
    parser.add_argument(
        "--wav",
        type=str,
        default=None,
        help="Path to a single input wav/mp3. If set, only this file is processed.",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default="pretrained_models/demos/lemas_edit_test/vocals",
        help="Directory containing input wavs/mp3s.",
    )
    parser.add_argument(
        "--align_dir",
        type=str,
        default="pretrained_models/demos/lemas_edit_test/align",
        help="Directory containing alignment JSONs.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="pretrained_models/demos/lemas_edit_test/Autoregressive_Edit",
        help="Directory to save edited wavs.",
    )

    # Model paths
    parser.add_argument(
        "--model_path",
        type=str,
        default="pretrained_models/ckpts/autoregressive/multilingual_330M.pth",
        help="Path to autoregressive edit model checkpoint.",
    )

    # Sampling parameters (roughly mirroring UI defaults)
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Top-p (nucleus) sampling for codec backend.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for autoregressive sampling.",
    )
    parser.add_argument(
        "--stop_repetition",
        type=int,
        default=1,
        help="Silence repetition control; <=0 disables, >0 reduces long silences.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.5,
        help="Penalty factor for repeated tokens (first codebook).",
    )
    parser.add_argument(
        "--left_margin",
        type=float,
        default=0.01,
        help="Extra seconds on the left side of the edit span (safety margin).",
    )
    parser.add_argument(
        "--right_margin",
        type=float,
        default=0.01,
        help="Extra seconds on the right side of the edit span (safety margin).",
    )
    parser.add_argument(
        "--kvcache",
        type=int,
        default=1,
        help="1 to enable KV cache (faster, more VRAM), 0 to disable.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for random).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s",
        level=logging.INFO,
    )

    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device)

    # Resolve paths relative to repo root if needed
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(repo_root, model_path)
    models_root = os.path.dirname(model_path)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Resolve data dirs
    if args.wav_dir is not None and not os.path.isabs(args.wav_dir):
        wav_dir = os.path.join(repo_root, args.wav_dir)
    else:
        wav_dir = args.wav_dir
    if args.align_dir is not None and not os.path.isabs(args.align_dir):
        align_dir = os.path.join(repo_root, args.align_dir)
    else:
        align_dir = args.align_dir
    if args.save_dir is not None and not os.path.isabs(args.save_dir):
        save_dir = os.path.join(repo_root, args.save_dir)
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    if wav_dir is None or align_dir is None:
        raise ValueError("Both --wav_dir and --align_dir must be set.")

    if args.wav is not None and not os.path.isabs(args.wav):
        wav_single = os.path.join(repo_root, args.wav)
    else:
        wav_single = args.wav

    handle: EditCodecHandle = load_edit_codec_model(
        model_path=model_path,
        models_root=models_root,
        device=device,
    )

    pairs = collect_pairs(
        wav=wav_single,
        wav_dir=wav_dir,
        align_dir=align_dir,
        save_dir=save_dir,
    )

    if not pairs:
        LOGGER.warning("No input audio found in %s", wav_dir)
        return

    for wav_path, json_path, save_path in tqdm(pairs):
        if not os.path.exists(wav_path):
            LOGGER.warning("wav not found: %s", wav_path)
            continue
        if not os.path.exists(json_path):
            LOGGER.warning("json not found: %s", json_path)
            continue

        try:
            run_edit_for_pair(
                handle=handle,
                wav_path=wav_path,
                json_path=json_path,
                save_path=save_path,
                top_p=args.top_p,
                temperature=args.temperature,
                stop_repetition=args.stop_repetition,
                repetition_penalty=args.repetition_penalty,
                left_margin=args.left_margin,
                right_margin=args.right_margin,
                kvcache=args.kvcache,
            )
        except Exception as exc:  # defensive logging
            LOGGER.exception("Error processing %s: %s", wav_path, exc)


if __name__ == "__main__":
    main()
