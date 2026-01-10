"""
Tokenizer utilities for the LEMAS-Edit codec backend.

This is a lightly adapted copy of the neural codec tokenizer used in our
editing pipeline, wrapped under the `lemas_edit` namespace so that it can
be imported without depending on the original repository layout.
"""

import os
import re
import logging
import math
from dataclasses import dataclass, asdict  # noqa: F401 (kept for compatibility)
from typing import Any, Dict, List, Optional, Pattern, Union  # noqa: F401

import numpy as np  # noqa: F401
import torch
import torchaudio
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

# Configure espeak-ng via espeakng_loader if available so that we can rely on
# the bundled `espeak-ng-data` / `espeak-ng-lib` instead of system installs.
try:
    import espeakng_loader

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    data_path = espeakng_loader.get_data_path()
    os.environ["ESPEAK_DATA_PATH"] = data_path
    os.environ["ESPEAKNG_DATA_PATH"] = data_path
    print("[LEMAS-Edit] espeak-ng configured via espeakng_loader")
except Exception as e:  # ImportError or runtime errors
    print(f"[LEMAS-Edit] espeakng_loader not available or failed ({e}); using system espeak-ng")


class TextTokenizer:
    """Phonemize text using the espeak backend."""

    def __init__(
        self,
        language: str = "en-us",
        backend: str = "espeak",
        separator: Separator = Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation: bool = True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        phonemizer = EspeakBackend(
            language,
            punctuation_marks=punctuation_marks,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress,
            tie=tie,
            language_switch=language_switch,
            words_mismatch=words_mismatch,
        )

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields: List[str] = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip: bool = True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]
        phones: List[List[str]] = []
        for txt in text:
            if txt == "":
                continue
            if txt[0] == "#":
                phones.append(txt)
            else:
                ipa = self.backend.phonemize(
                    [txt],
                    separator=self.separator,
                    strip=strip,
                    njobs=1,
                    logger=logging.basicConfig(level=logging.ERROR),
                )
                phones.append(self.to_list(ipa[0]))
        return phones


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols


_PAUSE_SYMBOL = {"、": ",", "，": ",", "。": ",", "！": "!", "？": "?", "：": ":"}


def _replace(match):
    word = match.group(0)
    return _PAUSE_SYMBOL[word]


def txt2phone(tokenizer: TextTokenizer, text: str) -> str:
    text = re.sub("|".join(_PAUSE_SYMBOL.keys()), _replace, text)
    text = re.split(r"(#\d)", text)
    phones: List[str] = []
    for txt in text:
        if txt == "":
            continue
        if txt[0] == "#":
            phones.append(txt)
        else:
            ipa = tokenizer.backend.phonemize(
                [txt], separator=tokenizer.separator, strip=True, njobs=1
            )
            phones += tokenizer.to_list(ipa[0])
    phones_str = "|".join(phones).replace("(|", "(").replace("|)", ")")
    return phones_str


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


class AudioTokenizer:
    """Neural codec tokenizer used by the LEMAS-Edit codec backend."""

    def __init__(self, device: Any = None, signature: Optional[str] = None) -> None:
        # Some versions of `audiocraft` unconditionally import
        # `EncodecModel` from `transformers`. Older `transformers`
        # releases don't expose this symbol, which would raise an
        # ImportError even though we only use local checkpoints.
        # To keep things self‑contained, we inject a lightweight
        # placeholder if needed before importing audiocraft.
        try:
            import transformers as _tfm  # type: ignore

            if not hasattr(_tfm, "EncodecModel"):
                class _DummyEncodecModel:  # minimal stub, never actually used
                    pass

                _tfm.EncodecModel = _DummyEncodecModel  # type: ignore[attr-defined]
        except Exception:
            # If transformers is not installed at all, audiocraft will
            # still work with local checkpoints; just let it proceed.
            pass

        # Newer `audiocraft` uses Dora/Hydra configs to resolve checkpoint
        # paths in `checkpoint.resolve_checkpoint_path`, which imports
        # `audiocraft.train` and expects a config directory on disk.
        # For simple inference from a concrete `.th` path, this indirection
        # is unnecessary and can fail when configs are not packaged.
        # We therefore monkey‑patch `resolve_checkpoint_path` so that it
        # just returns the given path, avoiding the Hydra dependency.
        try:
            from pathlib import Path as _Path
            import audiocraft.utils.checkpoint as _ac_ckpt  # type: ignore

            if not getattr(_ac_ckpt, "_lemas_simple_resolve", False):
                def _simple_resolve(sig_or_path, name=None, use_fsdp=False):
                    return _Path(str(sig_or_path))

                _ac_ckpt.resolve_checkpoint_path = _simple_resolve  # type: ignore[assignment]
                _ac_ckpt._lemas_simple_resolve = True  # type: ignore[attr-defined]
        except Exception:
            # If audiocraft is missing, the caller will fail later in a
            # clearer way; no need to stop here.
            pass

        from audiocraft.solvers import CompressionSolver

        model = CompressionSolver.model_from_checkpoint(signature)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device
        self.codec = model.to(device)

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        codes = self.codec.encode(wav.to(self.device))
        return [(codes[0], None)]

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames[0][0]  # [1,4,T]
        return self.codec.decode(frames)


def tokenize_audio(tokenizer: AudioTokenizer, audio, offset: int = -1, num_frames: int = -1):
    # Load and pre-process the audio waveform
    if isinstance(audio, str):
        if offset != -1 and num_frames != -1:
            wav, sr = torchaudio.load(audio, frame_offset=offset, num_frames=num_frames)
        else:
            wav, sr = torchaudio.load(audio)
        wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
        wav = wav.unsqueeze(0)
    else:
        wav = audio.unsqueeze(0).unsqueeze(0)
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames


class AudioSR:
    """DAC-based super-resolution codec used for upsampling generated frames."""

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        import dac

        self.codec = dac.DAC.load(model_path)
        self.codec.to(device)
        self.codec.eval()

        self.sample_rate = self.codec.sample_rate
        self.channels = 1
        self._device = device

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        length = wav.shape[-1]
        right_pad = math.ceil(length / self.codec.hop_length) * self.codec.hop_length - length
        wav = torch.nn.functional.pad(wav, (0, right_pad))
        z, codes, _, _, _ = self.codec.encode(wav.to(self._device))
        return [(codes, z)]

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        z = frames[0][1]  # [1, 2048, T]
        with torch.no_grad():
            y = self.codec.decode(z)
        return y
