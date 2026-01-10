import os, gc
import re, time
import logging
import importlib as _importlib
from num2words import num2words
import gradio as gr
import torch, torchaudio
import numpy as np
import random
from scipy.io import wavfile
import onnx
import onnxruntime as ort
import copy
import uroman as ur
import jieba, zhconv
from pypinyin.core import Pinyin
from pypinyin import Style
from cached_path import cached_path
from lemas_tts.api import TTS, PRETRAINED_ROOT, CKPTS_ROOT
from lemas_tts.infer.edit_multilingual import gen_wav_multilingual
from lemas_tts.infer.text_norm.txt2pinyin import (
    MyConverter,
    _PAUSE_SYMBOL,
    change_tone_in_bu_or_yi,
    get_phoneme_from_char_and_pinyin,
)
from lemas_tts.infer.text_norm.cn_tn import NSWNormalizer

# Codec-based autoregressive edit backend (vendored from lemas_edit).
_codec_backend = _importlib.import_module("lemas_edit.infer.edit_infer")
EditCodecHandle = _codec_backend.EditCodecHandle
load_edit_codec_model = _codec_backend.load_edit_codec_model
edit_codec_inference_one_sample = _codec_backend.edit_codec_inference_one_sample
audio_upsampling = _codec_backend.audio_upsampling
# import io
# import uuid
_JIEBA_DICT = os.path.join(
    os.path.dirname(__file__),
    "lemas_tts",
    "infer",
    "text_norm",
    "jieba_dict.txt",
)
if os.path.isfile(_JIEBA_DICT):
    jieba.set_dictionary(_JIEBA_DICT)

# from inference_tts_scale import inference_one_sample as inference_tts
import langid
langid.set_languages(['es','pt','zh','en','de','fr','it', 'ru', 'id', 'vi'])


os.environ['CURL_CA_BUNDLE'] = ''
DEMO_PATH = os.getenv("DEMO_PATH", "./pretrained_models/demos")
TMP_PATH = os.getenv("TMP_PATH", "./pretrained_models/demos/temp")
MODELS_PATH = os.getenv("MODELS_PATH", "./pretrained_models")

# HF location for large TTS checkpoints (too big for Space storage).
# Mirrors LEMAS-TTS `inference_gradio.py`.
HF_PRETRAINED_ROOT = "hf://LEMAS-Project/LEMAS-TTS/pretrained_models"

def _pick_device():
    forced = os.getenv("LEMAS_DEVICE")
    if forced:
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"

device = _pick_device()

whisper_model, align_model = None, None
tts_edit_model = None
codec_handle = None  # neural codec autoregressive edit backend

_whitespace_re = re.compile(r"\s+")
alpha_pattern = re.compile(r"[a-zA-Z]")

formatter = ("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s")
logging.basicConfig(format=formatter, level=logging.INFO)

# def get_random_string():
#     return "".join(str(uuid.uuid4()).split("-"))

def seed_everything(seed):
    if seed != -1:
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class UVR5:
    """Small wrapper around the bundled uvr5 implementation for denoising."""

    def __init__(self, model_dir):
        # Code directory is always the local `uvr5` folder in this repo
        self.code_dir = os.path.join(os.path.dirname(__file__), "uvr5")
        self.model_dir = model_dir
        self.model = None
        self.device = "cpu"
    
    def load_model(self, device="cpu"):
        import sys, json, os, torch
        if self.code_dir not in sys.path:
            sys.path.append(self.code_dir)

        # Reuse an already-loaded model if it matches the requested device.
        if self.model is not None and self.device == device:
            return self.model

        from multiprocess_cuda_infer import ModelData, Inference
        # In the minimal LEMAS-TTS layout, UVR5 weights live under:
        model_path = os.path.join(self.model_dir, "Kim_Vocal_1.onnx")
        config_path = os.path.join(self.model_dir, "MDX-Net-Kim-Vocal1.json")
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
        model_data = ModelData(
            model_path=model_path,
            audio_path=self.model_dir,
            result_path=self.model_dir,
            device=device,
            process_method="MDX-Net",
            # Keep base_dir and model_dir the same so all UVR5 metadata
            # (model_data.json, model_name_mapper.json, etc.) are resolved
            # under `pretrained_models/uvr5`, matching LEMAS-TTS inference.
            base_dir=self.model_dir,
            **configs,
        )

        uvr5_model = Inference(model_data, device)
        uvr5_model.load_model(model_path, 1)

        self.model = uvr5_model
        self.device = device
        return self.model
        
    def denoise(self, audio_info):
        model = self.load_model(device="cpu")
        input_audio = load_wav(audio_info, sr=44100, channel=2)
        output_audio = model.demix_base({0:input_audio.squeeze()}, is_match_mix=False, device="cpu")
        # transform = torchaudio.transforms.Resample(44100, 16000)
        # output_audio = transform(output_audio)
        return output_audio.squeeze().T.cpu().numpy(), 44100


class DeepFilterNet:
    def __init__(self, model_path):
        self.hop_size = 480
        self.fft_size = 960
        self.model = self.load_model(model_path)


    def load_model(self, model_path, threads=1):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = threads
        sess_options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        model = onnx.load_model(model_path)
        ort_session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options,
            providers=["CPUExecutionProvider"], # ["CUDAExecutionProvider"], #
        )

        input_names = ["input_frame", "states", "atten_lim_db"]
        output_names = ["enhanced_audio_frame", "new_states", "lsnr"]
        return ort_session
        

    def denoise(self, audio_info):
        wav = load_wav(audio_info, 48000)
        orig_len = wav.shape[-1]
        hop_size_divisible_padding_size = (self.hop_size - orig_len % self.hop_size) % self.hop_size
        orig_len += hop_size_divisible_padding_size
        wav = torch.nn.functional.pad(
            wav, (0, self.fft_size + hop_size_divisible_padding_size)
        )
        chunked_audio = torch.split(wav, self.hop_size)
        # chunked_audio = torch.split(wav, int(wav.shape[-1]/2))

        state = np.zeros(45304,dtype=np.float32)
        atten_lim_db = np.zeros(1,dtype=np.float32)
        enhanced = []
        for frame in chunked_audio:
            out = self.model.run(None,input_feed={"input_frame":frame.numpy(),"states":state,"atten_lim_db":atten_lim_db})
            enhanced.append(torch.tensor(out[0]))
            state = out[1]

        enhanced_audio = torch.cat(enhanced).unsqueeze(0)  # [t] -> [1, t] typical mono format

        d = self.fft_size - self.hop_size
        enhanced_audio = enhanced_audio[:, d: orig_len + d]

        return enhanced_audio.squeeze().numpy(), 48000


class TextNorm():
    def __init__(self):
        my_pinyin = Pinyin(MyConverter())
        self.pinyin_parser = my_pinyin.pinyin

    def sil_type(self, time_s):
        if round(time_s) < 0.4:
            return ""
        elif round(time_s) >= 0.4 and round(time_s) < 0.8:
            return "#1"
        elif round(time_s) >= 0.8 and round(time_s) < 1.5:
            return "#2"
        elif round(time_s) >= 1.5 and round(time_s) < 3.0:
            return "#3"
        elif round(time_s) >= 3.0:
            return "#4"


    def add_sil_raw(self, sub_list, start_time, end_time, target_transcript):
        txt = []
        txt_list = [x["word"] for x in sub_list]
        sil = self.sil_type(sub_list[0]["start"])
        if len(sil) > 0:
            txt.append(sil)
        txt.append(txt_list[0])
        for i in range(1, len(sub_list)):
            if sub_list[i]["start"] >= start_time and sub_list[i]["end"] <= end_time:
                txt.append(target_transcript)
                target_transcript = ""
            else:
                sil = self.sil_type(sub_list[i]["start"] - sub_list[i-1]["end"])
                if len(sil) > 0:
                    txt.append(sil)
                txt.append(txt_list[i])
        return ' '.join(txt)

    def add_sil(self, sub_list, start_time, end_time, target_transcript, src_lang, tar_lang):
        txts = []
        txt_list = [x["word"] for x in sub_list]
        sil = self.sil_type(sub_list[0]["start"])
        if len(sil) > 0:
            txts.append([src_lang, sil])

        if sub_list[0]["start"] < start_time:
            txts.append([src_lang, txt_list[0]])
        for i in range(1, len(sub_list)):
            if sub_list[i]["start"] >= start_time and sub_list[i]["end"] <= end_time:
                txts.append([tar_lang, target_transcript])
                target_transcript = ""
            else:
                sil = self.sil_type(sub_list[i]["start"] - sub_list[i-1]["end"])
                if len(sil) > 0:
                    txts.append([src_lang, sil])
                txts.append([src_lang, txt_list[i]])
                
        target_txt = [txts[0]]
        for txt in txts[1:]:
            if txt[1] == "":
                continue
            if txt[0] != target_txt[-1][0]:
                target_txt.append([txt[0], ""])
            target_txt[-1][-1] += " " + txt[1]
        
        return target_txt


    def get_prompt(self, sub_list, start_time, end_time, src_lang):
        txts = []
        txt_list = [x["word"] for x in sub_list]

        if start_time <= sub_list[0]["start"]:
            sil = self.sil_type(sub_list[0]["start"])
            if len(sil) > 0:
                txts.append([src_lang, sil])
            txts.append([src_lang, txt_list[0]])
        
        for i in range(1, len(sub_list)):
            # if sub_list[i]["start"] <= start_time and sub_list[i]["end"] <= end_time:
            #     txts.append([tar_lang, target_transcript])
            #     target_transcript = ""
            if sub_list[i]["start"] >= start_time and sub_list[i]["end"] <= end_time:
                sil = self.sil_type(sub_list[i]["start"] - sub_list[i-1]["end"])
                if len(sil) > 0:
                    txts.append([src_lang, sil])
                txts.append([src_lang, txt_list[i]])

        target_txt = [txts[0]]
        for txt in txts[1:]:
            if txt[1] == "":
                continue
            if txt[0] != target_txt[-1][0]:
                target_txt.append([txt[0], ""])
            target_txt[-1][-1] += " " + txt[1]
        return target_txt


    def txt2pinyin(self, text):
        txts, phonemes = [], []
        texts = re.split(r"(#\d)", text)
        print("before norm: ", texts)
        for text in texts:
            if text in {'#1', '#2', '#3', '#4'}:
                txts.append(text)
                phonemes.append(text)
                continue
            text = NSWNormalizer(text.strip()).normalize()
            
            text_list = list(jieba.cut(text))
            print("jieba cut: ", text, text_list)
            for words in text_list:
                if words in _PAUSE_SYMBOL:
                    # phonemes.append('#2')
                    phonemes[-1] += _PAUSE_SYMBOL[words]
                    txts[-1] += words
                elif re.search("[\u4e00-\u9fa5]+", words):
                    pinyin = self.pinyin_parser(words, style=Style.TONE3, errors="ignore")
                    new_pinyin = []
                    for x in pinyin:
                        x = "".join(x)
                        if "#" not in x:
                            new_pinyin.append(x)
                        else:
                            phonemes.append(words)
                            continue
                    new_pinyin = change_tone_in_bu_or_yi(words, new_pinyin) if len(words)>1 and words[-1] not in {"一","不"} else new_pinyin
                    phoneme = get_phoneme_from_char_and_pinyin(words, new_pinyin)
                    phonemes += phoneme
                    txts += list(words)
                elif re.search(r"[a-zA-Z]", words) or re.search(r"#[1-4]", words):
                    phonemes.append(words)
                    txts.append(words)
                    # phonemes.append("#1")
        # phones = " ".join(phonemes)
        return txts, phonemes
    


def chunk_text(text, max_chars=135):
    """
    Splits the input text into chunks, each with a maximum number of characters.
    Args:
        text (str): The text to be split.
        max_chars (int): The maximum number of characters per chunk.
    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


class MMSAlignModel:
    def __init__(self):
        from torchaudio.pipelines import MMS_FA as bundle
        self.mms_model = bundle.get_model()
        # Keep MMS on the same device as the main edit model unless overridden.
        self.mms_model.to(device)
        self.mms_tokenizer = bundle.get_tokenizer()
        self.mms_aligner = bundle.get_aligner()
        self.text_normalizer = ur.Uroman() 


    def text_normalization(self, text_list):
        text_normalized = []
        for word in text_list:
            text_char = ''
            for c in word:
                if c.isalpha() or c=="'":
                    text_char += c.lower()
                elif c == "-":
                    text_char += '*'
            text_char = text_char if len(text_char) > 0 else "*"
            text_normalized.append(text_char)
        assert len(text_normalized) == len(text_list), f"normalized text len != raw text len: {len(text_normalized)} != {text_list}"
        return text_normalized

    def compute_alignments(self, waveform: torch.Tensor, tokens):
        with torch.inference_mode():
            emission, _ = self.mms_model(waveform.to(device))
            token_spans = self.mms_aligner(emission[0], tokens)
        return emission, token_spans


    def align(self, data, wav):
        waveform = load_wav(wav, 16000).unsqueeze(0)
        raw_text = data['text'][0]
        text = " ".join(data['text'][1]).replace("-", " ")
        text = re.sub("\s+", " ", text)
        text_normed = self.text_normalizer.romanize_string(text, lcode=data["lang"])
        # text_normed = re.sub("[\d_.,!$£%?#−/]", '', text_normed)
        fliter = re.compile("[^a-z^*^'^ ]")
        text_normed = fliter.sub('', text_normed.lower())
        text_normed = re.sub("\s+", " ", text_normed)
        text_normed = text_normed.split()
        assert len(text_normed) == len(raw_text), f"normalized text len != raw text len: {len(text_normed)} != {len(raw_text)}"
        tokens = self.mms_tokenizer(text_normed)
        with torch.inference_mode():
            emission, _ = self.mms_model(waveform.to(device))
            token_spans = self.mms_aligner(emission[0], tokens)
        num_frames = emission.size(1)
        ratio = waveform.size(1) / num_frames
        res = []
        for i in range(len(token_spans)):
            score = round(sum([x.score for x in token_spans[i]]) / len(token_spans[i]), ndigits=3)
            start = round(waveform.size(-1) * token_spans[i][0].start / num_frames / 16000, ndigits=3)
            end = round(waveform.size(-1) * token_spans[i][-1].end / num_frames / 16000, ndigits=3)
            res.append({"word": raw_text[i], "start": start, "end": end, "score": score})

        res = {"lang":data["lang"], "start": 0, "end": round(waveform.shape[-1]/16000, ndigits=3), "text_raw":data["text_raw"], "text": text, "words": res}
        return res


class WhisperxModel:
    def __init__(self, model_name):
        # Lazily construct the WhisperX pipeline.
        self.model_name = model_name
        self.model = None        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_model(self):
        if self.model is not None:
            return
        from whisperx import load_model
        from pathlib import Path
        import hashlib
        import whisperx.vad as whisperx_vad
        import torch

        prompt = None  # "This might be a blend of Simplified Chinese and English speech, do not translate, only transcription be allowed."

        # In this repo layout, a local copy of the VAD segmentation weights is
        # bundled so that WhisperX does not need to download it at runtime.
        # We patch the expected SHA256 in whisperx.vad so checksum validation
        # passes and force `load_model` to use this local file.
        vad_fp = None
        candidates = [
            Path(PRETRAINED_ROOT) / "whisperx-vad-segmentation.bin",
            Path(PRETRAINED_ROOT) / "whisperx" / "whisperx-vad-segmentation.bin",
            Path(MODELS_PATH) / "whisperx-vad-segmentation.bin",
            Path(MODELS_PATH) / "whisperx" / "whisperx-vad-segmentation.bin",
        ]
        try:
            # Also check the default torch hub cache location used by whisperx.vad
            hub_root = Path(torch.hub._get_torch_home())
            candidates.append(hub_root / "whisperx-vad-segmentation.bin")
        except Exception:
            pass

        for candidate in candidates:
            if candidate.is_file():
                vad_fp = str(candidate)
                try:
                    with open(vad_fp, "rb") as f:
                        sha = hashlib.sha256(f.read()).hexdigest()
                    # whisperx.vad checks VAD_SEGMENTATION_URL.split('/')[-2]
                    whisperx_vad.VAD_SEGMENTATION_URL = f"https://local/{sha}/pytorch_model.bin"
                except Exception as e:
                    logging.warning("Failed to patch whisperx VAD checksum: %s", e)
                break

        self.model = load_model(
            self.model_name,
            self.device,
            compute_type="float32",
            asr_options={
                "suppress_numerals": False,
                "max_new_tokens": None,
                "clip_timestamps": None,
                "initial_prompt": prompt,
                "append_punctuations": ".。,，!！?？:：、",
                "hallucination_silence_threshold": None,
                "multilingual": True,
                "hotwords": None,
            },
            vad_model_fp=vad_fp,
        )

    def transcribe(self, audio_info, lang=None):
        # Lazily init the underlying WhisperX pipeline.
        self._ensure_model()

        audio = load_wav(audio_info).numpy()
        if lang is None:
            lang = self.model.detect_language(audio)
        
        segments = self.model.transcribe(audio, batch_size=8, language=lang)["segments"]
        transcript = " ".join([segment["text"] for segment in segments])

        if lang not in {'es','pt','zh','en','de','fr','it', 'ar', 'ru', 'ja', 'ko', 'hi', 'th', 'id', 'vi'}:
            lang = langid.classify(transcript)[0]
            segments = self.model.transcribe(audio, batch_size=8, language=lang)["segments"]
            transcript = " ".join([segment["text"] for segment in segments])
        logging.debug(f"whisperx: {segments}")
        
        transcript = zhconv.convert(transcript, 'zh-hans')
        transcript = transcript.replace("-", " ")
        transcript = re.sub(_whitespace_re, " ", transcript)
        transcript = transcript[1:] if transcript[0] == " " else transcript
        segments = {'lang':lang, 'text_raw':transcript}
        if lang == "zh":
            segments["text"] = text_norm.txt2pinyin(transcript)
        else:
            transcript = replace_numbers_with_words(transcript, lang=lang).split(' ')
            segments["text"] = (transcript, transcript)

        return align_model.align(segments, audio_info)


def load_wav(audio_info, sr=16000, channel=1):
    raw_sr, audio = audio_info
    audio = audio.T if len(audio.shape) > 1 and audio.shape[1] == 2 else audio
    audio = audio / np.max(np.abs(audio))
    audio = torch.from_numpy(audio).squeeze().float()
    if channel == 1 and len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(dim=0, keepdim=True)
    elif channel == 2 and len(audio.shape) == 1:
        audio = torch.stack((audio, audio)) # mono to stereo
    if raw_sr != sr: 
        audio = torchaudio.functional.resample(audio.squeeze(), raw_sr, sr)
    audio = torch.clip(audio, -0.999, 0.999).squeeze()
    return audio


def update_word_time(lst, cut_time, edit_start, edit_end):
    for i in range(len(lst)):
        lst[i]["start"] = round(lst[i]["start"] - cut_time, ndigits=3)
        lst[i]["end"] = round(lst[i]["end"] - cut_time,  ndigits=3)
    edit_start = max(round(edit_start - cut_time, ndigits=3), 0)
    edit_end = round(edit_end - cut_time, ndigits=3)
    return lst, edit_start, edit_end


# def update_word_time2(lst, cut_time, edit_start, edit_end):
#     for i in range(len(lst)):
#         lst[i]["start"] = round(lst[i]["start"] + cut_time, ndigits=3)
#     return lst, edit_start, edit_end


def get_audio_slice(audio, words_info, start_time, end_time, max_len=10, sr=16000, code_sr=50):
    audio_dur = audio.shape[-1] / sr
    sub_list = []
    # 如果尾部小于5s则保留后面全部，并截取前半段音频
    if audio_dur - end_time <= max_len/2:
        for word in reversed(words_info):
            if word['start'] > start_time or audio_dur - word['start'] < max_len:
                sub_list = [word] + sub_list

    # 如果头部小于5s则保留前面全部，并截取后半段音频
    elif start_time <=max_len/2:
        for word in words_info:
            if word['end'] < max(end_time, max_len):
                sub_list += [word]
                
    # 如果前后都大于5s，则前后各留5s
    else:
        for word in words_info:
            if word['start'] > start_time - max_len/2 and word['end'] < end_time + max_len/2:
                sub_list += [word]
    audio = audio.squeeze()

    start = int(sub_list[0]['start']*sr)
    end = int(sub_list[-1]['end']*sr)
    # print("wav cuts:", start, end, (end-start) % int(sr/code_sr))
    end -= (end-start) % int(sr/code_sr) # chunk取整

    sub_list, start_time, end_time = update_word_time(sub_list, sub_list[0]['start'], start_time, end_time)
    audio = audio.squeeze()
    # print("after update_word_time:", sub_list, start_time, end_time, (end-start)/sr)

    return (audio[:start], audio[start:end], audio[end:]), (sub_list, start_time, end_time)


def load_models(lemas_model_name, whisper_model_name, alignment_model_name, denoise_model_name):

    global transcribe_model, align_model, denoise_model, text_norm, tts_edit_model, codec_handle

    # When switching models, explicitly free the previous edit backend to
    # avoid CUDA OOM when loading a new large checkpoint.
    try:
        if tts_edit_model is not None:
            try:
                if hasattr(tts_edit_model, "ema_model"):
                    tts_edit_model.ema_model.to("cpu")
                if hasattr(tts_edit_model, "model"):
                    tts_edit_model.model.to("cpu")
                if hasattr(tts_edit_model, "vocoder"):
                    try:
                        tts_edit_model.vocoder.to("cpu")
                    except Exception:
                        pass
            except Exception as e:
                logging.warning("Failed to move previous LEMAS-TTS model to CPU: %s", e)
            tts_edit_model = None
        if codec_handle is not None and hasattr(codec_handle, "model"):
            try:
                codec_handle.model.to("cpu")
            except Exception as e:
                logging.warning("Failed to move previous codec edit model to CPU: %s", e)
            codec_handle = None
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    if denoise_model_name == "UVR5":
        # Simple layout: UVR5 assets live directly under:
        #   <MODELS_PATH>/uvr5
        # with files:
        #   Kim_Vocal_1.onnx
        #   MDX-Net-Kim-Vocal1.json
        #   model_data.json
        #   model_name_mapper.json
        from pathlib import Path
        uv_root = Path(MODELS_PATH) / "uvr5"
        denoise_model = UVR5(str(uv_root))
    elif denoise_model_name == "DeepFilterNet":
        denoise_model = DeepFilterNet("./pretrained_models/denoiser_model.onnx")

    if alignment_model_name == "MMS":
        align_model = MMSAlignModel()
    else:
        align_model = WhisperxAlignModel()

    text_norm = TextNorm()

    transcribe_model = WhisperxModel(whisper_model_name)
    # Warm up WhisperX here so the first Transcribe button press is fast.
    try:
        transcribe_model._ensure_model()
    except Exception as e:
        raise gr.Error(f"Failed to initialize WhisperX model: {e}")

    # Autoregressive codec backend: load codec model instead of LEMAS-TTS
    if lemas_model_name == "autoregressive":
        model_path = os.path.join(MODELS_PATH, "ckpts", "autoregressive", "multilingual_330M.pth")
        if not os.path.isfile(model_path):
            raise gr.Error(f"Codec ckpt not found: {model_path}")
        try:
            codec_handle = load_edit_codec_model(
                model_path=model_path,
                models_root=os.path.dirname(model_path),
                device=device,
            )
            logging.info("Loaded edit model from %s", model_path)
        except Exception as e:
            raise gr.Error(f"Failed to load autoregressive edit model: {e}")
        # Do not construct tts_edit_model in this case.
        tts_edit_model = None
        return gr.Accordion()

    # LEMAS-TTS editing model (selected multilingual variant)
    from pathlib import Path

    # Local ckpt search under the standard CKPTS_ROOT layout
    ckpt_dir = Path(CKPTS_ROOT) / lemas_model_name
    ckpt_candidates = sorted(
        list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.pt"))
    )
    # Fallbacks for simpler layouts: allow ckpts directly under CKPTS_ROOT,
    # e.g. ./pretrained_models/ckpts/multilingual_grl.safetensors
    if not ckpt_candidates:
        root_candidates = sorted(
            list(Path(CKPTS_ROOT).glob(f"{lemas_model_name}*.safetensors"))
            + list(Path(CKPTS_ROOT).glob(f"{lemas_model_name}*.pt"))
        )
        ckpt_candidates = root_candidates

    # If no local ckpt is found, fall back to remote HF checkpoints
    # (using the same mapping as LEMAS-TTS `inference_gradio.py`).
    if not ckpt_candidates:
        remote_ckpts = {
            "multilingual_grl": f"{HF_PRETRAINED_ROOT}/ckpts/multilingual_grl/multilingual_grl.safetensors",
            "multilingual_prosody": f"{HF_PRETRAINED_ROOT}/ckpts/multilingual_prosody/multilingual_prosody.safetensors",
        }
        remote_path = remote_ckpts.get(lemas_model_name)
        if remote_path is not None:
            try:
                resolved = cached_path(remote_path)
                ckpt_candidates = [Path(resolved)]
                logging.info("Resolved remote ckpt %s -> %s", remote_path, resolved)
            except Exception as e:
                raise gr.Error(f"Failed to download remote ckpt {remote_path}: {e}")

    if not ckpt_candidates:
        raise gr.Error(
            f"No LEMAS-TTS ckpt found for '{lemas_model_name}' under {ckpt_dir} "
            f"or {CKPTS_ROOT}"
        )
    ckpt_file = str(ckpt_candidates[-1])

    vocab_file = Path(PRETRAINED_ROOT) / "data" / lemas_model_name / "vocab.txt"
    if not vocab_file.is_file():
        raise gr.Error(f"Vocab file not found: {vocab_file}")

    prosody_cfg = Path(CKPTS_ROOT) / "prosody_encoder" / "pretssel_cfg.json"
    prosody_ckpt = Path(CKPTS_ROOT) / "prosody_encoder" / "prosody_encoder_UnitY2.pt"

    # Decide whether to enable the prosody encoder:
    # - multilingual_prosody: True (if assets exist)
    # - multilingual_grl: False (GRL-only variant)
    # - others: fall back to presence of assets.
    if lemas_model_name.endswith("prosody"):
        use_prosody = prosody_cfg.is_file() and prosody_ckpt.is_file()
    elif lemas_model_name.endswith("grl"):
        use_prosody = False
    else:
        use_prosody = prosody_cfg.is_file() and prosody_ckpt.is_file()

    tts_edit_model = TTS(
        model=lemas_model_name,
        ckpt_file=ckpt_file,
        vocab_file=str(vocab_file),
        device=device,
        use_prosody_encoder=use_prosody,
        prosody_cfg_path=str(prosody_cfg) if use_prosody else "",
        prosody_ckpt_path=str(prosody_ckpt) if use_prosody else "",
        ode_method="euler",
        use_ema=True,
        frontend="phone",
    )
    logging.info(f"Loaded edit model from {ckpt_file}")

    return gr.Accordion()


def get_transcribe_state(segments):
    logging.info("===========After Align===========")
    logging.info(segments)
    return {
        "segments": segments,
        "transcript": segments["text_raw"],
        "words_info": segments["words"], 
        "transcript_with_start_time": " ".join([f"{word['start']} {word['word']}" for word in segments["words"]]),
        "transcript_with_end_time": " ".join([f"{word['word']} {word['end']}" for word in segments["words"]]),
        "word_bounds": [f"{word['start']} {word['word']} {word['end']}" for word in segments["words"]]
    }

def transcribe(seed, audio_info):
    if transcribe_model is None:
        raise gr.Error("Transcription model not loaded")
    seed_everything(seed)

    segments = transcribe_model.transcribe(audio_info)
    state = get_transcribe_state(segments)

    return [
        state["transcript"], state["transcript_with_start_time"], state["transcript_with_end_time"],
        # gr.Dropdown(value=state["word_bounds"][-1], choices=state["word_bounds"], interactive=True), # prompt_to_word
        gr.Dropdown(value=state["word_bounds"][0], choices=state["word_bounds"], interactive=True), # edit_from_word
        gr.Dropdown(value=state["word_bounds"][-1], choices=state["word_bounds"], interactive=True), # edit_to_word
        state
    ]

def align(transcript, audio_info, state):
    lang = state["segments"]["lang"]
    # print("realign: ", transcript, state)
    transcript = re.sub(_whitespace_re, " ", transcript)
    transcript = transcript[1:] if transcript[0] == " " else transcript
    segments = {'lang':lang, 'text':transcript, 'text_raw':transcript}
    if lang == "zh":
        segments["text"] = text_norm.txt2pinyin(transcript)
    else:
        transcript = replace_numbers_with_words(transcript)
        segments["text"] = (transcript.split(' '), transcript.split(' '))
    # print("text:", segments["text"])
    segments = align_model.align(segments, audio_info)
    
    state = get_transcribe_state(segments)

    return [
        state["transcript"], state["transcript_with_start_time"], state["transcript_with_end_time"],
        # gr.Dropdown(value=state["word_bounds"][-1], choices=state["word_bounds"], interactive=True), # prompt_to_word
        gr.Dropdown(value=state["word_bounds"][0], choices=state["word_bounds"], interactive=True), # edit_from_word
        gr.Dropdown(value=state["word_bounds"][-1], choices=state["word_bounds"], interactive=True), # edit_to_word
        state
    ]


def denoise(audio_info):
    # Denoiser can be relatively heavy (especially UVR5), so schedule it on
    # GPU workers when running on HF Spaces.
    if denoise_model is None:
        return audio_info
    denoised_audio, sr = denoise_model.denoise(audio_info)
    denoised_audio = denoised_audio  # already numpy
    return (sr, denoised_audio)

def cancel_denoise(audio_info):
    return audio_info

def get_output_audio(audio_tensors, sr):
    result = torch.cat(audio_tensors, -1)
    result = result.squeeze().cpu().numpy()
    result = (result * np.iinfo(np.int16).max).astype(np.int16)
    print("save result:", result.shape)
    # wavfile.write(os.path.join(TMP_PATH, "output.wav"), sr, result)
    return (int(sr), result)


def get_edit_audio_part(audio_info, edit_start, edit_end):
    sr, raw_wav = audio_info
    raw_wav = raw_wav[int(edit_start*sr):int(edit_end*sr)]
    return (sr, raw_wav)


def crossfade_concat(chunk1, chunk2, overlap):
    # 计算淡入和淡出系数
    fade_out = torch.cos(torch.linspace(0, torch.pi / 2, overlap)) ** 2
    fade_in = torch.cos(torch.linspace(torch.pi / 2, 0, overlap)) ** 2
    chunk2[:overlap] = chunk1[-overlap:] * fade_out + chunk2[:overlap] * fade_in
    chunk = torch.cat((chunk1[:-overlap], chunk2), dim=0)
    return chunk

def replace_numbers_with_words(sentence, lang="en"):
    sentence = re.sub(r'(\d+)', r' \1 ', sentence)  # add spaces around numbers

    def replace_with_words(match):
        num = match.group(0)
        try:
            return num2words(num, lang=lang)  # Convert numbers to words
        except Exception:
            return num  # Fallback if num2words fails

    return re.sub(r'\b\d+\b', replace_with_words, sentence)


def run(
    seed,
    nfe_step,
    speed,
    cfg_strength,
    sway_sampling_coef,
    ref_ratio,
    stop_repetition,
    repetition_penalty,
    chars_cut,
    left_margin,
    right_margin,
    top_p,
    temperature,
    kvcache,
    audio_info,
    denoised_audio,
    transcribe_state,
    transcript,
    smart_transcript,
    mode,
    start_time,
    end_time,
    split_text,
    selected_sentence,
    audio_tensors,
    edit_model_name="multilingual_grl",
):
    """
    Shared run entry for both LEMAS-TTS (CFM) and autoregressive codec backends.

    edit_model_name:
      - 'multilingual_grl' / 'multilingual_prosody' -> LEMAS-TTS backend
      - 'autoregressive' -> codec-based autoregressive backend
    """
    global tts_edit_model, codec_handle

    if edit_model_name == "autoregressive":
        if codec_handle is None:
            raise gr.Error("Autoregressive edit model not loaded. Please click 'Load models' first.")
    else:
        if tts_edit_model is None:
            raise gr.Error("LEMAS-TTS edit model not loaded. Please click 'Load models' first.")

    if smart_transcript and (transcribe_state is None):
        raise gr.Error("Can't use smart transcript: whisper transcript not found")

    # On HF Spaces, keep CUDA usage inside this GPU worker: move the edit
    # model and vocoder to GPU here (the weights were loaded on CPU).
    if edit_model_name != "autoregressive" and torch.cuda.is_available():
        try:
            if getattr(tts_edit_model, "device", "cpu") != "cuda":
                if hasattr(tts_edit_model, "ema_model"):
                    tts_edit_model.ema_model.to("cuda")
                if hasattr(tts_edit_model, "vocoder"):
                    try:
                        tts_edit_model.vocoder.to("cuda")
                    except Exception:
                        pass
                tts_edit_model.device = "cuda"
        except Exception as e:
            logging.warning("Failed to move LEMAS-TTS model to CUDA: %s", e)

    # Choose base audio (denoised if duration matches)
    audio_base = audio_info
    audio_dur = round(audio_info[1].shape[0] / audio_info[0], ndigits=3)
    if denoised_audio is not None:
        denoised_dur = round(denoised_audio[1].shape[0] / denoised_audio[0], ndigits=3)
        if audio_dur == denoised_dur or (
            denoised_audio[0] != audio_info[0] and abs(audio_dur - denoised_dur) < 0.1
        ):
            audio_base = denoised_audio
            logging.info("use denoised audio")

    raw_sr, raw_wav = audio_base
    print("audio_dur: ", audio_dur, raw_sr, raw_wav.shape, start_time, end_time)

    # Build target text by replacing the selected span with `transcript`
    words = transcribe_state["words_info"]
    if not words:
        raise gr.Error("No word-level alignment found; please run Transcribe first.")

    start_time = float(start_time)
    end_time = float(end_time)
    if end_time <= start_time:
        raise gr.Error("Edit end time must be greater than start time.")

    # Find word indices covering the selected region
    start_idx = 0
    for i, w in enumerate(words):
        if w["end"] > start_time:
            start_idx = i
            break

    end_idx = len(words)
    for i in range(len(words) - 1, -1, -1):
        if words[i]["start"] < end_time:
            end_idx = i + 1
            break
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, len(words))

    word_start_sec = float(words[start_idx]["start"])
    word_end_sec = float(words[end_idx - 1]["end"])

    # Edit span in seconds (relative to full utterance)
    edit_start = max(0.0, word_start_sec - 0.1)
    edit_end = min(word_end_sec + 0.1, audio_dur)
    parts_to_edit = [(edit_start, edit_end)]

    display_text = transcribe_state["segments"]["text_raw"].strip()
    txt_list = display_text.split(" ") if display_text else [w["word"] for w in words]

    prefix = " ".join(txt_list[:start_idx]).strip()
    suffix = " ".join(txt_list[end_idx:]).strip()
    new_phrase = transcript.strip()

    pieces = []
    if prefix:
        pieces.append(prefix)
    if new_phrase:
        pieces.append(new_phrase)
    if suffix:
        pieces.append(suffix)
    target_text = " ".join(pieces)

    logging.info(
        "target_text: %s (start_idx=%d, end_idx=%d, parts_to_edit=%s)",
        target_text,
        start_idx,
        end_idx,
        parts_to_edit,
    )

    # Decide backend
    if edit_model_name == "autoregressive":
        # Restrict supported languages for autoregressive backend
        lang_asr = transcribe_state["segments"]["lang"]
        supported_langs = {"en", "zh", "es", "pt", "fr", "de", "it"}
        if lang_asr not in supported_langs:
            raise gr.Error(
                f"Autoregressive edit model currently supports {sorted(supported_langs)} "
                f"but ASR detected language '{lang_asr}'. "
                "Please switch to multilingual_grl/multilingual_prosody backend."
            )

        # Prepare audio at codec sampling rate
        codec_audio_sr = 16000
        codec_sr = 50
        audio = load_wav(audio_base, sr=codec_audio_sr)


        # Optional: adjust edit span boundaries as in codec backend
        audio_dur = round(audio.shape[-1] / codec_audio_sr, ndigits=3)
        if edit_start <= words[0]["start"]:
            edit_start = 0.0
        if edit_end >= words[-1]["end"]:
            edit_end = audio_dur

        # Build raw_text and target_transcript for codec model
        raw_text = [transcribe_state["segments"]["lang"], " ".join([w["word"] for w in words])]

        tar_lang = langid.classify(transcript)[0]
        if re.search("[\u4e00-\u9fa5]+", transcript):
            txts, phones = text_norm.txt2pinyin(transcript)
            transcript_norm = " ".join(phones)
        else:
            transcript_norm = transcript
        sentences = " ".join([transcript_norm.replace("\n", " ")])
        target_transcript = text_norm.add_sil(
            words,
            edit_start,
            edit_end,
            sentences,
            transcribe_state["segments"]["lang"],
            tar_lang,
        )

        # Mask interval in codec frames (apply configurable margins)
        morphed_span = (
            max(edit_start - float(left_margin), 1 / codec_sr),
            min(edit_end + float(right_margin), audio_dur),
        )
        mask_interval = [[round(morphed_span[0] * codec_sr), round(morphed_span[1] * codec_sr)]]
        mask_interval = torch.LongTensor(mask_interval)

        # Run codec infilling with VoiceCraft-style sampling parameters
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

        generated_sample, num_gen = edit_codec_inference_one_sample(
            codec_handle,
            audio,
            raw_text,
            target_transcript,
            mask_interval,
            decode_config,
        )
        gen_audio = generated_sample.cpu().squeeze()

        # Upsample and crossfade back into original waveform
        # Here we keep things simple: upsample entire generated segment and
        # splice it back into the original audio with a short crossfade.
        gen_audio_upsampled = audio_upsampling(
            gen_audio, codec_handle.audiosr_model, codec_audio_sr, raw_sr
        )
        gen_audio_upsampled = torch.clip(gen_audio_upsampled, -0.999, 0.999)
        gen_np = gen_audio_upsampled.cpu().numpy()
        gen_int16 = (gen_np * np.iinfo(np.int16).max).astype(np.int16)
        output_audio = (raw_sr, gen_int16)

        sentences = [f"0: {target_text}"]
        audio_tensors = [gen_audio_upsampled]
        component = gr.Dropdown(choices=sentences, value=sentences[0])
        return output_audio, target_text, component, audio_tensors

    # LEMAS-TTS backend (CFM)
    segment_audio = load_wav(audio_base, sr=tts_edit_model.target_sample_rate)

    # 根据字符数和 speed 自适应调整 edit span 时长：
    # 参考 utils_infer.process_batch 中 duration 的计算方式：
    #   duration ~ ref_audio_len + ref_audio_len / ref_text_len * gen_text_len / local_speed
    # 对编辑场景，我们只关心 edit span 的相对比例，因此设置：
    #   speed_effective = local_speed * ref_chars / gen_chars
    # 这样在 gen_wav_multilingual 中 orig_len / speed_effective ~= orig_len * gen_chars / (ref_chars * local_speed)
    local_speed = float(speed)
    # 原片段文本（被编辑的那一段）
    ref_phrase = " ".join(txt_list[start_idx:end_idx]).strip()
    ref_chars = max(1, len(ref_phrase.encode("utf-8")))
    gen_chars = max(1, len(new_phrase.encode("utf-8")))
    # 不再对短文本强行设置一个极慢的 local_speed=0.3，避免把未选中的上下文也拉入编辑区域
    adaptive_speed = local_speed * ref_chars / gen_chars

    seed_val = None if seed == -1 else int(seed)
    use_prosody_flag = bool(getattr(tts_edit_model, "use_prosody_encoder", False))

    wav_out, _ = gen_wav_multilingual(
        tts_edit_model,
        segment_audio,
        tts_edit_model.target_sample_rate,
        target_text,
        parts_to_edit,
        speed=float(adaptive_speed),
        nfe_step=int(nfe_step),
        cfg_strength=float(cfg_strength),
        sway_sampling_coef=float(sway_sampling_coef),
        ref_ratio=float(ref_ratio),
        no_ref_audio=False,
        use_acc_grl=False,
        use_prosody_encoder_flag=use_prosody_flag,
        seed=seed_val,
    )

    wav_np = wav_out.cpu().numpy()
    wav_np = np.clip(wav_np, -0.999, 0.999)
    wav_int16 = (wav_np * np.iinfo(np.int16).max).astype(np.int16)
    out_sr = int(tts_edit_model.target_sample_rate)

    output_audio = (out_sr, wav_int16)
    sentences = [f"0: {target_text}"]
    audio_tensors = [torch.from_numpy(wav_np)]
    component = gr.Dropdown(choices=sentences, value=sentences[0])
    return output_audio, target_text, component, audio_tensors
        

def update_input_audio(audio_info):
    if audio_info is None:
        return 0, 0, 0
    elif type(audio_info) is str:
        info = torchaudio.info(audio_path)
        max_time = round(info.num_frames / info.sample_rate, 2)
    elif type(audio_info) is tuple:
        max_time = round(audio_info[1].shape[0] / audio_info[0], 2)
    return [
        # gr.Slider(maximum=max_time, value=max_time),
        gr.Slider(maximum=max_time, value=0),
        gr.Slider(maximum=max_time, value=max_time),
    ]

def change_mode(mode):
    # tts_mode_controls, edit_mode_controls, edit_word_mode, split_text, long_tts_sentence_editor
    return [
        gr.Group(visible=mode != "Edit"),
        gr.Group(visible=mode == "Edit"),
        gr.Radio(visible=mode == "Edit"),
        gr.Radio(visible=mode == "Long TTS"),
        gr.Group(visible=mode == "Long TTS"),
    ]

def load_sentence(selected_sentence, audio_tensors):
    if selected_sentence is None:
        return None
    colon_position = selected_sentence.find(':')
    selected_sentence_idx = int(selected_sentence[:colon_position])
    # Use LEMAS-TTS target sample rate if available, otherwise default to 16000
    sr = getattr(tts_edit_model, "target_sample_rate", 16000)
    return get_output_audio([audio_tensors[selected_sentence_idx]], sr)


def update_bound_word(is_first_word, selected_word, edit_word_mode):
    if selected_word is None:
        return None

    word_start_time = float(selected_word.split(' ')[0])
    word_end_time = float(selected_word.split(' ')[-1])
    if edit_word_mode == "Replace half":
        bound_time = (word_start_time + word_end_time) / 2
    elif is_first_word:
        bound_time = word_start_time
    else:
        bound_time = word_end_time

    return bound_time


def update_bound_words(from_selected_word, to_selected_word, edit_word_mode):
    return [
        update_bound_word(True, from_selected_word, edit_word_mode),
        update_bound_word(False, to_selected_word, edit_word_mode),
    ]


smart_transcript_info = """
If enabled, the target transcript will be constructed for you:</br>
 - In Edit mode just write the text to replace selected editing segment.</br>
"""

demo_original_transcript = ""

demo_text = {
    "Edit": {
        "smart": "Write new words here.",
    },
}

all_demo_texts = {vv for k, v in demo_text.items() for kk, vv in v.items()}

demo_words = ['0.401 My 0.481', '0.521 son 0.661', '0.701 really 0.861', '0.921 thought 1.142', '1.142 of 1.202', '1.202 the 1.262', '1.282 entry 1.482', '1.522 about 1.682', '1.722 me 1.782', '1.883 after 2.223', '2.684 watching 2.964', '2.964 The 3.044', '3.064 Breaking 3.405', '3.445 Point, 3.805', '4.006 and 4.166', '4.246 he 4.306', '4.366 ended 4.586', '4.606 up 4.727', '4.727 writing 5.107', '5.488 a 5.508', '5.568 review 5.808', '5.848 of 5.908', '5.948 the 6.008', '6.068 movie 6.349', '6.389 as 6.489', '6.569 well. 6.829']

demo_words_info =[{'word': 'My', 'start': 0.401, 'end': 0.481, 'score': 0.992}, {'word': 'son', 'start': 0.521, 'end': 0.661, 'score': 0.974}, {'word': 'really', 'start': 0.701, 'end': 0.861, 'score': 0.17}, {'word': 'thought', 'start': 0.921, 'end': 1.142, 'score': 0.055}, {'word': 'of', 'start': 1.142, 'end': 1.202, 'score': 0.11}, {'word': 'the', 'start': 1.202, 'end': 1.262, 'score': 0.344}, {'word': 'entry', 'start': 1.282, 'end': 1.482, 'score': 0.642}, {'word': 'about', 'start': 1.522, 'end': 1.682, 'score': 0.842}, {'word': 'me', 'start': 1.722, 'end': 1.782, 'score': 0.886}, {'word': 'after', 'start': 1.883, 'end': 2.223, 'score': 0.959}, {'word': 'watching', 'start': 2.684, 'end': 2.964, 'score': 0.985}, {'word': 'The', 'start': 2.964, 'end': 3.044, 'score': 0.847}, {'word': 'Breaking', 'start': 3.064, 'end': 3.405, 'score': 0.705}, {'word': 'Point,', 'start': 3.445, 'end': 3.805, 'score': 0.919}, {'word': 'and', 'start': 4.006, 'end': 4.166, 'score': 0.816}, {'word': 'he', 'start': 4.246, 'end': 4.306, 'score': 0.591}, {'word': 'ended', 'start': 4.366, 'end': 4.586, 'score': 0.638}, {'word': 'up', 'start': 4.606, 'end': 4.727, 'score': 0.034}, {'word': 'writing', 'start': 4.727, 'end': 5.107, 'score': 0.665}, {'word': 'a', 'start': 5.488, 'end': 5.508, 'score': 0.47}, {'word': 'review', 'start': 5.568, 'end': 5.808, 'score': 0.703}, {'word': 'of', 'start': 5.848, 'end': 5.908, 'score': 0.85}, {'word': 'the', 'start': 5.948, 'end': 6.008, 'score': 0.955}, {'word': 'movie', 'start': 6.068, 'end': 6.349, 'score': 0.76}, {'word': 'as', 'start': 6.389, 'end': 6.489, 'score': 0.989}, {'word': 'well.', 'start': 6.569, 'end': 6.829, 'score': 0.997}]

def update_demo(mode, smart_transcript, edit_word_mode, transcript, edit_from_word, edit_to_word):
    if transcript not in all_demo_texts:
        return transcript, edit_from_word, edit_to_word

    replace_half = edit_word_mode == "Replace half"
    change_edit_from_word = edit_from_word == demo_words[2] or edit_from_word == demo_words[3]
    change_edit_to_word = edit_to_word == demo_words[11] or edit_to_word == demo_words[12]
    demo_edit_from_word_value = demo_words[2] if replace_half else demo_words[3]
    demo_edit_to_word_value = demo_words[12] if replace_half else demo_words[11]
    return [
        demo_text[mode]["smart" if smart_transcript else "regular"],
        demo_edit_from_word_value if change_edit_from_word else edit_from_word,
        demo_edit_to_word_value if change_edit_to_word else edit_to_word,
    ]

def get_app():
    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column(scale=2):
                load_models_btn = gr.Button(value="Load Models")
            with gr.Column(scale=5):
                with gr.Accordion("Select models", open=False) as models_selector:
                    # For LEMAS-TTS editing, we expose a simple model selector
                    # between the two multilingual variants.
                    with gr.Row():
                        lemas_model_choice = gr.Radio(
                            label="Edit Models",
                            choices=["autoregressive", "multilingual_grl", "multilingual_prosody"],
                            value="autoregressive",
                            interactive=True,
                            scale=4,
                        )
                        denoise_model_choice = gr.Radio(label="Denoise Models", scale=2, value="UVR5", choices=["UVR5", "DeepFilterNet"]) 
                        whisper_model_choice = gr.Radio(label="Whisper Models", scale=3, value="medium", choices=["base", "small", "medium", "large"])
                        align_model_choice = gr.Radio(label="Forced Alignment Model", scale=2, value="MMS", choices=["whisperX", "MMS"], visible=False)

        with gr.Row():
            with gr.Column(scale=2):
                # Use a numpy waveform as default value to avoid Gradio's
                # InvalidPathError with local filesystem paths.
                _demo_value = None
                demo_candidates = [
                    os.path.join(DEMO_PATH, "test.wav"),
                ]
                for demo_path in demo_candidates:
                    try:
                        if not os.path.isfile(demo_path):
                            continue
                        _demo_wav, _demo_sr = torchaudio.load(demo_path)
                        if _demo_wav.dim() > 1 and _demo_wav.shape[0] > 1:
                            _demo_wav = _demo_wav.mean(dim=0, keepdim=True)
                        _demo_value = (_demo_sr, _demo_wav.squeeze(0).numpy())
                        break
                    except Exception:
                        continue

                input_audio = gr.Audio(value=_demo_value, label="Input Audio", interactive=True, type="numpy", show_download_button=True, editable=True)

                with gr.Row():
                    transcribe_btn = gr.Button(value="Transcribe")
                    align_btn = gr.Button(value="ReAlign")
                with gr.Group():
                    original_transcript = gr.Textbox(label="Original transcript", lines=5, interactive=True, value=demo_original_transcript,
                                                    info="Use whisperx model to get the transcript. Fix and align it if necessary.")
                    with gr.Accordion("Word start time", open=False, visible=False):
                        transcript_with_start_time = gr.Textbox(label="Start time", lines=5, interactive=False, info="Start time before each word")
                    with gr.Accordion("Word end time", open=False, visible=False):
                        transcript_with_end_time = gr.Textbox(label="End time", lines=5, interactive=False, info="End time after each word")
                
                with gr.Row():
                    denoise_btn = gr.Button(value="Denoise")
                    cancel_btn = gr.Button(value="Cancel Denoise")
                denoise_audio = gr.Audio(label="Denoised Audio", value=None, interactive=False, type="numpy", show_download_button=True, editable=True)

            with gr.Column(scale=3):
                with gr.Group():
                    transcript_inbox = gr.Textbox(label="Text", lines=5, value=demo_text["Edit"]["smart"])
                    with gr.Row(visible=False):
                        smart_transcript = gr.Checkbox(label="Smart transcript", value=True)
                        with gr.Accordion(label="?", open=False):
                            info = gr.Markdown(value=smart_transcript_info)

                    mode = gr.Radio(label="Mode", choices=["Edit"], value="Edit", visible=False)
                    with gr.Row(visible=False):
                        split_text = gr.Radio(label="Split text", choices=["Newline", "Sentence"], value="Newline",
                                            info="Split text into parts and run TTS for each part.", visible=True)
                        edit_word_mode = gr.Radio(label="Edit word mode", choices=["Replace half", "Replace all"], value="Replace all",
                                                info="What to do with first and last word", visible=False)

                    with gr.Row():
                        edit_from_word = gr.Dropdown(label="First word to edit", choices=demo_words, value=demo_words[12], interactive=True)
                        edit_to_word = gr.Dropdown(label="Last word to edit", choices=demo_words, value=demo_words[18], interactive=True)
                    with gr.Row():
                        edit_start_time = gr.Slider(label="Edit from time", minimum=0, maximum=7.614, step=0.001, value=4.022)
                        edit_end_time = gr.Slider(label="Edit to time", minimum=0, maximum=7.614, step=0.001, value=5.768)
                    with gr.Row():
                        check_btn = gr.Button(value="Check edit words", scale=2)
                        edit_audio = gr.Audio(label="Edit word(s)", scale=3, type="numpy")
                    
                    run_btn = gr.Button(value="Run", variant="primary")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Output Audio", type="numpy", show_download_button=True, editable=True)
                with gr.Accordion("Inference transcript", open=True):
                    inference_transcript = gr.Textbox(
                        label="Inference transcript",
                        lines=4,
                        interactive=False,
                        info="Inference was performed on this transcript.",
                    )
                # Simple in-app README to guide users through the editing workflow.
                # Use HTML so we can cap the height (~12 lines) and enable scrolling.
                readme_help = gr.HTML(
                    value=(
                        '<div style="max-height: 12em; overflow-y: auto; white-space: pre-wrap;">'
                        "<h4>README: How to Use This Tool</h4>"
                        "<p><b>1. Load models</b><br>"
                        "Click <b>&ldquo;Load Models&rdquo;</b> and wait for all models to finish loading. "
                        "Note that <b>WhisperX</b> takes the longest to initialize, so please be patient.</p>"
                        "<p><b>2. Upload input audio</b><br>"
                        "Click <b>&ldquo;Input Audio&rdquo;</b> and upload the audio file you want to edit.</p>"
                        "<p><b>3. Transcribe and correct text</b><br>"
                        "Click <b>&ldquo;Transcribe&rdquo;</b> to perform speech recognition. If the transcription is inaccurate, "
                        "edit the text in <b>&ldquo;Original transcript&rdquo;</b>, then click <b>&ldquo;ReAlign&rdquo;</b> to recompute "
                        "word-level timestamps.</p>"
                        "<p><b>4. (Optional) Denoise noisy audio</b><br>"
                        "If the input audio is noisy and affects recognition or synthesis quality, click "
                        "<b>&ldquo;Denoise&rdquo;</b> to apply noise reduction. If you are not satisfied with the denoised result, "
                        "click <b>&ldquo;Cancel Denoise&rdquo;</b> to restore the original audio, or switch to a different denoiser "
                        "under <b>&ldquo;Select models&rdquo;</b> and reload.</p>"
                        "<p><b>5. Select the edit span</b><br>"
                        "Use <b>&ldquo;First word to edit&rdquo;</b> and <b>&ldquo;Last word to edit&rdquo;</b> to specify the region to modify, "
                        "then click <b>&ldquo;Check edit words&rdquo;</b> to preview the selection. For finer control, you may also adjust "
                        "<b>&ldquo;Edit from time&rdquo;</b> and <b>&ldquo;Edit to time&rdquo;</b>.</p>"
                        "<p><b>6. Enter the new text</b><br>"
                        "In the <b>&ldquo;Text&rdquo;</b> box, enter the text that should replace the selected segment.</p>"
                        "<p><b>7. Run the edit</b><br>"
                        "Click <b>&ldquo;Run&rdquo;</b> and wait for the model to generate the edited audio.</p>"
                        "<p><b>8. Inspect the result</b><br>"
                        "The edited waveform will appear in <b>&ldquo;Output Audio&rdquo;</b>, and the corresponding edited text will be "
                        "shown under <b>&ldquo;Inference transcript&rdquo;</b>.</p>"
                        "<p><b>9. Refine or change models</b><br>"
                        "If the result is not satisfactory, try adjusting the <b>&ldquo;Generation Parameters&rdquo;</b> or selecting a "
                        "different <b>&ldquo;Edit Model&rdquo;</b> under <b>&ldquo;Select models&rdquo;</b>, then run again.</p>"
                        "<p><b>10. Feedback</b><br>"
                        "For bug reports or feature requests, feel free to:<br>"
                        "1) Open a GitHub issue<br>"
                        "2) Post on the Hugging Face community page<br>"
                        "3) Contact us via email at <code>approximetal@gmail.com</code></p>"
                        "</div>"
                    )
                )
                with gr.Group(visible=False) as long_tts_sentence_editor:
                    sentence_selector = gr.Dropdown(label="Sentence", value=None,
                                                    info="Select sentence you want to regenerate")
                    sentence_audio = gr.Audio(label="Sentence Audio", scale=2, type="numpy")
                    rerun_btn = gr.Button(value="Rerun")

        with gr.Row():
            with gr.Accordion("Generation Parameters - change these if you are unhappy with the generation", open=False):
                gr.Markdown("**Parameters for autoregressive edit model**")
                with gr.Row():
                    stop_repetition = gr.Radio(
                        label="stop_repetition",
                        choices=[-1, 1, 2, 3, 4],
                        value=1,
                        info=(
                            "If there are long silences in the generated audio, "
                            "reduce this to 2 or 1. -1 = disabled."
                        ),
                    )
                    repetition_penalty = gr.Slider(
                        label="repetition penalty",
                        minimum=0.0,
                        maximum=3.0,
                        step=0.1,
                        value=1.5,
                        info="Penalize over-repeated tokens in the autoregressive codec.",
                    )
                    chars_cut = gr.Number(
                        label="chars_cut",
                        value=135,
                        precision=0,
                        info="Max characters to keep when cutting long sentences (for edit/AR backend).",
                    )

                    left_margin = gr.Number(
                        label="left_margin",
                        value=0.01,
                        precision=3,
                        info="Margin (seconds) to the left of the editing segment.",
                    )
                    right_margin = gr.Number(
                        label="right_margin",
                        value=0.01,
                        precision=3,
                        info="Margin (seconds) to the right of the editing segment.",
                    )
                    top_p = gr.Number(
                        label="top_p",
                        value=0.8,
                        info="Top-p (nucleus) sampling for the autoregressive codec.",
                    )
                    temperature = gr.Number(
                        label="temperature",
                        value=1.0,
                        info="Softmax temperature for sampling. 1.0 is usually fine.",
                    )
                    kvcache = gr.Radio(
                        label="kvcache",
                        choices=[0, 1],
                        value=1,
                        visible=False,
                        info="1 = use KV cache (faster, more VRAM); 0 = no cache (slower, less VRAM).",
                    )
                gr.Markdown("**Parameters for multilingual_grl / multilingual_prosody (LEMAS-TTS backend)**")
                with gr.Row():
                    nfe_step = gr.Number(
                        label="NFE Step",
                        value=32,
                        precision=0,
                        scale=1,
                        info="Sampling steps for the diffusion model.",
                    )
                    speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=1.5,
                        step=0.1,
                        value=1.0,
                        scale=2,
                        info="Adjust speed of the generated audio.",
                    )
                    cfg_strength = gr.Slider(
                        label="CFG Strength",
                        minimum=2.0,
                        maximum=10.0,
                        step=0.1,
                        value=5.0,
                        scale=2,
                        info="Classifier-free guidance strength.",
                    )

                    sway_sampling_coef = gr.Slider(
                        label="Sway-Sampling",
                        minimum=2.0,
                        maximum=5.0,
                        step=0.1,
                        value=3.0,
                        scale=2,
                        info="Sampling sway coefficient.",
                    )
                    ref_ratio = gr.Slider(
                        label="Ref Ratio",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=1.0,
                        visible=False,
                        info="How much to rely on reference audio (if used).",
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        scale=1,
                        info="-1 for random, otherwise fixed seed.",
                    )


        audio_tensors = gr.State()
        transcribe_state = gr.State(value={"words_info": demo_words_info, "lang":"zh"})


        edit_word_mode.change(fn=update_demo,
                            inputs=[mode, smart_transcript, edit_word_mode, transcript_inbox, edit_from_word, edit_to_word],
                            outputs=[transcript_inbox, edit_from_word, edit_to_word])
        smart_transcript.change(
            fn=update_demo,
            inputs=[mode, smart_transcript, edit_word_mode, transcript_inbox, edit_from_word, edit_to_word],
            outputs=[transcript_inbox, edit_from_word, edit_to_word],
        )

        load_models_btn.click(
            fn=load_models,
            inputs=[
                lemas_model_choice,
                whisper_model_choice,
                align_model_choice,
                denoise_model_choice,
            ],
            outputs=[models_selector],
        )

        input_audio.upload(fn=update_input_audio,
                        inputs=[input_audio],
                        outputs=[edit_start_time, edit_end_time])

        transcribe_btn.click(fn=transcribe,
                            inputs=[seed, input_audio],
                            outputs=[original_transcript, transcript_with_start_time, transcript_with_end_time,
                                    edit_from_word, edit_to_word, transcribe_state])
        align_btn.click(fn=align,
                        inputs=[original_transcript, input_audio, transcribe_state],
                        outputs=[original_transcript, transcript_with_start_time, transcript_with_end_time,
                                edit_from_word, edit_to_word, transcribe_state]) 

        denoise_btn.click(fn=denoise,
                        inputs=[input_audio],
                        outputs=[denoise_audio])

        cancel_btn.click(fn=cancel_denoise,
                        inputs=[input_audio],
                        outputs=[denoise_audio])

        check_btn.click(fn=get_edit_audio_part,
                        inputs=[input_audio, edit_start_time, edit_end_time],
                        outputs=[edit_audio])

        run_btn.click(
            fn=run,
            inputs=[
                seed,
                nfe_step,
                speed,
                cfg_strength,
                sway_sampling_coef,
                ref_ratio,
                stop_repetition,
                repetition_penalty,
                chars_cut,
                left_margin,
                right_margin,
                top_p,
                temperature,
                kvcache,
                input_audio,
                denoise_audio,
                transcribe_state,
                transcript_inbox,
                smart_transcript,
                mode,
                edit_start_time,
                edit_end_time,
                split_text,
                sentence_selector,
                audio_tensors,
                lemas_model_choice,  # select backend
            ],
            outputs=[output_audio, inference_transcript, sentence_selector, audio_tensors],
        )

        sentence_selector.change(
            fn=load_sentence,
            inputs=[sentence_selector, audio_tensors],
            outputs=[sentence_audio],
        )
        rerun_btn.click(
            fn=run,
            inputs=[
                seed,
                nfe_step,
                speed,
                cfg_strength,
                sway_sampling_coef,
                ref_ratio,
                stop_repetition,
                repetition_penalty,
                chars_cut,
                left_margin,
                right_margin,
                top_p,
                temperature,
                kvcache,
                input_audio,
                denoise_audio,
                transcribe_state,
                transcript_inbox,
                smart_transcript,
                gr.State(value="Rerun"),
                edit_start_time,
                edit_end_time,
                split_text,
                sentence_selector,
                audio_tensors,
                lemas_model_choice,
            ],
            outputs=[output_audio, inference_transcript, sentence_audio, audio_tensors],
        )

        edit_from_word.change(fn=update_bound_word,
                            inputs=[gr.State(True), edit_from_word, edit_word_mode],
                            outputs=[edit_start_time])
        edit_to_word.change(fn=update_bound_word,
                            inputs=[gr.State(False), edit_to_word, edit_word_mode],
                            outputs=[edit_end_time])
        edit_word_mode.change(fn=update_bound_words,
                            inputs=[edit_from_word, edit_to_word, edit_word_mode],
                            outputs=[edit_start_time, edit_end_time])

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LEMAS-Edit gradio app.")
    
    parser.add_argument("--demo-path", default="./pretrained_models/demos", help="Path to demo directory")
    parser.add_argument("--tmp-path", default="./pretrained_models/demos/tmp", help="Path to tmp directory")
    parser.add_argument("--port", default=41020, type=int, help="App port")
    parser.add_argument("--share", action="store_true", help="Launch with public url")
    parser.add_argument("--server_name", default="0.0.0.0", type=str, help="Server name for launching the app. 127.0.0.1 for localhost; 0.0.0.0 to allow access from other machines in the local network. Might also give access to external users depends on the firewall settings.")
    parser.add_argument(
        "--models-path",
        default="./pretrained_models",
        dest="models_path",
        help="Path to pretrained_models root (mirrors LEMAS-TTS layout).",
    )

    os.environ["USER"] = os.getenv("USER", "user")
    args = parser.parse_args()
    DEMO_PATH = args.demo_path
    TMP_PATH = args.tmp_path
    MODELS_PATH = args.models_path
    
    app = get_app()
    app.queue().launch(share=args.share, server_name=args.server_name, server_port=args.port)
