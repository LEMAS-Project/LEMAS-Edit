# LEMAS‑Edit: Multilingual Speech Editing System
[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://www.python.org/downloads/release/python-3100/)
[![Report](https://img.shields.io/badge/Arxiv-Report%20-red.svg)](https://arxiv.org/abs/2601.04233)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://lemas-project.github.io/LEMAS-Project/)
[![hfspace](https://img.shields.io/badge/🤗-Space%20Demo-yellow)](https://huggingface.co/spaces/LEMAS-Project/LEMAS-Edit)
[![hfmodel](https://img.shields.io/badge/🤗-Models%20Download-yellow)](https://huggingface.co/LEMAS-Project/LEMAS-Edit)
[![msmodel](https://img.shields.io/badge/MS-Models%20Download-purple)](https://www.modelscope.cn/models/LEMAS/LEMAS-Edit)

LEMAS‑Edit is a multilingual version speech editing system, supporting 10 languages:
- Chinese
- English
- Spanish
- Russian
- French
- German
- Italian
- Portuguese
- Indonesian
- Vietnamese

It bundles:
- the multilingual flow-matching backend (`lemas_tts`) 
- the decoder only edit backend (`lemas_edit`)
- pretrained checkpoints, vocabs and demo data (`pretrained_models/`)
- an end‑to‑end Gradio web UI (`gradio_mix.py`)

Compared to the original LEMAS‑TTS repo, this project focuses on **speech editing**
instead of pure TTS, and integrates both backends into a single interface.

## 1. Features

- **Autoregressive codec speech editing backend**  
  - Support 7 languages (zh / en / de / fr / pt / es / it)
  - Integrated with WhisperX + MMS alignment for “edit by text + span”
  - Uses UVR5 and DeepFilterNet for denoising (Optional Choice)

- **Multilingual speech editing (flow-matching backend)**  
  - Based on the LEMAS‑TTS models (`multilingual_grl`, `multilingual_prosody`)  
  - Supports the same languages as LEMAS‑TTS (zh / en / es / ru / fr / de / it / pt / id / vi)  

- **One Gradio UI for both backends**  
  - `Edit Model` selector: `multilingual_grl`, `multilingual_prosody`, `autoregressive`  
  - Shared transcription, alignment, denoise and visualization components  
  - All required models are expected under `pretrained_models/`  


## 2. Installation

### 2.1 Environment

```bash
git clone https://github.com/LEMAS-Project/LEMAS-Edit.git
cd ./LEMAS-Edit

conda create -n lemas-edit python=3.10
conda activate lemas-edit
```
### 2.2 System Dependencies

You can install system dependencies via apt or conda:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```
or
```bash
conda install -c conda-forge ffmpeg
```

### 2.3 Python Dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch + Torchaudio according to your device (CUDA / ROCm / CPU / MPS)
following the official PyTorch instructions.

### 3.4 Download Pretrained Models

Download the pretrained models for both backends from [https://huggingface.co/LEMAS-Project/LEMAS-Edit](https://huggingface.co/LEMAS-Project/LEMAS-Edit) 
and place `pretrained_models/` in the directory next to the `lemas_edit/` folder.

Once `pretrained_models/` is in place, both `lemas_tts` and `lemas_edit`
will automatically find the checkpoints and vocabs.

## 3. Usage

All commands below assume:

```bash
cd ./LEMAS-Edit
export PYTHONPATH="$PWD:${PYTHONPATH}"
```
### 3.1 Gradio Web UI (Integrated Editing Demo)

To launch the full editing UI locally:

```bash
python gradio_mix.py
```

You can customize host/port and sharing:
```bash
python gradio_mix.py --host 0.0.0.0 --port 7861 --share
```

### 3.2 CLI: Multilingual TTS and CFM Speech Editing

The `lemas_tts.scripts` entrypoints are kept for convenience and behave as in
the original LEMAS‑TTS repo:

- TTS from text:
  - Python: `lemas_tts.scripts.tts_multilingual`
  - Shell:  `lemas_tts/scripts/tts_multilingual.sh`

- speech editing:
  - Python: `lemas_tts.scripts.speech_edit_multilingual`
  - Shell:  `lemas_tts/scripts/speech_edit_multilingual.sh`

See those scripts for detailed CLI options (model choice, ckpt paths,
speed / NFE / CFG / Sway, etc.).

### 3.3 CLI: Autoregressive Codec Speech Editing (WIP)

A direct CLI for the autoregressive codec backend is provided as a starting point:

- Python entry: `lemas_edit.scripts.inference_lemas_editing`
- Shell helper: `lemas_edit/scripts/inference_lemas_editing.sh`

This script is a port of the original `VoiceCraft/inference_lemas_editing.py`
and is currently being adapted to the `lemas_edit` namespace. Its interface may
change; please refer to the script source for up‑to‑date arguments and usage.


### 3.4 Subjective Evaluation

We provide simple subjective listening tests (MUSHRA and ABX preference test) setup under `./eval`.

To install the extra dependencies for evaluation, run:

```bash
pip install git+https://github.com/descriptinc/audiotools
pip install joypy pandas
```

To start the ABX preference test, install the extra dependencies and launch the tools:

```bash
cd ./eval/abx
python abx.py      # launch Gradio ABX preference test UI
python plot.py     # aggregate results and plot preference distributions
```
To start the MUSHRA listening test, install the extra dependencies and launch the tools:

```bash
cd ./eval/mushra
python mushra.py      # launch Gradio MUSHRA listening test UI
```

## 4. Acknowledgements

This project builds on, and reuses code from, several open‑source projects:
- [VoiceCraft](https://github.com/jasonppy/VoiceCraft) – Autoregressive
  speech editing model.
- [F5‑TTS](https://github.com/SWivid/F5-TTS) – Flow Matching based TTS.
- [Vocos](https://github.com/gemelo-ai/vocos) – Fourier-based neural vocoder.
- [Seamless-Expressive](https://huggingface.co/facebook/seamless-expressive) – Prosody encoder.
- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) – Separate an audio file into various stems, using multiple models.
- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) – Noise supression using deep filtering.
- [audiotools](https://github.com/descriptinc/audiotools) – Audio tools for subjective evaluation.

If you use LEMAS‑Edit in your work, please also consider citing and acknowledging these upstream projects.

## 5. License

This repository is released under the **CC‑BY‑NC‑4.0** license.  
See https://creativecommons.org/licenses/by-nc/4.0/ for more details.
