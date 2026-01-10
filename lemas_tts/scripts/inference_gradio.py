import gc
import os
import platform
import psutil
import tempfile
from glob import glob
import traceback
import click
import gradio as gr
import torch

import sys
from pathlib import Path

# Add the local code directory so that `lemas_tts` can be imported when running this
# script directly without installing the package.
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[2]  # .../code
sys.path.append(str(SRC_ROOT))


def _find_repo_root(start: Path) -> Path:
    """Locate the repo root by looking for a `pretrained_models` folder upwards."""
    for p in [start, *start.parents]:
        if (p / "pretrained_models").is_dir():
            return p
    cwd = Path.cwd()
    if (cwd / "pretrained_models").is_dir():
        return cwd
    return start


REPO_ROOT = _find_repo_root(THIS_FILE)
PRETRAINED_ROOT = REPO_ROOT / "pretrained_models"
CKPTS_ROOT = PRETRAINED_ROOT / "ckpts"
DATA_ROOT = PRETRAINED_ROOT / "data"
UVR5_CODE_DIR = REPO_ROOT / "code" / "uvr5"
UVR5_MODEL_DIR = PRETRAINED_ROOT / "uvr5" / "models" / "MDX_Net_Models" / "model_data"

from lemas_tts.api import F5TTS
import torch, torchaudio
import soundfile as sf

# Global variables
tts_api = None
last_checkpoint = ""
last_device = ""
last_ema = None

# Device detection
device = (
    "cuda"
    if torch.cuda.is_available()
    else "xpu"
    if torch.xpu.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class UVR5:
    def __init__(self, model_dir):
        code_dir = str(UVR5_CODE_DIR)
        self.model = self.load_model(str(model_dir), code_dir)
    
    def load_model(self, model_dir, code_dir):
        import sys, json, os
        sys.path.append(code_dir)
        from multiprocess_cuda_infer import ModelData, Inference
        model_path = os.path.join(model_dir, 'Kim_Vocal_1.onnx')
        config_path = os.path.join(model_dir, 'MDX-Net-Kim-Vocal1.json')
        configs = json.loads(open(config_path, 'r', encoding='utf-8').read())
        model_data = ModelData(
            model_path=model_path,
            audio_path = model_dir,
            result_path = model_dir,
            device = 'cpu',
            process_method = "MDX-Net",
            base_dir=code_dir,
            **configs
        )

        uvr5_model = Inference(model_data, 'cpu')
        uvr5_model.load_model(model_path, 1)
        return uvr5_model
        
    def denoise(self, audio_info):
        print("denoise UVR5: ", audio_info)
        input_audio = load_wav(audio_info, sr=44100, channel=2)
        output_audio = self.model.demix_base({0:input_audio.squeeze()}, is_match_mix=False)
        # transform = torchaudio.transforms.Resample(44100, 16000)
        # output_audio = transform(output_audio)
        return output_audio.squeeze().T.numpy(), 44100


denoise_model = UVR5(UVR5_MODEL_DIR)

def load_wav(audio_info, sr=16000, channel=1):
    print("load audio:", audio_info)
    audio, raw_sr = torchaudio.load(audio_info)
    audio = audio.T if len(audio.shape) > 1 and audio.shape[1] == 2 else audio
    audio = audio / torch.max(torch.abs(audio))
    audio = audio.squeeze().float()
    if channel == 1 and len(audio.shape) == 2:  # stereo to mono
        audio = audio.mean(dim=0, keepdim=True)
    elif channel == 2 and len(audio.shape) == 1:
        audio = torch.stack((audio, audio)) # mono to stereo
    if raw_sr != sr:
        audio = torchaudio.functional.resample(audio.squeeze(), raw_sr, sr)
    audio = torch.clip(audio, -0.999, 0.999).squeeze()
    return audio


def denoise(audio_info):
    save_path = "./denoised_audio.wav"
    denoised_audio, sr = denoise_model.denoise(audio_info)
    sf.write(save_path, denoised_audio, sr, format='wav', subtype='PCM_24')
    print("save denoised audio:", save_path)
    return save_path

def cancel_denoise(audio_info):
    return audio_info


def get_checkpoints_project(project_name=None, is_gradio=True):
    """Get available checkpoint files"""
    checkpoint_dir = [str(CKPTS_ROOT)]
    if project_name is None:
        # Look for checkpoints in common locations
        files_checkpoints = []
        for path in checkpoint_dir:
            if os.path.isdir(path):
                files_checkpoints.extend(glob(os.path.join(path, "**/*.pt"), recursive=True))
                files_checkpoints.extend(glob(os.path.join(path, "**/*.safetensors"), recursive=True))
                break
    else:
        # project_name = project_name.replace("_pinyin", "").replace("_char", "")
        project_name = "_".join(["F5TTS_v1_Base", "vocos", "custom", project_name.replace("_custom", "")]) if project_name != "F5TTS_v1_Base" else project_name
        if os.path.isdir(checkpoint_dir[0]):
            files_checkpoints = glob(os.path.join(checkpoint_dir[0], project_name, "*.pt"))
            files_checkpoints.extend(glob(os.path.join(checkpoint_dir[0], project_name, "*.safetensors")))
        else:
            files_checkpoints = []
    print("files_checkpoints:", project_name, files_checkpoints)
    # Separate pretrained and regular checkpoints
    pretrained_checkpoints = [f for f in files_checkpoints if "pretrained_" in os.path.basename(f)]
    regular_checkpoints = [
        f
        for f in files_checkpoints
        if "pretrained_" not in os.path.basename(f) and "model_last.pt" not in os.path.basename(f)
    ]
    last_checkpoint = [f for f in files_checkpoints if "model_last.pt" in os.path.basename(f)]

    # Sort regular checkpoints by number
    try:
        regular_checkpoints = sorted(
            regular_checkpoints, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
        )
    except (IndexError, ValueError):
        regular_checkpoints = sorted(regular_checkpoints)

    # Combine in order: pretrained, regular, last
    files_checkpoints = pretrained_checkpoints + regular_checkpoints + last_checkpoint

    select_checkpoint = None if not files_checkpoints else files_checkpoints[-1]

    if is_gradio:
        return gr.update(choices=files_checkpoints, value=select_checkpoint)

    return files_checkpoints, select_checkpoint


def get_available_projects():
    """Get available project names from data directory"""
    data_path = str(DATA_ROOT)

    project_list = []
    if os.path.isdir(data_path):
        for folder in os.listdir(data_path):
            if "test" in folder:
                continue
            project_list.append(folder)

    # Fallback to a sensible default if no projects are found
    if not project_list:
        project_list = ["multilingual_acc_grl_custom"]

    return project_list


def infer(
    project, file_checkpoint, exp_name, ref_text, ref_audio, denoise_audio, gen_text, nfe_step, use_ema, separate_langs, frontend, speed, cfg_strength, use_acc_grl, ref_ratio, no_ref_audio, sway_sampling_coef, use_prosody_encoder, seed
):
    global last_checkpoint, last_device, tts_api, last_ema

    if not os.path.isfile(file_checkpoint):
        return None, "Checkpoint not found!", ""

    if denoise_audio:
        ref_audio = denoise_audio

    device_test = device  # Use the global device

    if last_checkpoint != file_checkpoint or last_device != device_test or last_ema != use_ema or tts_api is None:
        if last_checkpoint != file_checkpoint:
            last_checkpoint = file_checkpoint

        if last_device != device_test:
            last_device = device_test

        if last_ema != use_ema:
            last_ema = use_ema

        # Try to find vocab file
        vocab_file = None
        possible_vocab_paths = [
            str(DATA_ROOT / project / "vocab.txt"),
            # legacy fallbacks for older layouts
            f"./data/{project}/vocab.txt",
            f"../../data/{project}/vocab.txt",
            "./data/Emilia_ZH_EN_pinyin/vocab.txt",
            "../../data/Emilia_ZH_EN_pinyin/vocab.txt",
        ]
        
        for path in possible_vocab_paths:
            if os.path.isfile(path):
                vocab_file = path
                break
        
        if vocab_file is None:
            return None, "Vocab file not found!", ""

        try:
            tts_api = F5TTS(
                model=exp_name,
                ckpt_file=file_checkpoint,
                vocab_file=vocab_file,
                device=device_test,
                use_ema=use_ema,
                frontend=frontend,
                use_prosody_encoder=use_prosody_encoder,
                prosody_cfg_path=str(CKPTS_ROOT / "prosody_encoder" / "pretssel_cfg.json"),
                prosody_ckpt_path=str(CKPTS_ROOT / "prosody_encoder" / "prosody_encoder_UnitY2.pt"),
            )
        except Exception as e:
            traceback.print_exc()
            return None, f"Error loading model: {str(e)}", ""

        print("Model loaded >>", device_test, file_checkpoint, use_ema)

    if seed == -1:  # -1 used for random
        seed = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            tts_api.infer(
                ref_file=ref_audio,
                ref_text=ref_text.strip(),
                gen_text=gen_text.strip(),
                nfe_step=nfe_step,
                separate_langs=separate_langs,
                speed=speed,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                use_acc_grl=use_acc_grl,
                ref_ratio=ref_ratio,
                no_ref_audio=no_ref_audio,
                use_prosody_encoder=use_prosody_encoder,
                file_wave=f.name,
                seed=seed,
            )
            return f.name, f"Device: {tts_api.device}", str(tts_api.seed)
    except Exception as e:
        traceback.print_exc()
        return None, f"Inference error: {str(e)}", ""


def get_gpu_stats():
    """Get GPU statistics"""
    gpu_stats = ""

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_properties = torch.cuda.get_device_properties(i)
            total_memory = gpu_properties.total_memory / (1024**3)  # in GB
            allocated_memory = torch.cuda.memory_allocated(i) / (1024**2)  # in MB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024**2)  # in MB

            gpu_stats += (
                f"GPU {i} Name: {gpu_name}\n"
                f"Total GPU memory (GPU {i}): {total_memory:.2f} GB\n"
                f"Allocated GPU memory (GPU {i}): {allocated_memory:.2f} MB\n"
                f"Reserved GPU memory (GPU {i}): {reserved_memory:.2f} MB\n\n"
            )
    elif torch.xpu.is_available():
        gpu_count = torch.xpu.device_count()
        for i in range(gpu_count):
            gpu_name = torch.xpu.get_device_name(i)
            gpu_properties = torch.xpu.get_device_properties(i)
            total_memory = gpu_properties.total_memory / (1024**3)  # in GB
            allocated_memory = torch.xpu.memory_allocated(i) / (1024**2)  # in MB
            reserved_memory = torch.xpu.memory_reserved(i) / (1024**2)  # in MB

            gpu_stats += (
                f"GPU {i} Name: {gpu_name}\n"
                f"Total GPU memory (GPU {i}): {total_memory:.2f} GB\n"
                f"Allocated GPU memory (GPU {i}): {allocated_memory:.2f} MB\n"
                f"Reserved GPU memory (GPU {i}): {reserved_memory:.2f} MB\n\n"
            )
    elif torch.backends.mps.is_available():
        gpu_count = 1
        gpu_stats += "MPS GPU\n"
        total_memory = psutil.virtual_memory().total / (
            1024**3
        )  # Total system memory (MPS doesn't have its own memory)
        allocated_memory = 0
        reserved_memory = 0

        gpu_stats += (
            f"Total system memory: {total_memory:.2f} GB\n"
            f"Allocated GPU memory (MPS): {allocated_memory:.2f} MB\n"
            f"Reserved GPU memory (MPS): {reserved_memory:.2f} MB\n"
        )

    else:
        gpu_stats = "No GPU available"

    return gpu_stats


def get_cpu_stats():
    """Get CPU statistics"""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_used = memory_info.used / (1024**2)
    memory_total = memory_info.total / (1024**2)
    memory_percent = memory_info.percent

    pid = os.getpid()
    process = psutil.Process(pid)
    nice_value = process.nice()

    cpu_stats = (
        f"CPU Usage: {cpu_usage:.2f}%\n"
        f"System Memory: {memory_used:.2f} MB used / {memory_total:.2f} MB total ({memory_percent}% used)\n"
        f"Process Priority (Nice value): {nice_value}"
    )

    return cpu_stats


def get_combined_stats():
    """Get combined system stats"""
    gpu_stats = get_gpu_stats()
    cpu_stats = get_cpu_stats()
    combined_stats = f"### GPU Stats\n{gpu_stats}\n\n### CPU Stats\n{cpu_stats}"
    return combined_stats


# Create Gradio interface
with gr.Blocks(title="LEMAS-TTS Inference") as app:
    gr.Markdown(
        """
        # Zero-Shot TTS

        Set seed to -1 for random generation.
        """
    )
    with gr.Accordion("Model configuration", open=False):
    # Model configuration
        with gr.Row():
            exp_name = gr.Radio(
                label="Model", choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"], value="F5TTS_v1_Base", visible=False
            )
        # Project selection
        available_projects = get_available_projects()

        # Get initial checkpoints
        list_checkpoints, checkpoint_select = get_checkpoints_project(available_projects[0] if available_projects else None, False)

        with gr.Row():
            with gr.Column(scale=1):
                # load_models_btn = gr.Button(value="Load models")
                cm_project = gr.Dropdown(
                    choices=available_projects, 
                    value=available_projects[0] if available_projects else None,
                    label="Project", 
                    allow_custom_value=True, 
                    scale=4
                )
                
            with gr.Column(scale=5):
                cm_checkpoint = gr.Dropdown(
                    choices=list_checkpoints, value=checkpoint_select, label="Checkpoints", allow_custom_value=True # scale=4, 
)
            bt_checkpoint_refresh = gr.Button("Refresh", scale=1)

        with gr.Row():
            ch_use_ema = gr.Checkbox(label="Use EMA", value=True, scale=2, info="Turn off at early stage might offer better results")
            frontend = gr.Radio(label="Frontend", choices=["phone", "char", "bpe"], value="phone", scale=3)
            separate_langs = gr.Checkbox(label="Separate Languages", value=True, scale=2, info="separate language tokens")

        # Inference parameters
        with gr.Row():
            nfe_step = gr.Number(label="NFE Step", scale=1, value=64)
            speed = gr.Slider(label="Speed", scale=3, value=1.0, minimum=0.5, maximum=1.5, step=0.1)
            cfg_strength = gr.Slider(label="CFG Strength", scale=2, value=5.0, minimum=0.0, maximum=10.0, step=1)
            sway_sampling_coef = gr.Slider(label="Sway Sampling Coef", scale=2, value=3, minimum=-1, maximum=5, step=0.1)
            ref_ratio = gr.Slider(label="Ref Ratio", scale=2, value=1.0, minimum=0.0, maximum=1.0, step=0.1)
            no_ref_audio = gr.Checkbox(label="No Reference Audio", value=False, scale=1, info="No mel condition")
            use_acc_grl = gr.Checkbox(label="Use accent grl condition", value=False, scale=1, info="Use accent grl condition")
            use_prosody_encoder = gr.Checkbox(label="Use prosody encoder", value=False, scale=1, info="Use prosody encoder")
            seed = gr.Number(label="Random Seed", scale=1, value=5828684826493313192, minimum=-1)


    # Input fields
    ref_text = gr.Textbox(label="Reference Text", placeholder="Enter the text for the reference audio...")
    ref_audio = gr.Audio(label="Reference Audio", type="filepath", interactive=True, show_download_button=True, editable=True)


    with gr.Row():
        denoise_btn = gr.Button(value="Denoise")
        cancel_btn = gr.Button(value="Cancel Denoise")
    denoise_audio = gr.Audio(label="Denoised Audio", value=None, type="filepath", interactive=True, show_download_button=True, editable=True)

    gen_text = gr.Textbox(label="Text to Generate", placeholder="Enter the text you want to generate...")

    # Inference button and outputs
    with gr.Row():
        txt_info_gpu = gr.Textbox("", label="Device Info")
        seed_info = gr.Textbox(label="Used Random Seed")
        check_button_infer = gr.Button("Generate Audio", variant="primary")

    gen_audio = gr.Audio(label="Generated Audio", type="filepath", interactive=True, show_download_button=True, editable=True)

    # Examples
    examples = gr.Examples(
        examples=[
            [
                "Ich glaub, mein Schwein pfeift.",
                str(DATA_ROOT / "test_examples" / "de.wav"),
                "我觉得我的猪在吹口哨。",
            ],
            [
                "em, #1 I have a list of YouTubers, and I'm gonna be going to their houses and raiding them by.",
                str(DATA_ROOT / "test_examples" / "en.wav"),
                "我有一份 YouTuber 名单，我打算去他们家，对他们进行突袭。",
            ],
            [
                "Te voy a dar un tip #1 que le copia a John Rockefeller, uno de los empresarios más picudos de la historia.",
                str(DATA_ROOT / "test_examples" / "es.wav"),
                "我要给你一个从历史上最精明的商人之一约翰·洛克菲勒那里抄来的秘诀。",
            ],
            [
                "Per l'amor di Dio #1 fai, #2 se pensi di non poterti fermare, fallo #1 e fallo.",
                str(DATA_ROOT / "test_examples" / "it.wav"),
                "看在上帝的份上，去做吧，如果你认为你无法停止，那就去做吧，继续做下去。",
            ],
            [
                "Nova, #1 dia 25 desse mês vai rolar operação the last Frontier.",
                str(DATA_ROOT / "test_examples" / "pt.wav"),
                "新消息，本月二十五日，'最后的边疆行动'将启动。",
            ],
            # ["Good morning! #1 ",
            # "/mnt/code/lemas/F5-TTS/data/trueman/recognition_d0a02641c090813574a8ec398220339f_0.wav",
            # " #1"
            # ],
            # ["Good morning! #1 ",
            # "/mnt/code/lemas/F5-TTS/data/trueman/recognition_d0a02641c090813574a8ec398220339f_1.wav",
            # " #1",
            # ],
            # ["Good morning! #1 ",
            # "/mnt/code/lemas/F5-TTS/data/trueman/recognition_d0a02641c090813574a8ec398220339f_2.wav",
            # " #1",
            # ],
            # ["Oh, and in case I don't see ya, #1",
            # "/mnt/code/lemas/F5-TTS/data/trueman/recognition_d0a02641c090813574a8ec398220339f_3.wav",
            # " #1",
            # ],
            # ["Good afternoon, good evening, and good night. #1",
            # "/mnt/code/lemas/F5-TTS/data/trueman/recognition_d0a02641c090813574a8ec398220339f_4.wav",
            # " #1",
            # ],
        ],
        inputs=[
            ref_text,
            ref_audio,
            gen_text,
        ],
        outputs=[gen_audio, txt_info_gpu, seed_info],
        fn=infer,
        cache_examples=False
    )

    # System Info section at the bottom
    gr.Markdown("---")
    gr.Markdown("## System Information")
    with gr.Accordion("Update System Stats", open=False):
        update_button = gr.Button("Update System Stats", scale=1)
        output_box = gr.Textbox(label="GPU and CPU Information", lines=5, scale=5)

    def update_stats():
        return get_combined_stats()
        
    
    denoise_btn.click(fn=denoise,
                        inputs=[ref_audio],
                        outputs=[denoise_audio])

    cancel_btn.click(fn=cancel_denoise,
                        inputs=[ref_audio],
                        outputs=[denoise_audio])

    # Event handlers
    check_button_infer.click(
        fn=infer,
        inputs=[
            cm_project,
            cm_checkpoint,
            exp_name,
            ref_text,
            ref_audio,
            denoise_audio,
            gen_text,
            nfe_step,
            ch_use_ema,
            separate_langs,
            frontend,
            speed,
            cfg_strength,
            use_acc_grl,
            ref_ratio,
            no_ref_audio,
            sway_sampling_coef,
            use_prosody_encoder,
            seed,
        ],
        outputs=[gen_audio, txt_info_gpu, seed_info],
    )

    bt_checkpoint_refresh.click(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])
    cm_project.change(fn=get_checkpoints_project, inputs=[cm_project], outputs=[cm_checkpoint])

    ref_audio.change(
            fn=lambda x: None,
            inputs=[ref_audio],
            outputs=[denoise_audio]
        )

    update_button.click(fn=update_stats, outputs=output_box)
    
    # Auto-load system stats on startup
    app.load(fn=update_stats, outputs=output_box)


@click.command()
@click.option("--port", "-p", default=7860, type=int, help="Port to run the app on")
@click.option("--host", "-H", default="0.0.0.0", help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
def main(port, host, share, api):
    global app
    print("Starting LEMAS-TTS Inference Interface...")
    print(f"Device: {device}")
    app.queue(api_open=api).launch(
        server_name=host,
        server_port=port,
        share=share,
        show_api=api,
        allowed_paths=[str(DATA_ROOT)],
    )


if __name__ == "__main__":
    main()
