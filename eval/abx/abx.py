import csv
import math
import secrets
import string
from dataclasses import dataclass
from pathlib import Path

import gradio as gr
import numpy as np
import rich
import soundfile as sf

from audiotools import preference as pr


@dataclass
class Config:
    folder: str | None = None
    save_path: str = "results.csv"
    conditions: list[str] | None = None  # e.g. ["lemas-edit", "lemas-tts"]
    reference: str | None = None        # optional reference condition
    seed: int = 0


def random_sine(f: float):
    # Only used for toy data; real ABX will use existing audios with their own sr.
    fs = 24000
    duration = 5.0  # seconds
    volume = 0.1
    num_samples = int(fs * duration)
    samples = volume * np.sin(2 * math.pi * (f / fs) * np.arange(num_samples))
    return samples, fs


# Example config; in real experiments, point `folder` to the root directory
# that contains lemas-edit / lemas-tts / gt.
config = Config(
    folder=str(Path(__file__).parent),
    save_path="results.csv",                    # Global results CSV
    conditions=["lemas-edit", "lemas-tts"],     # Two systems: A / B
    reference=None,                             # Do not use Samples reference logic
)


results_root = Path("res")
results_root.mkdir(exist_ok=True, parents=True)


def format_user_table(rows: list[dict]) -> str:
    """Format per-user results as a small markdown table."""
    if not rows:
        return "No ratings for this user yet."

    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + " | ".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


gt_dir = Path(__file__).parent / "gt"


with gr.Blocks() as app:
    # Increase overall width and keep the layout centered
    app.css = pr.CUSTOM_CSS + "\n.gradio-container {max-width: 1100px !important; margin: 0 auto;}\n"

    save_path = config.save_path

    # Initialize Samples and keep only samples that exist for both systems
    base_samples = pr.Samples(config.folder)
    needed = set(config.conditions)
    good_names = [
        name for name, cond_map in base_samples.samples.items()
        if needed.issubset(cond_map.keys())
    ]
    base_samples.samples = {k: base_samples.samples[k] for k in good_names}
    base_samples.names = good_names
    base_samples.n_samples = len(good_names)
    samples = gr.State(base_samples)

    reference = config.reference
    conditions = config.conditions

    # Top instructions
    gr.Markdown("## ABX Preference Test")
    gr.Markdown(
        "For each sentence, you will hear synthesized audio from two systems: A and B, "
        "as well as a reference audio clip.\n\n"
        "Please choose the system you prefer based on overall speech quality "
        "(naturalness, clarity, etc.) and use the slider below to indicate your "
        "preference from **Strongly prefer A** to **Strongly prefer B**.\n"
        "The middle of the slider means the two systems sound almost the same.\n\n"
    )

    # User information
    gr.Markdown("### User information")

    # Visible, editable user id.
    user_box = gr.Textbox(
        label="User ID",
        placeholder="Leave empty to use an auto-generated browser ID",
        interactive=True,
    )
    # Persist resolved user id for this browser session.
    user_state = gr.State("")

    # Initialize user id once per session (no JS cookie, purely server-side).
    def init_user(state_val: str):
        uid = (state_val or "").strip()
        if not uid:
            uid = f"user_{secrets.token_hex(4)}"
        return uid, uid

    app.load(
        init_user,
        inputs=[user_state],
        outputs=[user_box, user_state],
    )

    # Reference text + reference audio (stacked vertically)
    gr.Markdown("### Reference")
    with gr.Column():
        ref_text_md = gr.Markdown(
            label="Reference text",
            value="Please click Submit to start the test.",
        )
        ref_audio = gr.Audio(
            label="Reference audio",
            interactive=False,
            type="filepath",
        )

    # Rainbow bar with 5 preference levels
    pref_bar_html = pr.slider_mushra.replace("bad\">bad", "bad\">strongly prefer A") \
        .replace("poor\">poor", "poor\">weakly prefer A") \
        .replace("fair\">fair", "fair\">No preference") \
        .replace("good\">good", "good\">weakly prefer B") \
        .replace("excellent\">excellent", "excellent\">strongly prefer B")

    gr.HTML(pref_bar_html)

    # ABX preference slider: 0..100, 50 is neutral
    pref_slider = gr.Slider(
        minimum=0,
        maximum=100,
        step=1,
        value=50,
        label="Preference (0=Strong A, 50=Borderline, 100=Strong B)",
        interactive=True,
    )

    # Playback area: Play A / Play B side by side (audio widgets include their own play buttons)
    with gr.Row():
        audio_A = gr.Audio(
            label="Play A",
            interactive=False,
            type="filepath",
        )
        audio_B = gr.Audio(
            label="Play B",
            interactive=False,
            type="filepath",
        )

    def build(user_box_val, user_state_val, samples, pref_value):
        # Resolve effective user id: visible box > previous state > random
        uid = (user_box_val or "").strip() or (user_state_val or "").strip()
        if not uid:
            uid = f"user_{secrets.token_hex(4)}"

        # Filter out samples this user already rated
        samples.filter_completed(uid, save_path)

        # Save current ratings (for previous sample)
        if samples.current > 0:
            name = samples.names[samples.current - 1]
            result = {"sample": name, "user": uid}
            # A/B condition order (Samples.order only contains the two systems)
            cond_A = samples.order[0]
            cond_B = samples.order[1]
            result["cond_A"] = cond_A
            result["cond_B"] = cond_B
            result["pref"] = int(pref_value)
            pr.save_result(result, save_path)
            rich.print(f"[green]Saved rating[/green]: {result}")

        # Get next sample to rate (only two conditions A/B)
        updates_ab, done_btn, pbar = samples.get_next_sample(None, conditions)

        # Determine current sample name for reference text + reference audio mp3 path
        text_value = ""
        ref_audio_path = None
        if 0 < samples.current <= len(samples):
            cur_name = samples.names[samples.current - 1]
            stem = Path(cur_name).stem
            txt_path = gt_dir / f"{stem}.txt"
            mp3_path = gt_dir / f"{stem}.mp3"
            if txt_path.exists():
                text_value = txt_path.read_text(encoding="utf-8")
            else:
                text_value = f"(No reference text found: {txt_path})"
            if mp3_path.exists():
                ref_audio_path = str(mp3_path)
        else:
            text_value = "All samples have been completed."
            ref_audio_path = None

        # Collect per-user rows from global CSV, only printed in the backend (not shown as table in Gradio)
        user_rows: list[dict] = []
        save_path_path = Path(save_path)
        if save_path_path.exists():
            with save_path_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("user") == uid:
                        user_rows.append(row)

        if user_rows:
            rich.print(f"[yellow]User {uid} already has {len(user_rows)} records[/yellow]")
            rich.print(format_user_table(user_rows))

        # If user has finished all samples, write their CSV to res/<user>.csv
        finished = samples.current == len(samples)
        if finished and user_rows:
            user_csv = results_root / f"{uid}.csv"
            with user_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=user_rows[0].keys())
                writer.writeheader()
                writer.writerows(user_rows)
            rich.print(f"[bold cyan]Saved per-user results[/bold cyan] -> {user_csv}")

        # Reset slider to 50, update progress + user state + result preview + reference text / reference audio
        slider_reset = gr.update(value=50)
        return (
            [gr.update(value=ref_audio_path)]  # Reference audio
            + updates_ab                      # Play A, Play B
            + [
                slider_reset,
                done_btn,
                samples,
                pbar,
                uid,
                gr.update(value=text_value),
            ]
        )

    progress = gr.HTML()
    begin = gr.Button("Submit", elem_id="start-survey")

    begin.click(
        fn=build,
        inputs=[user_box, user_state, samples, pref_slider],
        outputs=[ref_audio, audio_A, audio_B, pref_slider, begin, samples, progress, user_state, ref_text_md],
    )

    app.launch(server_name="0.0.0.0", server_port=51230, share=True)
