import string
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import List
import os
import argbind
import gradio as gr
import csv
import pandas as pd
from collections import defaultdict

from audiotools import preference as pr
import rich


@argbind.bind(without_prefix=True)
@dataclass
class Config:
    folder: str = None
    save_path: str = "results.csv"
    conditions: List[str] = None
    reference: str = None
    seed: int = 0
    share: bool = True
    n_samples: int = 10


def get_text(wav_file: str):
    txt_file = Path(wav_file).with_suffix(".txt")
    if Path(txt_file).exists():
        with open(txt_file, "r") as f:
            txt = f.read()
    else:
        txt = ""
    return f"""<div style="text-align:center;font-size:large;">{txt}</div>"""


def calculate_and_print_scores(save_path, conditions, user):
    """Compute and print per-condition score statistics for a given user."""
    if not Path(save_path).exists():
        print("Result file does not exist; cannot compute scores.")
        return

    try:
        # Read CSV file
        df = pd.read_csv(save_path)

        # Filter rows belonging to the current user
        user_data = df[df['user'] == user]

        if user_data.empty:
            print(f"User {user} has no rating data.")
            return

        print("="*60)
        print(f"MUSHRA summary for user {user}:")
        print("="*60)

        # Compute stats for each condition
        stats = {}
        for condition in conditions:
            if condition in user_data.columns:
                scores = user_data[condition].dropna()
                if len(scores) > 0:
                    stats[condition] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'min': scores.min(),
                        'max': scores.max(),
                        'count': len(scores)
                    }

        # Print results
        for condition, stat in stats.items():
            print(f"{condition}:")
            print(f"  Mean score: {stat['mean']:.2f}")
            print(f"  Std dev:    {stat['std']:.2f}")
            print(f"  Min score:  {stat['min']:.2f}")
            print(f"  Max score:  {stat['max']:.2f}")
            print(f"  #Samples:   {stat['count']}")
            print()

        # Sort by mean score descending
        sorted_conditions = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        print("Sorted by mean score (high to low):")
        for i, (condition, stat) in enumerate(sorted_conditions, 1):
            print(f"{i}. {condition}: {stat['mean']:.2f}")

        print("="*60)

    except Exception as e:
        print(f"Error while computing scores: {e}")


def format_user_table(rows: list[dict]) -> str:
    """Format per-user rating records as a small markdown table (similar to ABX)."""
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


def main(config: Config):
    # Center and widen the page (same style as ABX, via app.css override)
    with gr.Blocks() as app:
        app.css = (
            pr.CUSTOM_CSS
            + """
/* Enlarge and center the whole Gradio page container (override default styles) */
body .gradio-container {
    max-width: 1200px !important;
    width: 100% !important;
    margin: 0 auto;
}
/* Allow inner blocks to stretch and occupy more width */
body .gradio-container .block {
    max-width: 1100px !important;
}
/* Make audio widgets shorter on the MUSHRA page to save vertical space */
.audio-sm {
    width: 100% !important;
}
.audio-sm audio {
    width: 100% !important;
    height: 20px !important;
}
"""
        )

        save_path = config.save_path
        reference = config.reference
        conditions = config.conditions

        # Initialize Samples and keep only samples where all systems exist (wav-only, no gt dependency)
        base_samples = pr.Samples(config.folder)
        needed = set(conditions)
        good_names = [
            name for name, cond_map in base_samples.samples.items()
            if needed.issubset(cond_map.keys())
        ]
        base_samples.samples = {k: base_samples.samples[k] for k in good_names}
        # Truncate number of samples according to config
        if config.n_samples and config.n_samples < len(good_names):
            base_samples.names = good_names[: config.n_samples]
        else:
            base_samples.names = good_names
        base_samples.n_samples = len(base_samples.names)
        samples = gr.State(base_samples)

        # Per-user results directory (aligned with ABX logic)
        results_root = Path("./")
        results_root.mkdir(exist_ok=True, parents=True)

        # Top instructions
        gr.Markdown("## MUSHRA Listening Test")
        gr.Markdown(
            "In this test, each page corresponds to one sentence. You will hear:\n"
            "- A **reference** audio clip (usually high quality, often the ground-truth recording).\n"
            "- Several **system** audio clips (A, B, C, ...), each generated by a different model.\n\n"
            "For each system, please rate the **overall perceived audio quality** on a scale from 0 to 100.\n"
            "- 0 means \"very bad quality\" (strong artifacts, very unnatural, or hard to understand).\n"
            "- 100 means \"excellent quality\" (very natural, clean, and close to the reference).\n"
            "- Scores between 0 and 100 represent intermediate quality levels.\n\n"
            "You may replay the clips as many times as you want before submitting. "
            "Try to use the **full range of the scale** and be consistent across all pages."
        )

        # User information (editable ID, mirrored to internal state)
        gr.Markdown("### User information")
        user_box = gr.Textbox(
            label="User ID",
            placeholder="Leave empty to automatically generate a random ID",
            interactive=True,
        )
        user_state = gr.State("")

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

        # Reference audio + text (reference on top, then ref_text / tar_text)
        with gr.Column():
            ref_audio = gr.Audio(
                label="Reference",
                interactive=False,
                type="filepath",
                elem_classes=["audio-sm"],
            )
            ref_txt = gr.Markdown("ref_text: ")
            tar_txt = gr.Markdown("tar_text: ")

        # MUSHRA scale visualization (rainbow bar, placed under tar_text)
        gr.HTML(pr.slider_mushra)

        # System audios + rating sliders (slider under each audio)
        audio_components = []
        ratings = []
        for i in range(len(conditions)):
            x = string.ascii_uppercase[i]
            with gr.Column():
                audio = gr.Audio(
                    label=f"Play {x}",
                    interactive=False,
                    type="filepath",
                    elem_classes=["audio-sm"],
                )
                slider = gr.Slider(value=50, interactive=True)
                audio_components.append(audio)
                ratings.append(slider)

        # Output order: reference + all system audios
        output_audios = [ref_audio] + audio_components

        def build(user_box_val, user_state_val, samples, *rating_values):
            # Resolve user ID: text box > previous state > random
            uid = (user_box_val or "").strip() or (user_state_val or "").strip()
            if not uid:
                uid = f"user_{secrets.token_hex(4)}"

            # Filter out samples this user has done already, by looking in the CSV.
            samples.filter_completed(uid, save_path)

            # Save ratings for the previous sample (if any)
            if samples.current > 0:
                name = samples.names[samples.current - 1]
                result = {"sample": name, "user": uid}
                for k, r in zip(samples.order, rating_values):
                    result[k] = r
                pr.save_result(result, save_path)

            # Randomize only among system conditions; do not treat gt as a reference condition in pr.Samples
            updates_sys, done, pbar = samples.get_next_sample(None, conditions)

            # Compute current sample name and construct reference/target text and reference audio path
            ref_text_value = ""
            tar_text_value = ""
            ref_audio_path = None
            if 0 < samples.current <= len(samples):
                cur_name = samples.names[samples.current - 1]  # e.g. xxx.wav
                stem = Path(cur_name).stem
                # Parse src / tgt language ids from the filename
                # Rule: first two chars are src, first two chars after '--' are tgt
                src_lang = stem[:2]
                tgt_lang = ""
                if "--" in stem:
                    try:
                        right = stem.split("--", 1)[1]
                        tgt_lang = right[:2]
                    except Exception:
                        tgt_lang = ""
                mushra_root = Path(config.folder)
                gt_dir = mushra_root / (reference or "gt")
                txt_path = gt_dir / f"{stem}.txt"
                mp3_path = gt_dir / f"{stem}.mp3"

                # Use first line of text file as ref_text, second line (if present) as tar_text
                if txt_path.exists():
                    text_raw = txt_path.read_text(encoding="utf-8").splitlines()
                    ref_text_value = text_raw[0] if len(text_raw) > 0 else ""
                    tar_text_value = text_raw[1] if len(text_raw) > 1 else ""
                else:
                    ref_text_value = f"(No reference text found: {txt_path})"
                    tar_text_value = ""
                if mp3_path.exists():
                    ref_audio_path = str(mp3_path)
            else:
                ref_text_value = "All samples have been completed."
                tar_text_value = ""
                ref_audio_path = None

            # Collect per-user rows from the global CSV (printed in the backend, like ABX)
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

            # If this user finished all samples, save to <uid>.csv
            finished = samples.current == len(samples)
            if finished and user_rows:
                user_csv = results_root / f"{uid}.csv"
                with user_csv.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=user_rows[0].keys())
                    writer.writeheader()
                    writer.writerows(user_rows)
                rich.print(f"[bold cyan]Saved per-user results[/bold cyan] -> {user_csv}")
                # Also print condition-wise statistics
                calculate_and_print_scores(save_path, conditions, uid)

            # Update reference text / reference audio + reset sliders
            slider_reset = gr.update(value=50)
            ref_txt_update = gr.update(value=f"ref_text [{src_lang}]: {ref_text_value}")
            tar_txt_update = gr.update(value=f"tar_text [{tgt_lang}]: {tar_text_value}")
            ref_update = gr.update(value=ref_audio_path)

            return (
                [ref_update]               # Reference audio
                + updates_sys              # System audios
                + [slider_reset for _ in ratings]
                + [done, samples, pbar, ref_txt_update, tar_txt_update]
            )

        progress = gr.HTML()
        begin = gr.Button("Submit")
        begin.click(
            fn=build,
            inputs=[user_box, user_state, samples] + ratings,
            outputs=output_audios + ratings + [begin, samples, progress, ref_txt, tar_txt],
        )

        app.launch(server_name="0.0.0.0", server_port=51231, share=config.share)


if __name__ == "__main__":

    script_directory = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_directory, "res", "results.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    config = Config(folder=os.path.join(script_directory, "audios"), save_path=save_path, 
    conditions=["lemas-tts", "lemas-tts-prosody", "openaudio-s1-mini"], reference="gt", share=True)

    main(config)
