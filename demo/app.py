"""Nunchaku Creative Pipeline — interactive Gradio demo.

5 tabs: Text-to-Image, Edit Image, Text-to-Video, Image-to-Video, Pipeline.
The Pipeline tab chains: generate → edit → animate in one flow.

Usage:
    echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
    pip install gradio requests Pillow python-dotenv
    python demo/app.py
"""

import io
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nunchaku_demo")

# Allow importing nunchaku.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from nunchaku import NunchakuClient
from variation_factory import (
    PRESETS,
    build_prompt,
    extract_albedo_textures,
    replace_albedo_textures,
    sanitize_glb,
    snap_to_valid_size,
)

# ---------------------------------------------------------------------------
# Models & options
# ---------------------------------------------------------------------------

T2I_MODELS = [
    "nunchaku-qwen-image",
    "nunchaku-flux.2-klein-9b",
]

I2I_MODELS = [
    "nunchaku-qwen-image-edit",
    "nunchaku-flux.2-klein-9b-edit",
]

T2V_MODELS = [
    "nunchaku-wan2.2-lightning-t2v",
]

I2V_MODELS = [
    "nunchaku-wan2.2-lightning-i2v",
]

TIERS = ["fast", "radically_fast"]

# Default inference steps per model (from tier configs)
# Qwen: 28 steps (fast), 4 steps (radically_fast)
# FLUX: 4 steps (pre-distilled)
# Video Lightning: 4 steps, guidance_scale=1.0, 81 frames
MODEL_DEFAULTS = {
    "nunchaku-qwen-image":           {"steps": 28, "guidance": 0, "rf_steps": 4},
    "nunchaku-qwen-image-edit":      {"steps": 28, "guidance": 4.0, "rf_steps": 4},
    "nunchaku-flux.2-klein-9b":      {"steps": 4, "guidance": 1.0},
    "nunchaku-flux.2-klein-9b-edit": {"steps": 4, "guidance": 1.0},
    "nunchaku-wan2.2-lightning-t2v": {"steps": 4, "guidance": 1.0, "frames": 81},
    "nunchaku-wan2.2-lightning-i2v": {"steps": 4, "guidance": 1.0, "frames": 81},
}

IMAGE_SIZES = ["1024x1024", "1024x768", "768x1024"]
VIDEO_SIZES = ["1280x720", "720x1280"]


def get_client() -> NunchakuClient:
    return NunchakuClient()


# ---------------------------------------------------------------------------
# Tab 1: Text-to-Image
# ---------------------------------------------------------------------------


def tab_text_to_image(prompt, model, size, tier, seed):
    client = get_client()
    seed_val = int(seed) if seed and int(seed) >= 0 else None
    img_bytes = client.text_to_image(
        prompt=prompt, model=model, size=size, tier=tier, seed=seed_val
    )
    return Image.open(io.BytesIO(img_bytes))


# ---------------------------------------------------------------------------
# Tab 2: Edit Image
# ---------------------------------------------------------------------------


def tab_edit_image(image, prompt, model, tier):
    if image is None:
        raise gr.Error("Upload an image first.")
    client = get_client()
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    edited_bytes = client.edit_image(
        image=buf.getvalue(), prompt=prompt, model=model, tier=tier
    )
    return Image.open(io.BytesIO(edited_bytes))


# ---------------------------------------------------------------------------
# Tab 3: Text-to-Video
# ---------------------------------------------------------------------------


def tab_text_to_video(prompt, model, size):
    client = get_client()
    video_bytes = client.text_to_video(prompt=prompt, model=model, size=size)
    path = tempfile.mktemp(suffix=".mp4")
    with open(path, "wb") as f:
        f.write(video_bytes)
    return path


# ---------------------------------------------------------------------------
# Tab 4: Image-to-Video
# ---------------------------------------------------------------------------


def tab_image_to_video(image, prompt, model, size):
    if image is None:
        raise gr.Error("Upload an image first.")
    client = get_client()
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    video_bytes = client.image_to_video(
        image=buf.getvalue(), prompt=prompt, model=model, size=size
    )
    path = tempfile.mktemp(suffix=".mp4")
    with open(path, "wb") as f:
        f.write(video_bytes)
    return path


# ---------------------------------------------------------------------------
# Tab 5: Pipeline (generate → edit → animate)
# ---------------------------------------------------------------------------


def tab_pipeline(gen_prompt, edit_prompt, animate_prompt, tier):
    client = get_client()

    # Step 1: Generate
    yield "Step 1/3: Generating image...", None, None, None
    img_bytes = client.text_to_image(
        prompt=gen_prompt, tier=tier
    )
    gen_image = Image.open(io.BytesIO(img_bytes))

    # Step 2: Edit
    yield "Step 2/3: Editing image...", gen_image, None, None
    edited_bytes = client.edit_image(
        image=img_bytes, prompt=edit_prompt, tier=tier
    )
    edited_image = Image.open(io.BytesIO(edited_bytes))

    # Step 3: Animate
    yield "Step 3/3: Animating to video (this takes ~30s)...", gen_image, edited_image, None
    video_bytes = client.image_to_video(
        image=edited_bytes, prompt=animate_prompt
    )
    path = tempfile.mktemp(suffix=".mp4")
    with open(path, "wb") as f:
        f.write(video_bytes)

    yield "Done!", gen_image, edited_image, path


# ---------------------------------------------------------------------------
# Tab 6: Variation Factory (GLB albedo variants)
# ---------------------------------------------------------------------------

MAX_VARIANTS = 10
MAX_SLOTS = MAX_VARIANTS + 1  # variant_0 (original) + N variants


def tab_variation_factory(glb_file, n, preset, intensity, extra, quality, seed_in):
    run_started = time.time()
    if glb_file is None:
        logger.warning("variation_factory invoked with no file")
        raise gr.Error("Upload a .glb or .gltf file first.")

    glb_path = Path(glb_file if isinstance(glb_file, str) else glb_file.name)
    n = int(n)
    tier = "radically_fast" if "radically_fast" in quality else "fast"

    logger.info(
        "variation_factory start: file=%s n=%d preset=%s intensity=%s tier=%s seed_in=%s extra=%r",
        glb_path.name, n, preset, intensity, tier, seed_in, extra or "",
    )

    base_seed = random.randint(1, 2**31 - 1) if int(seed_in) == 0 else int(seed_in)
    output_root = Path("output")
    output_root.mkdir(exist_ok=True)
    out_dir = Path(tempfile.mkdtemp(prefix="variation_factory_", dir=output_root))

    stem = glb_path.stem
    sanitized_base = out_dir / f"{stem}_sanitized.glb"
    try:
        sanitize_glb(glb_path, sanitized_base)
    except Exception as e:
        logger.exception("sanitize_glb failed: %s", e)
        raise gr.Error(f"Could not parse GLB: {e}")

    try:
        extracted = extract_albedo_textures(sanitized_base)
    except ValueError as e:
        logger.error("texture extraction failed: %s", e)
        raise gr.Error(str(e))

    prompt = build_prompt(preset, intensity, extra or "")

    sample = Image.open(io.BytesIO(extracted[0][1]))
    w, h = snap_to_valid_size(*sample.size)
    size_str = f"{w}x{h}"
    logger.info(
        "prompt=%r  size=%s  base_seed=%d  out_dir=%s  source_dims=%dx%d",
        prompt, size_str, base_seed, out_dir, sample.size[0], sample.size[1],
    )

    total_slots = n + 1
    orig_out = out_dir / f"{stem}_variant_0.glb"
    shutil.copy(sanitized_base, orig_out)

    # Initial yield uses gr.update() to set visibility and labels; subsequent
    # yields pass plain values so Gradio preserves visibility and re-renders
    # the Model3D viewers correctly.
    init_models = [gr.update(value=None, visible=False) for _ in range(MAX_SLOTS)]
    init_errors = [gr.update(value="", visible=False) for _ in range(MAX_SLOTS)]
    for i in range(total_slots):
        init_models[i] = gr.update(value=None, visible=True, label=f"variant_{i}")
    init_models[0] = gr.update(value=str(orig_out), visible=True, label="variant_0 (original)")

    yield (
        f"1/{total_slots} ready. Generating variants at {size_str}, tier={tier}, seed={base_seed}...",
        *init_models,
        *init_errors,
    )

    # Track slot values as plain Python values from here on
    model_values: list[str | None] = [None] * MAX_SLOTS
    model_values[0] = str(orig_out)
    error_values: list[str] = [""] * MAX_SLOTS

    client = NunchakuClient()
    success = 1

    for v in range(1, n + 1):
        variant_seed = base_seed + v
        v_started = time.time()
        logger.info("variant %d/%d: starting (seed=%d)", v, n, variant_seed)
        try:
            replacements: dict[int, bytes] = {}
            for img_idx, orig_bytes, _mime in extracted:
                src = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
                if src.size != (w, h):
                    src = src.resize((w, h), Image.LANCZOS)
                buf = io.BytesIO()
                src.save(buf, format="PNG")
                input_bytes = buf.getvalue()

                api_started = time.time()
                logger.debug(
                    "  variant %d img[%d]: calling /v1/images/edits (%d bytes in, size=%s, seed=%d)",
                    v, img_idx, len(input_bytes), size_str, variant_seed,
                )
                result_bytes = client.edit_image(
                    image=input_bytes,
                    prompt=prompt,
                    model="nunchaku-qwen-image-edit",
                    tier=tier,
                    size=size_str,
                    seed=variant_seed,
                    output_format="png",
                )
                logger.info(
                    "  variant %d img[%d]: api ok in %.2fs (%d bytes out)",
                    v, img_idx, time.time() - api_started, len(result_bytes),
                )
                replacements[img_idx] = result_bytes

            variant_out = out_dir / f"{stem}_variant_{v}.glb"
            replace_albedo_textures(sanitized_base, replacements, variant_out, new_mime="image/png")

            model_values[v] = str(variant_out)
            error_values[v] = ""
            success += 1
            logger.info("variant %d/%d: done in %.2fs -> %s", v, n, time.time() - v_started, variant_out.name)
        except Exception as e:
            logger.exception("variant %d/%d: FAILED after %.2fs", v, n, time.time() - v_started)
            model_values[v] = None
            error_values[v] = f"Error: {e}"

        yield (
            f"{success}/{total_slots} ready (variant {v} of {n} processed)...",
            *model_values,
            *error_values,
        )

    logger.info(
        "variation_factory done: %d/%d variants in %.2fs, base_seed=%d, out_dir=%s",
        success, total_slots, time.time() - run_started, base_seed, out_dir,
    )
    yield (
        f"Done. {success}/{total_slots} variants produced. Base seed: {base_seed}. Files in {out_dir}",
        *model_values,
        *error_values,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Nunchaku Creative Pipeline", theme=gr.themes.Soft()) as app:
    gr.Markdown("# Nunchaku Creative Pipeline")
    gr.Markdown("Generate, edit, and animate images using the Nunchaku API.")

    # -- Tab 1: Text-to-Image --
    with gr.Tab("Text to Image"):
        with gr.Row():
            with gr.Column():
                t2i_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the image...")
                t2i_model = gr.Dropdown(T2I_MODELS, value=T2I_MODELS[0], label="Model")
                t2i_size = gr.Dropdown(IMAGE_SIZES, value="1024x1024", label="Size")
                t2i_tier = gr.Dropdown(TIERS, value="fast", label="Tier")
                t2i_seed = gr.Number(value=-1, label="Seed (-1 = random)")
                t2i_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                t2i_output = gr.Image(label="Result", type="pil")
        t2i_btn.click(tab_text_to_image, [t2i_prompt, t2i_model, t2i_size, t2i_tier, t2i_seed], t2i_output)

    # -- Tab 2: Edit Image --
    with gr.Tab("Edit Image"):
        with gr.Row():
            with gr.Column():
                i2i_input = gr.Image(label="Input Image", type="pil")
                i2i_prompt = gr.Textbox(label="Edit Prompt", lines=2, placeholder="Describe the edit...")
                i2i_model = gr.Dropdown(I2I_MODELS, value=I2I_MODELS[0], label="Model")
                i2i_tier = gr.Dropdown(TIERS, value="fast", label="Tier")
                i2i_btn = gr.Button("Edit", variant="primary")
            with gr.Column():
                i2i_output = gr.Image(label="Result", type="pil")
        i2i_btn.click(tab_edit_image, [i2i_input, i2i_prompt, i2i_model, i2i_tier], i2i_output)

    # -- Tab 3: Text-to-Video --
    with gr.Tab("Text to Video"):
        with gr.Row():
            with gr.Column():
                t2v_prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the video...")
                t2v_model = gr.Dropdown(T2V_MODELS, value=T2V_MODELS[0], label="Model")
                t2v_size = gr.Dropdown(VIDEO_SIZES, value="1280x720", label="Size")
                t2v_btn = gr.Button("Generate Video", variant="primary")
                gr.Markdown("*Video generation takes ~30 seconds.*")
            with gr.Column():
                t2v_output = gr.Video(label="Result")
        t2v_btn.click(tab_text_to_video, [t2v_prompt, t2v_model, t2v_size], t2v_output)

    # -- Tab 4: Image-to-Video --
    with gr.Tab("Image to Video"):
        with gr.Row():
            with gr.Column():
                i2v_input = gr.Image(label="Input Image", type="pil")
                i2v_prompt = gr.Textbox(label="Prompt", lines=2, placeholder="Describe the motion...")
                i2v_model = gr.Dropdown(I2V_MODELS, value=I2V_MODELS[0], label="Model")
                i2v_size = gr.Dropdown(VIDEO_SIZES, value="1280x720", label="Size")
                i2v_btn = gr.Button("Animate", variant="primary")
                gr.Markdown("*Video generation takes ~30 seconds.*")
            with gr.Column():
                i2v_output = gr.Video(label="Result")
        i2v_btn.click(tab_image_to_video, [i2v_input, i2v_prompt, i2v_model, i2v_size], i2v_output)

    # -- Tab 5: Pipeline --
    with gr.Tab("Pipeline"):
        gr.Markdown("### Generate → Edit → Animate")
        gr.Markdown("Chain all three endpoints into one creative flow.")
        with gr.Row():
            with gr.Column():
                pipe_gen = gr.Textbox(label="1. Generate prompt", lines=2, value="a cozy cabin in the mountains at sunset")
                pipe_edit = gr.Textbox(label="2. Edit prompt", lines=2, value="add snow falling and northern lights in the sky")
                pipe_animate = gr.Textbox(label="3. Animate prompt", lines=2, value="snow gently falling, lights dancing in the sky")
                pipe_tier = gr.Dropdown(TIERS, value="fast", label="Tier (for image steps)")
                pipe_btn = gr.Button("Run Pipeline", variant="primary")
            with gr.Column():
                pipe_status = gr.Textbox(label="Status", interactive=False)
                pipe_gen_out = gr.Image(label="Generated Image", type="pil")
                pipe_edit_out = gr.Image(label="Edited Image", type="pil")
                pipe_video_out = gr.Video(label="Final Video")
        pipe_btn.click(
            tab_pipeline,
            [pipe_gen, pipe_edit, pipe_animate, pipe_tier],
            [pipe_status, pipe_gen_out, pipe_edit_out, pipe_video_out],
        )

    # -- Tab 6: Variation Factory --
    with gr.Tab("Variation Factory"):
        gr.Markdown("### Game Asset Variation Factory")
        gr.Markdown(
            "Upload a `.glb` / `.gltf` file, pick a style, and generate N texture variants. "
            "Each variant is a full GLB with a re-skinned albedo — drop it into Unity, Unreal, Blender, or any viewer."
        )
        with gr.Row():
            with gr.Column(scale=1):
                vf_file = gr.File(label="GLB / GLTF file", file_types=[".glb", ".gltf"], type="filepath")
                vf_n = gr.Slider(1, MAX_VARIANTS, value=4, step=1, label="Number of variants")
                vf_preset = gr.Dropdown(list(PRESETS.keys()), value="Rusted", label="Style preset")
                vf_intensity = gr.Radio(["Subtle", "Moderate", "Heavy"], value="Subtle", label="Intensity")
                vf_extra = gr.Textbox(label="Additional prompt (optional)", placeholder="e.g., with green patina", lines=1)
                vf_quality = gr.Dropdown(
                    ["draft (radically_fast)", "final (fast)"],
                    value="draft (radically_fast)",
                    label="Quality",
                )
                vf_seed = gr.Number(value=0, label="Base seed (0 = random)", precision=0)
                vf_btn = gr.Button("Generate variants", variant="primary")
                vf_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=2):
                vf_models = []
                vf_errors = []
                for row_start in range(0, MAX_SLOTS, 3):
                    with gr.Row():
                        for slot_i in range(row_start, min(row_start + 3, MAX_SLOTS)):
                            with gr.Column():
                                m = gr.Model3D(label=f"variant_{slot_i}", visible=False, clear_color=[0.1, 0.1, 0.1, 1.0])
                                err = gr.Textbox(label="", visible=False, interactive=False, lines=2)
                                vf_models.append(m)
                                vf_errors.append(err)

        vf_btn.click(
            tab_variation_factory,
            [vf_file, vf_n, vf_preset, vf_intensity, vf_extra, vf_quality, vf_seed],
            [vf_status, *vf_models, *vf_errors],
        )

if __name__ == "__main__":
    if not os.environ.get("NUNCHAKU_API_KEY"):
        print("Warning: NUNCHAKU_API_KEY not set. Set it before using the app.")
    output_dir = Path("output").resolve()
    output_dir.mkdir(exist_ok=True)
    allowed = [str(output_dir), tempfile.gettempdir()]
    logger.info("Gradio allowed_paths: %s", allowed)
    app.queue(default_concurrency_limit=4).launch(
        share=False,
        allowed_paths=allowed,
    )
