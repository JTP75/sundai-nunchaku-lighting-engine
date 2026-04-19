"""Nunchaku Creative Pipeline — interactive Gradio demo.

5 tabs: Text-to-Image, Edit Image, Text-to-Video, Image-to-Video, Pipeline.
The Pipeline tab chains: generate → edit → animate in one flow.

Usage:
    echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
    pip install gradio requests Pillow python-dotenv
    python demo/app.py
"""

import io
import os
import sys
import tempfile

import gradio as gr
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Allow importing nunchaku.py from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from nunchaku import NunchakuClient

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

if __name__ == "__main__":
    if not os.environ.get("NUNCHAKU_API_KEY"):
        print("Warning: NUNCHAKU_API_KEY not set. Set it before using the app.")
    app.launch(share=False)
