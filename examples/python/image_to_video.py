"""Image-to-Video — animate a static image into a video.

Endpoint: POST /v1/video/animations   (note: /video/ not /videos/)
Model:    nunchaku-wan2.2-lightning-i2v

IMPORTANT: The input image is passed via the `messages` field using
multimodal content blocks (similar to chat completions), NOT via a
simple `image` field. This is the trickiest endpoint format.

Usage:
    echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
    python image_to_video.py input.jpg "the scene comes to life with gentle motion"
"""

import base64
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["NUNCHAKU_API_KEY"]
BASE_URL = "https://api.nunchaku.dev"

# Parse args
if len(sys.argv) < 2:
    print("Usage: python image_to_video.py <input_image> [prompt]")
    sys.exit(1)

input_path = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "the scene comes to life with gentle motion"

# Encode input image as data URI
with open(input_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

ext = input_path.rsplit(".", 1)[-1].lower()
mime = "image/png" if ext == "png" else "image/jpeg"
data_uri = f"data:{mime};base64,{img_b64}"

print("Generating video (this may take ~30 seconds)...")

response = requests.post(
    f"{BASE_URL}/v1/video/animations",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "nunchaku-wan2.2-lightning-i2v",
        "prompt": prompt,
        "n": 1,
        "size": "1280x720",
        "num_frames": 81,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "response_format": "b64_json",
        # The image MUST be passed inside `messages` with multimodal content blocks
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    },
    timeout=120,
)
response.raise_for_status()

video_bytes = base64.b64decode(response.json()["data"][0]["b64_json"])
with open("output_i2v.mp4", "wb") as f:
    f.write(video_bytes)
print(f"Saved output_i2v.mp4 ({len(video_bytes) / 1024 / 1024:.1f} MB)")
