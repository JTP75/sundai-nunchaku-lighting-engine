"""Text-to-Video — generate a video from a text prompt.

Endpoint: POST /v1/video/generations   (note: /video/ not /videos/)
Model:    nunchaku-wan2.2-lightning-t2v (4-step distilled, ~27s)

Defaults for Lightning models:
    num_frames=81 (3.4s at 24fps), num_inference_steps=4, guidance_scale=1.0

Usage:
    echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
    python text_to_video.py
"""

import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["NUNCHAKU_API_KEY"]
BASE_URL = "https://api.nunchaku.dev"

print("Generating video (this may take ~30 seconds)...")

response = requests.post(
    f"{BASE_URL}/v1/video/generations",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "nunchaku-wan2.2-lightning-t2v",
        "prompt": "A golden retriever running on a beach at sunset, cinematic, slow motion",
        "n": 1,
        "size": "1280x720",
        "num_frames": 81,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
        "response_format": "b64_json",
    },
    timeout=120,
)
response.raise_for_status()

video_bytes = base64.b64decode(response.json()["data"][0]["b64_json"])
with open("output_t2v.mp4", "wb") as f:
    f.write(video_bytes)
print(f"Saved output_t2v.mp4 ({len(video_bytes) / 1024 / 1024:.1f} MB)")
