"""Text-to-Image — generate an image from a text prompt.

Endpoint: POST /v1/images/generations
Model:    nunchaku-qwen-image (supports tiers: fast, radically_fast)

Usage:
    echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
    python text_to_image.py
"""

import base64
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["NUNCHAKU_API_KEY"]
BASE_URL = "https://api.nunchaku.dev"

response = requests.post(
    f"{BASE_URL}/v1/images/generations",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "nunchaku-qwen-image",
        "prompt": "a red apple on a wooden table, photorealistic, studio lighting",
        "n": 1,
        "size": "1024x1024",
        "tier": "fast",
        "num_inference_steps": 28,
        "response_format": "b64_json",
        "seed": 42,
    },
)
response.raise_for_status()

img_bytes = base64.b64decode(response.json()["data"][0]["b64_json"])
with open("output_t2i.jpg", "wb") as f:
    f.write(img_bytes)
print(f"Saved output_t2i.jpg ({len(img_bytes):,} bytes)")
