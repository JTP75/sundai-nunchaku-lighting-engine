"""Image-to-Image — edit an existing image with a text prompt.

Endpoint: POST /v1/images/edits
Model:    nunchaku-qwen-image-edit

NOTE: The input image is passed via the `url` field as a data URI,
      NOT as a separate `image` field.

Usage:
    echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
    python image_to_image.py input.jpg "make it look like a watercolor painting"
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
    print("Usage: python image_to_image.py <input_image> [prompt]")
    sys.exit(1)

input_path = sys.argv[1]
prompt = sys.argv[2] if len(sys.argv) > 2 else "transform this into a watercolor painting"

# Encode input image as data URI
with open(input_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

ext = input_path.rsplit(".", 1)[-1].lower()
mime = "image/png" if ext == "png" else "image/jpeg"

response = requests.post(
    f"{BASE_URL}/v1/images/edits",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "model": "nunchaku-qwen-image-edit",
        "prompt": prompt,
        "url": f"data:{mime};base64,{img_b64}",
        "n": 1,
        "size": "1024x1024",
        "num_inference_steps": 28,
        "tier": "fast",
        "response_format": "b64_json",
    },
)
response.raise_for_status()

edited_bytes = base64.b64decode(response.json()["data"][0]["b64_json"])
with open("output_i2i.jpg", "wb") as f:
    f.write(edited_bytes)
print(f"Saved output_i2i.jpg ({len(edited_bytes):,} bytes)")
