"""Using the Nunchaku API with the OpenAI Python SDK.

The image generation endpoints are OpenAI-compatible. Just swap the base_url.
Non-standard parameters (tier, num_frames, etc.) go in `extra_body`.

Install: pip install openai

Usage:
    echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
    python openai_sdk.py
"""

import base64
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://api.nunchaku.dev/v1",
    api_key=os.environ["NUNCHAKU_API_KEY"],
)

# --- Text-to-Image -----------------------------------------------------------

print("Generating image...")
response = client.images.generate(
    model="nunchaku-qwen-image",
    prompt="a red apple on a wooden table, photorealistic, studio lighting",
    n=1,
    size="1024x1024",
    response_format="b64_json",
    extra_body={"tier": "fast", "num_inference_steps": 28, "seed": 42},
)

img_bytes = base64.b64decode(response.data[0].b64_json)
with open("output_openai_t2i.jpg", "wb") as f:
    f.write(img_bytes)
print(f"Saved output_openai_t2i.jpg ({len(img_bytes):,} bytes)")

# --- Notes -------------------------------------------------------------------
#
# OpenAI SDK limitations with the Nunchaku API:
#
# - `client.images.edit(...)` sends multipart form data, but Nunchaku's edit
#   endpoint takes JSON with a `url` data URI. Use `requests` for i2i — see
#   examples/python/image_to_image.py.
#
# - Video endpoints (`/v1/video/*`) aren't in the OpenAI SDK at all — there
#   is no `client.videos.generate()`. Use `requests` for t2v/i2v — see
#   examples/python/text_to_video.py and image_to_video.py.
#
# For text-to-image the OpenAI SDK works cleanly; pass Nunchaku-specific
# parameters (`tier`, `num_inference_steps`, `seed`) via `extra_body`.

print("Done!")
