#!/usr/bin/env bash
# Text-to-image with cURL.
#
# Usage:
#   echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
#   bash text_to_image.sh

set -e

# Load .env if present
[ -f .env ] && export $(grep -v '^#' .env | xargs)

curl -s https://api.nunchaku.dev/v1/images/generations \
  -H "Authorization: Bearer $NUNCHAKU_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nunchaku-qwen-image",
    "prompt": "a red apple on a wooden table, photorealistic, studio lighting",
    "n": 1,
    "size": "1024x1024",
    "tier": "fast",
    "num_inference_steps": 28,
    "response_format": "b64_json",
    "seed": 42
  }' \
  | python3 -c "import sys, json, base64; d=json.load(sys.stdin); open('output_t2i.jpg','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print('Saved output_t2i.jpg')"
