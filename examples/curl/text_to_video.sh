#!/usr/bin/env bash
# Text-to-video with cURL.
# Note: endpoint is /v1/video/ (no 's'), not /v1/videos/.
#
# Usage:
#   echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
#   bash text_to_video.sh

set -e

# Load .env if present
[ -f .env ] && export $(grep -v '^#' .env | xargs)

echo "Generating video (this may take ~30 seconds)..."

curl -s https://api.nunchaku.dev/v1/video/generations \
  -H "Authorization: Bearer $NUNCHAKU_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nunchaku-wan2.2-lightning-t2v",
    "prompt": "A golden retriever running on a beach at sunset, cinematic, slow motion",
    "n": 1,
    "size": "1280x720",
    "num_frames": 81,
    "num_inference_steps": 4,
    "guidance_scale": 1.0,
    "response_format": "b64_json"
  }' \
  | python3 -c "import sys, json, base64; d=json.load(sys.stdin); open('output_t2v.mp4','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print('Saved output_t2v.mp4')"
