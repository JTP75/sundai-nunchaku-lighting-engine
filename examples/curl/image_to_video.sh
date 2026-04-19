#!/usr/bin/env bash
# Image-to-video with cURL.
# The input image is passed inside `messages` using OpenAI-style multimodal
# content blocks — NOT via a simple `image` field. Endpoint is /v1/video/animations.
#
# Usage:
#   echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
#   bash image_to_video.sh input.jpg "the scene comes to life"

set -e

# Load .env if present
[ -f .env ] && export $(grep -v '^#' .env | xargs)

INPUT_PATH="${1:-input.jpg}"
PROMPT="${2:-the scene comes to life with gentle motion}"

EXT="${INPUT_PATH##*.}"
case "$EXT" in
  png|PNG) MIME="image/png" ;;
  *)       MIME="image/jpeg" ;;
esac
IMG_B64=$(base64 -w0 "$INPUT_PATH")

echo "Generating video (this may take ~30 seconds)..."

curl -s https://api.nunchaku.dev/v1/video/animations \
  -H "Authorization: Bearer $NUNCHAKU_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"nunchaku-wan2.2-lightning-i2v\",
    \"prompt\": \"$PROMPT\",
    \"n\": 1,
    \"size\": \"1280x720\",
    \"num_frames\": 81,
    \"num_inference_steps\": 4,
    \"guidance_scale\": 1.0,
    \"response_format\": \"b64_json\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": [
          {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:$MIME;base64,$IMG_B64\"}},
          {\"type\": \"text\", \"text\": \"$PROMPT\"}
        ]
      }
    ]
  }" \
  | python3 -c "import sys, json, base64; d=json.load(sys.stdin); open('output_i2v.mp4','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print('Saved output_i2v.mp4')"
