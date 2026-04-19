#!/usr/bin/env bash
# Image-to-image (edit) with cURL.
# Input image goes in the `url` field as a data URI — not a file upload.
#
# Usage:
#   echo 'NUNCHAKU_API_KEY=sk-nunchaku-...' > .env
#   bash image_to_image.sh input.jpg "make it a watercolor painting"

set -e

# Load .env if present
[ -f .env ] && export $(grep -v '^#' .env | xargs)

INPUT_PATH="${1:-input.jpg}"
PROMPT="${2:-transform this into a watercolor painting}"

# Encode input image as base64 data URI
EXT="${INPUT_PATH##*.}"
case "$EXT" in
  png|PNG) MIME="image/png" ;;
  *)       MIME="image/jpeg" ;;
esac
IMG_B64=$(base64 -w0 "$INPUT_PATH")

curl -s https://api.nunchaku.dev/v1/images/edits \
  -H "Authorization: Bearer $NUNCHAKU_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"nunchaku-qwen-image-edit\",
    \"prompt\": \"$PROMPT\",
    \"url\": \"data:$MIME;base64,$IMG_B64\",
    \"n\": 1,
    \"size\": \"1024x1024\",
    \"tier\": \"fast\",
    \"num_inference_steps\": 28,
    \"response_format\": \"b64_json\"
  }" \
  | python3 -c "import sys, json, base64; d=json.load(sys.stdin); open('output_i2i.jpg','wb').write(base64.b64decode(d['data'][0]['b64_json'])); print('Saved output_i2i.jpg')"
