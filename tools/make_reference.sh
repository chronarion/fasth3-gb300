#!/usr/bin/env bash
# Make a 1344x768 reference image for Omni (warm-up and benchmarks) from the t2v service: the first frame of one clip.
# Run with the t2v service up; writes serve/media/in/reference.png (or the given path under serve/media/in).
#   tools/make_reference.sh [serve/media/in/reference.png]
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-serve/media/in/reference.png}
BASE=${BASE:-http://127.0.0.1:8101/v1}
mkdir -p "$(dirname "$OUT")"
body='{"model": "fasth3", "prompt": "A red fox stands in fresh snow at the edge of a pine forest at dawn, looking toward the camera, soft wind.", "seed": 1000}'
job=$(curl -fsS -m 60 "$BASE/videos" -H 'Content-Type: application/json' -d "$body")
id=$(python3 -I -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "$job")
while :; do
  st=$(python3 -I -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "$job")
  case $st in completed) break ;; failed) echo "generation failed: $job" >&2; exit 1 ;; esac
  sleep 1; job=$(curl -fsS -m 30 "$BASE/videos/$id")
done
tmp=$(mktemp --suffix .mp4)
curl -fsS -m 300 "$BASE/videos/$id/content" -o "$tmp"
# ffmpeg from the image, so the host needs none
docker run --rm --user "$(id -u):$(id -g)" -v "$tmp:/in.mp4:ro" -v "$PWD/$(dirname "$OUT"):/out" --entrypoint ffmpeg "${FASTH3_IMAGE:-fasth3-gb300}" \
  -loglevel error -y -i /in.mp4 -frames:v 1 "/out/$(basename "$OUT")"
rm -f "$tmp"
echo "wrote $OUT"
