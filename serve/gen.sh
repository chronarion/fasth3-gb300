#!/usr/bin/env bash
# Submit one text->video job to the FastH3 server, poll, download, and print timings.
#   ./gen.sh "prompt" [out.mp4]
#   REF_IMAGE=/media/in/reference.png MODEL=fasth3-omni ./gen.sh "<Picture 1> ..."   (Omni: refs required)
set -euo pipefail
BASE=${BASE:-http://127.0.0.1:8101/v1}
PROMPT=${1:-"A fox runs through fresh snow at dawn, its breath steaming, soft wind and crunching snow."}
OUT=${2:-./media/out/fasth3/test-$(date +%s).mp4}
t0=$(date +%s.%N)
job=$(curl -fsS -m 60 "$BASE/videos" -H 'Content-Type: application/json' \
  -d "$(python3 -I -c '
import json, os, sys
body = {"model": os.environ.get("MODEL", "fasth3"), "prompt": sys.argv[1]}
if os.environ.get("REF_IMAGE"): body["image_reference"] = [{"image_url": os.environ["REF_IMAGE"]}]
if os.environ.get("REF_AUDIO"): body["audio_reference"] = [{"audio_url": os.environ["REF_AUDIO"]}]
print(json.dumps(body))' "$PROMPT")")
id=$(python3 -I -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "$job")
while :; do
  st=$(python3 -I -c 'import json,sys; print(json.loads(sys.argv[1])["status"])' "$job")
  case $st in completed) break ;; failed) echo "FAILED: $job" >&2; exit 1 ;; esac
  sleep 1; job=$(curl -fsS -m 30 "$BASE/videos/$id")
done
t1=$(date +%s.%N)
curl -fsS -m 300 "$BASE/videos/$id/content" -o "$OUT"
printf 'job %s: %.1f s end-to-end -> %s (%s)\n' "$id" "$(echo "$t1 - $t0" | bc)" "$OUT" "$(du -h "$OUT" | cut -f1)"
echo "$job" | python3 -I -c 'import json,sys; j=json.load(sys.stdin); print({k: j[k] for k in j if k not in ("prompt",)})' | cut -c1-400
