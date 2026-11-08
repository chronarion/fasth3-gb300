"""Startup warmup for a FastH3 server (runs in the container next to `fastvideo serve`).

torch.compile specializes the DiT on the first prompt's text length; the first request with a
different length recompiles once with dynamic shapes (~17 s). Two warmup requests with different
prompt lengths take both compiles before real traffic. Warmup outputs are deleted.
   python3 -I warmup.py MODEL [REFERENCE_IMAGE]"""
import json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8101"
MODEL = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else None
PROMPTS = ["A lighthouse on a rocky coast at dusk.",
           "A paper boat drifts down a rain-filled street gutter past glowing shop windows at night, "
           "splashing softly as cars pass and neon signs flicker in the puddles."]
if REF:
    PROMPTS = ["The subject of <Picture 1> stands still in soft morning light.",
               "The subject of <Picture 1> walks slowly across a wide sunlit meadow while the wind moves "
               "the tall grass and birds circle overhead, the camera tracking alongside."]


def call(path, body=None):
    req = urllib.request.Request(URL + path, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


for _ in range(3600):
    try:
        urllib.request.urlopen(URL + "/health", timeout=3); break
    except Exception:
        time.sleep(2)
for prompt in PROMPTS:
    t = time.time()
    body = {"model": MODEL, "prompt": prompt, "seed": 1}
    if REF:
        body["image_reference"] = [{"image_url": REF}]
    try:
        job = call("/v1/videos", body)
        while job["status"] not in ("completed", "failed"):
            time.sleep(1); job = call(f"/v1/videos/{job['id']}")
    except Exception as e:  # warmup must never take the server down
        print(f"warmup: request failed: {e}", flush=True); break
    path = job.get("file_path")
    if path and os.path.isfile(path):
        os.remove(path)
    print(f"warmup: {job['status']} in {time.time() - t:.1f} s ({len(prompt)} chars)", flush=True)
