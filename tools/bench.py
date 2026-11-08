"""Benchmark a running FastH3 server: one warm-up clip, then N timed clips, all with fixed seeds.

Reports, per clip and as medians over the timed clips:
  done_s      submit -> job "completed" (the MP4 is written on the server), polled every 20 ms
  fetched_s   submit -> last byte of the MP4 received by the client (the scope of FastVideo's UniServe benchmark)
  pipeline_s  sum of the pipeline stage times the server logs: text encoding, latent setup, denoising, video and audio
              decode; no MP4 encode (the scope of NVIDIA's Sol-H3 table)
plus the median of each stage.

    python3 bench.py LABEL [--service t2v|omni] [--n 6] [--width 1344 --height 768 --frames 124] [--ref IMAGE]
                     [--prompt-file FILE] [--url http://127.0.0.1:8101/v1] [--container fasth3-t2v] [--out DIR]

Omni needs a reference image path the server can read (under one of its media_roots). With --prompt-file every clip
gets the file's text behind a distinct "Take N." prefix, so no clip can reuse another clip's text encoding.
"""
import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request

T2V_PROMPTS = [
    "A fox trots through fresh snow at dawn, breath steaming, wind and crunching snow.",
    "A drummer plays a fast rock beat in a small club, cymbals crashing, crowd cheering.",
    "Waves crash against black volcanic rocks at sunset, seabirds calling overhead.",
    "A chef flips vegetables in a flaming wok in a busy restaurant kitchen, sizzling loudly.",
]
OMNI_PROMPTS = [
    "The fox from <Picture 1> trots through the snowy forest at sunset and barks once.",
    "The fox from <Picture 1> curls up beneath a pine tree as snow begins to fall.",
    "The fox from <Picture 1> leaps into a snowdrift hunting a mouse, snow spraying.",
    "The fox from <Picture 1> drinks from a frozen stream, ice cracking softly.",
]
STAGE_RX = re.compile(r"(\d\d:\d\d:\d\d\.\d+) \[base\.py:\d+\] \[(\w+)\|\w+\] Execution completed in ([\d.]+) ms")
STAGES_PER_CLIP = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--service", default="t2v", choices=["t2v", "omni"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--ref", default=None, help="reference image for Omni (path the server can read)")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--reroll", action="store_true", help="every clip repeats the first prompt (re-rolls); by default "
                    "long prompts get a distinct 'Take N.' prefix per clip")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--lora", default=None, help="LoRA adapter name (a file in serve/loras without .safetensors)")
    ap.add_argument("--lora-scale", type=float, default=None, help="LoRA strength (the adapter's default when unset)")
    ap.add_argument("--url", default="http://127.0.0.1:8101/v1")
    ap.add_argument("--container", default=None, help="container whose logs carry the stage times "
                    "(default fasth3-SERVICE; '' skips them)")
    ap.add_argument("--out", default=None, help="directory for a JSON record per run")
    a = ap.parse_args()
    omni = a.service == "omni"
    if omni and not a.ref:
        ap.error("--ref is required for the omni service")
    if a.prompt_file:
        text = open(a.prompt_file).read().strip()
        prompts = [text if a.reroll else f"Take {i + 1}. {text}" for i in range(a.n + 1)]
    else:
        base = OMNI_PROMPTS if omni else T2V_PROMPTS
        prompts = [base[0]] * (a.n + 1) if a.reroll else [base[0]] + [base[1 + i % 3] for i in range(a.n)]
    container = f"fasth3-{a.service}" if a.container is None else a.container

    def call(path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(a.url + path, json.dumps(body).encode() if body else None,
                                     {"Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=3600))
        except urllib.error.HTTPError as error:
            raise SystemExit(f"HTTP {error.code} from {path}: {error.read().decode(errors='replace')[:2000]}")

    def generate(prompt: str) -> tuple[float, float, str | None]:
        body = {"model": "fasth3-omni" if omni else "fasth3", "prompt": prompt, "seed": a.seed}
        for key, value in (("width", a.width), ("height", a.height), ("num_frames", a.frames)):
            if value:
                body[key] = value
        if omni:
            body["image_reference"] = [{"image_url": a.ref}]
        if a.lora:
            body["lora"] = {"name": a.lora} if a.lora_scale is None else {"name": a.lora, "scale": a.lora_scale}
        t0 = time.time()
        job = call("/videos", body)
        while job["status"] not in ("completed", "failed"):
            time.sleep(0.02)
            job = call(f"/videos/{job['id']}")
        done = time.time() - t0
        if job["status"] != "completed":
            raise SystemExit(f"generation failed: {job}")
        with urllib.request.urlopen(a.url + f"/videos/{job['id']}/content", timeout=600) as r:
            while r.read(1 << 20):
                pass
        return done, time.time() - t0, job.get("file_path")

    since = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    first, _, _ = generate(prompts[0])
    done, fetched, paths = [], [], []
    for prompt in prompts[1:]:
        d, f, path = generate(prompt)
        done.append(d)
        fetched.append(f)
        paths.append(path)

    stages, pipeline = {}, []
    if container:
        logs = subprocess.run(["docker", "logs", "--since", since, container], capture_output=True, text=True)
        events = [(m[1], m[2], float(m[3])) for m in STAGE_RX.finditer(logs.stderr + logs.stdout)]
        timed = events[-(a.n * STAGES_PER_CLIP):]
        for _, name, ms in timed:
            stages.setdefault(name, []).append(ms)
        pipeline = [sum(ms for _, _, ms in timed[i:i + STAGES_PER_CLIP]) / 1000
                    for i in range(0, len(timed), STAGES_PER_CLIP)]
    med = lambda xs: round(statistics.median(xs), 3) if xs else None
    record = {
        "label": a.label, "service": a.service,
        "size": f"{a.width}x{a.height}" if a.width and a.height else "server default",
        "frames": a.frames or "server default",
        "prompt": (os.path.basename(a.prompt_file) if a.prompt_file else "short") + (", re-roll" if a.reroll else ""),
        "lora": None if not a.lora else f"{a.lora} @ {a.lora_scale if a.lora_scale is not None else 'default'}",
        "first_s": round(first, 2),
        "done_s": med(done), "fetched_s": med(fetched), "pipeline_s": med(pipeline),
        "runs_done": [round(x, 2) for x in done],
        "stages_ms": {k: round(statistics.median(v), 1) for k, v in stages.items()},
        "samples": paths,
    }
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, f"{a.service}-{a.label}.json"), "w") as f:
            json.dump(record, f, indent=2)
    print(json.dumps({k: v for k, v in record.items() if k != "samples"}), flush=True)


if __name__ == "__main__":
    main()
