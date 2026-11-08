"""Teacher-forced quality: run the four quality prompts (seed 1234) on a server started with H3Q_DUMP or H3Q_REPLAY, then
print the mean relative L2 of the candidate's velocities against the recording (video and audio), per step and overall.

A run with H3Q_REPLAY gets the recorded latents at every one of the 8 denoising steps, so each step's error is that of
one model evaluation, not an accumulated trajectory difference. Start the server with FV_WARMUP=0: the hook counts
requests, and warm-up requests would shift them.

    python3 tools/quality.py t2v                        # after: H3Q_REPLAY=/q/bf16 FV_WARMUP=0 docker compose up -d t2v
    python3 tools/quality.py omni --ref /media/in/reference.png
"""
import argparse
import json
import re
import subprocess
import time
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
LINE = re.compile(r"H3Q r=(\d+) s=(\d+) rel_v=([\d.e+-]+) rel_a=([\d.e+-]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("service", choices=["t2v", "omni"])
    ap.add_argument("--ref", default=None, help="Omni reference image (path the server can read)")
    ap.add_argument("--url", default="http://127.0.0.1:8101/v1")
    ap.add_argument("--container", default=None, help="default fasth3-SERVICE")
    ap.add_argument("--prompt-file", default=None, help="use this text (behind distinct \"Take N.\" prefixes) for the "
                    "four prompts instead of the short ones, e.g. tools/prompts/long-1k.txt")
    a = ap.parse_args()
    omni = a.service == "omni"
    if omni and not a.ref:
        ap.error("--ref is required for omni")

    def call(path: str, body: dict | None = None) -> dict:
        req = urllib.request.Request(a.url + path, json.dumps(body).encode() if body else None,
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=1800))

    prompts = OMNI_PROMPTS if omni else T2V_PROMPTS
    if a.prompt_file:
        text = open(a.prompt_file).read().strip()
        prompts = [f"Take {i + 1}. {text}" for i in range(4)]
    for prompt in prompts:
        body = {"model": "fasth3-omni" if omni else "fasth3", "prompt": prompt, "seed": 1234}
        if omni:
            body["image_reference"] = [{"image_url": a.ref}]
        job = call("/videos", body)
        while job["status"] not in ("completed", "failed"):
            time.sleep(0.2)
            job = call(f"/videos/{job['id']}")
        print(job["status"], prompt[:60], flush=True)

    logs = subprocess.run(["docker", "logs", a.container or f"fasth3-{a.service}"], capture_output=True, text=True)
    rows = [(int(r), int(s), float(v), float(au)) for r, s, v, au in LINE.findall(logs.stdout + logs.stderr)]
    if not rows:
        print("no H3Q lines: was the server started with H3Q_REPLAY? (H3Q_DUMP only records)")
        return
    steps = sorted({s for _, s, _, _ in rows})
    mean = lambda xs: sum(xs) / len(xs)
    print("video rel-L2 per step", [round(mean([v for _, s, v, _ in rows if s == st]), 4) for st in steps])
    print("audio rel-L2 per step", [round(mean([au for _, s, _, au in rows if s == st]), 4) for st in steps])
    print(f"mean rel-L2: video {mean([v for *_, v, _ in rows]):.5f}, audio {mean([au for *_, au in rows]):.5f} "
          f"({len(rows)} step evaluations)")


if __name__ == "__main__":
    main()
