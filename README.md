# FastH3 on one GB300

Make videos with sound from a text prompt, quickly, on a single NVIDIA GB300 GPU (the GPU in an NVIDIA DGX Station).

[FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2) is FastVideo's fast version of the MiniMax-H3
video model: it turns a prompt into a 1344×768 video with stereo audio in 8 steps. This repository makes it run faster.
It is a set of changes to [FastVideo](https://github.com/hao-ai-lab/FastVideo), the software FastH3 normally runs on,
plus a ready-to-run server. The model itself is unchanged.

On one GB300, a 5-second clip with sound takes **11.47 s** from request to finished video file, as accurate as
FastVideo's own standard 8-bit setting. FastVideo alone takes 17.32 s on the same GPU with all of its speed
options, or 34.59 s as installed. Presets that trade a little accuracy for speed go down to **7.35 s**, and
the optional token-reduction add-on takes that to 5.31 s, at the cost of a different video for the same seed
(see [Presets](#presets)).
FastH3 Omni, which also takes reference images, video or audio, takes 18.84 s per 5-second clip, down to
15.40 s.

At 832×480 (480p) a 5-second clip takes **4.01 s** (**2.71 s** with `max`), and at 640×640, the size
ComfyUI's FastH3 template uses by default, 3.79 s (2.49 s). For comparison, a published run of
ComfyUI's official FastH3 template on an RTX 5090 home GPU took 18.7 s for the same 5-second 832×480 clip and
50.8-52.8 s at 1344×768. The GPUs differ, so this compares setups, not chips; see
[Differences from ComfyUI's FastH3 workflow](#differences-from-comfyuis-fasth3-workflow).

- [Results](#results): every preset at 5, 10 and 15 seconds, FastVideo on the same GPU, other published systems
- [How the server is set up](#how-the-server-is-set-up)
- [Differences from ComfyUI's FastH3 workflow](#differences-from-comfyuis-fasth3-workflow)
- [What makes it fast](#what-makes-it-fast)
- [Quick start](#quick-start), [Measuring](#measuring), [Notes](#notes), [Glossary](#glossary), [Credits](#credits)

## Results

All times are from one GB300 making one video at a time, after the server has warmed up. Each number is the median of
several runs. Unless a row says otherwise: 1344×768 video at 24 frames per second, with sound, and a short prompt
(12-16 words). "Total" is the time from sending the request to having the finished MP4 file. Details:
[How we measured](#how-we-measured).

### Presets

A preset sets how many bits the model calculates with: fewer bits run faster but drift further from the full-precision
result (see the [glossary](#glossary)). You pick a preset when you start the server; clip size and length are picked
per request.

| Preset | Model math | Attention math | 5 s clip | 10 s clip | 15 s clip | Error (video / audio) |
|---|---|---|---:|---:|---:|---|
| Default | 8-bit | 16-bit | 11.47 s | 28.66 s | 54.71 s | 0.192 / 0.070 |
| `fast` | 8-bit | partly 8-bit | 10.80 s | 25.44 s | 47.14 s | 0.197 / 0.075 |
| `qad-nvfp4` | 4-bit (QAD model) | partly 8-bit | 8.37 s | 20.78 s | 39.98 s | 0.271 / 0.158 |
| `max` | 4-bit (QAD model) | 8-bit | 7.35 s | 17.86 s | 33.75 s | 0.282 / 0.167 |
| `preview` | 4-bit (QAD model) | 8-bit, lighter decoder | 6.68 s | 16.46 s | 31.47 s | 0.282 / 0.167 (same as `max`; softer decoder) |
| Default + token reduction | 8-bit | 16-bit; middle blocks at half width | 7.90 s | 18.80 s | 34.40 s | 0.346 / 0.115 |
| `max` + token reduction | 4-bit (QAD model) | 8-bit; middle blocks at half width | 5.31 s | 12.17 s | 22.00 s | 0.320 / 0.177 |
| Omni, 1 reference image | 4-bit (Omni QAD model) | partly 8-bit; prompt and reference in 16-bit | 18.84 s | 35.45 s | 55.49 s | 0.219 / 0.111 |
| `omni-fast` | 4-bit (Omni QAD model) | partly 8-bit; prompt and reference in 8-bit | 15.77 s | 29.45 s | 46.33 s | 0.226 / 0.119 |
| `omni-max` | 4-bit (Omni QAD model) | 8-bit | 15.40 s | 28.29 s | 44.16 s | 0.230 / 0.117 |
| Omni + token reduction | 4-bit (Omni QAD model) | partly 8-bit; middle blocks at half width | 14.62 s | 25.82 s | 38.97 s | 0.271 / 0.124 |
| `omni-max` + token reduction | 4-bit (Omni QAD model) | 8-bit; middle blocks at half width | 12.08 s | 20.89 s | 31.34 s | 0.277 / 0.128 |

**Error** is how far a preset's predictions drift from the full-precision (16-bit) model, given exactly the same input
at each of the 8 steps, averaged over four prompts. Lower is better; 0 would be identical. FastVideo's own standard
8-bit setting scores 0.190 / 0.071, so the default preset is just as accurate. `qad-nvfp4`, `max` and `preview` use
FastVideo's separately trained 4-bit version of the model, so part of their error is that model's own difference.
`preview` also uses a lighter video decoder, which gives softer detail. The Omni rows use one reference image,
processed again for every clip.

**Token reduction** (the "+ token reduction" rows) is an add-on for any preset: add
`--env-file profiles/token-reduction.env` after the preset's own file. In the middle of the model (blocks 4 to 29
of 50, and up to block 39 during the first 4 of the 8 steps, where the overall layout forms) every two side-by-side
pieces of the video are averaged into one, so those blocks work on a video of half the width. Afterwards each piece
gets back its own value plus the change its pair went through. The model was not trained this way, so the same seed
gives a different video: in our checks the picture stays as clean, while the framing or details of the shot can
change. The idea comes from antirez's h3.c; this is FastVideo's version of it, run on this repository's fast path.

Other sizes and prompt lengths. Long prompts take longer, because every part of the video looks at the whole prompt.
Error with a 1,000-token prompt: 0.243 / 0.082; with a 10,000-token prompt: 0.256 / 0.062.

| Preset | Request | Total |
|---|---|---:|
| Default | 832×480, 5 s | 4.01 s |
| Default | 768×1344 (portrait), 5 s | 11.81 s |
| Default | 640×640 (ComfyUI template's size), 5 s | 3.79 s |
| Default | 5 s, 1,000-token prompt | 12.41 s |
| Default | 10 s, 1,000-token prompt | 29.90 s |
| Default | 15 s, 1,000-token prompt | 55.48 s |
| Default | 5 s, 10,000-token prompt | 19.67 s |
| Default | 10 s, 10,000-token prompt | 41.81 s |
| Default | 15 s, 10,000-token prompt | 72.47 s |
| Default | 5 s, same prompt again (re-roll) | 11.46 s |
| Default | 5 s, same 10,000-token prompt again (re-roll) | 19.23 s |
| `fast` | 832×480, 5 s | 3.86 s |
| `qad-nvfp4` | 832×480, 5 s | 2.92 s |
| `max` | 832×480, 5 s | 2.71 s |
| `max` | 640×640, 5 s | 2.49 s |
| `max` | 5 s, 1,000-token prompt | 8.11 s |
| `max` | 10 s, 1,000-token prompt | 18.90 s |
| `max` | 15 s, 1,000-token prompt | 35.01 s |
| `fast` | 5 s, 10,000-token prompt | 18.91 s |
| `fast` | 10 s, 10,000-token prompt | 38.94 s |
| `fast` | 15 s, 10,000-token prompt | 65.51 s |
| `qad-nvfp4` | 5 s, 10,000-token prompt | 16.05 s |
| `max` | 5 s, 10,000-token prompt | 13.18 s |
| `max` | 10 s, 10,000-token prompt | 27.36 s |
| `max` | 15 s, 10,000-token prompt | 46.67 s |
| `preview` | 832×480, 5 s | 2.39 s |
| `preview` | 5 s, 10,000-token prompt | 12.43 s |
| Default + token reduction | 832×480, 5 s | 2.93 s |
| Default + token reduction | 5 s, 10,000-token prompt | 14.49 s |
| `max` + token reduction | 832×480, 5 s | 2.10 s |
| `max` + token reduction | 5 s, 10,000-token prompt | 9.93 s |
| Omni | 832×480, 5 s | 10.00 s |
| Omni | 5 s, same reference image again (processed once) | 18.31 s |
| `omni-max` | 832×480, 5 s | 8.36 s |

Per-step timings of every run (prompt reading, video generation, decoding): `results/`.

### FastVideo on the same GPU

The same model and requests on the same GB300, run by FastVideo alone. Total time:

| Software | Settings | 5 s clip | 10 s clip | 15 s clip | 5 s clip, 832×480 |
|---|---|---:|---:|---:|---:|
| FastVideo `a01c50a`, as installed | 16-bit model, FastVideo's standard decoder, nothing compiled; its Docker image with only the library fix from [Notes](#notes) | 34.59 s | 78.81 s | 137.3 s | 11.87 s |
| FastVideo `a01c50a`, all of its speed options | 8-bit model math, its combined GPU steps (`FUSIONS=mk`), its fastest sparse attention code, its compiled 4-bit quality decoder with FlashAttention-4, whole frames decoded in one batch | 17.32 s | 40.26 s | 73.75 s | 6.14 s |
| This repository, default | 8-bit model math, 16-bit attention | 11.47 s | 28.66 s | 54.71 s | 4.01 s |
| This repository, `max` | 4-bit model (QAD), 8-bit attention | 7.35 s | 17.86 s | 33.75 s | 2.71 s |

### Published numbers from other systems

Each published result is shown next to this repository's time for the same clip, timed the same way. The sources time
different things, so each row says what is included. Published numbers are copied from the linked pages, not
re-measured here; GPU counts and model versions differ (FastH3 V2 unless noted).

| System | GPUs | Number format | Clip | What is timed | Their time | This repository, 1 GB300 (default / `max`) |
|---|---|---|---|---|---:|---:|
| UniServe (FastVideo) | 4× GB200 | 16-bit | 5 s, 1,000-token prompt | request to downloaded MP4 | 5.62 s | 12.42 s / 8.12 s |
| UniServe (FastVideo) | 4× GB200 | 4-bit in part of the model and the decoder | 5 s, 1,000-token prompt | request to downloaded MP4 | 4.49 s | 12.42 s / 8.12 s |
| FastVideo, tuned by the UniServe team | 4× GB200 | 16-bit | 5 s, 1,000-token prompt | request to downloaded MP4 | 7.47 s | 12.42 s / 8.12 s |
| SGLang, tuned by the UniServe team | 4× GB200 | 16-bit | 5 s, 1,000-token prompt | request to downloaded MP4 | 7.07 s | 12.42 s / 8.12 s |
| vLLM-Omni, tuned by the UniServe team | 4× GB200 | 16-bit | 5 s, 1,000-token prompt | request to downloaded MP4 | 7.69 s | 12.42 s / 8.12 s |
| UniServe (FastVideo) | 8× GB200 (2 machines) | 16-bit | 5 s, 1,000-token prompt | request to downloaded MP4 | 3.13 s | 12.42 s / 8.12 s |
| UniServe (FastVideo) | 4× GB200 | 16-bit | 5 s, 10,000-token prompt | request to downloaded MP4 | 8.80 s | 19.69 s / 13.19 s |
| UniServe (FastVideo) | 4× GB200 | 16-bit | 10 s, 1,000-token prompt | request to downloaded MP4 | 11.73 s | 29.93 s / 18.92 s |
| UniServe (FastVideo) | 4× GB200 | 16-bit | 15 s, 1,000-token prompt | request to downloaded MP4 | 20.27 s | 55.51 s / 35.05 s |
| UniServe (FastVideo) | 4× GB200 | 16-bit | 10 s, 10,000-token prompt | request to downloaded MP4 | 16.75 s | 41.83 s / 27.38 s |
| UniServe (FastVideo) | 4× GB200 | 16-bit | 15 s, 10,000-token prompt | request to downloaded MP4 | 27.31 s | 72.51 s / 46.70 s |
| UniServe (FastVideo) | 8× RTX PRO 6000 Server | 16-bit | 5 s, 1,000-token prompt | request to downloaded MP4 | 19.12 s | 12.42 s / 8.12 s |
| Sol-H3 (NVIDIA), 4 steps, full attention | 1× B300 | not stated | 5 s (124 frames) | prompt reading to decoded video, no MP4 file | 11.48 s | 11.35 s / 7.23 s |
| Sol-H3 (NVIDIA), 4 steps, full attention | 1× B300 | not stated | 10 s (243 frames) | prompt reading to decoded video, no MP4 file | 28.18 s | 28.41 s / 17.59 s |
| Sol-H3 (NVIDIA), 4 steps, full attention | 1× B300 | not stated | 15 s (362 frames) | prompt reading to decoded video, no MP4 file | 50.71 s | 54.38 s / 33.39 s |
| Sol-H3 (NVIDIA), 4 steps, sparse attention | 8× B300 | partly 8-bit | 5 s (124 frames) | prompt reading to decoded video, no MP4 file | 1.65 s | 11.35 s / 7.23 s |
| ComfyUI 0.37, official template | 1× RTX 5090 | 4-bit | 5 s, text to video | submit to saved MP4 | 50.8-52.8 s | 11.47 s / 7.35 s |
| ComfyUI 0.37, official template | 1× RTX 5090 | 4-bit | 832×480, 5 s, text to video | submit to saved MP4 | 18.7 s | 4.01 s / 2.71 s |
| FastVideo (its own published figures) | 1× RTX 5090 | not stated | 5 s | prompt reading to MP4 | 38.6 s | 11.47 s / 7.35 s |
| FastVideo (its own published figures) | 1× RTX 5090 | not stated | 832×480, 5 s | prompt reading to MP4 | 14.8 s | 4.01 s / 2.71 s |

Sources: [UniServe](https://haoailab.com/blogs/uniserve-fasth3/) (FastVideo's own serving system; its comparison
systems were tuned by the same team), [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) (NVIDIA; an older
4-step version of FastH3, so half the steps), [ComfyUI and FastVideo on an RTX 5090](https://ai-muninn.com/en/blog/fasth3-rtx-5090-comfyui)
(ComfyUI 0.37 with its official FastH3 template; the FastVideo numbers on that page are FastVideo's own).

### Throughput per GPU

UniServe reports how much GPU time one second of finished video costs, over a mix of 5, 10 and 15-second clips with
1,000- and 10,000-token prompts. Lower is better. One GPU making one video at a time is always busy, so its cost is
its time per video divided by the video's length. Over the same mix:

| System | GPUs | Videos per day | GPU-seconds per second of video |
|---|---|---:|---:|
| UniServe (FastVideo), 16-bit | 4× GB200 | ~6,100 | 5.8 |
| UniServe (FastVideo), 16-bit | 8× GB200 | ~11,000 | 6.4 |
| UniServe (FastVideo), 16-bit | 8× RTX PRO 6000 Server | ~3,400 | 21.2 |
| This repository, default (8-bit) | 1× GB300 | 2,237 | 3.8 |
| This repository, `max` (4-bit) | 1× GB300 | 3,474 | 2.5 |

### How we measured

- **Hardware:** an NVIDIA DGX Station with one GB300 GPU (252 GB of memory) and a 72-core Grace CPU. The GPU is
  limited to 1,250 W here (the station's setting when its second GPU slot is in use), and the heaviest parts of each
  video run at that limit.
- **Software:** NVIDIA driver 615, CUDA 13.0, PyTorch 2.12.0, FastVideo `a01c50a`, FlashAttention-4 4.0.0b36,
  FlashInfer 0.7.4rc1, cuDNN 9.20 (the Docker image this repository builds).
- **Model versions:** FastVideo-FastH3-8-Step-V2 `3da2ddf`, FastH3-QAD-v2 `a0dc05a`, FastH3-Omni-8-Step-V1 `53e0d26`,
  FastH3-Omni-NVFP4-QAD `f01e6a2`, FastH3-NVFP4-VAE-Decoder `bd0c248`.
- **Requests:** the four short prompts are in `tools/bench.py`, the two long ones (956 and 9,771 tokens) in
  `tools/prompts/`. Each long-prompt clip gets a different "Take N." prefix so no clip can reuse another's work. Omni
  requests carry one 1344×768 reference image, processed again for every clip unless the row says "same reference
  again".
- **Runs:** each number is the median of 6 timed clips (3 for long prompts, 2 for 10 and 15-second clips), after one
  untimed clip, with seed 1234. The benchmark repeats its four short prompts, so the prompt cache is off for every
  row except the re-roll rows.
- **Which release:** each row comes from the release that last changed it. Later releases only made the older rows
  a little faster (the streamed video encode of 1.4 takes about 0.07 s off a 5-second clip).
- **What is timed:** *total* runs from the request until the server has written the MP4 file. *Downloaded* also
  includes downloading the file. *Without making the MP4* covers reading the prompt, generating the video and decoding
  it, but not writing the video file.

## How the server is set up

`serve/docker-compose.yml` runs FastVideo's server (`fastvideo serve`) on one GPU, with `serve/fasth3-8step.yaml` for
text to video or `serve/fasth3-omni.yaml` for Omni.

**Set by the model** (FastVideo reads these from the model's files and will not run it any other way):

| Setting | Text to video (FastH3 8-Step V2) | Omni (FastH3 Omni 8-Step V1) |
|---|---|---|
| Model passes per clip | 8 | 8 |
| Step schedule | the 8 steps the model was trained on | the model's own 8-step schedule |
| Prompt guidance | none: one pass per step, as trained | none |
| Sparse attention | each small chunk of the video looks only at the 20% of other video chunks most relevant to it, plus the whole prompt and audio (as trained) | the same with 10% of the video chunks |

<details>
<summary>Exact model settings</summary>

- **Steps:** text to video makes its 8 passes at the timesteps the model was trained on (t = 999, 874, 749, 624, 500,
  375, 250, 125, then 0; FastVideo's `num_inference_steps: 9` counts the 9 points), with an Euler step between them
  for video and audio together. Omni runs a 32-interval grid as 8 passes of 4 intervals each.
- **Shift (video / audio):** 10 / 3 for text to video, 12 / 3 for Omni.
- **Guidance:** scale 1.0: one pass per step, empty negative prompt.
- **Sparse attention (VSA):** text to video splits the sequence into 64-token tiles (4×4×4 cubes of the compressed
  video). Each video tile attends its top 20% of video tiles (sparsity 0.8) plus every text and audio tile, and a
  coarse branch over tile averages is added through the model's learned `to_gate_compress` gate. Omni uses 128-token
  tiles and sparsity 0.9, and keeps reference tiles at rate 0.1.
- **Output:** H.264 (libx264, preset ultrafast, yuv420p) and AAC (32 kHz stereo).

</details>

**Per request:** size (`width`, `height`), length (`num_frames`) and `seed`. The default is 1344×768 (the model's
native size) with 124 frames at 24 frames per second, which is 5.17 s. FastVideo accepts 124 to 362 frames in steps of
17 (124, 141, 158, ...; 243 frames is 10.1 s, 362 is 15.1 s). The output is an MP4 file with H.264 video and stereo
AAC audio.

**This repository's settings** (presets change the first three rows):

| Setting | Default | Notes |
|---|---|---|
| Model math | 8-bit (FP8), converted when the model loads | `qad-nvfp4`, `max`, `preview`: FastVideo's 4-bit model (NVFP4) |
| Attention math | 16-bit | `fast`, `qad-nvfp4`: partly 8-bit; `max`, `preview`: 8-bit (switch: `FASTVIDEO_H3_FP8_ATTN`) |
| Attention to the prompt and references | done separately with FlashAttention-4 in 16-bit, then combined exactly with the rest | `fast`, `qad-nvfp4`: only for long prompts (from 2,048 rows of prompt and audio); `max`, `preview`: the same, in 8-bit; `omni-fast`, `omni-max`: in 8-bit |
| Choosing which video chunks to attend to | slightly lower-precision math (TF32): 0.1 s faster, same accuracy | `FASTVIDEO_H3_TF32_SCORES=0` turns it off |
| Prompt reader (text encoder) | the model's Qwen3-VL 32B, 16-bit, with cuDNN's attention; a prompt used again is not read twice (`FASTVIDEO_H3_REF2VA_MEMO_ENTRIES=16`) | `FASTVIDEO_H3_ENCODER_SDPA=` (empty) uses PyTorch's FlashAttention-2 instead |
| Video decoder | FastVideo's 4-bit "quality" decoder (37.9 dB against the full-precision decoder, per its model card) | `preview`: FastVideo's "fast" decoder (28.5 dB, softer) |
| Writing the video file | frames are converted on the GPU and encoded while the rest of the clip is still decoding; the sound is added at the end | keep `pin_cpu_memory: true` (FastVideo's default): without it, copying frames off the GPU stalled for 10-25 s per clip on this machine |
| Memory | everything stays on the GPU (about 134 of 252 GB) | |
| Startup | two warm-up videos before taking requests | `FV_WARMUP=0` skips them |
| Token reduction | off | `profiles/token-reduction.env` turns it on for any preset (see [Presets](#presets)) |
| Omni | FastVideo's 4-bit Omni model, attention partly 8-bit; the reference image's tiles are encoded together instead of one by one (`FASTVIDEO_H3_VAE_ENCODE_TILE_BATCH=28`), and a reference image used again is not processed twice (`FASTVIDEO_H3_REF2VA_MEMO_ENTRIES=16`) | `OMNI_ARGS=` (empty) runs the 16-bit Omni model with 8-bit math |

**Presets** (`docker compose --env-file profiles/<name>.env up -d <service>`):

| Preset | Service | What it changes |
|---|---|---|
| (none) | text to video | the defaults above |
| `fast` | text to video | part of attention in 8-bit |
| `qad-nvfp4` | text to video | FastVideo's 4-bit model, part of attention in 8-bit |
| `max` | text to video | FastVideo's 4-bit model, attention in 8-bit |
| `preview` | text to video | `max` plus a lighter video decoder (softer detail) |
| (none) | Omni | the defaults above |
| `omni-fast` | Omni | attention to the prompt and references in 8-bit |
| `omni-max` | Omni | `omni-fast` plus the rest of attention in 8-bit |
| `token-reduction` | either (add-on) | the middle blocks work on a half-width video; combine it with a preset, e.g. `--env-file profiles/max.env --env-file profiles/token-reduction.env` |

## Differences from ComfyUI's FastH3 workflow

ComfyUI ships FastH3 as the template `video_fastvideo_fasth3_t2v` (and `_i2v` for first and last frames), made for
home GPUs with 24-32 GB of memory. Its defaults next to this server's:

| Setting | This server (FastVideo) | ComfyUI template |
|---|---|---|
| Model file | FastH3 8-Step V2, converted to 8-bit when loading | an 8-bit (INT8) file, 22.1 GB |
| Prompt reader | Qwen3-VL 32B, 16-bit | a 4-bit (NVFP4) file, 15.7 GB |
| Video decoder | FastVideo's 4-bit quality decoder | an 8-bit (INT8) file, 2.8 GB |
| Size | 1344×768 | 640×640 |
| Length | 124 frames (5.17 s); 5 to 15 seconds allowed | 2 s (56 frames) |
| Steps | 8, taken the way the model was trained | 8, with a different sampler (`res_multistep`) |
| Sparse attention | keeps 20% of the video chunks at every step, as trained | keeps 10% after the first 20% of the steps (everything before that) |
| Memory | everything on one 252 GB GPU | moves models in and out of a 24-32 GB GPU |
| Seed | fixed per request | random each run |

<details>
<summary>ComfyUI template files and settings</summary>

- Model: `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` (INT8, 22.1 GB)
- Prompt reader: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (NVFP4, 15.7 GB)
- Decoders: `minimax_h3_video_vae_int8_convrot.safetensors` (INT8, 2.8 GB) and `minimax_h3_audio_vae_fp32.safetensors`
- Size: the Resolution Selector at 1:1, 0.4 MP, multiple of 32 (640×640). Length: 2 s, rounded up to 56 frames
  (2.33 s)
- Steps: the `res_multistep` sampler with the `simple` scheduler. Shift 10 / 3 (`ModelSamplingMiniMaxH3`) and no
  guidance (`BasicGuider`), both as in FastVideo
- Sparse attention: sparsity 0.9 from 20% of the schedule on, full attention before that; sequences under 12,288
  tokens always use full attention
- Attention code: ComfyUI's "comfy kitchen" backend

</details>

Two of these change what you get, not only the speed. ComfyUI uses a different sampler than the one the model was
trained with. It also keeps half as many video chunks in attention as the model was trained with. On this repository's
code (version 1.0), keeping 10% instead of 20% raised the error from 0.194 / 0.071 to 0.284 / 0.089.

Speed next to a home PC. The ComfyUI column is a published run on an RTX 5090 (ComfyUI 0.37 with the official
template and the 4-bit V2 model file, text to video, timed from submit to saved file). The FastVideo column is
FastVideo's own published figures for that card, timed from reading the prompt to the MP4. This server's numbers
are from one GB300, request to finished MP4. Every row is a 5-second clip:

| Size | ComfyUI template, RTX 5090 | FastVideo, RTX 5090 | This server, default | This server, `max` |
|---|---:|---:|---:|---:|
| 832×480 (480p) | 18.7 s | 14.8 s | 4.01 s | 2.71 s |
| 640×640 (the template's default size) | - | - | 3.79 s | 2.49 s |
| 1344×768 | 50.8-52.8 s | 38.6 s | 11.47 s | 7.35 s |

The template itself makes 2-second clips at 640×640; FastVideo's server does not make H3 clips under 5 seconds, so
the 640×640 row is a 5-second clip and has no published ComfyUI time to set beside it.

## What makes it fast

Measured on 5-second 1344×768 text-to-video clips unless noted; each change builds on the ones above it.

| Change | Effect | Idea from |
|---|---|---|
| **Fewer, bigger GPU programs.** Each of the model's 50 blocks runs as 5 custom GPU programs plus the large matrix multiplications, instead of dozens of small steps, and the data stays in one compact layout for the whole pass (`fastvideo/models/dits/h3fast/`) | video generation 15.0 → 10.3 s per clip | FastVideo's own combined steps; NVIDIA's Sol-H3 |
| **8-bit math without extra passes.** Every row of numbers gets its own scale, and the next program applies it, so the extra accuracy costs no speed | same accuracy as FastVideo's 8-bit setting | FastVideo's 8-bit model |
| **Faster video file writing.** Frames are converted to the video format on the GPU and streamed to the encoder | 1.37 → 0.16 s after decoding | FastVideo's UniServe |
| **FlashAttention-4 in the video decoder** | decoding 1.26 → 1.00 s | FastVideo |
| **Lower-precision chunk selection** (TF32) | 0.1 s faster, same accuracy | |
| **8-bit attention** for `fast`, `qad-nvfp4`, `max` and `preview`, using features only the GB300 has (`h3fast/vsa_fp8.cuh`) | attention 10.7 → 6.7 ms per block | SageAttention; FlashAttention-4 |
| **Omni: attention on pairs of video chunks** at once, to use the GPU's full width (`h3fast/vsa_pair.cuh`) | Omni 23.9 → 21.0 s | FlashAttention-4; MC-Sparse |
| **Omni: prompt and reference attention done separately.** That part runs on FlashAttention-4's fast general-purpose code, the video part on the sparse code, and the two results are combined exactly | Omni 20.4 → 18.6 s, slightly more accurate | FlashAttention-4; Fireworks |
| **The same for long text prompts** (1.1) | 10,000-token prompt: 24.0 → 20.1 s for a 5 s clip, 83.1 → 73.3 s for 15 s | |
| **...and for the 8-bit presets** (1.2) | 10,000-token prompt, 5 s clip: `fast` 20.6 → 19.3 s, `max` 14.9 → 13.5 s | |
| **Omni: the reference image encoded in one batch** (1.3) instead of tile by tile | 1.3 → 0.6 s per new reference; Omni 5 s clip 19.8 → 18.9 s | FastVideo's tile-batch option |
| **cuDNN attention in the prompt reader** (1.4) | a 10,000-token prompt reads in 0.55 s instead of 0.68 s; short prompts unchanged | NVIDIA cuDNN |
| **The same prompt again is not read twice** (1.4) | a re-roll skips the prompt reader: -0.55 s with a 10,000-token prompt | FastVideo's reference memo, extended to text prompts |
| **Video encoded during decoding** (1.4): each finished chunk goes to the H.264 encoder while the next one decodes | 5 s clip: 0.17 → 0.12 s after decoding | |
| **Token reduction** (1.4, optional add-on): the middle blocks work on a half-width video | the 8 model passes 10.25 → 6.61 s, 5 s clip 11.57 → 7.94 s; a different video for the same seed | antirez's h3.c; FastVideo |

FastVideo's own speed options are on as well: its fast attention code for this GPU, decoding the whole frame in one
batch, its 4-bit quality decoder, 8-bit model math and a compiled video decoder.

<details>
<summary>Technical details</summary>

- **Fused block path** (`FASTVIDEO_H3_FAST=1`): the residual stream stays in compact tile order for the whole forward
  (one permute in and out, no padded rows in the GEMMs). Each block runs 5 fused CUDA kernels (norm + modulate + FP8
  quantization; q/k RMSNorm + partial RoPE + heads-first write + tile means; per-row top-k written straight into the
  attention kernel's index lists; attention untile + gate branch + quantization; SwiGLU + quantization) plus the GEMMs
  and the attention kernel. Block time 37.6 → 25.5 ms.
- **FP8 scales:** E4M3 weights per tensor (quantized at load), activations per row. The GEMMs run tensor-wise with
  scale 1 and each GEMM's consumer applies the row scales.
- **Output:** frames are converted to yuv420p on the GPU and piped raw into ffmpeg through pinned host buffers that
  live as long as the server.
- **8-bit attention** (`h3fast/vsa_fp8.cuh`): FastVideo's warp-specialized tcgen05 VSA kernel with `kind::f8f6f4`
  MMAs. Q and K use static per-layer scales from the RMSNorm bound; V has one scale per (tile, head), folded into P's
  exponent. Hardware `ex2` (Blackwell Ultra doubles the SFU rate) and the sm_103 `tcgen05.ld.red` row max.
  `FASTVIDEO_H3_FP8_ATTN=pv` runs PV in E4M3 ("partly 8-bit"), `=1` both QK^T and PV.
- **Omni pair kernel** (`h3fast/vsa_pair.cuh`): two 64-row query tiles on full-rate M=128 MMAs with per-row
  membership masks, E4M3 PV.
- **Dense prefix:** the prefix queries (text, audio and reference rows, against all keys) and the prefix keys (for all
  other queries) run as two dense FlashAttention-4 calls; the sparse kernel keeps only the selected video tiles, and a
  log-sum-exp merge in the gate kernel joins them. Both sparse kernels return their log-sum-exp for the merge (the
  8-bit one from its correction warps, which combine each row's two halves).
- **Tile selection** scores in TF32 (`FASTVIDEO_H3_TF32_SCORES=0`: fp32).

</details>

## Quick start

You need a GB300 (the custom GPU code uses features only it has), Docker set up to give containers a specific GPU
(CDI), and about 280 GB of disk for all models (about 140 GB for text to video only).

```bash
# 1. FastVideo's Docker image, version a01c50a, built for the GB300 (~10 min)
git clone https://github.com/hao-ai-lab/FastVideo && cd FastVideo && git checkout a01c50a
docker build -f docker/Dockerfile --build-arg TORCH_CUDA_ARCH_LIST=10.3a \
  --build-arg CMAKE_BUILD_PARALLEL_LEVEL=32 -t fastvideo:gb300-a01c50a .
cd ..

# 2. This repository's changes on top of it
git clone https://github.com/chronarion/fasth3-gb300 && cd fasth3-gb300
docker build -t fasth3-gb300 .

# 3. Download the models (exact versions) and arrange them the way the server expects
HF_HOME=/path/to/hf tools/download_models.sh            # or: ... download_models.sh t2v  (text to video only)

# 4. Start the server
cd serve && cp .env.example .env    # set FASTH3_GPU (the GB300's Docker device name; `docker info` lists them)
                                    # and FASTH3_HF (the folder the models went into)
docker compose up -d t2v            # the first start compiles the custom GPU code and warms up (~5 min)
./gen.sh "A fox trots through fresh snow at dawn, breath steaming, wind and crunching snow."

# Omni warms up with a reference image at serve/media/in/reference.png (any 1344x768 image; this makes one)
../tools/make_reference.sh && docker compose stop t2v && docker compose up -d omni
```

The server speaks an OpenAI-style video API: send `POST /v1/videos` with
`{"model": "fasth3", "prompt": ..., "width": 1344, "height": 768, "num_frames": 124, "seed": 1}`, check
`GET /v1/videos/{id}` until it says completed, then download `GET /v1/videos/{id}/content`. For Omni, use
`"model": "fasth3-omni"`, add `"image_reference": [{"image_url": "/media/in/<file>"}]`, and write `<Picture 1>` in
the prompt where the image belongs; reference files go in `serve/media/in`.

**LoRAs.** Put LoRA files (`.safetensors`) in `serve/loras/` and restart the service. FastVideo reads the layouts
that ComfyUI, fal, ai-toolkit and musubi-tuner write. Each file is offered under its file name without
`.safetensors`, and a request picks one with `"lora": {"name": "<file name>", "scale": 0.8}` (scale from 0 to 2,
0.8 when left out; `GET /v1/model_info` lists what is loaded). A LoRA runs beside the model's 8-bit or 4-bit weights
on the fast path instead of being merged into them, so each request can use a different LoRA, or none. fal's
realism LoRA (rank 32, on the attention projections) adds
1.1 s with the default preset and 0.6 s with `max` to a 5-second clip.
Most community LoRAs were trained on the original 50-step MiniMax-H3. Style, character and camera-motion LoRAs
usually carry over to FastH3, but the best strength can differ; "turbo" or step-reduction LoRAs are not meant for
it. LoRAs are third-party weights with their own licenses.

## Measuring

**Speed:** run `tools/bench.py LABEL` against a running server (options: `--service omni --ref /media/in/<image>`,
`--width W --height H --frames N`, `--prompt-file F`). It prints the median times and the time of each step.
`tools/baselines/` has the two FastVideo-only setups used above.

**Accuracy:** start the server once with `H3Q_DUMP=/q/<name>` to record a run, then again with
`H3Q_REPLAY=/q/<name>` and the settings you want to test, and run `tools/quality.py` (use `FV_WARMUP=0` both times).
The error numbers above compare against FastVideo's full-precision path (`FASTVIDEO_H3_FAST=0`,
`FV_ARGS="--generator.engine.compile.vae_enabled true"`, and for Omni also `OMNI_ARGS=`).

**GPU code:** `tests/test_kernels.py` checks each custom GPU program against plain PyTorch.

## Notes

- GB300 only: the custom GPU code uses features only the GB300 has. Anything it does not cover falls back to
  FastVideo's normal code.
- One video at a time, as in FastVideo's server; other requests wait in line.
- FastVideo's own Docker image (version `a01c50a`) does not start this model as built: the FlashAttention-4 version
  it installs brings `nvidia-cutlass-dsl` 4.6.0.dev0, which its FlashInfer 0.7.4rc1 cannot load. This repository's
  Dockerfile, and the FastVideo-only baseline in `tools/baselines/Dockerfile.stock`, move to cutlass-dsl 4.8.1 and
  FlashAttention-4 4.0.0b36.
- Size and length are per request: any multiple of 32 pixels (tested from 640×640 to 1344×768, portrait and
  landscape) and 124 to 362 frames in steps of 17. The first clip at a new size takes a few seconds longer while the
  decoder adapts.
- Every request runs on the same worker thread. With a new thread per request, the second video size used to crash.

## Glossary

- **16-bit, 8-bit, 4-bit (BF16, FP8 / E4M3, NVFP4):** how many bits each number has in the GPU's math. Fewer bits run
  faster and use less memory, but are less precise. 16-bit is the model's full precision here.
- **Attention:** the step where each part of the video looks at the other parts (and at the prompt) to decide what to
  draw. It is the most expensive part of the model. It has two big multiplications: one scores how relevant each part
  is, the other mixes in their content. **Partly 8-bit** attention does only the second in 8-bit.
- **Token:** a piece of a word. A 1,000-token prompt is roughly 750 English words.
- **Sparse attention (VSA):** FastH3 was trained so that each small chunk of the video only looks at the 20% of other
  chunks that matter most to it, which saves most of the attention work.
- **FlashAttention-4:** a fast open-source attention library for NVIDIA's newest GPUs.
- **Steps / model passes:** the model builds a video from noise in 8 steps; each step is one pass through the model.
- **QAD model:** FastVideo's version of FastH3 that was trained to run in 4-bit.
- **Decoder (VAE):** turns the model's compressed output into actual video frames.
- **Error:** see [Presets](#presets).

## Credits

- [FastVideo](https://github.com/hao-ai-lab/FastVideo) (Hao AI Lab): the FastH3 models (V2, the 4-bit QAD version,
  Omni, the 4-bit decoders), the sparse attention method and its GPU code (this repository's 8-bit and pair versions
  are based on it), the pipeline and server, and every FastVideo speed option used here.
- [MiniMax](https://huggingface.co/MiniMaxAI) for MiniMax-H3.
- [FlashAttention-4](https://github.com/Dao-AILab/flash-attention) (Tri Dao and contributors): the attention code
  used for prompts, references and the decoder, and GPU techniques reused in the 8-bit code.
- [SageAttention](https://github.com/thu-ml/SageAttention) for the design of 8-bit attention.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and NVIDIA cuDNN for 4-bit math and fast attention.
- NVIDIA's [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) write-up for the combined-step layout and the
  single-B300 reference point.
- antirez's [h3.c](https://github.com/antirez/h3.c) for token reduction, which FastVideo ported and this repository
  runs.

## License

Apache-2.0, as FastVideo. `fastvideo/models/dits/h3fast/vsa_primitives.cuh` is copied from FastVideo and
`vsa_fp8.cuh` / `vsa_pair.cuh` are based on its attention code; their headers say what changed. The model weights are
not part of this repository and have their own licenses (see each model's page).
