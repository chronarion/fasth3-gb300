# FastH3 on one GB300

A single-GPU inference path for [FastH3](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2), FastVideo's
8-step distillation of MiniMax-H3 (text to 1344×768 video with synchronized stereo audio), built for the NVIDIA GB300
in a DGX Station and served through FastVideo's OpenAI-style video API.

On one GB300, a 5-second 1344×768 clip with audio takes **11.57 s** from request to finished MP4, at the
quality of FastVideo's own FP8 path. FastVideo with all of its speed options takes 17.32 s on the same GPU, and
34.59 s out of the box. The lossy profiles go down to **7.44 s**. FastH3 Omni (text plus image, video or audio
references) takes 19.83 s per 5-second clip, down to 16.23 s.

The repository is a patch over [FastVideo](https://github.com/hao-ai-lab/FastVideo) at commit `a01c50a`: a fused DiT
block path with its own CUDA kernels (`fastvideo/models/dits/h3fast/`), 8-bit block-sparse attention kernels derived
from FastVideo's Blackwell VSA kernel, a dense path on FlashAttention-4 for the prompt and reference prefix, and a
faster output tail. Model weights are FastVideo's, unchanged.

- [Results](#results): presets, the same GPU under other engines, published numbers from other systems
- [How it is served](#how-it-is-served): every setting, and why
- [Differences from ComfyUI's FastH3 workflow](#differences-from-comfyuis-fasth3-workflow)
- [What makes it fast](#what-makes-it-fast)
- [Quick start](#quick-start), [Measuring](#measuring), [Notes](#notes), [Credits](#credits)

## Results

All numbers: one GB300, one request at a time, server warmed up, median of the timed clips (6 per preset at 5 s, 3 with
long prompts, 2 for 10 s and 15 s clips) after one untimed clip. Unless a row says otherwise: 1344×768, 124 frames at 24 fps
(5.17 s), 8 transformer forwards, short prompts (12-16 words), seed 1234. "End to end" runs from the API request to the
finished MP4 (H.264 + AAC) on disk. "Denoise" is the 8 transformer forwards. Conditions in full:
[Benchmark conditions](#benchmark-conditions).

### Presets

Server profiles (`serve/profiles/`) set the precision of the transformer's linear layers and of attention; the request
sets size and length. End to end, 1344×768, short prompts:

| Preset | Linears | Attention | 5 s (124 frames) | 10 s (243) | 15 s (362) | Quality (video / audio) |
|---|---|---|---:|---:|---:|---|
| Default | FP8 | BF16 | 11.57 s | 28.66 s | 54.71 s | 0.192 / 0.070 |
| `fast` | FP8 | BF16 QK^T, E4M3 PV | 10.80 s | 25.44 s | 47.14 s | 0.197 / 0.075 |
| `qad-nvfp4` | NVFP4 (QAD) | BF16 QK^T, E4M3 PV | 8.37 s | 20.78 s | 39.98 s | 0.271 / 0.158 |
| `max` | NVFP4 (QAD) | E4M3 | 7.44 s | 17.86 s | 33.75 s | 0.282 / 0.167 |
| `preview` | NVFP4 (QAD) | E4M3; 8-layer decoder | 6.68 s | 16.46 s | 31.47 s | 0.282 / 0.167 (as `max`; softer decoder) |
| Omni, 1 image reference | NVFP4 (QAD Omni) | E4M3 PV; prefix BF16 | 19.83 s | 36.28 s | 56.41 s | 0.219 / 0.111 |
| `omni-fast` | NVFP4 (QAD Omni) | E4M3 PV; prefix E4M3 | 16.59 s | 30.23 s | 47.35 s | 0.226 / 0.119 |
| `omni-max` | NVFP4 (QAD Omni) | E4M3 | 16.23 s | 29.08 s | 45.23 s | 0.230 / 0.117 |

Quality is the teacher-forced error against BF16: the relative L2 of the predicted velocity (video / audio) against
FastVideo's BF16 path (for Omni: the BF16 Omni transformer), at each of the 8 steps from identical inputs, averaged
over four prompts ([Measuring](#measuring)). FastVideo's own FP8 path scores 0.190 / 0.071 on the same test; the default
profile matches it. The QAD rows run FastVideo's quantization-aware NVFP4 checkpoint (FastH3-QAD-v2), a different set of
weights distilled from the same model, so their error includes that model's distance from the BF16 one. `preview`
changes only the video decoder (see [How it is served](#how-it-is-served)). Omni rows carry one 1344×768 reference image,
encoded on every clip.

Other sizes and prompt lengths (long prompts: teacher-forced error against BF16 with the 1K-token prompt 0.243 /
0.082 and with the 10K-token prompt 0.256 / 0.062; the dense text prefix of 1.1 moved it by less than 0.003):

| Preset | Request | End to end |
|---|---|---:|
| Default | 832×480, 5 s | 4.01 s |
| Default | 768×1344 (portrait), 5 s | 11.81 s |
| Default | 640×640 (ComfyUI template's size), 5 s | 3.79 s |
| Default | 5 s, 1K-token prompt | 12.41 s |
| Default | 10 s, 1K-token prompt | 29.90 s |
| Default | 15 s, 1K-token prompt | 55.48 s |
| Default | 5 s, 10K-token prompt | 20.08 s |
| Default | 10 s, 10K-token prompt | 42.38 s |
| Default | 15 s, 10K-token prompt | 73.25 s |
| `fast` | 832×480, 5 s | 3.86 s |
| `qad-nvfp4` | 832×480, 5 s | 2.92 s |
| `max` | 832×480, 5 s | 2.71 s |
| `max` | 640×640, 5 s | 2.49 s |
| `max` | 5 s, 1K-token prompt | 8.11 s |
| `max` | 10 s, 1K-token prompt | 18.90 s |
| `max` | 15 s, 1K-token prompt | 35.01 s |
| `fast` | 5 s, 10K-token prompt | 19.21 s |
| `fast` | 10 s, 10K-token prompt | 39.31 s |
| `fast` | 15 s, 10K-token prompt | 66.03 s |
| `qad-nvfp4` | 5 s, 10K-token prompt | 16.30 s |
| `max` | 5 s, 10K-token prompt | 13.42 s |
| `max` | 10 s, 10K-token prompt | 27.67 s |
| `max` | 15 s, 10K-token prompt | 47.08 s |
| `preview` | 832×480, 5 s | 2.39 s |
| `preview` | 5 s, 10K-token prompt | 12.65 s |
| Omni | 832×480, 5 s | 10.74 s |
| Omni | 5 s, same reference again (re-roll; reference encodes reused) | 18.31 s |
| `omni-max` | 832×480, 5 s | 9.13 s |

Per-stage times of every run (text encoding, denoising, decoders): `results/`.

### Same GPU, other engines

The same model and requests on the same GB300 with FastVideo alone, end to end (request to MP4 on disk):

| Engine | Settings | 5 s | 10 s | 15 s | 5 s, 832×480 |
|---|---|---:|---:|---:|---:|
| FastVideo `a01c50a`, out of the box | BF16 transformer, no compile, FastVideo's defaults (full VAE decoder, 8 tiles per decoder call); its image with only the FlashInfer import fixed | 34.59 s | 78.81 s | 137.3 s | 11.87 s |
| FastVideo `a01c50a`, all of its speed options | FP8 linears, compiled VAE, its fused kernels (`FUSIONS=mk`), tile-first tcgen05 VSA, FlashAttention-4 + NVFP4 quality decoder + 28 tiles per call in the VAE | 17.32 s | 40.26 s | 73.75 s | 6.14 s |
| This repository, default | FP8 linears, BF16 attention | 11.57 s | 28.66 s | 54.71 s | 4.01 s |
| This repository, `max` | QAD NVFP4 linears, E4M3 attention | 7.44 s | 17.86 s | 33.75 s | 2.71 s |

### Published numbers from other systems

Each published number next to this repository's time for the same clip, measured over the same span. Spans differ
between sources, so each row says what is timed. Published rows are copied from the cited pages and were not
re-measured here; GPU counts and models differ (FastH3 V2 unless noted).

| System | GPUs | Precision | Clip | Timed span | Published | This repo, 1× GB300 (default FP8 / `max` NVFP4) |
|---|---|---|---|---|---:|---:|
| UniServe (FastVideo) | 4× GB200 | BF16 | 5 s, 1K-token prompt | request → MP4 downloaded | 5.62 s | 12.42 s / 8.12 s |
| UniServe (FastVideo) | 4× GB200 | NVFP4 MLP + decoder | 5 s, 1K-token prompt | request → MP4 downloaded | 4.49 s | 12.42 s / 8.12 s |
| FastVideo (UniServe's tuned baseline) | 4× GB200 | BF16 | 5 s, 1K-token prompt | request → MP4 downloaded | 7.47 s | 12.42 s / 8.12 s |
| SGLang (UniServe's tuned baseline) | 4× GB200 | BF16 | 5 s, 1K-token prompt | request → MP4 downloaded | 7.07 s | 12.42 s / 8.12 s |
| vLLM-Omni (UniServe's tuned baseline) | 4× GB200 | BF16 | 5 s, 1K-token prompt | request → MP4 downloaded | 7.69 s | 12.42 s / 8.12 s |
| UniServe (FastVideo) | 8× GB200 (2 nodes) | BF16 | 5 s, 1K-token prompt | request → MP4 downloaded | 3.13 s | 12.42 s / 8.12 s |
| UniServe (FastVideo) | 4× GB200 | BF16 | 5 s, 10K-token prompt | request → MP4 downloaded | 8.80 s | 20.09 s / 13.43 s |
| UniServe (FastVideo) | 4× GB200 | BF16 | 10 s, 1K-token prompt | request → MP4 downloaded | 11.73 s | 29.93 s / 18.92 s |
| UniServe (FastVideo) | 4× GB200 | BF16 | 15 s, 1K-token prompt | request → MP4 downloaded | 20.27 s | 55.51 s / 35.05 s |
| UniServe (FastVideo) | 4× GB200 | BF16 | 10 s, 10K-token prompt | request → MP4 downloaded | 16.75 s | 42.41 s / 27.69 s |
| UniServe (FastVideo) | 4× GB200 | BF16 | 15 s, 10K-token prompt | request → MP4 downloaded | 27.31 s | 73.28 s / 47.12 s |
| UniServe (FastVideo) | 8× RTX PRO 6000 Server | BF16 | 5 s, 1K-token prompt | request → MP4 downloaded | 19.12 s | 12.42 s / 8.12 s |
| Sol-H3 (NVIDIA), 4 forwards, dense | 1× B300 | not stated | 5 s (124 frames) | text encoding → VAE decode, no MP4 | 11.48 s | 11.40 s / 7.26 s |
| Sol-H3 (NVIDIA), 4 forwards, dense | 1× B300 | not stated | 10 s (243 frames) | text encoding → VAE decode, no MP4 | 28.18 s | 28.41 s / 17.59 s |
| Sol-H3 (NVIDIA), 4 forwards, dense | 1× B300 | not stated | 15 s (362 frames) | text encoding → VAE decode, no MP4 | 50.71 s | 54.38 s / 33.39 s |
| Sol-H3 (NVIDIA), 4 forwards, sparse | 8× B300 | INT8 QKV, FP8 transport | 5 s (124 frames) | text encoding → VAE decode, no MP4 | 1.65 s | 11.40 s / 7.26 s |
| ComfyUI 0.37, official template | 1× RTX 5090 | NVFP4 | 5 s, text to video | submit → MP4 saved | 50.8-52.8 s | 11.57 s / 7.44 s |
| ComfyUI 0.37, official template | 1× RTX 5090 | NVFP4 | 832×480, 5 s, text to video | submit → MP4 saved | 18.7 s | 4.01 s / 2.71 s |
| FastVideo engine (FastVideo's figures) | 1× RTX 5090 | not stated | 5 s | text encoding → MP4 | 38.6 s | 11.57 s / 7.44 s |
| FastVideo engine (FastVideo's figures) | 1× RTX 5090 | not stated | 832×480, 5 s | text encoding → MP4 | 14.8 s | 4.01 s / 2.71 s |

Sources: [UniServe](https://haoailab.com/blogs/uniserve-fasth3/) (FastVideo's serving stack; BF16 unless noted, its
baselines tuned by the same team), [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) (NVIDIA; FastH3 Preview's
4-step dense adapter, 4 forwards), [ComfyUI and FastVideo on an RTX 5090](https://ai-muninn.com/en/blog/fasth3-rtx-5090-comfyui)
(ComfyUI 0.37 with the official template; the FastVideo-engine RTX figures quote FastVideo's own release numbers).

### Throughput per GPU

UniServe's throughput figure is GPU-seconds per second of finished video over a mix of 5, 10 and 15 s clips with 1K-
and 10K-token prompts. A single GPU running one request at a time is saturated, so its throughput is the inverse of
its latency; over an equal mix of the same six shapes:

| System | GPUs | Videos per day | GPU-s per video-s |
|---|---|---:|---:|
| UniServe (FastVideo), BF16 | 4× GB200 | ~6,100 | 5.8 |
| UniServe (FastVideo), BF16 | 8× GB200 | ~11,000 | 6.4 |
| UniServe (FastVideo), BF16 | 8× RTX PRO 6000 Server | ~3,400 | 21.2 |
| This repository, default (FP8) | 1× GB300 | 2,220 | 3.8 |
| This repository, `max` (NVFP4) | 1× GB300 | 3,452 | 2.5 |

### Benchmark conditions

- **Hardware:** NVIDIA DGX Station GB300: one GB300 (sm_103, 252 GB HBM3e) with a 72-core Grace CPU. The GB300 module
  reads a 1,250 W power limit (its static allocation when the station's second GPU slot is populated); tensor-heavy
  phases run against that cap.
- **Software:** driver 615, CUDA 13.0, PyTorch 2.12.0, FastVideo `a01c50a`, FlashAttention-4 4.0.0b36,
  FlashInfer 0.7.4rc1, cuDNN 9.20 (the image built from this repository).
- **Model revisions:** FastVideo-FastH3-8-Step-V2 `3da2ddf`, FastH3-QAD-v2 `a0dc05a`, FastH3-Omni-8-Step-V1 `53e0d26`,
  FastH3-Omni-NVFP4-QAD `f01e6a2`, FastH3-NVFP4-VAE-Decoder `bd0c248`.
- **Requests:** the four short prompts and the two long ones (956 and 9,771 tokens) are in `tools/bench.py` and
  `tools/prompts/`. Long-prompt clips each get a distinct "Take N." prefix, so no clip reuses another's text encoding.
  Omni requests carry one 1344×768 reference image; Omni rows encode it on every clip (reference memo off, FastVideo's
  default) unless marked "re-roll".
- **Spans:** *end to end* = request until the job reports completed (MP4 written on the server); *downloaded* = until
  the client has received the last byte of the MP4; *pipeline* = text encoding + denoising + video and audio decode,
  without MP4 encoding.

## How it is served

`serve/docker-compose.yml` runs FastVideo's `fastvideo serve` with `serve/fasth3-8step.yaml` (t2v) or
`serve/fasth3-omni.yaml` (Omni) on one GPU. The settings, grouped by what decides them:

**Fixed by the checkpoint** (FastVideo reads them from the model's `fastvideo_inference.json` and refuses others):

| Setting | t2v (FastH3 8-Step V2) | Omni (FastH3 Omni 8-Step V1) |
|---|---|---|
| Transformer forwards per clip | 8 | 8 (a 32-interval PDD grid fused into 8 blocks of 4) |
| Noise schedule | the trained DMD rungs t = 999, 874, 749, 624, 500, 375, 250, 125, then 0 (`num_inference_steps: 9` counts the 9 grid points) | 33-point grid, blocks at every 4th point |
| Sigma shift (video / audio) | 10 / 3 | 12 / 3 |
| Step | flow-matching Euler step between grid points, video and audio together | same |
| Guidance | none (scale 1.0, one forward per step, empty negative prompt) | none |
| Sparse attention (VSA) | 64-token tiles (4×4×4 latent cubes); each video query tile attends its top 20% of video key tiles (sparsity 0.8, as trained) plus every text and audio tile; a coarse branch over tile means is added through the learned `to_gate_compress` gate | 128-token tiles, sparsity 0.9; reference tiles kept at rate 0.1 |

**Request defaults** (per request: `width`, `height`, `num_frames`, `seed`): 1344×768 (H3's native 768-pixel short
edge), 124 frames at 24 fps (5.17 s). Frame counts follow H3's 17k + 5 grid, and FastVideo accepts 124 to 362 frames
(5 to 15 s; 243 frames = 10.1 s).
Output: MP4 with H.264 (libx264, preset ultrafast, yuv420p) and AAC (32 kHz stereo).

**Engine settings** (this repository's defaults; the profiles change the first two rows):

| Setting | Default | Notes |
|---|---|---|
| Transformer linears | FP8 E4M3: weights per tensor (quantized at load), activations per row | `qad-nvfp4` and `max` profiles: FastVideo's quantization-aware NVFP4 checkpoint (FastH3-QAD-v2), W4A4 |
| Attention | BF16 on FastVideo's tcgen05 VSA kernel | `FASTVIDEO_H3_FP8_ATTN=pv`: BF16 QK^T + E4M3 PV; `=1`: E4M3 QK^T and PV |
| Prefix attention (text, audio and reference rows) | dense on FlashAttention-4 (BF16), merged with the sparse rest by log-sum-exp | `omni-fast`, `omni-max`: E4M3; t2v `fast` / `qad-nvfp4`: BF16 and `max` / `preview`: E4M3, from 2,048 prefix rows (long prompts) |
| Tile selection scores | TF32 | `FASTVIDEO_H3_TF32_SCORES=0`: fp32 |
| Text encoder | H3's Qwen3-VL 32B, BF16; FA2 attention (t2v), cuDNN (Omni) | |
| Video decoder | FastVideo's NVFP4 quality decoder (36 layers; 37.9 dB / LPIPS 0.010 against the FP32 decoder on FastH3 V2 clips, per its model card), all 28 spatial tiles of a frame chunk per call, FlashAttention-4 attention, `torch.compile` with CUDA graphs | `preview`: the 8-layer fast tier (28.5 dB / 0.068) |
| Audio decoder | H3's audio VAE | |
| Output | frames converted to yuv420p on the GPU and piped raw into ffmpeg, through pinned host buffers that live as long as the server | keep `pin_cpu_memory: true` (FastVideo's default): device-to-host copies into fresh pageable memory stalled 10-25 s per clip on this platform |
| Memory | every component resident in HBM (~134 GB of 252 GB): no offload, no sequence parallelism | |
| Startup | two warm-up requests (two prompt lengths) before traffic | `FV_WARMUP=0` skips them |
| Omni | NVFP4 QAD Omni transformer (FastH3-Omni-NVFP4-QAD), E4M3 PV on the selected video tiles; repeated references reuse their encodes (`FASTVIDEO_H3_REF2VA_MEMO_ENTRIES=16`) | `OMNI_ARGS=` (empty): BF16 Omni transformer with FP8 linears |

**Profiles** (`docker compose --env-file profiles/<name>.env up -d <service>`):

| Profile | Service | Changes |
|---|---|---|
| (none) | t2v | the defaults above |
| `fast` | t2v | E4M3 PV |
| `qad-nvfp4` | t2v | FastH3-QAD-v2 NVFP4 linears + E4M3 PV |
| `max` | t2v | FastH3-QAD-v2 NVFP4 linears + E4M3 QK^T and PV |
| `preview` | t2v | `max` + FastVideo's 8-layer NVFP4 video decoder (fast tier, softer) |
| (none) | omni | the defaults above |
| `omni-fast` | omni | E4M3 dense prefix |
| `omni-max` | omni | E4M3 dense prefix + E4M3 QK^T and PV |

## Differences from ComfyUI's FastH3 workflow

ComfyUI ships FastH3 as the template `video_fastvideo_fasth3_t2v` (and `_i2v` for first/last frames), built for
consumer GPUs. Its defaults next to this server's:

| Setting | This server (FastVideo engine) | ComfyUI template defaults |
|---|---|---|
| Transformer weights | FastH3-8-Step-V2 (BF16 checkpoint), FP8 linears at load | `fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors` (INT8, 22.1 GB) |
| Text encoder | Qwen3-VL 32B, BF16 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (NVFP4, 15.7 GB) |
| Video VAE | full VAE, NVFP4 quality decoder | `minimax_h3_video_vae_int8_convrot.safetensors` (INT8, 2.8 GB) |
| Audio VAE | H3's | `minimax_h3_audio_vae_fp32.safetensors` |
| Size | 1344×768 | 640×640 (Resolution Selector: 1:1, 0.4 MP, multiple of 32) |
| Length | 124 frames (5.17 s); FastVideo accepts 124-362 (5-15 s) | 2 s, snapped up to 56 frames (2.33 s) |
| Steps | 8 forwards on the trained DMD rungs, Euler step | 8 steps, `res_multistep` sampler, `simple` scheduler |
| Sigma shift (video / audio) | 10 / 3 | 10 / 3 (`ModelSamplingMiniMaxH3`) |
| Guidance | none | none (`BasicGuider`) |
| Sparse attention | VSA, keep 20% (sparsity 0.8, as trained), every step and every block | VSA, keep 10% (sparsity 0.9) from 20% of the schedule on, dense before that; sequences under 12,288 tokens stay dense |
| Attention kernels | FastVideo's tcgen05 VSA kernel and this repository's 8-bit variants; FlashAttention-4 for dense parts | ComfyUI's "comfy kitchen" attention backend |
| Memory | everything resident on one 252 GB GPU | ComfyUI loads and unloads models to fit 24-32 GB of VRAM |
| Seed | fixed per request (`seed`) | randomized per run |

FastVideo does not generate H3 clips shorter than 5 s, so the presets table runs the template's 640×640 size at
124 frames.

Two of these change the result, not only the speed: ComfyUI's `res_multistep` sampler is a multistep solver where
FastVideo takes the Euler step the checkpoint was distilled with, and ComfyUI's VSA default keeps 10% of the video
tiles where the model card says the checkpoint was trained at 80% sparsity (20% kept). On this repository's path,
sparsity 0.9 raised the teacher-forced error from 0.194 / 0.071 to 0.284 / 0.089 (video / audio).

For timing on a regular PC: a published run of the ComfyUI template on an RTX 5090 (ComfyUI 0.37, NVFP4 V2 file,
1344×768, 124 frames, text to video) took 50.8-52.8 s from submit to saved file, and 18.7 s at 832×480. FastVideo's own
engine on the same card: 38.6 s and 14.8 s. This server on a GB300: 11.57 s and 4.01 s with the default
profile, 7.44 s and 2.71 s with `max`.

## What makes it fast

Measured on t2v (5 s, 1344×768) unless noted; each step on top of the previous ones.

| Change | Effect | Idea from |
|---|---|---|
| **Fused DiT block path** (`FASTVIDEO_H3_FAST=1`): the residual stream stays in compact tile order for the whole forward (one permute in and out, no padded rows in GEMMs); each block runs 5 fused CUDA kernels (norm + modulate + FP8 quantization; q/k RMSNorm + partial RoPE + heads-first write + tile means; per-row top-k written straight into the attention kernel's index lists; attention untile + gate branch + quantization; SwiGLU + quantization) plus the GEMMs and the attention kernel | denoise 15.0 → 10.3 s; block 37.6 → 25.5 ms | FastVideo's own fused kernels (`FASTVIDEO_MINIMAX_H3_FUSIONS`); NVIDIA's Sol-H3 fused chains |
| **Per-row FP8 activation scales applied by the consumer** of each GEMM: the GEMMs run tensor-wise with scale 1, so per-token scaling costs neither a second pass nor a slower GEMM | same error as FastVideo's FP8 path | per-token scaling in FastVideo's FP8 checkpoint |
| **GPU yuv420p + raw pipe into ffmpeg**, pinned and reused frame buffers | post-decode 1.37 → 0.16 s; no more 10-30 s stalls copying into fresh pageable memory | FastVideo UniServe's output path |
| **FlashAttention-4 in the VAE decoder** | decode 1.26 → 1.00 s | FastVideo (`FASTVIDEO_FA4`) |
| **TF32 tile-selection scores** | −0.1 s per clip, error unchanged | |
| **8-bit attention kernels** (`h3fast/vsa_fp8.cuh`): FastVideo's warp-specialized tcgen05 VSA kernel with `kind::f8f6f4` MMAs; Q and K with static per-layer scales from the RMSNorm bound, V with one scale per (tile, head) folded into P's exponent; hardware `ex2` (Blackwell Ultra doubles the SFU rate) and the sm_103 `tcgen05.ld.red` row max | kernel 10.7 → 6.7 ms (all E4M3); profiles `fast` / `max` | SageAttention (per-block 8-bit attention); FlashAttention-4 (`ld.red` row max, no exp emulation on sm_103) |
| **Omni: 128-token pair kernel** (`h3fast/vsa_pair.cuh`): two 64-row query tiles on full-rate M=128 MMAs with per-row membership masks, E4M3 PV | Omni 23.9 → 21.0 s | FlashAttention-4's two-query-tile CTAs; MC-Sparse's query grouping |
| **Omni: dense prefix on FlashAttention-4**: the reference/text prefix queries (against all keys) and the prefix keys (for all other queries) run as two dense FA4 calls; the sparse kernel keeps only the selected video tiles; a log-sum-exp merge in the gate kernel joins them | Omni 20.4 → 18.6 s with *lower* error (BF16 dense part); 15.3 s with E4M3 | FlashAttention-4 (dense kernel with LSE output); Fireworks' KV-stationary sparse-attention write-up (hot KV blocks) |
| **Dense text prefix for t2v** (1.1): the same split for the prompt's text and the audio rows, with FastVideo's BF16 sparse kernel returning its log-sum-exp for the merge | 5 s clip with a 10K-token prompt 24.0 → 20.1 s, 15 s 83.1 → 73.3 s; 1K-token prompt −2%; short prompts unchanged | |
| **Dense prefix on the 8-bit profiles** (1.2): the 64-token 8-bit kernel returns its log-sum-exp (written by the correction warps, which combine each row's two halves); from 2,048 prefix rows `fast` / `qad-nvfp4` use a BF16 dense prefix and `max` / `preview` an E4M3 one | 10K-token prompt, 5 s: `fast` 20.6 → 19.3 s, `max` 14.9 → 13.5 s; teacher-forced error unchanged (`max`) or lower (`fast`, BF16 instead of E4M3 PV on the prefix) | |

FastVideo's own options are also on: its tcgen05 VSA kernel, batched VAE tiles (all 28 tiles per call), the NVFP4
quality decoder, FP8 linears and the compiled VAE.

## Quick start

Requirements: a GB300 (the kernels are built for sm_103a), Docker with CDI GPU support, ~280 GB of disk for all models
(~140 GB for t2v alone).

```bash
# 1. FastVideo's image at a01c50a, built for sm_103a (~10 min)
git clone https://github.com/hao-ai-lab/FastVideo && cd FastVideo && git checkout a01c50a
docker build -f docker/Dockerfile --build-arg TORCH_CUDA_ARCH_LIST=10.3a \
  --build-arg CMAKE_BUILD_PARALLEL_LEVEL=32 -t fastvideo:gb300-a01c50a .
cd ..

# 2. This repository's layer
git clone https://github.com/chronarion/fasth3-gb300 && cd fasth3-gb300
docker build -t fasth3-gb300 .

# 3. Weights (pinned revisions) and the composed model directories
HF_HOME=/path/to/hf tools/download_models.sh            # or: ... download_models.sh t2v

# 4. Serve
cd serve && cp .env.example .env    # set FASTH3_GPU and FASTH3_HF
docker compose up -d t2v            # first start JIT-builds the kernels and warms up (~5 min)
./gen.sh "A fox trots through fresh snow at dawn, breath steaming, wind and crunching snow."

# Omni: its warm-up needs a reference image at serve/media/in/reference.png (any 1344x768 image; this makes one)
../tools/make_reference.sh && docker compose stop t2v && docker compose up -d omni
```

API: `POST /v1/videos` with `{"model": "fasth3", "prompt": ..., "width": 1344, "height": 768, "num_frames": 124,
"seed": 1}`, then poll `GET /v1/videos/{id}` and fetch `GET /v1/videos/{id}/content`. Omni: `"model": "fasth3-omni"`
and `"image_reference": [{"image_url": "/media/in/<file>"}]` with `<Picture 1>` in the prompt; reference files must sit
under `serve/media/in`.

## Measuring

**Speed:** `tools/bench.py LABEL [--service omni --ref /media/in/<image>] [--width W --height H --frames N]
[--prompt-file F]` against a running server prints the median end-to-end, downloaded and pipeline times and the
per-stage medians. The two FastVideo baselines are compose files in `tools/baselines/`.

**Quality:** start the server with `H3Q_DUMP=/q/<name>` to record every step's inputs and velocities, then with
`H3Q_REPLAY=/q/<name>` to replay them into a candidate, and run `tools/quality.py` (`FV_WARMUP=0` both times). The BF16
references above come from FastVideo's BF16 path (`FASTVIDEO_H3_FAST=0`,
`FV_ARGS="--generator.engine.compile.vae_enabled true"`; Omni also `OMNI_ARGS=`).

**Kernels:** `tests/test_kernels.py` checks each fused kernel against PyTorch on random data.

## Notes

- GB300 only. The CUDA sources target sm_103a (tcgen05 MMAs, `tcgen05.ld.red`); requests that the fused path does not
  cover (no FP8/NVFP4 linears, sequence parallelism, training) fall back to FastVideo's block loop.
- One request at a time, as FastVideo's server runs them; concurrent requests queue.
- FastVideo's own image at `a01c50a` does not start this model as built: its FlashAttention-4 pin brings
  `nvidia-cutlass-dsl` 4.6.0.dev0 and its FlashInfer 0.7.4rc1 then fails to import (`cute.FastDivmodDivisorV2`).
  This repository's Dockerfile, and the stock baseline's `tools/baselines/Dockerfile.stock`, move to cutlass-dsl
  4.8.1 and FlashAttention-4 4.0.0b36.
- Size and length are per request: multiples of 32 (tested 640×640 to 1344×768, portrait and landscape) and 124 to 362
  frames. The first clip at a new size also compiles and captures the VAE decoder for it (a few seconds).
- Running every forward on one long-lived thread is part of the patch: with a new thread per request, the compiled
  VAE's CUDA-graph capture for a second size failed (`cudagraph_trees.get_container`).

## Credits

- [FastVideo](https://github.com/hao-ai-lab/FastVideo) (Hao AI Lab): the FastH3 checkpoints (V2, QAD-v2, Omni, the NVFP4
  VAE decoders), Video Sparse Attention and its tcgen05 Blackwell kernel (the 8-bit and pair kernels here derive from
  it), the pipeline, server and every option listed under "FastVideo's own options".
- [MiniMax](https://huggingface.co/MiniMaxAI) for MiniMax-H3.
- [FlashAttention-4](https://github.com/Dao-AILab/flash-attention) (Tri Dao and contributors): the dense attention
  kernels used for Omni's prefix and the VAE, and the sm_103 techniques reused in the 8-bit kernels.
- [SageAttention](https://github.com/thu-ml/SageAttention) for the per-block 8-bit attention design.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) and cuDNN for the NVFP4 GEMMs and fused attention.
- NVIDIA's [Sol-H3](https://nvlabs.github.io/Sana/Sol-Engine/Sol-H3/) write-up for the fused-chain layout and the
  single-B300 reference point.

## License

Apache-2.0, as FastVideo. `fastvideo/models/dits/h3fast/vsa_primitives.cuh` is copied from FastVideo and
`vsa_fp8.cuh` / `vsa_pair.cuh` are derived from its VSA kernel; their headers say what changed. Model weights are not
part of this repository and carry their own licenses (see each model card).
