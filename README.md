# Wan2.2-S2V Docker

A Dockerised [Wan2.2-S2V-14B](https://huggingface.co/Wan-AI/Wan2.2-S2V-14B)
renderer built around one idea: **a performance needs a first frame and a
soundtrack, not a last frame.**

```
performance/
    start.jpg     the first frame; the only visual reference
    audio.wav     drives lips, gaze, head motion, gestures and timing
    prompt.txt    the scene, one paragraph
```

That folder renders to an mp4 with the audio muxed in. There is no `stop.jpg`
on purpose. Pinning an end pose is what makes image-to-video acting look
rubbery: the model spends the clip trying to arrive at a shape it was handed
instead of finishing the performance the track asked for. Wan2.2-S2V drives
the whole clip from the audio, so the ending is an outcome, not a constraint.

Status: **verified end to end on a rented H100 80 GB (2026-11-21)** — weights,
attention, sampling, VAE decode and mux all ran. The first run found two real
bugs (a missing `dashscope` import, and a three-frame tail loss in upstream's
own mux), both fixed and now covered by tests. Measured timings, VRAM and the
cost per clip are in [Performance](#performance); 116 tests, and the same suite
runs inside the built image.

- **One still, one track** — no end frame, no pose video, no keyframe pair.
- **Planned before it costs anything** — `plan` reports the exact resolution,
  clip count and output duration on CPU.
- **Clip count is computed, not guessed** — upstream asks for an extra clip at
  exact five-second boundaries; this renders the smallest count that covers
  the audio.
- **Verified artefacts** — a clip without an audio stream fails the job instead
  of being delivered silently.
- **Resumable batches** — a shot folder is the contract, on disk and over HTTP.
- **HTTP service and CLI** — same four inputs, same manifest.

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | 48 GB: short 480p clips only (measured 42.6 GiB peak) | 80 GB (A100/H100) |
| Driver | 570+ (CUDA 12.8) | 580+ |
| RAM | 64 GB | 96 GB |
| Disk | 80 GB | 150 GB (45.6 GB weights + image + outputs) |
| Bandwidth | **< $4/TB each direction** | lowest on offer |

Upstream's own single-GPU figure is **80 GB**, and the arithmetic says the same.
The checkpoint's DiT shards alone are **32.6 GB** in bf16 (14B parameters plus
the audio-injection and motion modules), and `--offload_model` still moves the
whole model onto the card for every clip; the UMT5 encoder (11.4 GB) also visits
the card before sampling starts. Measurements bear that out: **42.6 GiB peak**
for 576x576/48 frames, **49.3 GiB** for 768x768/80 frames. A 48 GB card has
about 44 GiB usable, so it cannot run the default 80-frame clip; 80 GB is the
supported configuration. `doctor` warns below 70 GiB and fails below 40 GiB
(the weights alone will not fit).

Host RAM is what holds the offloaded pieces between clips: the DiT (32.6 GB)
and the text encoder (11.4 GB) are both resident at once, so 64 GB is the real
minimum. With `--no-offload` on an 80 GB card, RAM drops to ~32 GB.

The image is CUDA 12.8 on Ubuntu 24.04, Python 3.12. flash-attention is a
prebuilt wheel (`cu12`, torch 2.8, cp312); nothing compiles at build time.
Weights are **not** baked in — they are 45.6 GB and live on the `/models`
volume.

## Quick start (this machine, Docker + GPU)

```bash
cp .env.example .env              # set WAN_API_KEY
./scripts/run.sh build
./scripts/run.sh download         # 45.6 GB, once (public repo; no token needed)
./scripts/run.sh                  # doctor
mkdir -p performances/shot-01
cp /path/to/start.jpg /path/to/audio.wav performances/shot-01/
$EDITOR performances/shot-01/prompt.txt
./scripts/run.sh plan    /performances/shot-01
./scripts/run.sh perform /performances/shot-01
# -> performances/shot-01/video.mp4 + manifest.json + render.log
```

`perform` writes into the performance folder unless `-o` is given, so a shot
folder is self-contained: inputs, video and provenance together.

## Quick start (GHCR image, no local build)

```bash
docker run --gpus all --rm -p 8000:8000 \
  -v $PWD/models:/models -v $PWD/performances:/performances -v $PWD/output:/output \
  -e WAN_API_KEY=change-me \
  ghcr.io/josephatwoosh/wan22-s2v-docker:latest
```

The image is built and pushed by
[`.github/workflows/build-image.yml`](.github/workflows/build-image.yml): from
the Actions tab ("Run workflow"), or by pushing a `build-*` tag (the trigger
that works with a contents-only token — `tools/publish.js --tag build-1`).

> GHCR packages start **private**, even when the repository is public. For
> Vast to pull without a registry token, open the package page → *Package
> settings* → *Change visibility* → **Public**. See [docs/VAST.md](docs/VAST.md).

---

## How a render is planned

`plan` is pure arithmetic on the still and the track — no GPU, no weights:

```text
$ wan22-s2v plan performances/shot-01
still        1920x1080 -> render 1280x704 (921600 px area)
audio        4.990 s
clips        1 x 80 frames
video        80 frames @ 16 fps = 5.000 s
params       seed 1293847561, 40 steps, guidance 4.5, shift 3.0, start_from_ref=True
```

Four numbers matter, and each comes from the model rather than a guess:

| Number | Where it comes from |
|---|---|
| **resolution** | Wan's `get_size_less_than_area()`: scale down (never up) to the preset's area, then pad to Wan's 64-pixel grid. A 16:9 still at `720p` renders **1280x704**, not 1280x720. |
| **fps** | 16, from `shared_config.sample_fps`. |
| **clip length** | `INFER_FRAMES` frames per clip (80 = 5 s). |
| **clip count** | `ceil(audio_seconds * 16 / INFER_FRAMES)` — the smallest count that covers the track. |

Upstream computes `int(audio_frames / batch_frames) + 1`, which asks for **two**
clips at exactly 5.000 s: 10 s of generation for 5 s of sound. This renderer
passes the smaller count as `--num_clip`, so the boundary case costs one clip.
The saving is one clip, not a quality trade — the video already covered the
audio.

`START_FROM_REF=1` (the default) makes the still frame 0, which is what a shot
needs, and keeps every generated frame. Without it, upstream drops the first
three frames of clip one, so a 5.0 s line renders 4.8125 s of picture and
ffmpeg's `-shortest` trims the last 0.1875 s of sound. That is why it is on.

### What is checked before "done"

The renderer never trusts the upstream exit code alone:

1. `generate.py` must exit 0 **and** write a non-empty file.
2. The picture is re-muxed onto the track with `-shortest` (same cut as
   upstream, but it raises on failure instead of logging one).
3. `ffprobe` confirms a video stream, an audio stream, the planned
   width/height, the expected frame rate, and a duration within 0.25 s of
   `min(video, audio)`.
4. `manifest.json` records input hashes, the resolved plan, the seed, the
   command, timings and environment.

A clip that lost its soundtrack is the failure mode worth engineering against:
Wan's `merge_video_audio()` catches its own ffmpeg error and only logs it.

---

## The contract, in detail

| File | Rules |
|---|---|
| `start.*` | Exactly one image: `.jpg`, `.jpeg`, `.png`, `.webp`. One still, because the model takes one reference. EXIF orientation is ignored by upstream too, so rotate assets before rendering. Small images are never upscaled — `plan` warns. |
| `audio.*` | Exactly one track: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aiff`. 16 kHz+ mono is plenty; the encoder resamples to 16 kHz. Speech with no music bed and no overlap is what the lip-sync is trained on. |
| `prompt.txt` | UTF-8, one paragraph, whitespace collapsed. 20–4000 characters. Name the place, light and props first, then the action, then the speaker's intent — the model builds the world from nouns before it places the cast. |
| `performance.json` | *Optional.* Render overrides for this folder only: `seed`, `size`, `infer_frames`, `max_clips`, `sample_steps`, `guide_scale`, `shift`, `start_from_ref`, `conform_fps`. Unknown keys are an error, not silently ignored. |

A folder with two images is rejected with both names: "keep exactly one — the
first frame". That rule is what keeps a shot's provenance unambiguous.

Prompt craft, audio notes and the "why is the acting like that" checklist are
in [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

---

## CLI

```text
wan22-s2v doctor                       preflight: GPU, VRAM, libraries, weights, ffmpeg
wan22-s2v download                     fetch the 45.6 GB checkpoint
wan22-s2v selftest                     synthetic 3 s clip: proves the GPU stack works
wan22-s2v plan    <performance>         resolution, clips, duration (no GPU)
wan22-s2v perform <performance>         render one, into the folder or -o
wan22-s2v batch   <root>                render every performance under a folder
wan22-s2v serve                        HTTP API on :8000
wan22-s2v version
```

Render flags are shared by `plan`, `perform` and `batch`:

```text
--size {720p,704p,480p}   render area preset (aspect follows the still)
--infer-frames N          frames per clip, multiple of 4, default 80
--max-clips N             preview: cap the clip count and cut the tail
--seed N                  -1 = pick and record
--steps N                 sampling steps, default 40
--guide-scale F           default 4.5
--shift F                 default 3.0
--conform-fps N           retime 16 fps to N by duplicating frames
--no-start-from-ref       let the model invent frame 0 (not recommended for shots)
--no-offload              only with enough VRAM to hold the whole DiT
```

`batch` writes `out/batch.csv` (name, status, duration, render time, seed) and
skips folders whose manifest still matches their inputs. `--dry-run` plans
without rendering; `--limit` renders the first N.

---

## HTTP API

```bash
curl -s -X POST localhost:8000/v1/performances \
  -H "Authorization: Bearer $KEY" \
  -F start=@start.jpg -F audio=@audio.wav \
  -F prompt="A grey dawn in the farmyard. The cockerel turns to the coop and speaks." \
  -F max_clips=1
# -> 202 {"id": "20261121-064000-a1b2c3", "status": "queued", ...}
```

| Endpoint | Purpose |
|---|---|
| `POST /v1/performances` | multipart upload → job (202) |
| `GET /v1/jobs` | recent jobs |
| `GET /v1/jobs/{id}` | status, plan, manifest when done |
| `GET /v1/jobs/{id}/video` | `video.mp4` (409 until done) |
| `GET /v1/jobs/{id}/manifest.json` | provenance |
| `GET /v1/jobs/{id}/log` | the upstream process log |
| `GET /health` | open: GPU, VRAM, engine readiness, queue depth |

Multipart fields: `start`, `audio` (files), `prompt` (text) or `prompt_file`
(file), plus optional `size`, `seed`, `max_clips`, `start_from_ref`,
`conform_fps`. The upload is written to disk as a performance folder before it
is validated, so a job folder can be downloaded and re-run with the CLI
unchanged.

Authentication is `Authorization: Bearer <WAN_API_KEY>` (or `X-API-Key`). For
`<video src>` URLs, `?token=<key>` is accepted too. `/health` is intentionally
open. Renders are queued one at a time; the API returns 202 and the caller
polls.

Full examples: [examples/curl.md](examples/curl.md).

---

## Connecting to the film pipeline

This repository does not modify the studio; it is the service the studio calls.
The mapping is one shot → one performance:

1. The studio already has a start still per shot and a dialogue WAV per shot
   (the Qwen-TTS service produces those).
2. Write a prompt paragraph per shot from the beat text — the same text that
   drove the still, so picture and performance agree.
3. Upload the three files, poll the job, download `video.mp4` and keep
   `manifest.json` with the shot.

The full design, including the PHP client, the retake policy and where the
16 fps conform belongs, is in [docs/STUDIO.md](docs/STUDIO.md).

---

## Vast.ai

CI builds → GHCR public → Vast template → rent → download weights → render →
destroy. The machine-readable recipe and the cost table are in
[docs/VAST.md](docs/VAST.md); [`tools/vast.js`](tools/vast.js) wraps
search/rent/ssh/scp and enforces a **$4/TB** ceiling on either direction of
internet transfer.

```bash
node tools/vast.js search "A100 SXM4,H100 PCIE" 79000 150
node tools/vast.js rent --offer <id> --image ghcr.io/josephatwoosh/wan22-s2v-docker:latest \
  --disk 150 --env WAN_API_KEY=... --yes
```

Weights download once (~45.6 GB) and are the largest transfer; keep them on a
persistent volume or re-download into the instance disk and budget ~$0.20.

---

## Performance

Measured on **one H100 SXM 80 GB** (driver 615.71.09), 2026-11-21, 40 sampling
steps, guidance 4.5, 16 fps, one clip per job:

| Render | Output | Sampling | Wall | Peak VRAM |
|---|---|---|---|---|
| 480p, 48-frame clip | 576x576, 3.0 s | 4.85 s/step (3:13) | 321 s | **42.6 GiB** |
| 480p, 80-frame clip | 576x576, 3.0 s | 8.8 s/step (5:50) | 475 s | **42.6 GiB** |
| 720p, 80-frame clip | 768x768, 3.0 s | 20.6 s/step (13:44) | 954 s | **49.3 GiB** |

Model load is ~90 s of each job and is paid once per process, which is why
`batch` renders a scene in one process and `perform` is for retakes. Sampling
cost scales with frames x pixels: the 80-frame 720p clip is ~3x the 48-frame
480p one. At $1.56/h that is about **$0.14**, **$0.21** and **$0.41** for the
three rows — the expensive part of a film is still the stills, not the video.

The 720p row was rendered before the mux fix (its file came back 45 frames
instead of 48, which is how the bug was found); its timing and VRAM are still
valid. The 480p/80-frame row is the post-fix run: 48 frames, 3.000 s of picture
and 3.000 s of track.

What to expect from the manifest: `timing.render_seconds` includes model load,
`timing.realtime_factor` is `render_seconds / video_seconds` (106× for the
smallest run), and `params.seed` is the seed that reproduces it.

Caveats: one card, one synthetic still, 40 steps. No claim is made for 720P
1280x704 on a 16:9 still (that is ~1.5× the pixels of the 768x768 run above, so
expect ~60 GiB peak), for multi-GPU, or for `--no-offload`.

## Troubleshooting

**`doctor` says nvidia-smi is missing.** Start the container with `--gpus all`
(or `deploy.resources` in compose) and make sure the host driver is 570+.

**OOM at 720p on a 48 GB card.** Use `--size 480p` and fewer clips
(`--max-clips 1` for a test), keep `OFFLOAD_MODEL=1`, and do not pass
`--no-offload`. The DiT is 32.6 GB before activations, and the text encoder is
on the card with it during prompt encoding.

**`flash_attn` import fails.** The image ships a prebuilt wheel for cp312 +
torch 2.8 + cu12. A rebuilt image with a different torch needs a matching
wheel; `doctor` names the import.

**The clip has no sound.** It cannot: the renderer verifies the audio stream and
fails the job otherwise. If ffmpeg is missing, `doctor` says so before the
render.

**Audio is cut at the end.** That is `-shortest`: the performance ends with the
track. If the cut is meant to be longer than the audio, generate silence to the
required length — do not stretch the render.

**The acting looks stiff or rubbery.** See the checklist in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md): usually a music bed, overlapping
speakers, or a prompt that describes a pose instead of an intention.

**`plan` warns the still is small.** Wan does not upscale references. Render
from a bigger still or accept the smaller frame.

**`generate.py` exits 1 with `No module named 'dashscope'`.** `generate.py`
imports `wan.utils.prompt_extend` at module load, and that imports `dashscope`
even though this service never uses prompt extension. The image pins it; a
custom build that trims requirements will break here.

**The clip ends a frame or two before the audio.** Fixed in 0.1.1: upstream
muxes its own raw video with `-shortest` before this service sees it, which
dropped the last frames whenever the picture outlasted the track. The renderer
now hands the model a silence-padded copy and cuts the final mux with an
explicit duration and frame count.

## Licensing

Code: MIT ([LICENSE](LICENSE)). Weights: Apache-2.0, downloaded from
Hugging Face, not redistributed here. Upstream inference: Apache-2.0
([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)). Generated video is
yours, subject to those licences and to the consent of anyone depicted.
