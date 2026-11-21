# Rendering a whole scene on a rented GPU

Three small pieces turn a scene's clips, speaker turns and prompts into one
`scene.mp4`:

| File | Runs on | What it does |
|---|---|---|
| [`build_scene_pack.py`](build_scene_pack.py) | the GPU box | builds one performance folder per shot: frame 0 → `start.png`, sliced soundtrack → `audio.wav`, prompt copied in |
| [`scene_run.sh`](scene_run.sh) | the GPU box | renders every folder with `wan22-s2v perform`, skips finished shots, then concatenates to `scene.mp4` |
| [`run_scene.ps1`](run_scene.ps1) | your workstation | packs the sources, uploads, runs both, downloads the mp4 |

## One command

```powershell
pwsh tools/run_scene.ps1 -SshHost ssh8.vast.ai -SshPort 16096
```

Defaults match the film work in this workspace: `film/ltx-scene01/clips`,
`film/ltx-scene01/audio-scene01`, `scene01-pack/`, 116.1 s, **480p, 20 steps,
48-frame clips, 24 fps, seed 42**. Override any of them:

```powershell
pwsh tools/run_scene.ps1 -SshHost ssh8.vast.ai -SshPort 16096 `
  -Size 720p -Steps 30 -InferFrames 80 -OutFile .tmp/scene01-720p.mp4
```

## What it assumes

- The instance runs `ghcr.io/josephatwoosh/wan22-s2v-docker` and SSH works.
- Weights are on the box: `wan22-s2v download` (45.6 GB).
- The clips and turns use the same shot names (`sc01_001` …). Prompts are one
  `prompt.txt` per shot plus `manifest.csv` (`shot,duration_s[,start_s]`).

## The soundtrack

`build_scene_pack.py` concatenates the speaker turns in filename order with a
0.45 s gap, pads to the scene length, writes one `soundtrack.wav`, then slices
it at the shot boundaries. No music, no effects. If a line spans a cut, both
shots get their half of it, which is what the edit does anyway.

## VRAM and the clip length

`InferFrames` is the knob that decides whether the card fits:

| Card | `InferFrames` | Note |
|---|---|---|
| 80 GB (A100/H100) | 80 (5 s) | upstream's default, least over-generation |
| 48 GB (L40S/A6000) | **48 (3 s)** | measured peak ~39–43 GiB; 80-frame clips can exceed 48 GB |

Steps trade time for quality roughly linearly. Measured on an L40S: **1 s of
finished 480p at 20 steps ≈ 0.6–0.7 min of GPU time**, so a 116 s scene is
4–5 hours.

## Resuming

`scene_run.sh` skips any shot that already has `video.mp4`, so after a dropped
connection or a re-rent you run the same command again and it continues. It
refuses to assemble unless **every** shot exists — a partial scene is a bug, not
a deliverable.

## Verifying the result

Assembly prints `ffprobe` for the finished file: duration, resolution, frame
rate and stream count. Each shot is already verified individually (audio stream
present, exact frame count, planned resolution, 24 fps), so a wrong-shaped file
at this point means the concat, not the renderer.
