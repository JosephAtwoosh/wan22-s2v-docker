#!/usr/bin/env bash
# Render a scene's performance folders and assemble them, on a rented GPU.
#
#   tools/scene_run.sh /scene01 /scene01-out
#
# Expects one folder per shot (start.png + audio.wav + prompt.txt), as produced
# by tools/build_scene_pack.py. Skips shots that already have a video, so a
# dropped connection or a re-rent can resume.
#
# Env knobs: SIZE (480p) STEPS (20) INFER_FRAMES (48) FPS (24) SEED (42)
#            MODEL_ID, OUT_NAME (scene.mp4)
set -euo pipefail

ROOT="${1:-/scene01}"
OUT="${2:-/scene01-out}"
SIZE="${SIZE:-480p}"
STEPS="${STEPS:-20}"
INFER_FRAMES="${INFER_FRAMES:-48}"
FPS="${FPS:-24}"
SEED="${SEED:-42}"
MODEL_ID="${MODEL_ID:-/models/Wan2.2-S2V-14B}"
OUT_NAME="${OUT_NAME:-scene.mp4}"

command -v wan22-s2v >/dev/null || { echo "error: wan22-s2v not on PATH (run inside the image)" >&2; exit 1; }
if [ ! -f "$MODEL_ID/diffusion_pytorch_model.safetensors.index.json" ]; then
  echo "error: weights not found under $MODEL_ID; run: wan22-s2v download" >&2
  exit 1
fi

mkdir -p "$OUT"
shots=0
for folder in "$ROOT"/sc01_*; do
  [ -d "$folder" ] || continue
  name=$(basename "$folder")
  if [ -f "$OUT/$name/video.mp4" ]; then
    echo "skip $name (already rendered)"
    shots=$((shots + 1))
    continue
  fi
  echo "=== $name $(date -u +%H:%M:%S)"
  wan22-s2v perform "$folder" \
    --size "$SIZE" --infer-frames "$INFER_FRAMES" --steps "$STEPS" \
    --seed "$SEED" --conform-fps "$FPS" -o "$OUT/$name" --force
  shots=$((shots + 1))
done
if [ "$shots" -eq 0 ]; then
  echo "error: no shot folders under $ROOT" >&2
  exit 1
fi

list="$OUT/concat.txt"
: > "$list"
for folder in "$ROOT"/sc01_*; do
  [ -d "$folder" ] || continue
  name=$(basename "$folder")
  file="$OUT/$name/video.mp4"
  if [ ! -f "$file" ]; then
    echo "error: $name has no video; refusing to assemble an incomplete scene" >&2
    exit 1
  fi
  echo "file '$file'" >> "$list"
done

echo "=== assembling $OUT/$OUT_NAME"
ffmpeg -y -v error -f concat -safe 0 -i "$list" -c copy -movflags +faststart "$OUT/$OUT_NAME"
ffprobe -v error -show_entries format=duration,size:stream=codec_type,width,height,avg_frame_rate,nb_frames \
  -of default=nw=1 "$OUT/$OUT_NAME"
echo "done: $OUT/$OUT_NAME"
