"""Assemble performance folders for a scene from clips, speaker turns and prompts.

Input layout (the one this repository's film work uses):

    clips/            <shot>.mp4, one per shot (frame 0 becomes start.png)
    turns/            speaker turns, sorted, concatenated in order
    prompts/          <shot>/prompt.txt, plus manifest.csv (shot,duration_s[,start_s])

Output: one folder per shot — start.png, audio.wav, prompt.txt — ready for
`wan22-s2v perform` or the HTTP API.

The soundtrack is the turns concatenated with a short gap, padded to the scene
length, then sliced at the shot boundaries: no music, no effects.

Usage:
    python3 tools/build_scene_pack.py \
        --clips clips --turns turns --prompts prompts \
        --timings prompts/manifest.csv --out /scene01
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import shutil
import subprocess
import wave

RATE = 24000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clips", required=True, help="directory of <shot>.mp4")
    parser.add_argument("--turns", required=True, help="directory of speaker-turn WAVs")
    parser.add_argument("--prompts", required=True, help="directory of <shot>/prompt.txt + manifest.csv")
    parser.add_argument("--out", required=True, help="output root for performance folders")
    parser.add_argument("--timings", default=None, help="manifest.csv (default: <prompts>/manifest.csv)")
    parser.add_argument("--gap", type=float, default=0.45, help="silence between turns when packing (s)")
    parser.add_argument(
        "--starts",
        default=None,
        help="CSV of planned start_s per turn (one per turn, filename order). "
        "With this, each turn is anchored at its planned start and delayed on "
        "collision instead of being packed back-to-back; a shot then carries only "
        "the line that belongs in it.",
    )
    parser.add_argument("--scene-seconds", type=float, default=None, help="pad to this length (default: sum of shots)")
    parser.add_argument("--rate", type=int, default=RATE)
    return parser.parse_args()


def read_pcm(path: pathlib.Path, rate: int) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != rate or handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise SystemExit(f"{path.name}: expected mono 16-bit {rate} Hz")
        return handle.readframes(handle.getnframes())


def shot_rows(timings: pathlib.Path) -> list[dict[str, str]]:
    with timings.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    start = 0.0
    for row in rows:
        row.setdefault("start_s", "")
        if not row["start_s"]:
            row["start_s"] = f"{start:.3f}"
        start = float(row["start_s"]) + float(row["duration_s"])
    return rows


def frame_zero(clip: pathlib.Path, destination: pathlib.Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(clip), "-frames:v", "1", str(destination)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    clips = pathlib.Path(args.clips)
    turns = pathlib.Path(args.turns)
    prompts = pathlib.Path(args.prompts)
    out = pathlib.Path(args.out)
    timings = pathlib.Path(args.timings) if args.timings else prompts / "manifest.csv"
    rows = shot_rows(timings)
    if not rows:
        raise SystemExit(f"{timings}: no shots")
    scene_seconds = args.scene_seconds or sum(float(row["duration_s"]) for row in rows)

    gap = b"\x00\x00" * int(args.gap * args.rate)
    turn_files = sorted(turns.glob("*.wav"))
    planned: list[float] | None = None
    if args.starts:
        with open(args.starts, encoding="utf-8") as handle:
            rows_in = list(csv.reader(handle))
        planned = []
        for row in rows_in:
            if not row or not row[0].strip() or row[0].strip().lower() == "start_s":
                continue
            planned.append(float(row[0]))
        if len(planned) != len(turn_files):
            raise SystemExit(f"{args.starts}: {len(planned)} starts for {len(turn_files)} turns")

    # Two ways to lay the dialogue down:
    #  * packing (default): turns back-to-back with a gap. Simple, but the
    #    dialogue drifts earlier than the cut and a shot can inherit the start
    #    of the next line (shot 1 carried Mildred's narration).
    #  * anchored (--starts): each turn starts at its planned time, delayed only
    #    if the previous line is still running. The picture's line timings are
    #    respected and no shot carries a line that belongs to a later shot.
    parts: list[bytes] = []
    position = 0.0
    for index, turn in enumerate(turn_files):
        pcm = read_pcm(turn, args.rate)
        if planned is not None:
            start = max(planned[index], position)
        else:
            start = 0.0 if index == 0 else position + args.gap
        silence = start - position
        if silence > 0:
            parts.append(b"\x00\x00" * int(round(silence * args.rate)))
        parts.append(pcm)
        position = start + len(pcm) / 2 / args.rate
    track = b"".join(parts)
    target = int(round(scene_seconds * args.rate)) * 2
    if len(track) > target:
        raise SystemExit(f"soundtrack {len(track) / 2 / args.rate:.2f}s exceeds scene {scene_seconds:.2f}s")
    track += b"\x00\x00" * ((target - len(track)) // 2)

    out.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out / "soundtrack.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(args.rate)
        handle.writeframes(track)
    print(f"soundtrack {len(track) / 2 / args.rate:.2f}s from {len(turn_files)} turns")

    for row in rows:
        shot = row["shot"]
        start = float(row["start_s"])
        duration = float(row["duration_s"])
        folder = out / shot
        folder.mkdir(exist_ok=True)
        begin = int(round(start * args.rate)) * 2
        end = begin + int(round(duration * args.rate)) * 2
        with wave.open(str(folder / "audio.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(args.rate)
            handle.writeframes(track[begin:end])
        shutil.copyfile(prompts / shot / "prompt.txt", folder / "prompt.txt")
        frame_zero(clips / f"{shot}.mp4", folder / "start.png")
        print(f"{shot}: {duration:5.1f}s audio")


if __name__ == "__main__":
    main()
