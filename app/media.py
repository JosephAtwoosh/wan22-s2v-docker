"""ffprobe/ffmpeg helpers.

Wan's own ``merge_video_audio()`` catches its ffmpeg failure and only logs it,
which can leave a silent mp4 behind. Everything here raises instead, so a clip
that lost its soundtrack cannot be mistaken for a finished one.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    """ffmpeg or ffprobe refused, or the file is not what it claims to be."""


@dataclass(frozen=True)
class VideoInfo:
    duration: float | None
    width: int | None
    height: int | None
    fps: float | None
    has_video: bool
    has_audio: bool
    video_codec: str | None
    audio_codec: str | None
    bytes: int

    def as_dict(self) -> dict:
        return {
            "duration": None if self.duration is None else round(self.duration, 3),
            "width": self.width,
            "height": self.height,
            "fps": None if self.fps is None else round(self.fps, 3),
            "video": self.has_video,
            "audio": self.has_audio,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "bytes": self.bytes,
        }


def _fraction(value: str | None) -> float | None:
    if not value:
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_f = float(denominator)
            return float(numerator) / denominator_f if denominator_f else None
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def probe(path: str | Path, ffprobe: str = "ffprobe") -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"{path} does not exist")
    command = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except FileNotFoundError as exc:
        raise MediaError(f"{ffprobe} not found; install ffmpeg") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffprobe timed out on {path.name}") from exc

    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()[:3]
        raise MediaError(f"ffprobe failed on {path.name}: {'; '.join(detail)}")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {path.name}") from exc

    streams = payload.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    duration = None
    raw_duration = (payload.get("format") or {}).get("duration")
    try:
        duration = float(raw_duration) if raw_duration is not None else None
    except ValueError:
        duration = None

    return VideoInfo(
        duration=duration,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        fps=_fraction(video.get("r_frame_rate")) if video else None,
        has_video=video is not None,
        has_audio=audio is not None,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
        bytes=path.stat().st_size,
    )


def _run(command: list[str], *, timeout: int = 3600) -> None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise MediaError(f"{command[0]} not found; install ffmpeg") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{command[0]} timed out after {timeout}s") from exc
    if result.returncode != 0:
        detail = (result.stderr or "").strip().splitlines()[-4:]
        raise MediaError(f"{command[0]} failed: {' | '.join(detail)}")


def mux_audio(
    video: str | Path,
    audio: str | Path,
    destination: str | Path | None = None,
    *,
    duration: float | None = None,
    frames: int | None = None,
    ffmpeg: str = "ffmpeg",
    timeout: int = 3600,
) -> Path:
    """Mux the track onto the picture and end both at the same instant.

    With ``duration``/``frames`` the cut is explicit. ``-t`` alone is not
    enough for a copied stream: B-frames make DTS trail PTS, so packets past
    the cut are still written (measured: 50 frames for a 3.000 s cut) and
    ``-shortest`` is worse — it truncates by the shortest stream's packet end
    and drops the last frames even when both streams nominally agree.
    ``-frames:v`` caps the picture exactly; ``-t`` trims the re-encoded audio.
    """
    video = Path(video)
    audio = Path(audio)
    destination = Path(destination) if destination else video
    temp = destination.with_name(destination.stem + ".muxing" + destination.suffix)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ]
    if frames is not None:
        command += ["-frames:v", str(int(frames))]
    if duration is not None:
        command += ["-t", f"{duration:.3f}"]
    elif frames is None:
        command.append("-shortest")
    command += ["-movflags", "+faststart", str(temp)]
    _run(command, timeout=timeout)
    os.replace(temp, destination)
    return destination


def pad_audio(
    audio: str | Path,
    destination: str | Path,
    seconds: float,
    *,
    ffmpeg: str = "ffmpeg",
    timeout: int = 600,
) -> Path:
    """Copy the track with silence appended to ``seconds``.

    The model is handed this copy so that upstream's own ``merge_video_audio``
    (``-shortest``) never cuts the generated picture: the padded track is
    longer than the video, so nothing is trimmed there, and the real track is
    muxed afterwards with an explicit ``-t``. The tail the model generates on
    silence is discarded by that cut.
    """
    audio = Path(audio)
    destination = Path(destination)
    temp = destination.with_name(destination.stem + ".padding" + destination.suffix)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(audio),
        "-af",
        "apad",
        "-t",
        f"{seconds:.3f}",
        "-c:a",
        "pcm_s16le",
        str(temp),
    ]
    _run(command, timeout=timeout)
    os.replace(temp, destination)
    return destination


def conform_fps(
    video: str | Path,
    fps: int,
    destination: str | Path | None = None,
    *,
    ffmpeg: str = "ffmpeg",
    timeout: int = 3600,
) -> Path:
    """Retime a 16 fps clip to an edit's frame rate by duplicating frames.

    Wan2.2-S2V samples at 16 fps. Cutting a 24 fps timeline against it otherwise
    means a timeline-wide conform; this makes the clip natively 24 fps without
    changing its duration or its sync.
    """
    video = Path(video)
    destination = Path(destination) if destination else video
    temp = destination.with_name(destination.stem + ".fps" + destination.suffix)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={int(fps)}",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    _run(command, timeout=timeout)
    os.replace(temp, destination)
    return destination
