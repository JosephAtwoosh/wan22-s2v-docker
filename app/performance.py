"""The performance contract: one folder, three files.

    <performance>/
        start.jpg     the first frame; the only visual reference
        audio.wav     drives lips, gaze, head motion, gestures and timing
        prompt.txt    the scene, in one paragraph (UTF-8)

There is no ``stop.jpg`` on purpose. Pinning an end pose is what makes
image-to-video acting look rubbery: the model has to arrive at a shape it was
given rather than finish the performance the audio asked for. Wan2.2-S2V drives
the whole clip from the track, so the end pose is an outcome.

Optional ``performance.json`` may override render parameters for this folder
(see :data:`ALLOWED_OVERRIDES`); its fields are listed in docs/PERFORMANCE.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aiff", ".aif")
PROMPT_NAME = "prompt.txt"
SPEC_NAME = "performance.json"

CORE_INPUTS = ("start", "audio", "prompt")

# performance.json may only carry render parameters, never new inputs: the
# three files stay the contract.
ALLOWED_OVERRIDES: dict[str, type] = {
    "seed": int,
    "size": str,
    "infer_frames": int,
    "max_clips": int,
    "sample_steps": int,
    "guide_scale": float,
    "shift": float,
    "start_from_ref": bool,
    "conform_fps": int,
}


class PerformanceError(ValueError):
    """A performance folder does not satisfy the contract."""


@dataclass(frozen=True)
class Performance:
    directory: Path
    image: Path
    audio: Path
    prompt_path: Path
    prompt: str
    overrides: dict[str, Any] = field(default_factory=dict)
    spec_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "image": self.image.name,
            "audio": self.audio.name,
            "prompt_file": self.prompt_path.name,
            "prompt_chars": len(self.prompt),
            "overrides": dict(self.overrides),
        }


def _media_files(directory: Path, extensions: tuple[str, ...]) -> list[Path]:
    found = [
        entry
        for entry in directory.iterdir()
        if entry.is_file()
        and not entry.name.startswith(".")
        and entry.suffix.lower() in extensions
    ]
    return sorted(found, key=lambda path: path.name.lower())


def normalise_prompt(raw: str) -> str:
    """Collapse a prompt file to the single paragraph the encoder expects.

    Text files arrive with a trailing newline, CRLF, or hard-wrapped lines from
    an editor. UMT5 sees the collapse; keeping the newlines would waste tokens
    and, worse, change the prompt between two renders of the same folder.
    """
    text = raw.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    words = " ".join(part for part in text.split())
    return words.strip()


def _read_spec(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PerformanceError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PerformanceError(f"{path.name} must contain a JSON object")
    spec: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue  # comments, e.g. "_note"
        expected = ALLOWED_OVERRIDES.get(key)
        if expected is None:
            allowed = ", ".join(sorted(ALLOWED_OVERRIDES))
            raise PerformanceError(
                f"{path.name}: unknown key {key!r} (allowed: {allowed})"
            )
        if isinstance(value, bool) and expected is int:
            raise PerformanceError(f"{path.name}: {key!r} must be an integer, not a boolean")
        if not isinstance(value, expected):
            raise PerformanceError(
                f"{path.name}: {key!r} must be {expected.__name__}, got {type(value).__name__}"
            )
        spec[key] = value
    return spec


def discover(directory: str | Path, *, max_prompt_chars: int = 4000) -> Performance:
    """Validate a performance folder and read the prompt."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PerformanceError(f"{directory} is not a directory")

    images = _media_files(directory, IMAGE_EXTENSIONS)
    if not images:
        raise PerformanceError(
            f"{directory}: no start image. Expected one of {', '.join(IMAGE_EXTENSIONS)}"
        )
    if len(images) > 1:
        names = ", ".join(image.name for image in images)
        raise PerformanceError(
            f"{directory}: {len(images)} images ({names}). Keep exactly one: the first frame. "
            "The model takes a single reference still, and a folder with two is ambiguous."
        )

    audios = _media_files(directory, AUDIO_EXTENSIONS)
    if not audios:
        raise PerformanceError(
            f"{directory}: no audio. Expected one of {', '.join(AUDIO_EXTENSIONS)}"
        )
    if len(audios) > 1:
        names = ", ".join(audio.name for audio in audios)
        raise PerformanceError(
            f"{directory}: {len(audios)} audio files ({names}). Keep exactly one track."
        )

    prompt_path = directory / PROMPT_NAME
    if not prompt_path.is_file():
        matches = [entry for entry in directory.iterdir() if entry.name.lower() == PROMPT_NAME]
        if not matches:
            raise PerformanceError(f"{directory}: no {PROMPT_NAME}")
        prompt_path = matches[0]

    try:
        raw = prompt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise PerformanceError(f"{prompt_path} is not UTF-8: {exc}") from exc
    prompt = normalise_prompt(raw)
    if not prompt:
        raise PerformanceError(f"{prompt_path} is empty")
    if len(prompt) > max_prompt_chars:
        raise PerformanceError(
            f"{prompt_path} is {len(prompt)} characters; the limit is {max_prompt_chars}. "
            "A scene paragraph is enough — Wan's text encoder is 512 tokens."
        )

    spec_path = directory / SPEC_NAME
    overrides = _read_spec(spec_path) if spec_path.is_file() else {}

    return Performance(
        directory=directory,
        image=images[0],
        audio=audios[0],
        prompt_path=prompt_path,
        prompt=prompt,
        overrides=overrides,
        spec_path=spec_path if spec_path.is_file() else None,
    )


def looks_like_performance(directory: Path) -> bool:
    """Cheap check used by batch scanning: one image, one audio, a prompt."""
    try:
        if not directory.is_dir():
            return False
        images = _media_files(directory, IMAGE_EXTENSIONS)
        audios = _media_files(directory, AUDIO_EXTENSIONS)
        prompt = any(entry.name.lower() == PROMPT_NAME for entry in directory.iterdir())
        return len(images) == 1 and len(audios) == 1 and prompt
    except OSError:
        return False


def has_performance_artifacts(directory: Path) -> bool:
    """A folder that *tries* to be a performance, even if it is incomplete.

    Batch mode reports these as invalid instead of silently skipping them: a
    shot folder with a still and no audio is a mistake worth seeing.
    """
    try:
        if not directory.is_dir():
            return False
        if _media_files(directory, IMAGE_EXTENSIONS) or _media_files(directory, AUDIO_EXTENSIONS):
            return True
        return any(entry.name.lower() == PROMPT_NAME for entry in directory.iterdir())
    except OSError:
        return False


def probe_image_size(path: str | Path) -> tuple[int, int]:
    """(height, width) of the reference still, as Wan will read it.

    No EXIF transpose: upstream does ``Image.open(...).convert("RGB")``, which
    ignores the orientation tag. A phone photo that relies on EXIF will render
    sideways in Wan too, so the plan should say so rather than hide it.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a hard dependency
        raise PerformanceError("Pillow is required to read the start image") from exc
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError as exc:
        raise PerformanceError(f"cannot read image {path}: {exc}") from exc
    if width < 64 or height < 64:
        raise PerformanceError(f"{path} is {width}x{height}; too small for a 64-pixel VAE grid")
    return height, width


def probe_audio_seconds(path: str | Path, ffprobe: str = "ffprobe") -> float:
    """Duration in seconds, via ffprobe (wav fallback when ffprobe is missing)."""
    path = Path(path)
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None

    if result is not None and result.returncode == 0:
        try:
            seconds = float(result.stdout.strip())
        except ValueError:
            seconds = 0.0
        if seconds > 0:
            return seconds

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                if rate > 0 and frames > 0:
                    return frames / float(rate)
        except (wave.Error, EOFError, OSError):
            pass

    detail = (result.stderr or "").strip().splitlines()[:2] if result is not None else []
    suffix = f" ({'; '.join(detail)})" if detail else ""
    raise PerformanceError(f"cannot read audio duration from {path.name}{suffix}")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def copy_into(source: Performance, destination: str | Path) -> Performance:
    """Copy a performance into a job directory under canonical names.

    The API writes uploads this way, so a downloaded job folder is itself a
    valid performance folder: the same contract on disk and over HTTP.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    image = destination / f"start{source.image.suffix.lower()}"
    audio = destination / f"audio{source.audio.suffix.lower()}"
    prompt = destination / PROMPT_NAME
    shutil.copyfile(source.image, image)
    shutil.copyfile(source.audio, audio)
    prompt.write_text(source.prompt + "\n", encoding="utf-8")
    if source.spec_path is not None:
        shutil.copyfile(source.spec_path, destination / SPEC_NAME)
    return Performance(
        directory=destination,
        image=image,
        audio=audio,
        prompt_path=prompt,
        prompt=source.prompt,
        overrides=dict(source.overrides),
        spec_path=destination / SPEC_NAME if source.spec_path else None,
    )
