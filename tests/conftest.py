"""Shared fixtures: a real performance folder, built byte by byte.

The tests never need a GPU, ffmpeg or the model weights. Images come from
Pillow, audio from the stdlib ``wave`` module, and renders use the mock engine.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings, get_settings, reset_settings  # noqa: E402


def write_png(path: Path, width: int = 1920, height: int = 1080, colour=(120, 90, 60)) -> Path:
    from PIL import Image

    Image.new("RGB", (width, height), colour).save(path)
    return path


def write_wav(path: Path, seconds: float = 4.99, rate: int = 16000) -> Path:
    frames = max(1, int(round(seconds * rate)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


def build_performance(
    directory: Path,
    *,
    prompt: str = "A grey dawn in the farmyard. The cockerel turns to the coop and speaks.",
    seconds: float = 4.99,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    write_png(directory / "start.png", width, height)
    write_wav(directory / "audio.wav", seconds)
    (directory / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    return directory


@pytest.fixture
def performance(tmp_path: Path) -> Path:
    return build_performance(tmp_path / "shot-001")


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Settings pointed at a temp tree, with the mock engine and no auth."""
    monkeypatch.setenv("ENGINE", "mock")
    monkeypatch.setenv("WAN_API_KEY", "")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("MODEL_ID", str(tmp_path / "model"))
    monkeypatch.setenv("WAN_DIR", str(tmp_path / "wan"))
    monkeypatch.setenv("SIZE", "480p")
    monkeypatch.setenv("FFPROBE", "ffprobe-not-installed")
    monkeypatch.setenv("FFMPEG", "ffmpeg-not-installed")
    reset_settings()
    settings = get_settings()
    (tmp_path / "model").mkdir(exist_ok=True)
    (tmp_path / "wan").mkdir(exist_ok=True)
    yield settings
    reset_settings()
