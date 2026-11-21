"""The three-file contract: discovery, prompt handling, overrides."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.performance import (
    PerformanceError,
    copy_into,
    discover,
    looks_like_performance,
    normalise_prompt,
    probe_audio_seconds,
    probe_image_size,
    sha256_file,
)
from tests.conftest import build_performance, write_png, write_wav


def test_discovers_a_complete_folder(performance: Path):
    found = discover(performance)
    assert found.image.name == "start.png"
    assert found.audio.name == "audio.wav"
    assert found.prompt_path.name == "prompt.txt"
    assert found.prompt.startswith("A grey dawn in the farmyard.")
    assert found.overrides == {}


def test_allows_any_supported_extension(tmp_path: Path):
    folder = tmp_path / "shot"
    folder.mkdir()
    write_png(folder / "frame.jpeg", 1280, 720)
    write_wav(folder / "line.WAV", 2.0)
    (folder / "prompt.txt").write_text("A cat on a wall.", encoding="utf-8")
    found = discover(folder)
    assert found.image.name == "frame.jpeg"
    assert found.audio.name == "line.WAV"


def test_rejects_two_images(tmp_path: Path):
    folder = build_performance(tmp_path / "shot")
    write_png(folder / "stop.jpg", 1920, 1080)
    with pytest.raises(PerformanceError, match="2 images"):
        discover(folder)


def test_rejects_missing_audio(tmp_path: Path):
    folder = tmp_path / "shot"
    folder.mkdir()
    write_png(folder / "start.png")
    (folder / "prompt.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(PerformanceError, match="no audio"):
        discover(folder)


def test_rejects_missing_prompt(tmp_path: Path):
    folder = tmp_path / "shot"
    folder.mkdir()
    write_png(folder / "start.png")
    write_wav(folder / "audio.wav")
    with pytest.raises(PerformanceError, match="prompt.txt"):
        discover(folder)


def test_rejects_empty_prompt(tmp_path: Path):
    folder = build_performance(tmp_path / "shot", prompt="   \n\n ")
    with pytest.raises(PerformanceError, match="empty"):
        discover(folder)


def test_rejects_overlong_prompt(tmp_path: Path):
    folder = build_performance(tmp_path / "shot", prompt="word " * 900)
    with pytest.raises(PerformanceError, match="characters"):
        discover(folder, max_prompt_chars=1000)


def test_normalises_whitespace_and_crlf():
    assert normalise_prompt("\ufeffA  grey\r\ndawn\r\nthe   yard \n") == "A grey dawn the yard"


def test_unknown_override_is_an_error(tmp_path: Path):
    folder = build_performance(tmp_path / "shot")
    (folder / "performance.json").write_text(json.dumps({"end_image": "stop.jpg"}), encoding="utf-8")
    with pytest.raises(PerformanceError, match="unknown key"):
        discover(folder)


def test_override_type_is_checked(tmp_path: Path):
    folder = build_performance(tmp_path / "shot")
    (folder / "performance.json").write_text(json.dumps({"infer_frames": "80"}), encoding="utf-8")
    with pytest.raises(PerformanceError, match="infer_frames"):
        discover(folder)


def test_boolean_is_not_an_integer_override(tmp_path: Path):
    folder = build_performance(tmp_path / "shot")
    (folder / "performance.json").write_text(json.dumps({"seed": True}), encoding="utf-8")
    with pytest.raises(PerformanceError, match="seed"):
        discover(folder)


def test_override_is_read(tmp_path: Path):
    folder = build_performance(tmp_path / "shot")
    (folder / "performance.json").write_text(
        json.dumps({"seed": 7, "size": "480p", "_note": "commented"}), encoding="utf-8"
    )
    assert discover(folder).overrides == {"seed": 7, "size": "480p"}


def test_looks_like_performance(performance: Path, tmp_path: Path):
    assert looks_like_performance(performance) is True
    assert looks_like_performance(tmp_path) is False
    empty = tmp_path / "empty"
    empty.mkdir()
    assert looks_like_performance(empty) is False


def test_copy_into_creates_a_valid_folder(performance: Path, tmp_path: Path):
    found = discover(performance)
    destination = tmp_path / "job"
    copied = copy_into(found, destination)
    assert copied.image.name == "start.png"
    assert copied.prompt == found.prompt
    again = discover(destination)
    assert again.prompt == found.prompt
    assert sha256_file(again.image) == sha256_file(found.image)


def test_copy_into_keeps_file_hashes(tmp_path: Path):
    folder = tmp_path / "shot"
    folder.mkdir()
    write_png(folder / "still.webp")
    write_wav(folder / "track.wav")
    (folder / "prompt.txt").write_text("prompt", encoding="utf-8")
    found = discover(folder)
    copied = copy_into(found, tmp_path / "job")
    assert sha256_file(copied.audio) == sha256_file(found.audio)


def test_probe_image_size(performance: Path):
    assert probe_image_size(performance / "start.png") == (1080, 1920)


def test_probe_image_size_rejects_tiny_images(tmp_path: Path):
    tiny = write_png(tmp_path / "tiny.png", 32, 32)
    with pytest.raises(PerformanceError, match="too small"):
        probe_image_size(tiny)


def test_probe_audio_duration_falls_back_to_wave(performance: Path):
    assert probe_audio_seconds(performance / "audio.wav", "definitely-not-ffprobe") == pytest.approx(
        4.99, abs=0.01
    )


def test_sha256_changes_with_content(tmp_path: Path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    assert sha256_file(first) != sha256_file(second)


def test_prompt_is_read_from_any_case_of_filename(tmp_path: Path):
    folder = tmp_path / "shot"
    folder.mkdir()
    write_png(folder / "start.png")
    write_wav(folder / "audio.wav")
    (folder / "PROMPT.TXT").write_text("shout it", encoding="utf-8")
    assert discover(folder).prompt == "shout it"


def test_directory_with_subfolders_is_ignored(tmp_path: Path):
    folder = tmp_path / "shot"
    folder.mkdir()
    write_png(folder / "start.png")
    write_wav(folder / "audio.wav")
    (folder / "prompt.txt").write_text("hello", encoding="utf-8")
    (folder / "frames").mkdir()
    write_png(folder / "frames" / "other.png", 320, 240)
    assert discover(folder).image.name == "start.png"


def test_copy_into_does_not_move_the_source(performance: Path, tmp_path: Path):
    found = discover(performance)
    copy_into(found, tmp_path / "job")
    assert found.image.is_file()
    assert found.audio.is_file()
