"""ffmpeg command construction: the cut must be explicit, the pad must exist.

The bug these tests pin: upstream's `merge_video_audio()` runs `-shortest`
before this service sees the raw file, and when the generated picture outlasts
the track that drops the last frames. Padding the track for the model and
cutting the final mux with `-t` is the fix, so both commands are asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import media


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, timeout: int = 3600) -> None:
        commands.append(command)
        Path(command[-1]).write_bytes(b"stub")

    monkeypatch.setattr(media, "_run", fake_run)
    return commands


def test_mux_uses_an_explicit_duration(captured, tmp_path: Path):
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    video.write_bytes(b"v")
    audio.write_bytes(b"a")
    destination = tmp_path / "out.mp4"

    media.mux_audio(video, audio, destination, duration=3.0)

    command = captured[0]
    assert "-t" in command
    assert command[command.index("-t") + 1] == "3.000"
    assert "-shortest" not in command
    assert command[command.index("-c:v") + 1] == "copy"


def test_mux_falls_back_to_shortest_without_a_duration(captured, tmp_path: Path):
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    video.write_bytes(b"v")
    audio.write_bytes(b"a")

    media.mux_audio(video, audio, tmp_path / "out.mp4")

    assert "-shortest" in captured[0]
    assert "-t" not in captured[0]


def test_pad_audio_appends_silence_to_the_given_length(captured, tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"a")
    destination = tmp_path / "audio.padded.wav"

    media.pad_audio(audio, destination, 6.0)

    command = captured[0]
    assert "apad" in command[command.index("-af") + 1]
    assert command[command.index("-t") + 1] == "6.000"
    assert destination.is_file()


def test_pad_audio_keeps_pcm_so_nothing_is_re_encoded_lossily(captured, tmp_path: Path):
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"a")
    media.pad_audio(audio, tmp_path / "p.wav", 5.0)
    assert "pcm_s16le" in captured[0]
