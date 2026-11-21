"""CLI surface: plan, perform, batch."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app import __version__
from app.cli import main
from app.config import reset_settings
from tests.conftest import build_performance


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENGINE", "mock")
    monkeypatch.setenv("WAN_API_KEY", "")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("MODEL_ID", str(tmp_path / "model"))
    monkeypatch.setenv("WAN_DIR", str(tmp_path / "wan"))
    monkeypatch.setenv("SIZE", "480p")
    monkeypatch.setenv("FFPROBE", "ffprobe-not-installed")
    reset_settings()
    yield
    reset_settings()


def test_version(capsys):
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_plan_prints_json(performance: Path, capsys):
    assert main(["plan", str(performance), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["clips"] == 1
    assert payload["plan"]["out_width"] == 832
    assert payload["params"]["seed"] >= 0


def test_plan_human_output(performance: Path, capsys):
    assert main(["plan", str(performance)]) == 0
    out = capsys.readouterr().out
    assert "832x448" in out
    assert "clips" in out


def test_plan_missing_folder_is_an_error(tmp_path: Path, capsys):
    assert main(["plan", str(tmp_path / "does-not-exist")]) == 2
    assert "error" in capsys.readouterr().err


def test_perform_renders(performance: Path, capsys):
    assert main(["perform", str(performance), "--quiet"]) == 0
    assert (performance / "video.mp4").is_file()
    assert (performance / "manifest.json").is_file()


def test_perform_reports_cached(performance: Path, capsys):
    assert main(["perform", str(performance), "--quiet"]) == 0
    capsys.readouterr()
    assert main(["perform", str(performance)]) == 0
    out = capsys.readouterr().out
    assert "already rendered" in out


def test_perform_force_redraws(performance: Path, capsys):
    assert main(["perform", str(performance), "--quiet"]) == 0
    assert main(["perform", str(performance), "--force", "--quiet"]) == 0
    assert (performance / "video.mp4").is_file()


def test_perform_override_seed(performance: Path, capsys):
    assert main(["perform", str(performance), "--seed", "123", "--quiet", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["params"]["seed"] == 123


def test_perform_out_dir(tmp_path: Path, performance: Path, capsys):
    out = tmp_path / "renders"
    assert main(["perform", str(performance), "-o", str(out), "--quiet"]) == 0
    assert (out / "video.mp4").is_file()


def test_selftest_renders_without_assets(tmp_path: Path, capsys):
    work = tmp_path / "selftest"
    assert main(["selftest", "--workdir", str(work)]) == 0
    folder = work / "performance"
    assert (folder / "start.png").is_file()
    assert (folder / "audio.wav").is_file()
    assert (folder / "prompt.txt").is_file()
    assert (folder / "video.mp4").is_file()
    assert (folder / "manifest.json").is_file()
    assert "selftest OK" in capsys.readouterr().out


def test_batch_dry_run(tmp_path: Path, capsys):
    root = tmp_path / "scene"
    build_performance(root / "shot-a", seconds=2.0)
    build_performance(root / "shot-b", seconds=7.0)
    assert main(["batch", str(root), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 performance(s)" in out
    report = root / "out" / "batch.csv"
    assert report.is_file()
    rows = list(csv.DictReader(report.open(encoding="utf-8")))
    assert [row["status"] for row in rows] == ["planned", "planned"]


def test_batch_renders(tmp_path: Path, capsys):
    root = tmp_path / "scene"
    build_performance(root / "shot-a", seconds=2.0)
    build_performance(root / "shot-b", seconds=7.0)
    assert main(["batch", str(root)]) == 0
    assert (root / "out" / "shot-a" / "video.mp4").is_file()
    assert (root / "out" / "shot-b" / "video.mp4").is_file()
    rows = list(csv.DictReader((root / "out" / "batch.csv").open(encoding="utf-8")))
    assert {row["status"] for row in rows} == {"done"}


def test_batch_reports_invalid_folder(tmp_path: Path, capsys):
    root = tmp_path / "scene"
    build_performance(root / "good", seconds=2.0)
    broken = root / "broken"
    broken.mkdir()
    (broken / "prompt.txt").write_text("no image, no audio", encoding="utf-8")
    assert main(["batch", str(root)]) == 1
    rows = list(csv.DictReader((root / "out" / "batch.csv").open(encoding="utf-8")))
    assert any(row["status"] == "invalid" for row in rows)


def test_batch_with_no_performances(tmp_path: Path, capsys):
    root = tmp_path / "scene"
    root.mkdir()
    assert main(["batch", str(root)]) == 1
    assert "no performance folders" in capsys.readouterr().err


def test_limit(tmp_path: Path, capsys):
    root = tmp_path / "scene"
    for name in ("a", "b", "c"):
        build_performance(root / name, seconds=2.0)
    assert main(["batch", str(root), "--dry-run", "--limit", "2"]) == 0
    assert "2 performance(s)" in capsys.readouterr().out
