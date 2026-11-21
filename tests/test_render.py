"""Rendering with the mock engine: manifest, resume, parameter precedence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import Settings
from app.engine import EngineUnavailable, MockEngine, WanS2VEngine
from app.performance import PerformanceError, discover
from app.render import MANIFEST_NAME, VIDEO_NAME, is_complete, plan_for, render, resolve_params
from tests.conftest import build_performance


@pytest.fixture
def engine() -> MockEngine:
    return MockEngine()


def test_render_writes_video_and_manifest(performance: Path, settings: Settings, engine: MockEngine):
    found = discover(performance)
    out = performance / "out"
    manifest = render(found, out, settings, engine)

    assert (out / VIDEO_NAME).is_file()
    assert (out / MANIFEST_NAME).is_file()
    assert manifest["schema"] == "wan22-s2v/1"
    assert manifest["output"]["synthetic"] is True
    assert manifest["inputs"]["prompt"]["chars"] == len(found.prompt)
    assert manifest["plan"]["clips"] == 1
    assert manifest["params"]["seed"] >= 0
    assert manifest["timing"]["render_seconds"] >= 0
    assert manifest["engine"]["name"] == "mock"


def test_manifest_is_valid_json_on_disk(performance: Path, settings: Settings, engine: MockEngine):
    out = performance / "out"
    render(discover(performance), out, settings, engine)
    payload = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert payload["inputs"]["image"]["sha256"]
    assert payload["inputs"]["audio"]["seconds"] == pytest.approx(4.99, abs=0.01)


def test_second_render_is_skipped(performance: Path, settings: Settings, engine: MockEngine):
    found = discover(performance)
    out = performance / "out"
    render(found, out, settings, engine)
    again = render(found, out, settings, engine)
    assert again.get("skipped") is True


def test_changed_prompt_invalidates_the_cache(performance: Path, settings: Settings, engine: MockEngine):
    found = discover(performance)
    out = performance / "out"
    render(found, out, settings, engine)
    assert is_complete(out, found) is True
    (performance / "prompt.txt").write_text("A different scene entirely.", encoding="utf-8")
    changed = discover(performance)
    assert is_complete(out, changed) is False
    manifest = render(changed, out, settings, engine)
    assert not manifest.get("skipped")


def test_force_redraws(performance: Path, settings: Settings, engine: MockEngine):
    found = discover(performance)
    out = performance / "out"
    render(found, out, settings, engine)
    manifest = render(found, out, settings, engine, force=True)
    assert not manifest.get("skipped")


def test_missing_video_invalidates_the_cache(performance: Path, settings: Settings, engine: MockEngine):
    found = discover(performance)
    out = performance / "out"
    render(found, out, settings, engine)
    (out / VIDEO_NAME).unlink()
    assert is_complete(out, found) is False


def test_out_dir_may_be_the_performance_folder(performance: Path, settings: Settings, engine: MockEngine):
    found = discover(performance)
    render(found, performance, settings, engine)
    assert (performance / VIDEO_NAME).is_file()
    assert (performance / "manifest.json").is_file()


def test_parameters_precedence(performance: Path, settings: Settings):
    (performance / "performance.json").write_text(
        json.dumps({"size": "720p", "seed": 7}), encoding="utf-8"
    )
    found = discover(performance)
    params = resolve_params(found, settings, {"seed": 9, "max_clips": 1})
    assert params["seed"] == 9
    assert params["max_clips"] == 1
    assert params["size"] == "720p"


def test_parameters_use_settings_when_nothing_else_is_set(performance: Path, settings: Settings):
    params = resolve_params(discover(performance), settings)
    assert params["size"] == settings.size
    assert params["infer_frames"] == settings.infer_frames
    assert params["start_from_ref"] == settings.start_from_ref
    assert params["seed"] >= 0


def test_unknown_size_is_rejected(performance: Path, settings: Settings):
    (performance / "performance.json").write_text(json.dumps({"size": "1080p"}), encoding="utf-8")
    with pytest.raises(PerformanceError, match="size must be one of"):
        resolve_params(discover(performance), settings)


def test_short_audio_budget_is_enforced(performance: Path, settings: Settings):
    strict = replace(settings, max_audio_seconds=1.0)
    with pytest.raises(PerformanceError, match="MAX_AUDIO_SECONDS"):
        plan_for(discover(performance), strict, resolve_params(discover(performance), strict))


def test_conform_fps_validation(performance: Path, settings: Settings):
    (performance / "performance.json").write_text(json.dumps({"conform_fps": 2}), encoding="utf-8")
    with pytest.raises(PerformanceError, match="conform_fps"):
        resolve_params(discover(performance), settings)


def test_plan_uses_the_preset_area(performance: Path, settings: Settings):
    found = discover(performance)
    params = resolve_params(found, settings, {"size": "480p"})
    plan = plan_for(found, settings, params)
    assert (plan.out_width, plan.out_height) == (832, 448)
    assert plan.target_area == 832 * 480


def test_real_engine_reports_missing_checkout(performance: Path, settings: Settings):
    import types

    real = WanS2VEngine(replace(settings, engine="wan"))
    ok, detail = real.available()
    assert ok is False
    assert "generate.py" in detail
    found = discover(performance)
    params = resolve_params(found, settings)
    request = types.SimpleNamespace(
        performance=found,
        plan=plan_for(found, settings, params),
        params=params,
        raw_path=performance / "raw.mp4",
        output_path=performance / "video.mp4",
        log_path=performance / "render.log",
    )
    with pytest.raises(EngineUnavailable):
        real.render(request)


def test_batch_style_multiple_folders(tmp_path: Path, settings: Settings, engine: MockEngine):
    first = build_performance(tmp_path / "a", seconds=2.0)
    second = build_performance(tmp_path / "b", seconds=6.0)
    for folder in (first, second):
        manifest = render(discover(folder), folder / "out", settings, engine)
        assert manifest["plan"]["clips"] >= 1
    assert (first / "out" / VIDEO_NAME).is_file()
    assert (second / "out" / VIDEO_NAME).is_file()
