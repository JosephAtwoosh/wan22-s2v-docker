"""The job queue's on-disk state and crash recovery."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.engine import MockEngine
from app.jobs import JobManager
from app.performance import discover


def _manager(settings: Settings) -> JobManager:
    return JobManager(settings, MockEngine(settings))


def test_submit_writes_request_and_status(settings: Settings, performance: Path):
    manager = _manager(settings)
    job = manager.submit(discover(performance))
    assert job.status in {"queued", "running"}
    directory = job.directory
    assert (directory / "request.json").is_file()
    assert (directory / "status.json").is_file()
    assert (directory / "start.png").is_file()
    assert (directory / "prompt.txt").is_file()
    assert job.done.wait(10.0) is True
    state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    assert state["status"] in {"done", "error"}
    manager.shutdown()


def test_public_dict_has_plan(settings: Settings, performance: Path):
    manager = _manager(settings)
    job = manager.submit(discover(performance))
    manager.wait(job, 10.0)
    payload = manager.public(job)
    assert payload["plan"]["clips"] == 1
    assert payload["status"] == "done"
    assert payload["urls"]["video"].endswith("/video")
    manager.shutdown()


def test_recover_marks_interrupted_jobs(settings: Settings):
    manager = _manager(settings)
    directory = manager.root / "20260101-000000-abcdef"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(
        json.dumps({"id": "20260101-000000-abcdef", "status": "running"}), encoding="utf-8"
    )
    recovered = manager.recover()
    assert "20260101-000000-abcdef" in recovered
    state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    assert state["status"] == "error"
    assert "interrupted" in state["error"]
    manager.shutdown()


def test_recover_leaves_finished_jobs_alone(settings: Settings):
    manager = _manager(settings)
    directory = manager.root / "20260101-000000-finished"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(
        json.dumps({"id": "done", "status": "done"}), encoding="utf-8"
    )
    assert manager.recover() == []
    manager.shutdown()


def test_unknown_job_raises(settings: Settings):
    manager = _manager(settings)
    try:
        manager.get("nope")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected KeyError")
    manager.shutdown()
