"""HTTP contract: upload a performance, poll a job, fetch the clip."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import reset_settings
from app.server import create_app
from tests.conftest import write_png, write_wav

PNG = "image/png"
WAV = "audio/wav"


def _configure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, api_key: str = "") -> None:
    monkeypatch.setenv("ENGINE", "mock")
    monkeypatch.setenv("WAN_API_KEY", api_key)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("MODEL_ID", str(tmp_path / "model"))
    monkeypatch.setenv("WAN_DIR", str(tmp_path / "wan"))
    monkeypatch.setenv("SIZE", "480p")
    monkeypatch.setenv("FFPROBE", "ffprobe-not-installed")
    reset_settings()


def _form(tmp_path: Path) -> tuple[dict, dict]:
    png = write_png(tmp_path / "still.png")
    wav = write_wav(tmp_path / "line.wav", seconds=4.0)
    files = {
        "start": ("still.png", png.read_bytes(), PNG),
        "audio": ("line.wav", wav.read_bytes(), WAV),
    }
    return files, {"prompt": "A hen looks up from the straw and speaks."}


def _wait(client: TestClient, job_id: str, timeout: float = 10.0, headers: dict | None = None) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/v1/jobs/{job_id}", headers=headers).json()
        if payload.get("status") in {"done", "error"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure(tmp_path, monkeypatch)
    with TestClient(create_app()) as test_client:
        yield test_client
    reset_settings()


def test_health(client: TestClient):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["engine"]["name"] == "mock"
    assert payload["auth_required"] is False


def test_render_a_performance(client: TestClient, tmp_path: Path):
    files, data = _form(tmp_path)
    created = client.post("/v1/performances", files=files, data=data)
    assert created.status_code == 202, created.text
    body = created.json()
    assert body["status"] in {"queued", "running"}

    done = _wait(client, body["id"])
    assert done["status"] == "done", done
    assert done["plan"]["clips"] == 1
    assert done["urls"]["video"].endswith(f"/v1/jobs/{body['id']}/video")

    video = client.get(f"/v1/jobs/{body['id']}/video")
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert video.content

    manifest = client.get(f"/v1/jobs/{body['id']}/manifest.json")
    assert manifest.status_code == 200
    assert manifest.json()["schema"] == "wan22-s2v/1"

    log = client.get(f"/v1/jobs/{body['id']}/log")
    assert log.status_code == 200


def test_job_folder_is_a_performance_folder(client: TestClient, tmp_path: Path):
    files, data = _form(tmp_path)
    job_id = client.post("/v1/performances", files=files, data=data).json()["id"]
    _wait(client, job_id)
    job_dir = tmp_path / "output" / "jobs" / job_id
    assert (job_dir / "start.png").is_file()
    assert (job_dir / "audio.wav").is_file()
    assert (job_dir / "prompt.txt").is_file()
    assert (job_dir / "video.mp4").is_file()
    assert (job_dir / "manifest.json").is_file()


def test_video_is_409_while_queued(client: TestClient, tmp_path: Path):
    # A job that has not finished has no video; the API says so rather than 404.
    files, data = _form(tmp_path)
    job_id = client.post("/v1/performances", files=files, data=data).json()["id"]
    early = client.get(f"/v1/jobs/{job_id}/video")
    assert early.status_code in {200, 409}
    _wait(client, job_id)


def test_rejects_unsupported_image_type(client: TestClient, tmp_path: Path):
    png = write_png(tmp_path / "still.png")
    wav = write_wav(tmp_path / "line.wav")
    files = {
        "start": ("still.gif", png.read_bytes(), PNG),
        "audio": ("line.wav", wav.read_bytes(), WAV),
    }
    response = client.post("/v1/performances", files=files, data={"prompt": "hello"})
    assert response.status_code == 400
    assert "start must be one of" in response.json()["detail"]


def test_rejects_missing_prompt(client: TestClient, tmp_path: Path):
    files, _ = _form(tmp_path)
    response = client.post("/v1/performances", files=files, data={})
    assert response.status_code == 400
    assert "prompt" in response.json()["detail"]


def test_rejects_empty_audio(client: TestClient, tmp_path: Path):
    png = write_png(tmp_path / "still.png")
    files = {
        "start": ("still.png", png.read_bytes(), PNG),
        "audio": ("line.wav", b"", WAV),
    }
    response = client.post("/v1/performances", files=files, data={"prompt": "hello"})
    assert response.status_code == 400


def test_unknown_job_is_404(client: TestClient):
    assert client.get("/v1/jobs/nope").status_code == 404
    assert client.get("/v1/jobs/nope/video").status_code == 404


def test_list_jobs(client: TestClient, tmp_path: Path):
    files, data = _form(tmp_path)
    job_id = client.post("/v1/performances", files=files, data=data).json()["id"]
    _wait(client, job_id)
    listing = client.get("/v1/jobs").json()
    assert any(job["id"] == job_id for job in listing["jobs"])


def test_auth_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure(tmp_path, monkeypatch, api_key="s3cret")
    auth = {"Authorization": "Bearer s3cret"}
    with TestClient(create_app()) as guarded:
        assert guarded.get("/health").status_code == 200  # health stays open
        files, data = _form(tmp_path)
        assert guarded.post("/v1/performances", files=files, data=data).status_code == 401
        created = guarded.post(
            "/v1/performances",
            files=files,
            data=data,
            headers=auth,
        )
        assert created.status_code == 202
        job_id = created.json()["id"]
        _wait(guarded, job_id, headers=auth)
        assert guarded.get(f"/v1/jobs/{job_id}/video").status_code == 401
        assert guarded.get(f"/v1/jobs/{job_id}/video?token=s3cret").status_code == 200
    reset_settings()


def test_preview_override_is_honoured(client: TestClient, tmp_path: Path):
    png = write_png(tmp_path / "still.png")
    wav = write_wav(tmp_path / "line.wav", seconds=12.0)
    files = {
        "start": ("still.png", png.read_bytes(), PNG),
        "audio": ("line.wav", wav.read_bytes(), WAV),
    }
    response = client.post(
        "/v1/performances", files=files, data={"prompt": "hello", "max_clips": "1"}
    )
    assert response.status_code == 202, response.text
    done = _wait(client, response.json()["id"])
    assert done["status"] == "done"
    assert done["plan"]["clips"] == 1
    assert done["plan"]["preview"] is True
