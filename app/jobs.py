"""A one-at-a-time performance queue.

A five-second clip is minutes of GPU time, so the HTTP API returns a job id and
the caller polls. Jobs are directories under ``/output/jobs``: each one holds the
uploaded performance (so it can be re-run by hand), ``status.json``, and on
success ``video.mp4`` + ``manifest.json``. A crashed container leaves its state
readable rather than lost.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Settings
from .engine import Engine
from .performance import Performance, copy_into
from .plan import Plan
from .render import MANIFEST_NAME, VIDEO_NAME, LOG_NAME, plan_for, render, resolve_params

log = logging.getLogger("wan22_s2v.jobs")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Job:
    id: str
    directory: Path
    request: dict[str, Any]
    plan: Plan
    status: str = "queued"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    manifest: dict[str, Any] | None = None
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)


class JobManager:
    def __init__(self, settings: Settings, engine: Engine):
        self.settings = settings
        self.engine = engine
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="s2v-render"
        )
        self.root = settings.output_dir / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ submit
    def submit(
        self,
        performance: Performance,
        *,
        overrides: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Job:
        """Validate, plan, then queue. Raises PerformanceError before queueing."""
        params = resolve_params(performance, self.settings, overrides)
        plan = plan_for(performance, self.settings, params)

        job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        directory = self.root / job_id
        directory.mkdir(parents=True, exist_ok=False)
        stored = copy_into(performance, directory)

        request = {
            "job_id": job_id,
            "created_at": _now(),
            "meta": meta or {},
            "overrides": {key: value for key, value in params.items() if key != "seed"}
            | {"seed": params["seed"]},
            "plan": plan.as_dict(),
            "performance": stored.as_dict(),
        }
        (directory / "request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        job = Job(id=job_id, directory=directory, request=request, plan=plan)
        with self._lock:
            self._jobs[job_id] = job
        self._write(job)
        self._executor.submit(self._run, job, stored, params)
        return job

    # ------------------------------------------------------------------ status
    def _write(self, job: Job) -> None:
        job.updated_at = _now()
        state = {
            "id": job.id,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "error": job.error,
            "meta": job.request.get("meta", {}),
            "clips": job.plan.clips,
            "video_seconds": job.plan.video_seconds,
            "audio_seconds": job.plan.audio_seconds,
            "output_width": job.plan.out_width,
            "output_height": job.plan.out_height,
        }
        try:
            (job.directory / "status.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            log.warning("could not write status.json for %s", job.id, exc_info=True)

    def _run(self, job: Job, performance: Performance, params: dict[str, Any]) -> None:
        job.status = "running"
        self._write(job)
        try:
            manifest = render(
                performance,
                job.directory,
                self.settings,
                self.engine,
                params=params,
                job_id=job.id,
            )
            job.manifest = manifest
            job.status = "done"
        except Exception as exc:
            job.error = str(exc)
            job.status = "error"
            log.error("job %s failed: %s", job.id, exc)
            log.debug("%s", traceback.format_exc())
        finally:
            self._write(job)
            job.done.set()

    # ------------------------------------------------------------------- reads
    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def recent(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[: max(0, limit)]

    def wait(self, job: Job, timeout: float) -> bool:
        return job.done.wait(timeout=max(0.0, timeout))

    def public(self, job: Job, base_url: str = "") -> dict[str, Any]:
        base = base_url.rstrip("/")
        state = {
            "id": job.id,
            "status": job.status,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "error": job.error,
            "meta": job.request.get("meta", {}),
            "plan": job.plan.as_dict(),
        }
        if job.status == "done" and job.manifest is not None:
            state["manifest"] = {
                key: job.manifest.get(key)
                for key in ("schema", "inputs", "params", "output", "timing")
            }
            state["urls"] = {
                "video": f"{base}/v1/jobs/{job.id}/video",
                "manifest": f"{base}/v1/jobs/{job.id}/manifest.json",
                "log": f"{base}/v1/jobs/{job.id}/log",
                "directory": f"{base}/v1/jobs/{job.id}/files",
            }
        return state

    def output_file(self, job: Job, name: str) -> Path | None:
        allowed = {VIDEO_NAME, MANIFEST_NAME, LOG_NAME, "status.json", "request.json"}
        if name not in allowed:
            return None
        path = job.directory / name
        return path if path.is_file() else None

    # ---------------------------------------------------------------- recovery
    def recover(self) -> list[str]:
        """Mark jobs the previous container left mid-render as interrupted."""
        recovered: list[str] = []
        for status_path in sorted(self.root.glob("*/status.json")):
            try:
                state = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("status") not in {"queued", "running"}:
                continue
            state["status"] = "error"
            state["error"] = (
                "interrupted: the container stopped while this job was running; no video was produced"
            )
            state["updated_at"] = _now()
            status_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            recovered.append(str(state.get("id")))
        if recovered:
            log.warning("marked %d interrupted job(s) on startup: %s", len(recovered), ", ".join(recovered))
        return recovered

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
