"""FastAPI service.

Endpoints
---------
GET  /health                          liveness, GPU/weights state (no auth)
POST /v1/performances                 start.jpg + audio.wav + prompt -> job (202)
GET  /v1/jobs                         recent jobs
GET  /v1/jobs/{id}                    status, plan and (when done) the manifest
GET  /v1/jobs/{id}/video              video.mp4
GET  /v1/jobs/{id}/manifest.json      manifest.json
GET  /v1/jobs/{id}/log                render.log (the upstream process output)

An upload is written to disk as a performance folder *before* it is validated,
so a rejected request never queues work and an accepted one can be downloaded
and re-run with the CLI unchanged.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import socket
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .config import Settings, get_settings
from .engine import build_engine
from .jobs import Job, JobManager
from .performance import (
    AUDIO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    PerformanceError,
    discover,
)
from .render import environment

log = logging.getLogger("wan22_s2v.server")

_CHUNK = 1 << 20
_FILES = {
    "video": ("video.mp4", "video/mp4"),
    "manifest.json": ("manifest.json", "application/json"),
    "log": ("render.log", "text/plain"),
}


async def _save_upload(upload: UploadFile, destination: Path, limit_bytes: int) -> int:
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "wb") as handle:
        while True:
            chunk = await upload.read(_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > limit_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"upload exceeds MAX_UPLOAD_MB={limit_bytes // (1024 * 1024)}",
                )
            handle.write(chunk)
    return total


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = get_settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    engine = build_engine(settings)
    jobs = JobManager(settings, engine)
    app.state.settings = settings
    app.state.engine = engine
    app.state.jobs = jobs
    app.state.started_at = time.time()
    jobs.recover()
    log.info(
        "starting wan22-s2v %s: engine=%s model=%s output=%s",
        __version__,
        engine.name,
        settings.model_id,
        settings.output_dir,
    )
    try:
        yield
    finally:
        jobs.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Wan2.2-S2V Docker",
        version=__version__,
        description=(
            "Audio-driven performance from one still: start image + audio + prompt in, "
            "a synced clip out. No end frame."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _supplied_key(request: Request) -> str:
        supplied = (request.headers.get("x-api-key") or "").strip()
        authorization = request.headers.get("authorization") or ""
        if not supplied and authorization:
            scheme, _, value = authorization.partition(" ")
            if scheme.lower() == "bearer":
                supplied = value.strip()
        return supplied

    def require_key(request: Request) -> None:
        expected = request.app.state.settings.api_key
        if not expected:
            return
        if not secrets.compare_digest(_supplied_key(request), expected):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    router = APIRouter(dependencies=[Depends(require_key)])

    def base(request: Request) -> str:
        return str(request.base_url).rstrip("/")

    def job_or_404(mgr: JobManager, job_id: str) -> Job:
        try:
            return mgr.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}") from exc

    # ------------------------------------------------------------------ health
    @app.get("/health", tags=["meta"])
    def health(request: Request) -> dict[str, Any]:
        state = request.app.state
        settings: Settings = state.settings
        engine = state.engine
        available, detail = engine.available()
        status = "ok" if available or engine.name == "mock" else "degraded"
        env = environment()
        return {
            "status": status,
            "version": __version__,
            "engine": engine.describe(),
            "engine_ready": available,
            "engine_detail": detail,
            "auth_required": settings.auth_required,
            "gpu": env.get("gpu"),
            "vram_gib": env.get("vram_gib"),
            "cuda": env.get("cuda"),
            "running_jobs": sum(
                1 for job in state.jobs.recent(50) if job.status in {"queued", "running"}
            ),
            "hostname": socket.gethostname(),
            "uptime_seconds": round(time.time() - state.started_at, 1),
        }

    # ------------------------------------------------------------ performances
    @router.post("/v1/performances", tags=["performances"], status_code=202)
    async def create_performance(
        request: Request,
        start: Annotated[UploadFile, File(description="first frame: jpg/png/webp")],
        audio: Annotated[UploadFile, File(description="the performance track: wav/mp3/...")],
        prompt: Annotated[str | None, Form(description="scene prompt, one paragraph")] = None,
        prompt_file: Annotated[UploadFile | None, File(description="prompt.txt instead of the form field")] = None,
        size: Annotated[str | None, Form(description="720p | 704p | 480p")] = None,
        seed: Annotated[int | None, Form(description="-1 for a recorded random seed")] = None,
        max_clips: Annotated[int | None, Form(description="cap clips for a preview")] = None,
        start_from_ref: Annotated[bool | None, Form(description="use the still as frame 0")] = None,
        conform_fps: Annotated[int | None, Form(description="0 = leave at 16 fps")] = None,
    ) -> JSONResponse:
        state = request.app.state
        settings: Settings = state.settings

        start_name = Path(start.filename or "start").name
        audio_name = Path(audio.filename or "audio").name
        if Path(start_name).suffix.lower() not in IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"start must be one of {', '.join(IMAGE_EXTENSIONS)} (got {start_name!r})",
            )
        if Path(audio_name).suffix.lower() not in AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"audio must be one of {', '.join(AUDIO_EXTENSIONS)} (got {audio_name!r})",
            )

        limit = settings.max_upload_mb * 1024 * 1024
        staging = Path(tempfile.mkdtemp(prefix="upload-", dir=settings.output_dir))
        try:
            await _save_upload(start, staging / f"start{Path(start_name).suffix.lower()}", limit)
            await _save_upload(audio, staging / f"audio{Path(audio_name).suffix.lower()}", limit)

            text = prompt or ""
            if prompt_file is not None:
                raw = await prompt_file.read()
                if len(raw) > settings.max_prompt_chars * 4:
                    raise HTTPException(status_code=413, detail="prompt file is too large")
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise HTTPException(status_code=400, detail="prompt file must be UTF-8") from exc
            if not text.strip():
                raise HTTPException(
                    status_code=400, detail="send a prompt: form field `prompt` or file `prompt_file`"
                )
            (staging / "prompt.txt").write_text(text, encoding="utf-8")

            # Optional per-folder overrides; keeps the three-file contract and
            # lets the caller pin parameters for one shot.
            file_overrides: dict[str, Any] = {}
            for key, value in (
                ("size", size),
                ("seed", seed),
                ("max_clips", max_clips),
                ("start_from_ref", start_from_ref),
                ("conform_fps", conform_fps),
            ):
                if value is not None:
                    file_overrides[key] = value
            if file_overrides:
                import json

                (staging / "performance.json").write_text(
                    json.dumps(file_overrides, indent=2) + "\n", encoding="utf-8"
                )

            try:
                performance = discover(staging, max_prompt_chars=settings.max_prompt_chars)
                job = state.jobs.submit(
                    performance,
                    meta={
                        "source": "upload",
                        "start_filename": start_name,
                        "audio_filename": audio_name,
                    },
                )
            except PerformanceError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return JSONResponse(state.jobs.public(job, base(request)), status_code=202)

    # -------------------------------------------------------------------- jobs
    @router.get("/v1/jobs", tags=["jobs"])
    def list_jobs(request: Request, limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
        state = request.app.state
        return {"jobs": [state.jobs.public(job, base(request)) for job in state.jobs.recent(limit)]}

    @router.get("/v1/jobs/{job_id}", tags=["jobs"])
    def get_job(job_id: str, request: Request) -> dict[str, Any]:
        return request.app.state.jobs.public(job_or_404(request.app.state.jobs, job_id), base(request))

    @app.get("/v1/jobs/{job_id}/{name}", tags=["jobs"])
    def get_file(
        job_id: str, name: str, request: Request, token: str | None = Query(default=None)
    ) -> FileResponse:
        # A browser opening a <video src> cannot send an Authorization header,
        # so `?token=` is accepted as an alternative to the header.
        expected = request.app.state.settings.api_key
        if expected and not (
            token and secrets.compare_digest(token, expected)
        ) and not secrets.compare_digest(_supplied_key(request), expected):
            raise HTTPException(status_code=401, detail="missing or invalid API key")

        mgr = request.app.state.jobs
        job = job_or_404(mgr, job_id)
        if name == "video":
            if job.status != "done":
                raise HTTPException(status_code=409, detail=f"job is {job.status}")
            path, media = _FILES["video"]
        elif name in _FILES:
            path, media = _FILES[name]
        else:
            raise HTTPException(status_code=404, detail=f"no such file {name!r}")
        full = job.directory / path
        if not full.is_file():
            raise HTTPException(status_code=404, detail=f"{path} is not ready")
        headers = {"Content-Disposition": f'inline; filename="{path}"'}
        return FileResponse(full, media_type=media, headers=headers)

    app.include_router(router)
    return app


app = create_app()


def main() -> None:
    import uvicorn

    settings = get_settings()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "info").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=os.environ.get("LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
