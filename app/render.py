"""Turn a performance folder into ``video.mp4`` plus a manifest.

The order is deliberate: validate and plan first (cheap, CPU-only), render
second, then *verify* the artifact before calling it done. Verification is what
catches Wan's silent-audio failure mode and any resolution drift.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import __version__
from .config import SIZE_PRESETS, Settings
from .engine import Engine, RenderRequest, RenderResult, pick_seed
from .media import MediaError, conform_fps, mux_audio, probe
from .performance import (
    Performance,
    PerformanceError,
    discover,
    probe_audio_seconds,
    probe_image_size,
    sha256_file,
)
from .plan import Plan, plan as build_plan

SCHEMA = "wan22-s2v/1"
VIDEO_NAME = "video.mp4"
MANIFEST_NAME = "manifest.json"
LOG_NAME = "render.log"

# A render may be shorter than the audio by a frame or two of container
# rounding; more than this and something is wrong.
DURATION_TOLERANCE_S = 0.25


class RenderError(RuntimeError):
    """A render was produced but does not satisfy the contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _check_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PerformanceError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise PerformanceError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def _check_float(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerformanceError(f"{name} must be a number")
    if value < minimum or value > maximum:
        raise PerformanceError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return float(value)


def resolve_params(
    performance: Performance, settings: Settings, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Defaults <- performance.json <- explicit flags.

    Only the keys in :data:`performance.ALLOWED_OVERRIDES` (plus the settings'
    own defaults) can appear; everything else was already rejected while
    reading the folder.
    """
    params: dict[str, Any] = {
        "size": settings.size,
        "infer_frames": settings.infer_frames,
        "max_clips": None,
        "sample_steps": settings.sample_steps,
        "guide_scale": settings.guide_scale,
        "shift": settings.shift,
        "start_from_ref": settings.start_from_ref,
        "conform_fps": settings.conform_fps,
        "seed": settings.seed,
    }
    merged = dict(performance.overrides)
    merged.update({key: value for key, value in (overrides or {}).items() if value is not None})

    if "size" in merged:
        size = str(merged["size"]).strip().lower()
        if size not in SIZE_PRESETS:
            raise PerformanceError(
                f"size must be one of {', '.join(sorted(SIZE_PRESETS))}, got {size!r}"
            )
        params["size"] = size
    if "infer_frames" in merged:
        frames = _check_int("infer_frames", merged["infer_frames"], 24, 160)
        if frames % 4:
            raise PerformanceError(f"infer_frames must be a multiple of 4, got {frames}")
        params["infer_frames"] = frames
    if "max_clips" in merged and merged["max_clips"] is not None:
        params["max_clips"] = _check_int("max_clips", merged["max_clips"], 1, 100)
    if "sample_steps" in merged:
        params["sample_steps"] = _check_int("sample_steps", merged["sample_steps"], 1, 200)
    if "guide_scale" in merged:
        params["guide_scale"] = _check_float("guide_scale", merged["guide_scale"], 0.0, 30.0)
    if "shift" in merged:
        params["shift"] = _check_float("shift", merged["shift"], 0.1, 30.0)
    if "start_from_ref" in merged:
        params["start_from_ref"] = bool(merged["start_from_ref"])
    if "conform_fps" in merged:
        fps = _check_int("conform_fps", merged["conform_fps"], 0, 120)
        if fps and fps < 8:
            raise PerformanceError("conform_fps must be 0 (off) or at least 8")
        params["conform_fps"] = fps
    if "seed" in merged:
        params["seed"] = _check_int("seed", merged["seed"], -1, 2**32 - 1)

    params["seed"] = pick_seed(int(params["seed"]))
    return params


def plan_for(performance: Performance, settings: Settings, params: dict[str, Any]) -> Plan:
    image_size = probe_image_size(performance.image)
    audio_seconds = probe_audio_seconds(performance.audio, settings.ffprobe)
    if audio_seconds > settings.max_audio_seconds:
        raise PerformanceError(
            f"{performance.audio.name} is {audio_seconds:.1f}s; MAX_AUDIO_SECONDS="
            f"{settings.max_audio_seconds:.0f}. Split the performance into shots."
        )
    return build_plan(
        image_size,
        audio_seconds,
        target_area=SIZE_PRESETS[params["size"]],
        infer_frames=params["infer_frames"],
        start_from_ref=params["start_from_ref"],
        max_clips=params["max_clips"],
    )


@lru_cache(maxsize=1)
def environment() -> dict[str, Any]:
    """Facts worth having in a manifest six months later."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch  # noqa: PLC0415 - optional on CPU-only hosts

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - exercised only with torch installed
        info["torch"] = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        first = (result.stdout or "").strip().splitlines()[:1]
        if result.returncode == 0 and first:
            name, vram_mib, driver = (part.strip() for part in first[0].split(","))
            info["gpu"] = name
            info["vram_gib"] = round(float(vram_mib) / 1024, 1)
            info["driver"] = driver
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return info


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def is_complete(out_dir: Path, performance: Performance) -> bool:
    """True when this exact performance has already been rendered here."""
    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("schema") != SCHEMA:
        return False
    output = manifest.get("output") or {}
    video = out_dir / str(output.get("file") or VIDEO_NAME)
    if not video.is_file() or video.stat().st_size == 0:
        return False
    inputs = manifest.get("inputs") or {}
    for key, path in (
        ("image", performance.image),
        ("audio", performance.audio),
        ("prompt", performance.prompt_path),
    ):
        expected = (inputs.get(key) or {}).get("sha256")
        if not expected or expected != sha256_file(path):
            return False
    return True


def render(
    performance: Performance,
    out_dir: str | Path,
    settings: Settings,
    engine: Engine,
    *,
    overrides: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    force: bool = False,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Render one performance into ``out_dir``; returns its manifest.

    ``out_dir`` may be the performance folder itself (the API does that), so a
    job directory on disk is always a complete, re-runnable performance.
    ``params`` is passed already-resolved by the job queue; otherwise it is
    resolved here from the folder and the overrides.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    wall_start = time.monotonic()

    if not force and is_complete(out_dir, performance):
        manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest.setdefault("skipped", True)
        return manifest

    if params is None:
        params = resolve_params(performance, settings, overrides)
    else:
        params = dict(params)
    plan = plan_for(performance, settings, params)

    raw_path = out_dir / "render.raw.mp4"
    output_path = out_dir / VIDEO_NAME
    log_path = out_dir / LOG_NAME
    log_path.write_text("", encoding="utf-8")

    request = RenderRequest(
        performance=performance,
        plan=plan,
        params=params,
        raw_path=raw_path,
        output_path=output_path,
        log_path=log_path,
    )
    result: RenderResult = engine.render(request)

    if result.synthetic:
        output_path.write_bytes(result.path.read_bytes())
    else:
        # Always re-mux: it guarantees an audio stream (upstream's merge can
        # fail quietly), cuts to the shorter of picture and sound, and writes
        # a faststart mp4.
        mux_audio(
            result.path,
            performance.audio,
            output_path,
            ffmpeg=settings.ffmpeg,
        )
        if params["conform_fps"]:
            conform_fps(
                output_path,
                params["conform_fps"],
                ffmpeg=settings.ffmpeg,
            )

    output_info: dict[str, Any] = {"synthetic": True} if result.synthetic else _verify(
        output_path, plan, settings, conform_fps_value=params["conform_fps"]
    )

    if not result.synthetic and raw_path.exists():
        raw_path.unlink()

    finished_at = _now()
    manifest = {
        "schema": SCHEMA,
        "job_id": job_id,
        "created_at": started_at,
        "tool": {"name": "wan22-s2v-docker", "version": __version__},
        "engine": engine.describe(),
        "inputs": {
            "image": {
                "file": performance.image.name,
                "sha256": sha256_file(performance.image),
                "width": plan.image_width,
                "height": plan.image_height,
            },
            "audio": {
                "file": performance.audio.name,
                "sha256": sha256_file(performance.audio),
                "seconds": round(plan.audio_seconds, 3),
            },
            "prompt": {
                "file": performance.prompt_path.name,
                "sha256": sha256_file(performance.prompt_path),
                "chars": len(performance.prompt),
                "text": performance.prompt,
            },
        },
        "plan": plan.as_dict(),
        "params": {
            "size": params["size"],
            "infer_frames": params["infer_frames"],
            "max_clips": params["max_clips"],
            "sample_steps": params["sample_steps"],
            "guide_scale": params["guide_scale"],
            "shift": params["shift"],
            "start_from_ref": params["start_from_ref"],
            "conform_fps": params["conform_fps"],
            "seed": params["seed"],
        },
        "output": {
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            **output_info,
        },
        "timing": {
            "started_at": started_at,
            "finished_at": finished_at,
            "wall_seconds": round(time.monotonic() - wall_start, 1),
            "render_seconds": round(result.seconds, 1),
            "realtime_factor": round(result.seconds / plan.video_seconds, 2)
            if plan.video_seconds
            else None,
        },
        "environment": environment(),
    }
    _write_json(out_dir / MANIFEST_NAME, manifest)
    return manifest


def _verify(
    path: Path, plan: Plan, settings: Settings, *, conform_fps_value: int
) -> dict[str, Any]:
    info = probe(path, settings.ffprobe)
    if not info.has_video:
        raise RenderError(f"{path.name} has no video stream")
    if not info.has_audio:
        raise RenderError(
            f"{path.name} has no audio stream: the track was dropped in muxing"
        )
    if info.width != plan.out_width or info.height != plan.out_height:
        raise RenderError(
            f"{path.name} came back {info.width}x{info.height}, planned "
            f"{plan.out_width}x{plan.out_height}"
        )
    expected_seconds = min(plan.video_seconds, plan.audio_seconds)
    if info.duration is not None and abs(info.duration - expected_seconds) > DURATION_TOLERANCE_S:
        raise RenderError(
            f"{path.name} is {info.duration:.3f}s; expected {expected_seconds:.3f}s "
            "(picture and audio should end together)"
        )
    expected_fps = conform_fps_value or plan.fps
    if info.fps is not None and abs(info.fps - expected_fps) > 0.1:
        raise RenderError(f"{path.name} is {info.fps:.2f} fps; expected {expected_fps}")
    return info.as_dict()


def discover_performance(path: str | Path, settings: Settings) -> Performance:
    return discover(path, max_prompt_chars=settings.max_prompt_chars)
