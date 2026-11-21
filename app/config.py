"""Environment-driven configuration.

Every value has a container-appropriate default, so the image runs with an
empty environment and the host wrapper only has to set paths and a key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(_env(name, str(default)))
    except ValueError:
        return default
    value = max(minimum, value)
    return min(maximum, value) if maximum is not None else value


def _float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(_env(name, str(default)))
    except ValueError:
        return default
    return value if value >= minimum else default


SIZE_PRESETS: dict[str, int] = {
    # name -> pixel area handed to Wan's get_size_less_than_area(). The output
    # width/height themselves follow the reference image's aspect ratio.
    "720p": 1280 * 720,
    "704p": 1024 * 704,
    "480p": 832 * 480,
}

# The --size argument has to be one of Wan's SIZE_CONFIGS keys; for the s2v task
# only its area is used (the aspect comes from the still), so these stand in for
# the presets above.
SIZE_KEYS: dict[str, str] = {
    "720p": "1280*720",
    "704p": "1024*704",
    "480p": "832*480",
}


@dataclass(frozen=True)
class Settings:
    # where the upstream Wan2.2 checkout and the weights live
    wan_dir: Path
    model_id: Path
    output_dir: Path
    performances_dir: Path

    # engine: "wan" (real model, needs CUDA) or "mock" (GPU-less smoke tests)
    engine: str

    # generation defaults
    size: str
    infer_frames: int
    start_from_ref: bool
    sample_steps: int
    guide_scale: float
    shift: float
    offload_model: bool
    convert_model_dtype: bool
    gpus: int
    conform_fps: int
    seed: int
    render_timeout_s: int

    # tooling
    ffmpeg: str
    ffprobe: str
    hf_endpoint: str

    # HTTP service
    api_key: str
    host: str
    port: int
    max_upload_mb: int
    max_audio_seconds: float
    max_prompt_chars: int

    @property
    def auth_required(self) -> bool:
        return bool(self.api_key)

    @property
    def target_area(self) -> int:
        return SIZE_PRESETS[self.size]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    size = _env("SIZE", "720p")
    if size not in SIZE_PRESETS:
        size = "720p"
    engine = _env("ENGINE", "wan").strip().lower()
    if engine not in {"wan", "mock"}:
        engine = "wan"
    return Settings(
        wan_dir=Path(_env("WAN_DIR", "/opt/Wan2.2")),
        model_id=Path(_env("MODEL_ID", "/models/Wan2.2-S2V-14B")),
        output_dir=Path(_env("OUTPUT_DIR", "/output")),
        performances_dir=Path(_env("PERFORMANCES_DIR", "/performances")),
        engine=engine,
        size=size,
        infer_frames=_int("INFER_FRAMES", 80, 24, 160),
        start_from_ref=_flag("START_FROM_REF", True),
        sample_steps=_int("SAMPLE_STEPS", 40, 1, 200),
        guide_scale=_float("GUIDE_SCALE", 4.5),
        shift=_float("SHIFT", 3.0),
        offload_model=_flag("OFFLOAD_MODEL", True),
        convert_model_dtype=_flag("CONVERT_MODEL_DTYPE", True),
        gpus=_int("GPUS", 1, 1, 8),
        conform_fps=_int("CONFORM_FPS", 0, 0, 120),
        seed=_int("SEED", -1, -1),
        render_timeout_s=_int("RENDER_TIMEOUT_S", 14400, 60),
        ffmpeg=_env("FFMPEG", "ffmpeg"),
        ffprobe=_env("FFPROBE", "ffprobe"),
        hf_endpoint=_env("HF_ENDPOINT", "https://huggingface.co"),
        api_key=_env("WAN_API_KEY", ""),
        host=_env("HOST", "0.0.0.0"),
        port=_int("PORT", 8000, 1, 65535),
        max_upload_mb=_int("MAX_UPLOAD_MB", 200, 1),
        max_audio_seconds=_float("MAX_AUDIO_SECONDS", 300.0, 1.0),
        max_prompt_chars=_int("MAX_PROMPT_CHARS", 4000, 20),
    )


def reset_settings() -> None:
    """Test helper: forget the cached environment."""
    get_settings.cache_clear()
