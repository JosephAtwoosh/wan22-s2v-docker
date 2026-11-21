"""Render engines.

``WanS2VEngine`` shells out to the upstream ``generate.py`` with an argument
vector (never a shell string), so prompts containing quotes, apostrophes and
non-ASCII text survive unchanged. ``MockEngine`` exists so the HTTP API and the
job queue can be tested without a GPU.
"""

from __future__ import annotations

import os
import random
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .config import SIZE_KEYS, Settings
from .performance import Performance
from .plan import Plan


class EngineError(RuntimeError):
    """The renderer could not produce a clip."""


class EngineUnavailable(EngineError):
    """The engine cannot run in this container (missing weights, driver, ...)."""


@dataclass(frozen=True)
class RenderRequest:
    performance: Performance
    plan: Plan
    params: dict
    raw_path: Path
    output_path: Path
    log_path: Path
    # Silence-padded copy handed to the model, so upstream's own -shortest mux
    # cannot shave the tail of the picture. See media.pad_audio().
    audio_for_model: Path | None = None

    @property
    def model_audio(self) -> Path:
        return self.audio_for_model or self.performance.audio

    @property
    def work_dir(self) -> Path:
        return self.raw_path.parent


@dataclass
class RenderResult:
    path: Path
    log_path: Path
    seconds: float
    command: list[str] = field(default_factory=list)
    synthetic: bool = False
    returncode: int = 0


def build_command(
    *,
    wan_dir: str | Path,
    ckpt_dir: str | Path,
    prompt: str,
    image: str | Path,
    audio: str | Path,
    save_file: str | Path,
    size_key: str,
    infer_frames: int,
    clips: int,
    seed: int,
    start_from_ref: bool = True,
    sample_steps: int | None = None,
    guide_scale: float | None = None,
    shift: float | None = None,
    offload_model: bool = True,
    convert_model_dtype: bool = True,
    gpus: int = 1,
    master_port: int = 29500,
    python: str = sys.executable,
) -> list[str]:
    """Build the upstream command.

    ``size_key`` is one of Wan's ``SIZE_CONFIGS`` names ("1280*720", ...): for
    the s2v task it only supplies the pixel *area*, while the aspect ratio comes
    from the reference still.

    With more than one GPU, upstream's own recommended form is FSDP + Ulysses
    with neither ``--offload_model`` nor ``--convert_model_dtype``: it materialises
    the shard on each device, so the offload flags that make a single card fit
    are counter-productive there.
    """
    base = [python, str(Path(wan_dir) / "generate.py")]
    if gpus > 1:
        base = [
            "torchrun",
            "--nproc_per_node",
            str(gpus),
            "--master_port",
            str(master_port),
            *base,
        ]

    command = [
        *base,
        "--task",
        "s2v-14B",
        "--size",
        size_key,
        "--ckpt_dir",
        str(ckpt_dir),
        "--prompt",
        prompt,
        "--image",
        str(image),
        "--audio",
        str(audio),
        "--infer_frames",
        str(infer_frames),
        "--num_clip",
        str(clips),
        "--base_seed",
        str(seed),
        "--save_file",
        str(save_file),
    ]
    if start_from_ref:
        command.append("--start_from_ref")
    if sample_steps:
        command += ["--sample_steps", str(sample_steps)]
    if guide_scale is not None:
        command += ["--sample_guide_scale", str(guide_scale)]
    if shift is not None:
        command += ["--sample_shift", str(shift)]

    if gpus > 1:
        command += ["--dit_fsdp", "--t5_fsdp", "--ulysses_size", str(gpus)]
    else:
        if offload_model:
            command += ["--offload_model", "True"]
        if convert_model_dtype:
            command.append("--convert_model_dtype")
    return command


def _tail(path: Path, lines: int = 25) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(text[-lines:])


def _write_log_header(handle, command: list[str], plan: Plan) -> None:
    handle.write(f"# wan22-s2v render {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    handle.write("# plan\n")
    for line in plan.summary().splitlines():
        handle.write(f"#   {line}\n")
    handle.write("# command\n")
    handle.write(shlex.join(command) + "\n")
    handle.write("#\n")
    handle.flush()


def _child_env(settings: Settings) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    roots = [str(settings.wan_dir)]
    if existing:
        roots.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HOME", "/models/.hf")
    return env


class Engine(Protocol):
    name: str

    def describe(self) -> dict: ...

    def available(self) -> tuple[bool, str]: ...

    def render(self, request: RenderRequest) -> RenderResult: ...


class WanS2VEngine:
    name = "wan"

    def __init__(self, settings: Settings):
        self.settings = settings

    def describe(self) -> dict:
        return {
            "name": self.name,
            "wan_dir": str(self.settings.wan_dir),
            "model_id": str(self.settings.model_id),
            "gpus": self.settings.gpus,
            "offload_model": self.settings.offload_model,
        }

    def available(self) -> tuple[bool, str]:
        generate = self.settings.wan_dir / "generate.py"
        if not generate.is_file():
            return False, f"{generate} is missing; the image should carry the Wan2.2 checkout"
        if not self.settings.model_id.is_dir():
            return False, f"{self.settings.model_id} is missing; mount /models and run download"
        return True, "ready"

    def _command(self, request: RenderRequest) -> list[str]:
        settings = self.settings
        params = request.params
        return build_command(
            wan_dir=settings.wan_dir,
            ckpt_dir=settings.model_id,
            prompt=request.performance.prompt,
            image=request.performance.image,
            audio=request.model_audio,
            save_file=request.raw_path,
            size_key=SIZE_KEYS[params["size"]],
            infer_frames=params["infer_frames"],
            clips=request.plan.clips,
            seed=params["seed"],
            start_from_ref=params["start_from_ref"],
            sample_steps=params.get("sample_steps"),
            guide_scale=params.get("guide_scale"),
            shift=params.get("shift"),
            offload_model=settings.offload_model,
            convert_model_dtype=settings.convert_model_dtype,
            gpus=settings.gpus,
        )

    def render(self, request: RenderRequest) -> RenderResult:
        ok, detail = self.available()
        if not ok:
            raise EngineUnavailable(detail)
        command = self._command(request)
        request.log_path.parent.mkdir(parents=True, exist_ok=True)
        request.raw_path.parent.mkdir(parents=True, exist_ok=True)
        if request.raw_path.exists():
            request.raw_path.unlink()

        started = time.monotonic()
        with open(request.log_path, "a", encoding="utf-8", buffering=1) as log:
            _write_log_header(log, command, request.plan)
            # start_new_session puts torchrun's ranks in their own process
            # group, so a timeout can take the whole tree down rather than
            # leaving an orphan holding the GPU.
            process = subprocess.Popen(
                command,
                cwd=str(self.settings.wan_dir),
                env=_child_env(self.settings),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=self.settings.render_timeout_s)
            except subprocess.TimeoutExpired:
                _terminate(process)
                raise EngineError(
                    f"render exceeded RENDER_TIMEOUT_S={self.settings.render_timeout_s}s; "
                    f"log: {request.log_path}"
                ) from None
        elapsed = time.monotonic() - started

        if returncode != 0:
            raise EngineError(
                f"generate.py exited {returncode} after {elapsed:.0f}s. Last lines:\n"
                f"{_tail(request.log_path)}"
            )
        if not request.raw_path.is_file() or request.raw_path.stat().st_size == 0:
            raise EngineError(
                f"generate.py exited 0 but wrote no video. Last lines:\n{_tail(request.log_path)}"
            )
        return RenderResult(
            path=request.raw_path,
            log_path=request.log_path,
            seconds=elapsed,
            command=command,
            returncode=returncode,
        )


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (AttributeError, OSError):
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (AttributeError, OSError):
            process.kill()


class MockEngine:
    """Writes a placeholder file so queue/API tests need neither CUDA nor ffmpeg."""

    name = "mock"

    def __init__(self, settings: Settings | None = None):
        self.settings = settings

    def describe(self) -> dict:
        return {"name": self.name, "synthetic": True}

    def available(self) -> tuple[bool, str]:
        return True, "mock engine"

    def render(self, request: RenderRequest) -> RenderResult:
        started = time.monotonic()
        request.raw_path.parent.mkdir(parents=True, exist_ok=True)
        request.raw_path.write_bytes(b"mock engine: no video was rendered\n")
        request.log_path.write_text(
            "# mock engine\n# no subprocess, no GPU\n", encoding="utf-8"
        )
        return RenderResult(
            path=request.raw_path,
            log_path=request.log_path,
            seconds=time.monotonic() - started,
            command=["mock"],
            synthetic=True,
        )


def build_engine(settings: Settings) -> Engine:
    if settings.engine == "mock":
        return MockEngine(settings)
    return WanS2VEngine(settings)


def pick_seed(requested: int) -> int:
    """A known seed, always. -1 becomes a concrete number recorded in the manifest."""
    return requested if requested >= 0 else random.randint(0, 2**31 - 1)
