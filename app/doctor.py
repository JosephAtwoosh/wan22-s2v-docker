"""Preflight checks: is this container able to render a performance?

Read-only. Every check says what it looked for, so a failure is actionable
rather than "GPU error". Run it before renting a card, and again on the card.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .config import Settings

OK = "ok"
WARN = "warn"
FAIL = "fail"
INFO = "info"

# (relative path, minimum bytes, label) for a complete Wan2.2-S2V-14B folder.
# Sizes come from the Hugging Face file listing; the minimums leave room for
# LFS metadata pointer files being detected as truncated.
WEIGHT_FILES: tuple[tuple[str, int, str], ...] = (
    ("diffusion_pytorch_model-00001-of-00004.safetensors", 9_500_000_000, "DiT shard 1/4"),
    ("diffusion_pytorch_model-00002-of-00004.safetensors", 9_400_000_000, "DiT shard 2/4"),
    ("diffusion_pytorch_model-00003-of-00004.safetensors", 9_500_000_000, "DiT shard 3/4"),
    ("diffusion_pytorch_model-00004-of-00004.safetensors", 2_600_000_000, "DiT shard 4/4"),
    ("diffusion_pytorch_model.safetensors.index.json", 1_000, "DiT shard index"),
    ("models_t5_umt5-xxl-enc-bf16.pth", 11_000_000_000, "UMT5-XXL text encoder"),
    ("Wan2.1_VAE.pth", 450_000_000, "Wan2.1 VAE"),
    ("config.json", 100, "model config"),
    ("google/umt5-xxl/tokenizer.json", 1_000_000, "UMT5 tokenizer"),
    ("wav2vec2-large-xlsr-53-english/config.json", 100, "wav2vec2 config"),
    ("wav2vec2-large-xlsr-53-english/model.safetensors", 1_000_000_000, "wav2vec2 weights"),
)

# What the image installs, and why it matters if an import fails.
LIBRARIES = (
    ("torch", "PyTorch (CUDA build)"),
    ("torchvision", "image tensors and video writing"),
    ("diffusers", "Wan model loading"),
    ("transformers", "UMT5 and wav2vec2"),
    ("flash_attn", "the attention kernel S2V calls unconditionally"),
    ("decord", "pose-video reader, imported with the pipeline"),
    ("librosa", "audio decoding"),
    ("cv2", "OpenCV, imported by the Wan package"),
    ("fastapi", "HTTP service"),
)


@dataclass
class Check:
    section: str
    status: str
    message: str
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, section: str, status: str, message: str, detail: str = "") -> None:
        self.checks.append(Check(section, status, message, detail))

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [check for check in self.checks if check.status == WARN]

    def exit_code(self) -> int:
        if self.failures:
            return 2
        if self.warnings:
            return 3
        return 0

    def as_dict(self) -> dict:
        return {
            "ready": not self.failures,
            "exit_code": self.exit_code(),
            "failures": len(self.failures),
            "warnings": len(self.warnings),
            "checks": [
                {"section": c.section, "status": c.status, "message": c.message, "detail": c.detail}
                for c in self.checks
            ],
        }


def _run(command: list[str], timeout: int = 20) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _free_gb(path: Path) -> float | None:
    try:
        usage = shutil.disk_usage(str(path))
    except OSError:
        return None
    return usage.free / (1024**3)


def run_checks(settings: Settings) -> Report:
    report = Report()

    # ------------------------------------------------------------------ system
    import platform
    import sys

    report.add("system", INFO, f"python {platform.python_version()} on {platform.platform()}")
    report.add("system", INFO, f"wan22-s2v {__version__}")

    # --------------------------------------------------------------------- gpu
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi is None or smi.returncode != 0:
        report.add(
            "gpu",
            FAIL,
            "nvidia-smi is not available; start the container with --gpus all",
            (smi.stderr.strip() if smi else ""),
        )
    else:
        first = (smi.stdout or "").strip().splitlines()[:1]
        if not first:
            report.add("gpu", FAIL, "nvidia-smi ran but reported no GPU")
        else:
            name, vram_mib, driver = (part.strip() for part in first[0].split(","))
            vram_gib = float(vram_mib) / 1024
            report.add("gpu", OK, f"{name}, {vram_gib:.0f} GiB, driver {driver}")
            if vram_gib >= 70:
                report.add("gpu", OK, "VRAM meets the upstream 80 GB single-GPU figure")
            elif vram_gib >= 40:
                report.add(
                    "gpu",
                    WARN,
                    f"{vram_gib:.0f} GiB is below upstream's 80 GB figure: the DiT shards are "
                    "32.6 GB in bf16 and are materialised on the card for every clip, with the "
                    "11.4 GB text encoder also resident during prompt encoding. 480p with short "
                    "clips may finish; 720p is expected to OOM",
                )
            else:
                report.add(
                    "gpu",
                    FAIL,
                    f"{vram_gib:.0f} GiB cannot hold the DiT shards (32.6 GB) plus the CUDA "
                    "context and activations; 48 GB is the practical floor and 80 GB is "
                    "recommended",
                )

    # --------------------------------------------------------------- libraries
    for module, label in LIBRARIES:
        version = _module_version(module)
        if version is None:
            status = FAIL if module in {"torch", "transformers", "diffusers", "flash_attn"} else WARN
            report.add("libraries", status, f"{module} is not importable ({label})")
        else:
            report.add("libraries", OK, f"{module} {version} ({label})")

    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            report.add("libraries", OK, f"torch sees {torch.cuda.device_count()} CUDA device(s)")
        else:
            report.add("libraries", FAIL, "torch is installed but torch.cuda.is_available() is False")
    except Exception:
        pass

    # ------------------------------------------------------------- wan checkout
    generate = settings.wan_dir / "generate.py"
    if generate.is_file():
        report.add("wan", OK, f"upstream generate.py at {generate}")
    else:
        report.add("wan", FAIL, f"{generate} is missing")
    try:
        from wan.configs import WAN_CONFIGS  # noqa: PLC0415

        if "s2v-14B" in WAN_CONFIGS:
            config = WAN_CONFIGS["s2v-14B"]
            report.add(
                "wan",
                OK,
                f"s2v-14B config: {config.sample_steps} steps, shift {config.sample_shift}, "
                f"guidance {config.sample_guide_scale}, {config.sample_fps} fps",
            )
        else:
            report.add("wan", FAIL, "the checkout does not define the s2v-14B task")
    except Exception as exc:
        report.add("wan", FAIL, f"cannot import wan: {exc}")

    # ----------------------------------------------------------------- weights
    model_dir = settings.model_id
    if not model_dir.is_dir():
        report.add("weights", FAIL, f"{model_dir} does not exist; mount /models and run download")
    else:
        for relative, minimum, label in WEIGHT_FILES:
            path = model_dir / relative
            if not path.is_file():
                report.add("weights", FAIL, f"missing {label}: {relative}")
            elif path.stat().st_size < minimum:
                report.add(
                    "weights",
                    FAIL,
                    f"{label} looks truncated ({path.stat().st_size / 1e9:.2f} GB): {relative}",
                )
            else:
                report.add("weights", OK, f"{label} ({path.stat().st_size / 1e9:.2f} GB)")
        free = _free_gb(model_dir)
        if free is not None:
            if free < 5:
                report.add("weights", FAIL, f"{free:.1f} GiB free under {model_dir}; renders need room")
            else:
                report.add("weights", OK, f"{free:.0f} GiB free under {model_dir}")

    # ------------------------------------------------------------- media tools
    for tool, label in ((settings.ffmpeg, "ffmpeg"), (settings.ffprobe, "ffprobe")):
        result = _run([tool, "-version"])
        if result is None or result.returncode != 0:
            report.add("media", FAIL, f"{label} is not runnable")
        else:
            first = (result.stdout or "").strip().splitlines()[:1]
            report.add("media", OK, first[0] if first else label)

    # ------------------------------------------------------------------ storage
    free = _free_gb(settings.output_dir)
    if free is None:
        report.add("storage", WARN, f"cannot read free space for {settings.output_dir}")
    elif free < 2:
        report.add("storage", FAIL, f"{free:.1f} GiB free under {settings.output_dir}")
    else:
        report.add("storage", OK, f"{free:.0f} GiB free under {settings.output_dir}")

    # ------------------------------------------------------------------ service
    if settings.auth_required:
        report.add("service", OK, "WAN_API_KEY is set; the API requires a bearer token")
    else:
        report.add(
            "service",
            WARN,
            "WAN_API_KEY is empty: anyone who can reach the port can queue renders",
        )
    report.add("service", INFO, f"engine={settings.engine}, gpus={settings.gpus}, size={settings.size}")

    return report


def format_report(report: Report) -> str:
    colours = {
        OK: "\033[32m",
        WARN: "\033[33m",
        FAIL: "\033[31m",
        INFO: "\033[2m",
    }
    reset = "\033[0m"
    if not os.isatty(1):
        colours = {key: "" for key in colours}
        reset = ""

    symbols = {OK: "ok ", WARN: "!! ", FAIL: "XX ", INFO: "-- "}
    lines = ["Wan2.2-S2V container preflight", "=" * 29]
    section = None
    for check in report.checks:
        if check.section != section:
            section = check.section
            lines += ["", section.capitalize()]
        colour = colours.get(check.status, "")
        lines.append(f"  {colour}{symbols.get(check.status, '')}{reset} {check.message}")
        if check.detail:
            lines.append(f"       {check.detail}")
    lines.append("")
    lines.append("=" * 29)
    if report.failures:
        lines.append(f"NOT READY: {len(report.failures)} blocking problem(s), {len(report.warnings)} warning(s).")
    elif report.warnings:
        lines.append(f"READY (with warnings): {len(report.warnings)} warning(s).")
    else:
        lines.append("READY: all checks passed.")
    return "\n".join(lines)
