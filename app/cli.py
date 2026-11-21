"""Command-line interface.

    wan22-s2v doctor                       preflight report
    wan22-s2v download                     fetch the 50 GB checkpoint
    wan22-s2v plan  <performance>          what a render will produce (no GPU)
    wan22-s2v perform <performance>        render one performance
    wan22-s2v batch <dir>                  render every performance under a folder
    wan22-s2v serve                        HTTP API
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from . import __version__
from .config import SIZE_PRESETS, Settings, get_settings
from .doctor import WEIGHT_FILES, format_report, run_checks
from .engine import EngineError, build_engine
from .media import MediaError
from .performance import (
    PerformanceError,
    discover,
    has_performance_artifacts,
    looks_like_performance,
)
from .render import LOG_NAME, VIDEO_NAME, plan_for, render, resolve_params

log = logging.getLogger("wan22_s2v.cli")


def _add_render_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", choices=sorted(SIZE_PRESETS), default=None,
                        help="render area preset; aspect follows the still")
    parser.add_argument("--infer-frames", type=int, default=None,
                        help="frames per clip (multiple of 4; default 80 = 5 s)")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="cap clips for a preview; the tail of the audio is cut")
    parser.add_argument("--seed", type=int, default=None, help="-1 for a recorded random seed")
    parser.add_argument("--steps", type=int, default=None, help="sampling steps (default 40)")
    parser.add_argument("--guide-scale", type=float, default=None,
                        help="classifier-free guidance (default 4.5)")
    parser.add_argument("--shift", type=float, default=None, help="flow-matching shift (default 3.0)")
    parser.add_argument("--conform-fps", type=int, default=None,
                        help="retime the clip to this fps by duplicating frames (0 = 16 fps)")
    parser.add_argument("--start-from-ref", action=argparse.BooleanOptionalAction, default=None,
                        help="use the still as frame 0 (default on)")
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=None,
                        help="offload the model between forwards (default on; needed below 80 GB)")


def _overrides(args: argparse.Namespace) -> dict:
    mapping = {
        "size": getattr(args, "size", None),
        "infer_frames": getattr(args, "infer_frames", None),
        "max_clips": getattr(args, "max_clips", None),
        "seed": getattr(args, "seed", None),
        "sample_steps": getattr(args, "steps", None),
        "guide_scale": getattr(args, "guide_scale", None),
        "shift": getattr(args, "shift", None),
        "conform_fps": getattr(args, "conform_fps", None),
        "start_from_ref": getattr(args, "start_from_ref", None),
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _settings_with(args: argparse.Namespace) -> Settings:
    settings = get_settings()
    if getattr(args, "offload", None) is not None:
        settings = replace(settings, offload_model=bool(args.offload))
    return settings


def _print_plan(plan, params: dict) -> None:
    print(plan.summary())
    print(
        f"params       seed {params['seed']}, {params['sample_steps']} steps, "
        f"guidance {params['guide_scale']}, shift {params['shift']}, "
        f"start_from_ref={params['start_from_ref']}"
    )


def _follow_log(path: Path, stop: threading.Event) -> None:
    """Stream the upstream process log to the terminal while it renders."""
    position = 0
    while not stop.is_set():
        try:
            if path.is_file():
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
        except OSError:
            pass
        stop.wait(1.0)


def _manifest_summary(manifest: dict) -> str:
    output = manifest.get("output") or {}
    timing = manifest.get("timing") or {}
    plan = manifest.get("plan") or {}
    return (
        f"{output.get('width')}x{output.get('height')} "
        f"{output.get('duration') or plan.get('video_seconds')}s "
        f"{output.get('fps')} fps | {timing.get('render_seconds')}s render "
        f"({timing.get('realtime_factor')}x realtime) | {manifest.get('output', {}).get('file')}"
    )


# --------------------------------------------------------------------- commands
def cmd_doctor(args: argparse.Namespace) -> int:
    settings = get_settings()
    report = run_checks(settings)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print(format_report(report))
    return report.exit_code()


def cmd_download(args: argparse.Namespace) -> int:
    settings = get_settings()
    model_dir = Path(args.model_dir) if args.model_dir else settings.model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    ignore = [
        "assets/*",
        "*.md",
        "wav2vec2-large-xlsr-53-english/flax_model.msgpack",
        "wav2vec2-large-xlsr-53-english/pytorch_model.bin",
        "wav2vec2-large-xlsr-53-english/language_model/*",
        "wav2vec2-large-xlsr-53-english/log_*",
        "wav2vec2-large-xlsr-53-english/*_eval_results*.txt",
        "wav2vec2-large-xlsr-53-english/full_eval.sh",
        "wav2vec2-large-xlsr-53-english/eval.py",
    ]
    print(f"Downloading {args.repo} -> {model_dir}")
    print("  ~45.6 GB of weights; the UMT5 encoder alone is 11.4 GB")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("error: huggingface_hub is not installed", file=sys.stderr)
        return 1
    try:
        snapshot_download(
            repo_id=args.repo,
            local_dir=str(model_dir),
            ignore_patterns=ignore,
            max_workers=args.workers,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as exc:
        print(f"error: download failed: {exc}", file=sys.stderr)
        return 1

    missing = [
        relative
        for relative, minimum, _ in WEIGHT_FILES
        if not (model_dir / relative).is_file()
        or (model_dir / relative).stat().st_size < minimum
    ]
    if missing:
        print("download finished but these files are missing or truncated:", file=sys.stderr)
        for relative in missing:
            print(f"  {relative}", file=sys.stderr)
        return 1
    print("download complete and verified")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    settings = _settings_with(args)
    performance = discover(args.target, max_prompt_chars=settings.max_prompt_chars)
    params = resolve_params(performance, settings, _overrides(args))
    plan = plan_for(performance, settings, params)
    if args.json:
        print(json.dumps({"plan": plan.as_dict(), "params": params}, indent=2))
    else:
        print(f"performance  {performance.directory}")
        print(f"still        {performance.image.name}")
        print(f"audio        {performance.audio.name}")
        print(f"prompt       {len(performance.prompt)} chars: {performance.prompt[:120]}"
              + ("..." if len(performance.prompt) > 120 else ""))
        _print_plan(plan, params)
    return 0


def cmd_perform(args: argparse.Namespace) -> int:
    settings = _settings_with(args)
    engine = build_engine(settings)
    performance = discover(args.target, max_prompt_chars=settings.max_prompt_chars)
    params = resolve_params(performance, settings, _overrides(args))
    plan = plan_for(performance, settings, params)
    out_dir = Path(args.out) if args.out else performance.directory
    if not args.quiet:
        print(f"performance  {performance.directory}")
        _print_plan(plan, params)
        print(f"output       {out_dir / VIDEO_NAME}")

    stop = threading.Event()
    follower = None
    if args.follow and not args.quiet:
        follower = threading.Thread(target=_follow_log, args=(out_dir / LOG_NAME, stop), daemon=True)
        follower.start()
    started = time.monotonic()
    try:
        manifest = render(
            performance,
            out_dir,
            settings,
            engine,
            params=params,
            force=args.force,
            job_id=None,
        )
    except (PerformanceError, EngineError, MediaError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        stop.set()
        if follower is not None:
            follower.join(timeout=2)

    if manifest.get("skipped"):
        print("already rendered (manifest inputs match); use --force to redo")
    if not args.quiet:
        print("done         " + _manifest_summary(manifest))
        print(f"wall         {time.monotonic() - started:.0f}s")
    if args.json:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    settings = _settings_with(args)
    engine = build_engine(settings)
    root = Path(args.root)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1
    candidates = sorted(
        (
            entry
            for entry in root.iterdir()
            if entry.is_dir() and (looks_like_performance(entry) or has_performance_artifacts(entry))
        ),
        key=lambda path: path.name.lower(),
    )
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        print(f"error: no performance folders under {root}", file=sys.stderr)
        print("a performance folder needs exactly one image, one audio file and prompt.txt", file=sys.stderr)
        return 1

    out_root = Path(args.out) if args.out else root / "out"
    rows: list[dict] = []
    failures = 0
    skipped_problems = 0
    print(f"{len(candidates)} performance(s) under {root} -> {out_root}")

    for index, directory in enumerate(candidates, start=1):
        label = directory.name
        try:
            performance = discover(directory, max_prompt_chars=settings.max_prompt_chars)
            params = resolve_params(performance, settings, _overrides(args))
            plan = plan_for(performance, settings, params)
        except PerformanceError as exc:
            skipped_problems += 1
            failures += 1
            print(f"[{index}/{len(candidates)}] {label}: INVALID: {exc}", file=sys.stderr)
            rows.append({"name": label, "status": "invalid", "error": str(exc)})
            if args.stop_on_error:
                break
            continue

        if args.dry_run:
            print(
                f"[{index}/{len(candidates)}] {label}: {plan.out_width}x{plan.out_height}, "
                f"{plan.audio_seconds:.2f}s audio, {plan.clips} clip(s), {plan.video_seconds:.2f}s video"
            )
            rows.append(
                {
                    "name": label,
                    "status": "planned",
                    "clips": plan.clips,
                    "audio_seconds": plan.audio_seconds,
                    "video_seconds": plan.video_seconds,
                    "width": plan.out_width,
                    "height": plan.out_height,
                }
            )
            continue

        out_dir = out_root / label
        print(
            f"[{index}/{len(candidates)}] {label}: {plan.out_width}x{plan.out_height}, "
            f"{plan.audio_seconds:.2f}s audio, {plan.clips} clip(s)",
            flush=True,
        )
        try:
            manifest = render(performance, out_dir, settings, engine, params=params, force=args.force)
        except (PerformanceError, EngineError, MediaError, OSError) as exc:
            failures += 1
            print(f"    failed: {exc}", file=sys.stderr, flush=True)
            rows.append({"name": label, "status": "error", "error": str(exc)})
            if args.stop_on_error:
                break
            continue

        output = manifest.get("output") or {}
        timing = manifest.get("timing") or {}
        status = "cached" if manifest.get("skipped") else "done"
        print(
            f"    {status}: {output.get('file')} {output.get('duration')}s "
            f"({timing.get('render_seconds')}s render)",
            flush=True,
        )
        rows.append(
            {
                "name": label,
                "status": status,
                "file": str(out_dir / VIDEO_NAME),
                "duration": output.get("duration"),
                "width": output.get("width"),
                "height": output.get("height"),
                "render_seconds": timing.get("render_seconds"),
                "seed": (manifest.get("params") or {}).get("seed"),
            }
        )

    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "batch.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary      {len(rows)} processed, {failures} failed, {skipped_problems} invalid")
    print(f"report       {csv_path}")
    return 1 if failures else 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import app

    settings = get_settings()
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "info").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(app, host=args.host or settings.host, port=args.port or settings.port,
                log_level=os.environ.get("LOG_LEVEL", "info"))
    return 0


def cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


# ----------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wan22-s2v",
        description="Audio-driven performance from a single still (Wan2.2-S2V-14B).",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="preflight report")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    download = sub.add_parser("download", help="fetch the checkpoint from Hugging Face")
    download.add_argument("--repo", default="Wan-AI/Wan2.2-S2V-14B")
    download.add_argument("--model-dir", default=None)
    download.add_argument("--workers", type=int, default=8)
    download.set_defaults(func=cmd_download)

    plan = sub.add_parser("plan", help="plan a render without touching the GPU")
    plan.add_argument("target")
    plan.add_argument("--json", action="store_true")
    _add_render_flags(plan)
    plan.set_defaults(func=cmd_plan)

    perform = sub.add_parser("perform", help="render one performance")
    perform.add_argument("target")
    perform.add_argument("-o", "--out", default=None, help="output folder (default: the performance folder)")
    perform.add_argument("--force", action="store_true", help="re-render even if the manifest matches")
    perform.add_argument("--follow", action="store_true", help="stream the renderer log")
    perform.add_argument("--quiet", action="store_true")
    perform.add_argument("--json", action="store_true", help="print the manifest when finished")
    _add_render_flags(perform)
    perform.set_defaults(func=cmd_perform)

    batch = sub.add_parser("batch", help="render every performance under a folder")
    batch.add_argument("root")
    batch.add_argument("-o", "--out", default=None, help="output root (default: <root>/out)")
    batch.add_argument("--force", action="store_true")
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--limit", type=int, default=None)
    batch.add_argument("--stop-on-error", action="store_true")
    _add_render_flags(batch)
    batch.set_defaults(func=cmd_batch)

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=cmd_serve)

    version = sub.add_parser("version", help="print the version")
    version.set_defaults(func=cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PerformanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
