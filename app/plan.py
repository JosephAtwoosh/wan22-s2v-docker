"""Pure planning math for one performance.

Nothing here imports torch, so `plan` works on a laptop and in CI. The two
functions that matter mirror upstream Wan2.2 exactly:

* :func:`output_size` is a port of ``WanS2V.get_size_less_than_area()``
  (``wan/speech2video.py``). It answers "what resolution will this still be
  rendered at?" without loading a model.
* :func:`clip_count` replaces upstream's ``int(n / batch) + 1`` bucket count
  with the smallest number of 80-frame clips that still covers the audio. At an
  exact multiple of the clip length upstream asks for one extra clip (a 5.000 s
  line becomes two clips, 10 s of generation for 5 s of sound); the renderer
  passes ``--num_clip`` so that extra clip is not paid for.

Both are pinned by tests/test_plan.py against values published here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

# Wan2.2-S2V samples at 16 fps (config shared_config.sample_fps) and its motion
# window is 73 frames (wan_s2v_14B.transformer.motion_frames).
FPS = 16
MOTION_FRAMES = 73
DEFAULT_INFER_FRAMES = 80

# 1024*704 is the default target area in upstream get_size_less_than_area().
DEFAULT_TARGET_AREA = 1024 * 704


def output_size(
    height: int,
    width: int,
    target_area: int = DEFAULT_TARGET_AREA,
    divisor: int = 64,
) -> tuple[int, int]:
    """Return the (height, width) Wan will render for a reference still.

    Port of ``WanS2V.get_size_less_than_area``. The still is scaled down (never
    up) until it fits ``target_area``, then padded up to a multiple of
    ``divisor`` in each axis. The aspect ratio is therefore preserved to within
    a padding row/column; upstream centre-crops the still to this size.
    """
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if target_area <= 0:
        raise ValueError("target_area must be positive")

    if height * width <= target_area:
        # Already small enough: pad only. Never upscale a reference still.
        max_upper_area = target_area
        min_scale = 0.1
        max_scale = 1.0
    else:
        max_upper_area = target_area
        d = divisor - 1
        b = d * (height + width)
        a = height * width
        c = d**2 - max_upper_area
        min_scale = (-b + math.sqrt(b**2 - 2 * a * c)) / (2 * a)
        max_scale = math.sqrt(max_upper_area / (height * width))

    find_it = False
    padded_height = padded_width = 0
    for i in range(100):
        scale = max_scale - (max_scale - min_scale) * i / 100
        new_height, new_width = int(height * scale), int(width * scale)
        pad_height = (divisor - new_height % divisor) % divisor
        pad_width = (divisor - new_width % divisor) % divisor
        padded_height, padded_width = new_height + pad_height, new_width + pad_width
        if padded_height * padded_width <= max_upper_area:
            find_it = True
            break

    if find_it:
        return padded_height, padded_width

    # Fallback: derive dimensions from the aspect ratio, aligned to divisor.
    aspect_ratio = width / height
    target_width = int((target_area * aspect_ratio) ** 0.5 // divisor * divisor)
    target_height = int((target_area / aspect_ratio) ** 0.5 // divisor * divisor)
    if target_width >= width or target_height >= height:
        target_width = int(width // divisor * divisor)
        target_height = int(height // divisor * divisor)
    return target_height, target_width


def upstream_clip_count(audio_seconds: float, infer_frames: int = DEFAULT_INFER_FRAMES) -> int:
    """The clip count upstream would pick: ``int(audio_frames / batch) + 1``.

    audio frames are at 16 fps once the wav2vec2 features have been resampled,
    so ``audio_frames / batch_frames == audio_seconds * 16 / infer_frames``.
    """
    if audio_seconds <= 0:
        raise ValueError("audio duration must be positive")
    return int(audio_seconds * FPS / infer_frames) + 1


def clip_count(audio_seconds: float, infer_frames: int = DEFAULT_INFER_FRAMES) -> int:
    """Smallest clip count whose generated video covers the audio.

    A clip yields ``infer_frames`` frames at 16 fps, i.e. ``infer_frames / 16``
    seconds. One clip is the floor: the model cannot render less.
    """
    if audio_seconds <= 0:
        raise ValueError("audio duration must be positive")
    # Round the ratio before ceil so 5.000 s at 80 frames does not become 2
    # clips because of binary-float noise in the probe's duration.
    ratio = round(audio_seconds * FPS, 6) / infer_frames
    return max(1, math.ceil(ratio))


def output_frames(
    clips: int, infer_frames: int = DEFAULT_INFER_FRAMES, start_from_ref: bool = True
) -> int:
    """Frames the pipeline returns.

    With ``--start_from_ref`` the reference still is the first frame and every
    clip contributes ``infer_frames`` frames. Without it upstream drops the
    first three frames of the first clip (``drop_first_motion``), so the video
    is 3 frames (0.1875 s) shorter than the clip count implies.
    """
    if clips < 1:
        raise ValueError("clips must be >= 1")
    dropped = 0 if start_from_ref else 3
    return clips * infer_frames - dropped


@dataclass(frozen=True)
class Plan:
    """What a render will cost and produce, calculated before the GPU starts."""

    target_area: int
    infer_frames: int
    fps: int
    start_from_ref: bool

    out_height: int
    out_width: int
    image_height: int
    image_width: int

    audio_seconds: float
    clips: int
    frames: int
    video_seconds: float

    # What upstream would have generated, and what this planner saves.
    upstream_clips: int
    clips_saved: int

    preview: bool
    audio_truncated: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"still        {self.image_width}x{self.image_height} -> render {self.out_width}x{self.out_height}"
            f" ({self.target_area} px area)",
            f"audio        {self.audio_seconds:.3f} s",
            f"clips        {self.clips} x {self.infer_frames} frames"
            + (f" (upstream would ask for {self.upstream_clips})" if self.clips_saved else ""),
            f"video        {self.frames} frames @ {self.fps} fps = {self.video_seconds:.3f} s",
        ]
        if self.preview:
            lines.append(
                f"preview      capped at {self.clips} clip(s); "
                f"{max(0.0, self.audio_seconds - self.video_seconds):.3f} s of audio will be cut"
            )
        if self.audio_truncated:
            lines.append("audio cut    the render is shorter than the audio track (-shortest)")
        for warning in self.warnings:
            lines.append(f"warning      {warning}")
        return "\n".join(lines)


def plan(
    image_size: tuple[int, int],
    audio_seconds: float,
    *,
    target_area: int = DEFAULT_TARGET_AREA,
    infer_frames: int = DEFAULT_INFER_FRAMES,
    start_from_ref: bool = True,
    max_clips: int | None = None,
) -> Plan:
    """Plan one performance.

    ``image_size`` is (height, width) of the reference still. ``max_clips`` caps
    the render for previews; the audio beyond the cap is cut by ffmpeg's
    ``-shortest`` in the same way an over-long track is.
    """
    image_height, image_width = image_size
    if infer_frames % 4 != 0:
        raise ValueError(f"infer_frames must be a multiple of 4, got {infer_frames}")
    if infer_frames < 12 or infer_frames > 160:
        raise ValueError("infer_frames must be between 24 and 160")
    if audio_seconds <= 0:
        raise ValueError("audio duration must be positive")
    if max_clips is not None and max_clips < 1:
        raise ValueError("max_clips must be >= 1")

    height, width = output_size(image_height, image_width, target_area=target_area)
    wanted_clips = clip_count(audio_seconds, infer_frames)
    clips = min(wanted_clips, max_clips) if max_clips else wanted_clips
    frames = output_frames(clips, infer_frames, start_from_ref)
    seconds = frames / FPS

    warnings: list[str] = []
    if audio_seconds < 1.0:
        warnings.append(
            "audio is under a second; the model still renders a full clip, so expect a long tail"
        )
    if audio_seconds / (infer_frames / FPS) >= 40:
        warnings.append(
            f"{audio_seconds:.0f} s is {wanted_clips} clips and one process; split long "
            "performances so a failure does not lose the whole take"
        )
    if clips == wanted_clips and seconds + 1e-6 < audio_seconds:
        warnings.append(
            "generated video is shorter than the audio by "
            f"{audio_seconds - seconds:.3f} s; the tail will be cut"
        )

    if image_width * image_height < 0.5 * target_area:
        warnings.append(
            f"still is {image_width}x{image_height}; Wan never upscales a reference, so the "
            f"clip renders at {width}x{height} rather than the preset's full area"
        )

    return Plan(
        target_area=target_area,
        infer_frames=infer_frames,
        fps=FPS,
        start_from_ref=start_from_ref,
        out_height=height,
        out_width=width,
        image_height=image_height,
        image_width=image_width,
        audio_seconds=round(audio_seconds, 3),
        clips=clips,
        frames=frames,
        video_seconds=round(seconds, 3),
        upstream_clips=upstream_clip_count(audio_seconds, infer_frames),
        clips_saved=max(0, upstream_clip_count(audio_seconds, infer_frames) - wanted_clips),
        preview=clips < wanted_clips,
        audio_truncated=clips < wanted_clips or seconds + 1e-6 < audio_seconds,
        warnings=tuple(warnings),
    )
