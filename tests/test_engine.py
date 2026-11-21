"""The upstream command vector.

Quoting is the classic failure: prompts contain apostrophes, quotes, commas and
non-ASCII text, and a shell string loses them. The command is a list, and these
tests hold it there.
"""

from __future__ import annotations

from pathlib import Path

from app.engine import build_command

BASE = dict(
    wan_dir="/opt/Wan2.2",
    ckpt_dir="/models/Wan2.2-S2V-14B",
    prompt="A grey dawn; the cockerel says 'not yet'.",
    image="/performances/shot/start.jpg",
    audio="/performances/shot/audio.wav",
    save_file="/output/shot/render.raw.mp4",
    size_key="1280*720",
    infer_frames=80,
    clips=2,
    seed=1234,
)


def test_required_arguments():
    command = build_command(**BASE)
    assert Path(command[1]) == Path(BASE["wan_dir"]) / "generate.py"
    for token in ("--task", "s2v-14B", "--size", "1280*720", "--ckpt_dir", "--prompt", "--image", "--audio"):
        assert token in command
    assert command[command.index("--num_clip") + 1] == "2"
    assert command[command.index("--base_seed") + 1] == "1234"


def test_prompt_keeps_its_punctuation_in_one_argument():
    command = build_command(**BASE)
    assert command[command.index("--prompt") + 1] == BASE["prompt"]


def test_prompt_is_not_shell_quoted():
    command = build_command(**BASE)
    assert not any(argument.startswith("'") or argument.endswith("'") for argument in command)


def test_start_from_ref_flag():
    assert "--start_from_ref" in build_command(**BASE, start_from_ref=True)
    assert "--start_from_ref" not in build_command(**BASE, start_from_ref=False)


def test_single_gpu_offload_flags():
    command = build_command(**BASE, gpus=1, offload_model=True, convert_model_dtype=True)
    assert command[command.index("--offload_model") + 1] == "True"
    assert "--convert_model_dtype" in command
    assert "--dit_fsdp" not in command


def test_multi_gpu_uses_fsdp_without_offload_flags():
    command = build_command(**BASE, gpus=4, offload_model=True, convert_model_dtype=True)
    assert command[0] == "torchrun"
    assert command[command.index("--nproc_per_node") + 1] == "4"
    assert "--dit_fsdp" in command
    assert "--t5_fsdp" in command
    assert command[command.index("--ulysses_size") + 1] == "4"
    # Offloading a sharded model is counter-productive; upstream's own
    # multi-GPU examples omit both flags.
    assert "--offload_model" not in command
    assert "--convert_model_dtype" not in command


def test_sampling_overrides_are_optional():
    command = build_command(**BASE)
    assert "--sample_steps" not in command
    tuned = build_command(**BASE, sample_steps=30, guide_scale=4.0, shift=3.0)
    assert tuned[tuned.index("--sample_steps") + 1] == "30"
    assert tuned[tuned.index("--sample_guide_scale") + 1] == "4.0"
    assert tuned[tuned.index("--sample_shift") + 1] == "3.0"


def test_paths_are_passed_as_given():
    command = build_command(**BASE)
    assert command[command.index("--ckpt_dir") + 1] == BASE["ckpt_dir"]
    assert command[command.index("--save_file") + 1] == BASE["save_file"]
