"""Every import on the S2V render path must be installed in the image.

This is the check that would have caught the first GPU verification failure:
``generate.py`` imports ``wan.utils.prompt_extend`` at module load, which imports
``dashscope`` unconditionally, even though this service never uses prompt
extension. Nothing that imports ``wan`` can run on a GPU-less CI runner (loading
``flash_attn`` initialises CUDA and aborts), so the test is static: parse the
files the render path touches, resolve every top-level third-party module with
``importlib.util.find_spec`` (which does not execute the module), and fail on a
missing one.

It only runs where the upstream checkout exists, i.e. inside the built image;
elsewhere it is skipped.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import sys

import pytest

WAN_DIR = pathlib.Path(os.environ.get("WAN_DIR", "/opt/Wan2.2"))

pytestmark = pytest.mark.skipif(
    not (WAN_DIR / "generate.py").is_file(),
    reason="needs the Wan2.2 checkout (this test runs inside the image)",
)

# Imported only by paths this service does not take: CosyVoice TTS, the
# ModelScope/git download helpers, local-VLM prompt extension, FlashAttention 3,
# and Wan-Animate's preprocessing (needs sam2, hydra, moviepy, ...).
OPTIONAL = {
    "cosyvoice",
    "flash_attn_interface",
    "git",
    "modelscope",
    "qwen_vl_utils",
}
OPTIONAL_PATHS = ("wan/modules/animate/preprocess/",)


def _render_path_files() -> list[pathlib.Path]:
    files = [WAN_DIR / "generate.py"]
    for path in sorted((WAN_DIR / "wan").rglob("*.py")):
        relative = path.relative_to(WAN_DIR).as_posix()
        if any(fragment in relative for fragment in OPTIONAL_PATHS):
            continue
        files.append(path)
    return files


def _top_level_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_every_import_on_the_render_path_is_installed():
    found: dict[str, set[str]] = {}
    for path in _render_path_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - upstream is valid Python
            pytest.fail(f"cannot parse {path}: {exc}")
        for name in _top_level_imports(tree):
            if name in sys.stdlib_module_names or name == "wan":
                continue
            found.setdefault(name, set()).add(path.relative_to(WAN_DIR).as_posix())

    missing = {
        name: sorted(where)
        for name, where in found.items()
        if name not in OPTIONAL and importlib.util.find_spec(name) is None
    }
    assert not missing, (
        "generate.py cannot import these modules, so every render would fail: "
        + ", ".join(f"{name} ({', '.join(where[:3])})" for name, where in sorted(missing.items()))
    )
