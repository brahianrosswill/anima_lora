"""Shared path constants for the GUI package.

Foundation module — imported by every other ``gui`` submodule and by
``gui/__init__.py``. Deliberately dependency-free (no Qt, no other ``gui``
imports) so it can be imported first without risking a cycle.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

METHODS_DIR = CONFIGS_DIR / "methods"
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
CUSTOM_DIR = CONFIGS_DIR / "custom"
# User-created variants live alongside the curated gui-methods files but in
# their own subdirectory so they're easy to find and don't pollute the
# built-in family list.
CUSTOM_VARIANTS_DIR = GUI_METHODS_DIR / "custom"


_METHOD_ORDER = (
    "lora",
    "tlora",
    "hydralora",
    "fera",
    "chimera",
    "soft_tokens",
    "ip_adapter",
    "easycontrol",
)
