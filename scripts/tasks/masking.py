"""Mask generation: SAM3 + MIT/ComicTextDetector → merged.

``make mask`` is a one-shot orchestrator: it runs SAM and MIT into a
``tempfile.TemporaryDirectory()`` (cross-platform — honors ``TMPDIR`` /
``TEMP``) and writes only the merged result to
``post_image_dataset/masks/<rel>/{stem}_mask.png``. Per-tool intermediates
are never persisted under the project root.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from ._common import PY, ROOT, run

MASK_OUTPUT_DIR = ROOT / "post_image_dataset" / "masks"
RESIZED_IMAGE_DIR = ROOT / "post_image_dataset" / "resized"


def _run_sam(image_dir: Path, out_dir: Path, extra: list[str]) -> None:
    run(
        [
            PY,
            "preprocess/generate_masks.py",
            "--config",
            "configs/sam_mask.yaml",
            "--image-dir",
            str(image_dir),
            "--mask-dir",
            str(out_dir),
            "--checkpoint",
            "models/sam3/sam3.pt",
            "--batch-size",
            "2",
            "--recursive",
            *extra,
        ]
    )


def _run_mit(image_dir: Path, out_dir: Path, extra: list[str]) -> None:
    # MIT_TEXT_THRESHOLD / MIT_DILATE let the GUI's Preprocessing tab tune
    # the MIT masker without editing this file. Defaults match the script's
    # own argparse defaults so direct CLI use is unchanged.
    cmd = [
        PY,
        "preprocess/generate_masks_mit.py",
        "--image-dir",
        str(image_dir),
        "--mask-dir",
        str(out_dir),
        "--model-path",
        "models/mit/model.pth",
        "--recursive",
    ]
    text_threshold = os.environ.get("MIT_TEXT_THRESHOLD")
    if text_threshold:
        cmd += ["--text-threshold", text_threshold]
    dilate = os.environ.get("MIT_DILATE")
    if dilate:
        cmd += ["--dilate", dilate]
    cmd += list(extra)
    run(cmd)


def cmd_mask(extra):
    """Run SAM + MIT into a tempdir, merge, write to post_image_dataset/masks/."""
    with tempfile.TemporaryDirectory(prefix="anima-masks-") as tmp_root:
        tmp_sam = Path(tmp_root) / "sam"
        tmp_mit = Path(tmp_root) / "mit"
        _run_sam(RESIZED_IMAGE_DIR, tmp_sam, [])
        _run_mit(RESIZED_IMAGE_DIR, tmp_mit, [])
        MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        run(
            [
                PY,
                "preprocess/merge_masks.py",
                str(tmp_sam),
                str(tmp_mit),
                "--output-dir",
                str(MASK_OUTPUT_DIR),
                *extra,
            ]
        )


def cmd_mask_clean(_extra):
    if MASK_OUTPUT_DIR.exists():
        shutil.rmtree(MASK_OUTPUT_DIR)
        print(f"  Removed {MASK_OUTPUT_DIR.relative_to(ROOT)}/")
