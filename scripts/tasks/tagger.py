"""Anima Tagger task entry-points: preprocess (vocab + feature cache + resized
cache), train (two-stage: head-only → PE-LoRA warm-start), predict (single-
image debug).

All four invoke ``python -m scripts.anima_tagger.cli`` with the appropriate
``--mode`` flag. Extra args are forwarded verbatim, so per-mode knobs
(``--epochs``, ``--image``, ``--show_scores``, …) work as documented in
``scripts/anima_tagger/cli.py``.
"""

from __future__ import annotations

from pathlib import Path

from ._common import PY, run

_DEFAULT_OUT_DIR = "models/captioners/anima-tagger-v1"
_DEFAULT_ENCODER = "pe"


def _tagger(mode: str, extra):
    run([PY, "-m", "scripts.anima_tagger.cli", "--mode", mode, *extra])


def _extract_value(extra, flag: str, default: str) -> str:
    """Read ``--flag VAL`` (or ``--flag=VAL``) from ``extra`` without consuming
    it — both training stages still need the original list. First match wins
    (argparse takes the last; the discrepancy only matters if the user passes
    the flag twice, which is unusual)."""
    for i, tok in enumerate(extra):
        if tok == flag:
            return extra[i + 1] if i + 1 < len(extra) else default
        if tok.startswith(f"{flag}="):
            return tok.split("=", 1)[1]
    return default


def cmd_preprocess_tagger(extra):
    """Build the tagger vocab/manifest + cache pooled PE features + cache resized images.

    Three idempotent stages:

    1. ``--mode build_vocab`` — scans caption sources, emits ``vocab.json`` +
       ``dataset.json``.
    2. ``--mode build_features`` — encodes each manifest image through frozen
       PE-Core, mean-pools patches, writes per-stem safetensors. Consumed by
       the Stage-1 cached-encoder training path.
    3. ``--mode build_resized`` — LANCZOS-resizes each manifest image to its
       PE bucket, writes uint8 safetensors. Consumed by the Stage-2 PE-LoRA
       training path (where the encoder is unfrozen and pre-pooled features
       can't track it).

    Requires ``CAPTION_CORPUS_DIR`` set in ``anima_lora/.env`` (or the relevant
    paths passed via flags). Extra args are forwarded to ALL three stages —
    pass only flags they share (e.g. ``--out_dir``, ``--encoder``, ``--device``).
    """
    _tagger("build_vocab", extra)
    _tagger("build_features", extra)
    _tagger("build_resized", extra)


def cmd_tagger(extra):
    """Two-stage Anima Tagger train: head-only on cached features → PE-LoRA warm-start.

    **Stage 1** (``--pe_lora_rank 0``) — head only, encoder frozen.
        Reads pre-pooled features from ``<out_dir>/.cache/pooled-pe/``. Fast
        (no encoder forward per step). Saves the head to
        ``<out_dir>/model.safetensors``.

    **Stage 2** (``--pe_lora_rank > 0``) — PE-LoRA, warm-started from Stage 1.
        Reads pre-resized images from ``<out_dir>/.cache/resized-<encoder>/``
        (auto-built via ``build_resized`` if missing). Loads Stage 1's head via
        ``--init_head_from``, then jointly fine-tunes the head + the trailing
        PE-Core blocks. Overwrites ``<out_dir>/model.safetensors`` with the
        Stage-2 best, and writes ``<out_dir>/pe_lora.safetensors``.

    Stage-specific defaults (epochs, batch_size, lr, pe_lora_*) are applied
    first; ``extra`` flags follow so they override (argparse last-wins). A
    single flag in ``extra`` (e.g. ``--epochs 50``) hits BOTH stages — for
    fine-grained per-stage tuning, invoke
    ``python -m scripts.anima_tagger.cli --mode train ...`` directly per
    stage.

    Prerequisite: ``make preprocess-tagger`` (build_vocab + build_features +
    build_resized).
    """
    out_dir = _extract_value(extra, "--out_dir", _DEFAULT_OUT_DIR)
    encoder = _extract_value(extra, "--encoder", _DEFAULT_ENCODER)
    head_path = Path(out_dir) / "model.safetensors"
    resized_dir = Path(out_dir) / ".cache" / f"resized-{encoder}"

    # Stage 1: head-only on cached pooled features.
    stage1_defaults = [
        "--pe_lora_rank", "0",
        "--epochs", "75",
        "--batch_size", "64",
        "--lr", "2e-4",
    ]
    print(f"[tagger] stage 1 / 2: head-only train on cached pooled features → {head_path}")
    _tagger("train", [*stage1_defaults, *extra])

    if not head_path.exists():
        raise SystemExit(
            f"stage 1 finished but {head_path} missing — refusing to run stage 2"
        )

    # Stage 2 reads from the resized image cache. Build it if missing
    # (idempotent — only resizes stems that aren't already cached).
    if not resized_dir.exists() or not any(resized_dir.iterdir()):
        print(f"[tagger] resized cache missing at {resized_dir}, building …")
        _tagger("build_resized", extra)

    # Stage 2: PE-LoRA warm-started from Stage 1.
    stage2_defaults = [
        "--pe_lora_rank", "16",
        "--pe_lora_layers", "4",
        "--epochs", "12",
        "--batch_size", "32",
        "--lr", "1e-4",
        "--pe_lora_lr", "5e-5",
        "--init_head_from", str(head_path),
    ]
    print(f"[tagger] stage 2 / 2: PE-LoRA fine-tune, warm-starting head from {head_path}")
    _tagger("train", [*stage2_defaults, *extra])


def cmd_test_tagger(extra):
    """Single-image debug entry — runs the trained head and prints the caption.

    Without ``--image``, samples a random stem from the val split for a
    side-by-side comparison against ground-truth tags. Pass ``--show_scores``
    to also print rating distribution + top-K kept tags.
    """
    _tagger("predict", extra)
