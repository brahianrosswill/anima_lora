"""Pre-launch confirmation dialogs + checkpoint/cache discovery.

The Qt-facing half of the train/preprocess launch flow: resume prompts, cache
reassurance popups, and the on-disk probes (``find_resumable_checkpoint`` /
``count_preprocess_caches``) that decide whether to show them. Split out of the
package root so the Qt-free config logic in ``gui.config_io`` doesn't pull
QMessageBox in.
"""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QWidget

from gui._paths import ROOT
from gui.i18n import t
from library.datasets.path_filter import filter_paths_by_glob


def confirm_resumable_checkpoint(parent: QWidget | None, merged: dict) -> bool:
    """Prompt the user when a checkpoint is on disk; return whether to launch.

    Returns True if training should proceed (Yes = let train.py auto-resume,
    No = wipe the state dir + adapter sidecar so train.py starts fresh),
    False if the user cancelled. Returns True with no prompt when there is
    nothing to resume from — the call site can wrap every train launch in
    this helper unconditionally.
    """
    found = find_resumable_checkpoint(merged)
    if found is None:
        return True
    state_dir, step = found
    choice = QMessageBox.question(
        parent,
        t("resume_checkpoint_title"),
        t("resume_checkpoint_question", step=step),
        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
        QMessageBox.Yes,
    )
    if choice == QMessageBox.Cancel:
        return False
    if choice == QMessageBox.Yes:
        return True
    # No → start fresh. Wipe both the state dir and the sibling
    # ``-checkpoint.safetensors`` adapter so train.py's auto_resume sees
    # nothing on disk. Bail with a warning if the deletion fails — better
    # than silently launching a resume the user explicitly opted out of.
    import shutil

    sidecar = state_dir.parent / f"{state_dir.name.removesuffix('-state')}.safetensors"
    try:
        shutil.rmtree(state_dir)
        if sidecar.is_file():
            sidecar.unlink()
    except OSError as e:
        QMessageBox.warning(
            parent,
            t("error"),
            t("resume_checkpoint_delete_failed", error=str(e)),
        )
        return False
    return True


# Cache-file suffixes written by the preprocess scripts. Kept in sync with
# scripts/preprocess/cache_latents.py, cache_text_embeddings.py, cache_pe_encoder.py.
_LATENT_SUFFIX = "_anima.npz"
_TE_SUFFIX = "_anima_te.safetensors"
_PE_SUFFIX = "_anima_pe.safetensors"


def count_preprocess_caches(
    cache_dir: Path, path_pattern: str | None = None
) -> dict[str, int]:
    """Count existing latent / TE / PE cache sidecars under a cache directory.

    Returns zeros (without raising) if the directory does not exist. Used to
    surface a reassurance popup that ``make preprocess`` reuses existing caches
    rather than wiping them — a recurring point of confusion for new users.

    Walks recursively so nested caches (mirroring a subfoldered source tree)
    are counted.
    """
    out = {"latents": 0, "te": 0, "pe": 0}
    if not cache_dir.is_dir():
        return out
    paths = [p for p in cache_dir.rglob("*") if p.is_file()]
    if path_pattern and path_pattern != "*":
        keep = filter_paths_by_glob(
            [str(p) for p in paths],
            str(cache_dir),
            path_pattern,
        )
        paths = [p for p, k in zip(paths, keep) if k]
    for p in paths:
        n = p.name
        if n.endswith(_TE_SUFFIX):
            out["te"] += 1
        elif n.endswith(_PE_SUFFIX):
            out["pe"] += 1
        elif n.endswith(_LATENT_SUFFIX):
            out["latents"] += 1
    return out


def confirm_existing_caches(
    parent: QWidget | None, cache_dir: Path, require_pe: bool = False
) -> bool:
    """Reassure the user that existing preprocess caches will be reused, not
    deleted. Returns True to proceed, False if the user cancelled.

    No-op (returns True without prompting) when the cache directory is empty
    or missing, so the call site can wrap every preprocess launch in this.
    """
    counts = count_preprocess_caches(cache_dir)
    has_any = (
        counts["latents"] > 0 or counts["te"] > 0 or (require_pe and counts["pe"] > 0)
    )
    if not has_any:
        return True

    parts: list[str] = []
    if counts["latents"]:
        parts.append(t("preprocess_cache_count_latents", n=counts["latents"]))
    if counts["te"]:
        parts.append(t("preprocess_cache_count_te", n=counts["te"]))
    if require_pe and counts["pe"]:
        parts.append(t("preprocess_cache_count_pe", n=counts["pe"]))

    body = t(
        "preprocess_existing_caches_body",
        cache_dir=str(cache_dir),
        items="  • " + "\n  • ".join(parts),
    )
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Information)
    box.setWindowTitle(t("preprocess_existing_caches_title"))
    box.setText(body)
    box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Ok)
    return box.exec() == QMessageBox.Ok


def confirm_train_using_cache(
    parent: QWidget | None, cache_dir: Path, require_pe: bool = False
) -> bool | None:
    """Train-side cache confirmation: returns True to launch training against
    the existing cache, False if the user cancelled, or None when no cache was
    found on disk (caller should auto-chain a preprocess run instead).

    Distinct from ``confirm_existing_caches`` (which reassures during
    Preprocess) — this gates Train and exposes the empty-cache case as a
    separate ``None`` so the caller can branch into the auto-preprocess flow.

    ``require_pe`` (set when ``use_repa`` is on) makes the PE feature cache
    mandatory: a core latent/TE cache that lacks PE sidecars still returns
    ``None`` so the caller auto-chains a (PE-caching) preprocess pass, rather
    than launching a REPA run whose alignment target is silently absent. This
    is the common "preprocessed before enabling REPA" case.
    """
    counts = count_preprocess_caches(cache_dir)
    has_core = counts["latents"] > 0 or counts["te"] > 0
    # REPA on + a built core cache but no PE sidecars → treat as cache-missing
    # so Train rebuilds the PE caches. preprocess is idempotent (it skips the
    # latents/TE already on disk), so this only adds the missing PE pass.
    if require_pe and has_core and counts["pe"] == 0:
        return None
    has_any = has_core or (require_pe and counts["pe"] > 0)
    if not has_any:
        return None

    parts: list[str] = []
    if counts["latents"]:
        parts.append(t("preprocess_cache_count_latents", n=counts["latents"]))
    if counts["te"]:
        parts.append(t("preprocess_cache_count_te", n=counts["te"]))
    if require_pe and counts["pe"]:
        parts.append(t("preprocess_cache_count_pe", n=counts["pe"]))

    body = t(
        "train_using_cache_body",
        cache_dir=str(cache_dir),
        items="  • " + "\n  • ".join(parts),
    )
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle(t("train_using_cache_title"))
    box.setText(body)
    box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
    box.setDefaultButton(QMessageBox.Yes)
    return box.exec() == QMessageBox.Yes


def find_resumable_checkpoint(merged: dict) -> tuple[Path, int] | None:
    """If the merged config has a writable ``checkpointing_epochs`` and an
    on-disk checkpoint state directory exists with a usable ``train_state.json``,
    return ``(state_dir, current_step)``. Returns ``None`` when there is
    nothing to resume — that's the common case and callers should treat it as
    "just launch training normally".

    Mirrors ``library.training.checkpoints.AnimaCheckpointer.auto_resume``: the
    same ``<output_dir>/<output_name>-checkpoint-state/`` path that ``train.py``
    would auto-pick up. We deliberately do NOT enforce ``current_step <
    max_train_steps`` here — that check varies with dataset size and is
    re-evaluated at launch; the GUI prompt only needs to know "is there
    something on disk that train.py would consider resumable".
    """
    if not merged.get("checkpointing_epochs"):
        return None
    output_dir = merged.get("output_dir")
    output_name = merged.get("output_name") or "last"
    if not output_dir:
        return None
    state_dir = ROOT / output_dir / f"{output_name}-checkpoint-state"
    train_state_file = state_dir / "train_state.json"
    if not train_state_file.is_file():
        return None
    try:
        data = json.loads(train_state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    step = int(data.get("current_step", 0))
    return state_dir, step
