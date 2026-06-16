"""Config discovery, load/save, merge, and lint for the GUI.

Qt-free: this module reads/writes TOML and resolves the base → preset →
variant merge chain that drives the Config tab, plus the dataset-blueprint
linter. Kept free of any PySide6 import so it stays unit-testable headless and
so the cheap config queries don't pull the widget stack in.
"""

from __future__ import annotations

import re

import toml

from gui._paths import (
    CONFIGS_DIR,
    CUSTOM_DIR,
    CUSTOM_VARIANTS_DIR,
    GUI_METHODS_DIR,
    METHODS_DIR,
    PRESETS_FILE,
    _METHOD_ORDER,
)
from gui.validation import (
    _base_folder_repeats,
    _base_validation_enabled,
    _base_validation_split_num,
    _variant_folder_repeats_override,
    _variant_validation_override,
    _variant_validation_split_num,
)

# Built-in variant families are discovered from each gui-methods/*.toml file's
# ``[variant]`` table (``family`` / ``label`` / ``description`` / optional
# ``order`` / ``experimental``). The hand-curated _FAMILY_VARIANTS map was
# retired in the Track 2 refactor — adding or renaming a variant is now a
# one-file change.
#
# Display order within a family is ``[variant].order`` (ascending; ties broken
# by file stem). Family ordering in the method combo stays curated via
# ``_METHOD_ORDER``; a family omitted from it is kept off the GUI without
# renaming its file (the picker also accepts an explicit ``methods=`` subset,
# which is how the experimental MethodsTab surfaces soft_tokens / spd / turbo).
# Customs under
# ``configs/gui-methods/custom/`` are intentionally permissive — they don't
# need a ``[variant]`` block and are surfaced under every family the same way
# they were before.


def _read_variant_metadata(path) -> dict:
    """Return the ``[variant]`` table from a gui-methods TOML, or ``{}``.

    Failures (missing file, parse error, missing table) yield an empty dict
    so callers can treat "no metadata" uniformly — built-in validation is
    handled by ``tests/test_gui_variants.py``, not here.
    """
    if not path.is_file():
        return {}
    try:
        data = toml.loads(path.read_text(encoding="utf-8"))
    except (toml.TomlDecodeError, OSError):
        return {}
    meta = data.get("variant")
    return meta if isinstance(meta, dict) else {}


def _builtin_variants_by_family() -> dict[str, list[tuple[int, str, str]]]:
    """Map family → list of (order, stem, label) tuples for built-in variants.

    Built-in = directly under ``configs/gui-methods/`` (not the ``custom/``
    subdir). Files without a ``[variant].family`` are dropped silently —
    they're either malformed or intentionally hidden, and listing them under
    a guessed family would just re-introduce the stale-map problem.
    """
    by_family: dict[str, list[tuple[int, str, str]]] = {}
    if not GUI_METHODS_DIR.is_dir():
        return by_family
    for path in GUI_METHODS_DIR.glob("*.toml"):
        meta = _read_variant_metadata(path)
        family = meta.get("family")
        if not isinstance(family, str) or not family:
            continue
        order = meta.get("order")
        order_int = order if isinstance(order, int) else 100
        label = meta.get("label") if isinstance(meta.get("label"), str) else path.stem
        by_family.setdefault(family, []).append((order_int, path.stem, label))
    for entries in by_family.values():
        entries.sort(key=lambda e: (e[0], e[1]))
    return by_family


def variant_metadata(variant: str) -> dict:
    """Return the ``[variant]`` metadata for a built-in or ``custom/<name>``
    variant. Empty dict when the file has no ``[variant]`` block (custom
    variants may legitimately omit it)."""
    return _read_variant_metadata(variant_path(variant))


def list_methods() -> list[str]:
    """Method families, in a user-friendly order (lora first)."""
    return list(_METHOD_ORDER)


def list_gui_variants(method: str) -> list[str]:
    """gui-methods/*.toml files for the method family + all user customs.

    Built-in variants are filtered to those whose ``[variant].family`` matches
    ``method``, sorted by ``[variant].order`` then by file stem. Custom
    variants in ``configs/gui-methods/custom/*.toml`` are surfaced for every
    family — users name them freely and we don't try to bind a file to a
    specific family.
    """
    by_family = _builtin_variants_by_family()
    ordered = [stem for _, stem, _ in by_family.get(method, [])]
    if CUSTOM_VARIANTS_DIR.exists():
        for p in sorted(CUSTOM_VARIANTS_DIR.glob("*.toml")):
            ordered.append(f"custom/{p.stem}")
    return ordered


def is_custom_variant(name: str) -> bool:
    return name.startswith("custom/")


def custom_variant_path(name: str):
    """Resolve 'custom/<name>' (or bare '<name>') to the on-disk file path."""
    stem = name[len("custom/") :] if name.startswith("custom/") else name
    return CUSTOM_VARIANTS_DIR / f"{stem}.toml"


def variant_path(variant: str):
    """Resolve a variant identifier (built-in or 'custom/<name>') to its file."""
    return GUI_METHODS_DIR / f"{variant}.toml"


def _load_all_presets() -> dict:
    """Built-in sections in ``configs/presets.toml`` plus user-created flat
    files under ``configs/custom/<name>.toml`` (one preset per file)."""
    presets: dict = {}
    if PRESETS_FILE.exists():
        data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
        presets.update({k: v for k, v in data.items() if isinstance(v, dict)})
    if CUSTOM_DIR.exists():
        for p in sorted(CUSTOM_DIR.glob("*.toml")):
            try:
                presets[p.stem] = toml.loads(p.read_text(encoding="utf-8"))
            except (toml.TomlDecodeError, OSError):
                continue
    return presets


def list_presets() -> list[str]:
    return sorted(_load_all_presets())


def is_custom_preset(name: str) -> bool:
    return (CUSTOM_DIR / f"{name}.toml").exists()


def custom_preset_path(name: str):
    return CUSTOM_DIR / f"{name}.toml"


_GROUPS = {
    "Architecture": {
        "network_dim",
        "network_alpha",
        "network_module",
        "network_args",
        "use_ortho",
        "use_timestep_mask",
        "use_moe_style",
        "route_per_layer",
        "router_source",
        "min_rank",
        "alpha_rank_scale",
        "num_experts",
        "balance_loss_weight",
        "balance_loss_warmup_ratio",
        "sigma_feature_dim",
        "router_targets",
        "per_bucket_balance_weight",
        "num_sigma_buckets",
        "specialize_experts_by_sigma_buckets",
        "sigma_bucket_boundaries",
        "use_repa",
        "repa_target_dog",
        "network_train_unet_only",
    },
    "Paths": {
        "pretrained_model_name_or_path",
        "qwen3",
        "vae",
        "path_scope",
        "output_dir",
        "output_name",
        "save_model_as",
        "source_image_dir",
        "resized_image_dir",
        "lora_cache_dir",
        "path_pattern",
    },
    "Training": {
        "learning_rate",
        "max_train_epochs",
        "save_every_n_epochs",
        "checkpointing_epochs",
        "gradient_accumulation_steps",
        "use_shuffled_caption_variants",
        "caption_dropout_rate",
        "optimizer_type",
        "lr_scheduler",
        "timestep_sampling",
        "discrete_flow_shift",
        "masked_loss",
        "use_valid",
        "validation_split_num",
        "repeat_by_folder_name",
    },
    "Samples": {
        "sample_prompts",
        "sample_every_n_epochs",
        "sample_at_first",
        "sample_decode_inline",
    },
    "Performance": {
        "attn_mode",
        "gradient_checkpointing",
        "unsloth_offload_checkpointing",
        "activation_memory_budget",
        "blocks_to_swap",
        "torch_compile",
        "cache_llm_adapter_outputs",
        "mixed_precision",
        "vae_chunk_size",
        "vae_disable_cache",
        "use_vae_cache",
        "use_text_cache",
        "skip_cache_check",
        "layer_start",
        "use_cmmd",
    },
}
_K2G = {k: g for g, ks in _GROUPS.items() for k in ks}
# Preprocess-time knobs (target_res, drop_lowres_images, min_pixels) are owned by
# the Preprocess tab. The tab persists GUI-profile overrides in the selected
# gui-method variant metadata and uses configs/preprocess.toml only as the CLI
# default/fallback. target_res is dual-use (train.py reads it only to size the
# compile cache). Hide them from the config form to keep a single source of truth
# and avoid the two surfaces silently drifting.
_SKIP = {
    "base_config",
    "dataset_config",
    "general",
    "datasets",
    "variant",
    "target_res",
    "drop_lowres_images",
    "min_pixels",
}

# Virtual keys appear in the form like normal fields but don't round-trip as
# flat TOML keys — they're derived from / written into structured sections
# (e.g. ``use_valid`` toggles a `[[datasets]]` validation_split_num override).
# The save loop in ConfigTab skips these, and per-key apply helpers handle the
# structured write.
_VIRTUAL_KEYS = {"use_valid", "validation_split_num", "repeat_by_folder_name"}

# Fields shown under the "Basic" section. Everything else falls under the
# collapsible "Advanced" section. Picked to cover the knobs a first-time user
# realistically wants to touch (rate/length/output, headline architecture
# size, headline VRAM knobs) without exposing the long tail of regularizer /
# router / adapter-internal parameters. Keep only the common GUI path controls
# in Basic; concrete path overrides stay in Advanced for users who need them.
_BASIC = {
    "learning_rate",
    "max_train_epochs",
    "save_every_n_epochs",
    "checkpointing_epochs",
    "network_dim",
    "network_alpha",
    "network_weights",
    "num_experts",
    "use_shuffled_caption_variants",
    "caption_dropout_rate",
    "masked_loss",
    "gradient_checkpointing",
    "blocks_to_swap",
    "path_scope",
    "output_name",
    "path_pattern",
    "use_valid",
    "validation_split_num",
    "sample_prompts",
    "sample_every_n_epochs",
    "sample_at_first",
    "sample_decode_inline",
}


def is_basic_field(key: str) -> bool:
    return key in _BASIC


# ── Helpers ────────────────────────────────────────────────────


def _load(p) -> dict:
    return toml.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _load_base() -> dict:
    """``base.toml`` overlaid on ``configs/preprocess.toml`` (the split-out
    preprocess-only knobs: source_image_dir / drop_lowres_images / min_pixels).

    Mirrors ``load_path_overrides``' preprocess→base layering so the GUI form
    baseline matches what preprocess/training actually read: a legacy key still
    in base.toml wins; otherwise preprocess.toml supplies it. Use this anywhere
    the GUI needs base.toml as a flat-key baseline."""
    merged = _load(CONFIGS_DIR / "preprocess.toml")
    merged.update(_load(CONFIGS_DIR / "base.toml"))
    return merged


def _save(p, d: dict):
    p.write_text(toml.dumps(d), encoding="utf-8")


def _dataset_lint_sources(variant: str):
    """The (path, label) pairs the dataset-blueprint linter scans: shared
    ``base.toml`` plus the active variant file. ``label`` is what shows up in
    the banner and must match the ``source=`` passed to ``lint_dataset_sections``."""
    return (
        (CONFIGS_DIR / "base.toml", "base.toml"),
        (variant_path(variant), f"gui-methods/{variant}.toml"),
    )


def lint_variant_configs(variant: str) -> list:
    """Scan the dataset-blueprint sections of ``base.toml`` and the variant
    file for keys the trainer's validator will reject (e.g. a stale
    ``resolution`` in ``[[datasets]]``). Returns a list of
    ``library.config.dataset_keys.DatasetKeyIssue``.

    Torch-free: imports only the static allow-list module, so it's safe to run
    on every Config-tab reload without dragging the training stack into the GUI
    process.
    """
    from library.config.dataset_keys import lint_dataset_sections

    issues: list = []
    for path, label in _dataset_lint_sources(variant):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            raw = toml.loads(text)
        except (OSError, toml.TomlDecodeError):
            continue
        issues.extend(lint_dataset_sections(raw, source=label, text=text))
    return issues


def remove_unknown_dataset_keys(variant: str) -> list[str]:
    """Surgically delete the lines flagged by :func:`lint_variant_configs` from
    their source files, returning a list of ``"key (label)"`` descriptions of
    what was removed.

    Comment- and formatting-preserving on purpose: ``base.toml`` is heavily
    commented and the flat ``_save`` round-trips through ``toml.dumps`` (which
    drops all comments), so we edit the raw text line-by-line instead. Each
    flagged line carries its own line number from the linter; we delete in
    descending order so earlier deletions don't shift later targets, and we
    re-verify the line still starts with ``<key> =`` before cutting it.
    """
    from library.config.dataset_keys import lint_dataset_sections

    removed: list[str] = []
    for path, label in _dataset_lint_sources(variant):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            raw = toml.loads(text)
        except (OSError, toml.TomlDecodeError):
            continue
        issues = lint_dataset_sections(raw, source=label, text=text)
        targets = sorted(
            (i for i in issues if i.line is not None),
            key=lambda i: i.line,
            reverse=True,
        )
        if not targets:
            continue
        lines = text.splitlines(keepends=True)
        changed = False
        for issue in targets:
            idx = issue.line - 1
            if 0 <= idx < len(lines) and re.match(
                rf"^\s*{re.escape(issue.key)}\s*=", lines[idx]
            ):
                del lines[idx]
                removed.append(f"{issue.key} ({label})")
                changed = True
        if changed:
            path.write_text("".join(lines), encoding="utf-8")
    return removed


def merged_method_preset(method: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Return (merged_dict, origin_map). origin_map[key] is 'base' | 'preset' | 'method'."""
    base = _load_base()
    pset = _load_all_presets().get(preset, {})
    meth = _load(METHODS_DIR / f"{method}.toml")
    merged: dict = {}
    origin: dict[str, str] = {}
    for k, v in base.items():
        merged[k] = v
        origin[k] = "base"
    for k, v in pset.items():
        merged[k] = v
        origin[k] = "preset"
    for k, v in meth.items():
        merged[k] = v
        origin[k] = "method"
    return merged, origin


def merged_gui_variant_preset(variant: str, preset: str) -> tuple[dict, dict[str, str]]:
    """Merge base + preset + gui-methods/<variant>.toml. The GUI uses this
    instead of `merged_method_preset` so edits/training target the clean
    per-variant file, not the toggle-block methods/ tree."""
    base = _load_base()
    pset = _load_all_presets().get(preset, {})
    meth = _load(GUI_METHODS_DIR / f"{variant}.toml")
    merged: dict = {}
    origin: dict[str, str] = {}
    for k, v in base.items():
        merged[k] = v
        origin[k] = "base"
    for k, v in pset.items():
        merged[k] = v
        origin[k] = "preset"
    for k, v in meth.items():
        merged[k] = v
        origin[k] = "method"

    # GUI-only path scope is stored under [variant] so CLI config loading strips
    # it with the rest of the variant metadata. The Config tab surfaces it as a
    # normal field and expands it into concrete paths at submit time.
    meta = meth.get("variant")
    if isinstance(meta, dict) and isinstance(meta.get("path_scope"), str):
        merged["path_scope"] = meta["path_scope"]
        origin["path_scope"] = "method"
    elif "path_scope" not in merged:
        merged["path_scope"] = ""
        origin["path_scope"] = "base"

    # Inject the `use_valid` virtual key derived from the [[datasets]] block.
    # The variant file may shallow-override base.toml's validation_split_num /
    # validation_split via _apply_dataset_overrides in library/config/io.py; we
    # surface that as a single checkbox the user can flip in the form.
    variant_override = _variant_validation_override(meth)
    if variant_override is not None:
        merged["use_valid"] = variant_override
        origin["use_valid"] = "method"
    else:
        merged["use_valid"] = _base_validation_enabled(base)
        origin["use_valid"] = "base"

    # Inject `validation_split_num` (integer) from the same [[datasets]] block.
    # Shown as a basic field so users can resize the held-out slice directly
    # without dropping to base.toml. When the variant doesn't override it, the
    # value comes from base.toml.
    variant_vsn = _variant_validation_split_num(meth)
    if variant_vsn is not None:
        merged["validation_split_num"] = variant_vsn
        origin["validation_split_num"] = "method"
    else:
        merged["validation_split_num"] = _base_validation_split_num(base)
        origin["validation_split_num"] = "base"

    # Inject `repeat_by_folder_name` (Kohya-style {n}_folder repeats) the same
    # way — it's a dataset-blueprint key, not a flat TOML key, so the form
    # surfaces it as a virtual checkbox written back into the variant's
    # [[datasets]] override.
    variant_rbf = _variant_folder_repeats_override(meth)
    if variant_rbf is not None:
        merged["repeat_by_folder_name"] = variant_rbf
        origin["repeat_by_folder_name"] = "method"
    else:
        merged["repeat_by_folder_name"] = _base_folder_repeats(base)
        origin["repeat_by_folder_name"] = "base"

    # Surface the sample-image knobs as first-class fields even when no TOML in
    # the chain has set them (their argparse defaults are None/None/False, so
    # they'd otherwise never appear in `merged`). A variant that does set them
    # keeps its value + "method" origin from the merge loop above. Cadence uses
    # 0 as the "disabled" sentinel (train.py coerces non-positive → None).
    for _k, _default in (
        ("sample_prompts", []),
        ("sample_every_n_epochs", 0),
        ("sample_at_first", False),
        ("sample_decode_inline", "false"),
    ):
        if _k not in merged:
            merged[_k] = _default
            origin[_k] = "base"
    return merged, origin
