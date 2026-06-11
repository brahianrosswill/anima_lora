"""PreprocessingTab — caption shuffle/dropout + SAM3/MIT mask config.

Layout mirrors ConfigTab:
- Top action bar (refresh + per-step Run buttons + Save + Stop)
- Horizontal split: form on left (clickable labels show help on the right),
  explanation panel on the right
- Log panel below in a vertical splitter

Surfaces the knobs that the bare ``make preprocess`` / ``make mask`` paths
hardcode: caption shuffle variant count, per-tag dropout rate, SAM prompt
list / threshold / dilate, MIT text-threshold / dilate.

Settings persist to:
- ``configs/sam_mask.yaml`` — SAM prompts / threshold / dilate (existing
  canonical location, read directly by ``scripts/preprocess/generate_masks.py``).
- ``gui/gui_settings.json`` — TE-cache and MIT knobs, picked up by this
  tab on launch and forwarded to subprocesses via env vars
  (``CAPTION_SHUFFLE_VARIANTS``, ``CAPTION_TAG_DROPOUT_RATE``,
  ``MIT_TEXT_THRESHOLD``, ``MIT_DILATE``) consumed by
  ``scripts/tasks/preprocess.py`` and ``scripts/tasks/masking.py``.
"""

from __future__ import annotations

import html
import json
import re
import sys
import copy
from pathlib import Path

import toml
import yaml
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import (
    IMAGE_EXTS,
    ROOT,
    LazyTabMixin,
    _TargetResWidget,
    _load,
    _save,
    count_preprocess_caches,
    list_gui_variants,
    merged_gui_variant_preset,
    variant_path,
)
from gui import daemon as gui_daemon
from gui.explanations import preprocess_field_help, preprocess_guide
from gui.i18n import t
from gui.progress import TQDM_RE, TqdmProgressTracker, make_progress_bar
from gui.tabs.config_tab import ClickableLabel, ConfigTab, SplitButtonStyle
from library.datasets.subsets import filter_paths_by_glob

SAM_YAML = ROOT / "configs" / "sam_mask.yaml"
PREPROCESS_TOML = ROOT / "configs" / "preprocess.toml"
SETTINGS_FILE = Path(__file__).resolve().parent.parent / "gui_settings.json"

# Defaults match the historical hardcoded values in scripts/tasks/preprocess.py
# and scripts/preprocess/generate_masks_mit.py so a freshly installed GUI runs the
# same pipeline as the bare CLI.
DEFAULT_SOURCE_IMAGE_DIR = "image_dataset"
DEFAULT_PREPROCESS_PATH_PATTERN = "*"
DEFAULT_DROP_LOWRES_IMAGES = True
DEFAULT_MIN_PIXELS = 500000
DEFAULT_TARGET_RES = [1024]
DEFAULT_TE_SHUFFLE_VARIANTS = 4
DEFAULT_TE_TAG_DROPOUT = 0.1
DEFAULT_SAM_PROMPTS = ("speech bubble", "text bubble")
DEFAULT_SAM_THRESHOLD = 0.5
DEFAULT_SAM_DILATE = 5
DEFAULT_MASK_PATH_PATTERN = "*"
DEFAULT_MIT_TEXT_THRESHOLD = 0.8
DEFAULT_MIT_DILATE = 5
DEFAULT_RUN_SAM_MASK = True
DEFAULT_RUN_MIT_MASK = True
PREPROCESS_METHODS = ["lora", "tlora", "hydralora"]
_GUI_PREPROCESS_KEYS = {
    "preprocess_path_pattern",
    "drop_lowres_images",
    "min_pixels",
    "target_res",
    "caption_shuffle_variants",
    "caption_tag_dropout_rate",
}

RESIZED_DIR = ROOT / "post_image_dataset" / "resized"
LORA_CACHE_DIR = ROOT / "post_image_dataset" / "lora"
# Merged masks now live under the cache root alongside the resized tree
# (the SAM/MIT intermediates run through a tempdir during `make mask`).
MASK_DIR = ROOT / "post_image_dataset" / "masks"


def _load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_settings(updates: dict) -> None:
    """Merge ``updates`` into the existing settings JSON, preserving other keys."""
    data = _load_settings()
    data.update(updates)
    SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_preprocess_toml() -> dict:
    """Read configs/preprocess.toml (the preprocess-only knobs split out of
    base.toml). Returns {} if absent/unparseable so callers fall back to
    defaults. The GUI uses this only as the CLI-default fallback; GUI edits are
    stored on the selected gui-method variant."""
    if not PREPROCESS_TOML.exists():
        return {}
    try:
        return toml.loads(PREPROCESS_TOML.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}


def _load_sam_yaml() -> dict:
    if not SAM_YAML.exists():
        return {}
    try:
        return yaml.safe_load(SAM_YAML.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


class _IndentedListDumper(yaml.SafeDumper):
    """SafeDumper that indents list items under mapping keys.

    PyYAML's default dumper writes list items flush with the parent key,
    which is valid YAML but doesn't match the canonical sam_mask.yaml
    formatting (2-space indent on the dash). Overriding ``increase_indent``
    to disable ``indentless`` mode gives us the indented form so saving
    from the GUI doesn't churn the file's whitespace.
    """

    def increase_indent(self, flow=False, indentless=False):  # noqa: D401
        return super().increase_indent(flow, False)


def _load_rules(sam_yaml: dict) -> list[dict]:
    """Normalize either schema into a list of per-card rule dicts.

    A ``rules:`` array is returned card-for-card (per-rule threshold/dilate
    fall back to the top-level values). A flat config (no ``rules:`` key)
    collapses to one catch-all card carrying the top-level prompts.
    """
    default_threshold = float(sam_yaml.get("threshold", DEFAULT_SAM_THRESHOLD))
    default_dilate = int(sam_yaml.get("dilate", DEFAULT_SAM_DILATE))
    raw = sam_yaml.get("rules")
    if raw is None:
        return [
            {
                "path_pattern": "",
                "prompts": sam_yaml.get("prompts") or list(DEFAULT_SAM_PROMPTS),
                "focus_prompts": sam_yaml.get("focus_prompts") or [],
                "threshold": default_threshold,
                "dilate": default_dilate,
            }
        ]
    return [
        {
            "path_pattern": r.get("path_pattern") or "",
            "prompts": r.get("prompts") or [],
            "focus_prompts": r.get("focus_prompts") or [],
            "threshold": float(r.get("threshold", default_threshold)),
            "dilate": int(r.get("dilate", default_dilate)),
        }
        for r in raw
    ]


def _save_sam_yaml(
    rules: list[dict],
    path_pattern: str = DEFAULT_MASK_PATH_PATTERN,
) -> None:
    SAM_YAML.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Each rule routes a subset of images (by path_pattern) to its prompt
        # set; `prompts` mask OUT, `focus_prompts` keep ONLY. Matching rules
        # compose. See scripts/preprocess/generate_masks.py for the full rule.
        "rules": rules,
        # Read by scripts/tasks/masking.py and forwarded to BOTH the SAM and
        # MIT backends; "*" (the default) masks every resized image.
        "path_pattern": path_pattern or DEFAULT_MASK_PATH_PATTERN,
    }
    text = yaml.dump(
        payload,
        Dumper=_IndentedListDumper,
        default_flow_style=False,
        sort_keys=False,
    )
    # Blank line before the trailing global path_pattern, matching the
    # canonical layout's separation between the list and the scalar settings.
    text = text.replace("\npath_pattern:", "\n\npath_pattern:", 1)
    SAM_YAML.write_text(text, encoding="utf-8")


def _filtered_files(root: Path, pattern: str | None, predicate) -> list[Path]:
    if not root.is_dir():
        return []
    paths = [p for p in root.rglob("*") if p.is_file() and predicate(p)]
    if pattern and pattern != "*":
        keep = filter_paths_by_glob([str(p) for p in paths], str(root), pattern)
        paths = [p for p, k in zip(paths, keep) if k]
    return paths


def _count_masks(mask_dir: Path, path_pattern: str | None = None) -> int:
    if not mask_dir.is_dir():
        return 0
    # rglob picks up the nested `<rel>/` subtrees produced by `make mask`
    # under the consolidated layout; legacy flat trees still count correctly.
    return len(
        _filtered_files(
            mask_dir,
            path_pattern,
            lambda p: p.name.endswith("_mask.png"),
        )
    )


def _count_resized(resized_dir: Path, path_pattern: str | None = None) -> int:
    if not resized_dir.is_dir():
        return 0
    # rglob picks up the nested `<rel>/` subtrees produced by recursive
    # resize_images.py; flat trees still count correctly.
    return len(
        _filtered_files(
            resized_dir,
            path_pattern,
            lambda p: p.suffix.lower() in IMAGE_EXTS,
        )
    )


class _RuleCard(QGroupBox):
    """One SAM mask rule editor: path_pattern + prompts/focus + threshold/dilate.

    ``prompts`` mask OUT (ignored in the loss); ``focus_prompts`` keep ONLY
    that subject (reversed polarity). An empty / ``*`` path_pattern is a
    catch-all. Emits ``removed(self)`` when its Remove button is clicked.
    """

    removed = Signal(object)

    def __init__(self, rule: dict, help_cb):
        super().__init__(t("preprocess_sam_rule"))
        self._help_cb = help_cb
        form = QFormLayout(self)

        self.path_pattern_edit = QLineEdit(rule.get("path_pattern", ""))
        self.path_pattern_edit.setPlaceholderText("*")
        self.path_pattern_edit.setToolTip(t("preprocess_sam_rule_path_pattern_tip"))
        form.addRow(
            self._label("sam_rule_path_pattern", t("preprocess_sam_rule_path_pattern")),
            self.path_pattern_edit,
        )

        self.prompts_edit = QPlainTextEdit("\n".join(rule.get("prompts") or []))
        self.prompts_edit.setMaximumHeight(70)
        self.prompts_edit.setStyleSheet("font-family:monospace;")
        self.prompts_edit.setToolTip(t("preprocess_sam_prompts_tip"))
        form.addRow(
            self._label("sam_prompts", t("preprocess_sam_prompts")), self.prompts_edit
        )

        self.focus_prompts_edit = QPlainTextEdit(
            "\n".join(rule.get("focus_prompts") or [])
        )
        self.focus_prompts_edit.setMaximumHeight(70)
        self.focus_prompts_edit.setStyleSheet("font-family:monospace;")
        self.focus_prompts_edit.setToolTip(t("preprocess_sam_focus_prompts_tip"))
        form.addRow(
            self._label("sam_focus_prompts", t("preprocess_sam_focus_prompts")),
            self.focus_prompts_edit,
        )

        self.threshold_edit = QLineEdit(
            f"{float(rule.get('threshold', DEFAULT_SAM_THRESHOLD)):g}"
        )
        self.threshold_edit.setToolTip(t("preprocess_sam_threshold_tip"))
        form.addRow(
            self._label("sam_threshold", t("preprocess_sam_threshold")),
            self.threshold_edit,
        )

        self.dilate_spin = QSpinBox()
        self.dilate_spin.setRange(0, 64)
        self.dilate_spin.setValue(int(rule.get("dilate", DEFAULT_SAM_DILATE)))
        self.dilate_spin.wheelEvent = lambda e: e.ignore()
        form.addRow(self._label("sam_dilate", t("preprocess_dilate")), self.dilate_spin)

        self.remove_btn = QPushButton(t("preprocess_sam_remove_rule"))
        self.remove_btn.clicked.connect(lambda: self.removed.emit(self))
        form.addRow("", self.remove_btn)

    def _label(self, key: str, text: str) -> ClickableLabel:
        """Clickable field label that routes this field's help to the panel."""
        lbl = ClickableLabel(text)
        lbl.setStyleSheet("color:#f0f0f0; text-decoration: underline dotted;")
        help_text = preprocess_field_help(key)
        lbl.clicked.connect(lambda _t=text, _h=help_text: self._help_cb(_t, _h))
        return lbl

    def to_dict(self) -> dict:
        """Serialize to a rule dict. Raises ValueError on an unparseable threshold."""
        text = self.threshold_edit.text().strip()
        try:
            threshold = float(text)
        except ValueError as exc:
            raise ValueError(text) from exc
        rule: dict = {}
        pattern = self.path_pattern_edit.text().strip()
        if pattern and pattern != "*":
            rule["path_pattern"] = pattern
        prompts = [
            line.strip()
            for line in self.prompts_edit.toPlainText().splitlines()
            if line.strip()
        ]
        if prompts:
            rule["prompts"] = prompts
        focus = [
            line.strip()
            for line in self.focus_prompts_edit.toPlainText().splitlines()
            if line.strip()
        ]
        if focus:
            rule["focus_prompts"] = focus
        rule["threshold"] = threshold
        rule["dilate"] = int(self.dilate_spin.value())
        return rule


class PreprocessingTab(LazyTabMixin, QWidget):
    def __init__(self):
        super().__init__()
        # Daemon-backed preprocessing (mirrors ConfigTab's Train button): each
        # Run submits a "command" job to the local daemon — not a child of this
        # tab — so a long cache build / mask pass survives the GUI closing and
        # shares the daemon's serial queue with training (one GPU, one job at a
        # time). The tab observes the job by polling the per-job files the
        # daemon writes (job.json for state, stdout.log for the log/bar) off a
        # single timer; no SSE thread (daemon is localhost-only).
        self._job_id: str | None = None
        self._stdout_tailer = gui_daemon.FileTailer()
        self._stdout_buf = ""
        self._job_timer = QTimer(self)
        self._job_timer.setInterval(400)
        self._job_timer.timeout.connect(self._poll_job)
        self._run_buttons: list[QToolButton] = []
        # Custom QStyle instances for the split Run buttons — kept alive here
        # because setStyle() does not take ownership.
        self._split_styles: list[SplitButtonStyle] = []
        self._variant: str | None = None
        self._loading_variant = False

        outer = QVBoxLayout(self)

        # ── Top action bar ────────────────────────────────────────
        # Mirrors ConfigTab: Save + per-step Run buttons + Stop, all under
        # the tab strip on a single row. No manual refresh — the status
        # one-liner is rebuilt automatically when a job finishes.
        top = QHBoxLayout()

        # Color semantics (matches ConfigTab):
        #   Save           → neutral (default styling, no background tint)
        #   Cache / mask   → blue   (#2980b9) — run a specific preprocess step
        #   Stop           → red    (#c0392b) — abort the running subprocess
        # Split Run buttons (matches ConfigTab's Train button): SplitButtonStyle
        # widens the dropdown indicator and paints its divider + tint from the
        # style, so the label stays centred in the action segment. Symmetric
        # padding only — the style owns the arrow geometry.
        run_step_style = (
            "QToolButton{background:#2980b9;color:white;font-weight:bold;"
            "padding:4px 16px;}"
        )

        self._method_label = QLabel("Method")
        top.addWidget(self._method_label)
        self.method_combo = QComboBox()
        self.method_combo.addItems(PREPROCESS_METHODS)
        self.method_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.method_combo.setMinimumContentsLength(
            max((len(m) for m in PREPROCESS_METHODS), default=10)
        )
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        top.addWidget(self.method_combo)

        self._variant_label = QLabel(t("variant"))
        top.addWidget(self._variant_label)
        self.variant_combo = QComboBox()
        self.variant_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.variant_combo.setMinimumContentsLength(20)
        self.variant_combo.currentTextChanged.connect(self._on_variant_changed)
        top.addWidget(self.variant_combo, 1)
        self._refresh_variant_row(self.method_combo.currentText())

        self.save_btn = QPushButton(t("preprocess_save_settings"))
        self.save_btn.setToolTip(t("preprocess_save_settings_tip"))
        self.save_btn.clicked.connect(self._save_all_clicked)
        top.addWidget(self.save_btn)

        # Per-step Run buttons. Save is implicit on each Run (same pattern
        # as ConfigTab's auto-save before Train/Preprocess). Each is a split
        # button: the main action runs the step now (attaches this tab to the
        # job); the dropdown queues it on the daemon without attaching, so the
        # user can stack the next step / variant before anything starts.
        self.run_te_btn = self._make_run_button(
            t("preprocess_run_te"), run_step_style, self._run_te
        )
        top.addWidget(self.run_te_btn)

        self.run_mask_btn = self._make_run_button(
            t("preprocess_run_mask"), run_step_style, self._run_mask
        )
        top.addWidget(self.run_mask_btn)

        top.addStretch()
        self.stop_btn = QPushButton(t("stop"))
        self.stop_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        top.addWidget(self.stop_btn)
        outer.addLayout(top)

        # tqdm bar (same look as ConfigTab — shared QSS in gui/progress.py).
        # Shown when the observed daemon job emits a parseable tqdm line, hidden
        # again when the job finishes.
        self.progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.progress)
        outer.addWidget(self.progress)

        # Status one-liner stays directly under the progress bar.
        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color:#dcdcdc; padding: 2px 0;")
        outer.addWidget(self.status_lbl)

        # ── Body: vertical splitter (form+explain top, log bottom) ──
        vsplit = QSplitter(Qt.Vertical)

        # Horizontal splitter: form on left, explanation panel on right.
        hsplit = QSplitter(Qt.Horizontal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form_layout = QVBoxLayout(form_host)
        form_layout.setContentsMargins(0, 0, 0, 0)

        settings = _load_settings()
        pp_cfg = _load_preprocess_toml()
        sam_yaml = _load_sam_yaml()
        # Normalize either schema (flat or rules array) into a list of rule
        # dicts, one per editor card below. Flat configs collapse to a single
        # catch-all card; saving always re-emits the rules form.
        sam_rules = _load_rules(sam_yaml)
        mask_path_pattern = sam_yaml.get("path_pattern") or DEFAULT_MASK_PATH_PATTERN

        # Image preprocessing group. GUI-specific cache knobs are stored on the
        # selected gui-method variant; configs/preprocess.toml remains the CLI
        # default/fallback.
        img_box = QGroupBox(t("preprocess_image_prep"))
        img_form = QFormLayout()

        self.source_dir_edit = QLineEdit(
            str(pp_cfg.get("source_image_dir", DEFAULT_SOURCE_IMAGE_DIR))
        )
        self.source_dir_edit.setPlaceholderText(DEFAULT_SOURCE_IMAGE_DIR)
        self.source_dir_edit.setToolTip(t("preprocess_source_image_dir_tip"))
        self.source_dir_edit.setReadOnly(True)
        img_form.addRow(
            self._field_label("source_image_dir", t("preprocess_source_image_dir")),
            self.source_dir_edit,
        )

        self.preprocess_path_pattern_edit = QLineEdit(
            str(pp_cfg.get("preprocess_path_pattern", DEFAULT_PREPROCESS_PATH_PATTERN))
        )
        self.preprocess_path_pattern_edit.setPlaceholderText("*")
        self.preprocess_path_pattern_edit.setToolTip(t("preprocess_path_pattern_tip"))
        img_form.addRow(
            self._field_label("preprocess_path_pattern", t("preprocess_path_pattern")),
            self.preprocess_path_pattern_edit,
        )

        self.drop_lowres_chk = QCheckBox(t("preprocess_drop_lowres"))
        self.drop_lowres_chk.setToolTip(t("preprocess_drop_lowres_tip"))
        self.drop_lowres_chk.setChecked(
            bool(pp_cfg.get("drop_lowres_images", DEFAULT_DROP_LOWRES_IMAGES))
        )
        img_form.addRow(
            self._field_label("drop_lowres_images", t("preprocess_drop_lowres")),
            self.drop_lowres_chk,
        )

        self.min_pixels_spin = QSpinBox()
        self.min_pixels_spin.setRange(0, 100_000_000)
        self.min_pixels_spin.setSingleStep(50_000)
        self.min_pixels_spin.setGroupSeparatorShown(True)
        self.min_pixels_spin.setValue(int(pp_cfg.get("min_pixels", DEFAULT_MIN_PIXELS)))
        self.min_pixels_spin.wheelEvent = lambda e: e.ignore()
        # min_pixels only applies when the filter is on (mirrors the CLI:
        # drop_lowres=false → --min_pixels 0). Grey it out when unchecked.
        self.min_pixels_spin.setEnabled(self.drop_lowres_chk.isChecked())
        self.drop_lowres_chk.toggled.connect(self.min_pixels_spin.setEnabled)
        img_form.addRow(
            self._field_label("min_pixels", t("preprocess_min_pixels")),
            self.min_pixels_spin,
        )

        # Multi-scale constant-token tiers. Dual-use: preprocess resizes each
        # image into the tier that resizes it the least, and train.py reads the
        # same value back (via load_method_preset) to size the compile cache, so
        # this is the single source of truth — the config form no longer shows it.
        self.target_res_widget = _TargetResWidget(
            pp_cfg.get("target_res", DEFAULT_TARGET_RES)
        )
        # Live-persist tier checkboxes to the selected GUI method on every
        # toggle. The Config tab's Train auto-chain snapshots method values
        # without touching this widget, so without an immediate write the
        # auto-chain would preprocess at the stale/default tier whenever the
        # user changed tiers here but didn't click Save first.
        self.target_res_widget.changed.connect(self.persist_target_res)
        img_form.addRow(
            self._field_label("target_res", t("preprocess_target_res")),
            self.target_res_widget,
        )
        img_box.setLayout(img_form)
        form_layout.addWidget(img_box)

        # Text caching group
        text_box = QGroupBox(t("preprocess_text_caching"))
        text_form = QFormLayout()
        self.shuffle_spin = QSpinBox()
        self.shuffle_spin.setRange(0, 64)
        self.shuffle_spin.setValue(
            int(settings.get("caption_shuffle_variants", DEFAULT_TE_SHUFFLE_VARIANTS))
        )
        # Block scroll-wheel changes (matches gui/__init__.py::_widget convention).
        self.shuffle_spin.wheelEvent = lambda e: e.ignore()
        text_form.addRow(
            self._field_label(
                "caption_shuffle_variants",
                t("preprocess_caption_shuffle_variants"),
            ),
            self.shuffle_spin,
        )

        self.dropout_edit = QLineEdit(
            f"{float(settings.get('caption_tag_dropout_rate', DEFAULT_TE_TAG_DROPOUT)):g}"
        )
        text_form.addRow(
            self._field_label(
                "caption_tag_dropout_rate",
                t("preprocess_caption_tag_dropout_rate"),
            ),
            self.dropout_edit,
        )
        text_box.setLayout(text_form)
        form_layout.addWidget(text_box)

        # SAM masking group
        sam_box = QGroupBox(t("preprocess_masking_sam"))
        sam_outer = QVBoxLayout()

        # Top form: run toggle + global scope (forwarded to BOTH backends).
        sam_form = QFormLayout()
        self.run_sam_mask_chk = QCheckBox(t("preprocess_run_sam_mask"))
        self.run_sam_mask_chk.setToolTip(t("preprocess_run_sam_mask_tip"))
        self.run_sam_mask_chk.setChecked(
            bool(settings.get("run_sam_mask", DEFAULT_RUN_SAM_MASK))
        )
        sam_form.addRow(
            self._field_label("run_sam_mask", t("preprocess_run_sam_mask")),
            self.run_sam_mask_chk,
        )
        # Stored in sam_mask.yaml but scopes BOTH backends — masking.py reads
        # it and forwards --path-pattern to SAM and MIT alike.
        self.mask_path_pattern_edit = QLineEdit(mask_path_pattern)
        self.mask_path_pattern_edit.setPlaceholderText("*")
        self.mask_path_pattern_edit.setToolTip(t("preprocess_mask_path_pattern_tip"))
        sam_form.addRow(
            self._field_label("mask_path_pattern", t("preprocess_mask_path_pattern")),
            self.mask_path_pattern_edit,
        )
        sam_outer.addLayout(sam_form)

        # One card per rule: each routes a subset of images (by path_pattern)
        # to its own prompt set. Rules whose pattern matches an image compose.
        self._rule_cards: list[_RuleCard] = []
        self._rules_layout = QVBoxLayout()
        self._rules_layout.setContentsMargins(0, 0, 0, 0)
        sam_outer.addLayout(self._rules_layout)
        for rule in sam_rules:
            self._add_rule_card(rule)

        self.add_rule_btn = QPushButton(t("preprocess_sam_add_rule"))
        self.add_rule_btn.setToolTip(t("preprocess_sam_add_rule_tip"))
        self.add_rule_btn.clicked.connect(lambda: self._add_rule_card())
        sam_outer.addWidget(self.add_rule_btn)

        sam_box.setLayout(sam_outer)
        form_layout.addWidget(sam_box)

        # MIT masking group
        mit_box = QGroupBox(t("preprocess_masking_mit"))
        mit_form = QFormLayout()
        self.run_mit_mask_chk = QCheckBox(t("preprocess_run_mit_mask"))
        self.run_mit_mask_chk.setToolTip(t("preprocess_run_mit_mask_tip"))
        self.run_mit_mask_chk.setChecked(
            bool(settings.get("run_mit_mask", DEFAULT_RUN_MIT_MASK))
        )
        mit_form.addRow(
            self._field_label("run_mit_mask", t("preprocess_run_mit_mask")),
            self.run_mit_mask_chk,
        )

        self.mit_threshold_edit = QLineEdit(
            f"{float(settings.get('mit_text_threshold', DEFAULT_MIT_TEXT_THRESHOLD)):g}"
        )
        mit_form.addRow(
            self._field_label("mit_text_threshold", t("preprocess_mit_threshold")),
            self.mit_threshold_edit,
        )

        self.mit_dilate_spin = QSpinBox()
        self.mit_dilate_spin.setRange(0, 64)
        self.mit_dilate_spin.setValue(
            int(settings.get("mit_dilate", DEFAULT_MIT_DILATE))
        )
        self.mit_dilate_spin.wheelEvent = lambda e: e.ignore()
        mit_form.addRow(
            self._field_label("mit_dilate", t("preprocess_dilate")),
            self.mit_dilate_spin,
        )
        mit_box.setLayout(mit_form)
        form_layout.addWidget(mit_box)

        form_layout.addStretch()
        scroll.setWidget(form_host)
        hsplit.addWidget(scroll)

        # Right panel — same QTextBrowser style as ConfigTab's explain panel
        # so the look matches across tabs.
        self._explain = QTextBrowser()
        self._explain.setOpenExternalLinks(True)
        self._explain.setStyleSheet(
            "QTextBrowser { font-size: 13px; padding: 12px; "
            "background: #2b2b2b; color: #e0e0e0; }"
        )
        self._explain.setMinimumWidth(320)
        self._show_default_explain()
        hsplit.addWidget(self._explain)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([720, 420])
        vsplit.addWidget(hsplit)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family:monospace;font-size:11px;")
        self.log.setPlaceholderText(t("preprocess_log_placeholder"))
        vsplit.addWidget(self.log)
        vsplit.setSizes([520, 200])
        outer.addWidget(vsplit, 1)

    def _lazy_init(self) -> None:
        # Cache-count scan deferred to first show of the tab.
        self._refresh_status()
        # Re-bind to a preprocess/mask job still running from a previous GUI
        # session (or one submitted by the CLI) so closing+reopening re-attaches.
        self._try_reattach()

    def _refresh_variant_row(self, method: str) -> None:
        variants = list_gui_variants(method)
        current = [
            self.variant_combo.itemText(i) for i in range(self.variant_combo.count())
        ]
        if current == variants:
            return
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        if variants:
            self.variant_combo.addItems(variants)
        self.variant_combo.blockSignals(False)

    def _on_method_changed(self, method: str) -> None:
        if self._loading_variant:
            return
        self._refresh_variant_row(method)
        self.set_variant(self.variant_combo.currentText(), method=method)

    def _on_variant_changed(self, variant: str) -> None:
        if self._loading_variant:
            return
        self.set_variant(variant, method=self.method_combo.currentText())

    def set_variant(self, variant: str, *, method: str | None = None) -> None:
        """Load GUI preprocess controls for the selected training variant."""
        if not variant:
            return
        if method:
            self._loading_variant = True
            try:
                if self.method_combo.currentText() != method:
                    self.method_combo.setCurrentText(method)
                self._refresh_variant_row(method)
                if self.variant_combo.currentText() != variant:
                    self.variant_combo.setCurrentText(variant)
            finally:
                self._loading_variant = False
        self._variant = variant
        meta = self._variant_preprocess_meta(variant)
        settings = _load_settings()
        pp_cfg = _load_preprocess_toml()

        try:
            merged, _ = merged_gui_variant_preset(variant, "default")
            source_dir = ConfigTab._gui_scoped_paths(merged).get(
                "source_image_dir", DEFAULT_SOURCE_IMAGE_DIR
            )
        except Exception:
            source_dir = DEFAULT_SOURCE_IMAGE_DIR

        target_res = meta.get(
            "target_res", pp_cfg.get("target_res", DEFAULT_TARGET_RES)
        )
        path_pattern = meta.get(
            "preprocess_path_pattern", DEFAULT_PREPROCESS_PATH_PATTERN
        )
        drop_lowres = meta.get(
            "drop_lowres_images",
            pp_cfg.get("drop_lowres_images", DEFAULT_DROP_LOWRES_IMAGES),
        )
        min_pixels = meta.get(
            "min_pixels", pp_cfg.get("min_pixels", DEFAULT_MIN_PIXELS)
        )
        shuffle_variants = meta.get(
            "caption_shuffle_variants",
            settings.get("caption_shuffle_variants", DEFAULT_TE_SHUFFLE_VARIANTS),
        )
        tag_dropout = meta.get(
            "caption_tag_dropout_rate",
            settings.get("caption_tag_dropout_rate", DEFAULT_TE_TAG_DROPOUT),
        )

        self._loading_variant = True
        try:
            self.source_dir_edit.setText(str(source_dir or DEFAULT_SOURCE_IMAGE_DIR))
            self.preprocess_path_pattern_edit.setText(str(path_pattern or "*"))
            self.drop_lowres_chk.setChecked(bool(drop_lowres))
            self.min_pixels_spin.setValue(int(min_pixels))
            self.min_pixels_spin.setEnabled(self.drop_lowres_chk.isChecked())
            self._set_target_res_widget(target_res)
            self.shuffle_spin.setValue(int(shuffle_variants))
            self.dropout_edit.setText(f"{float(tag_dropout):g}")
        finally:
            self._loading_variant = False
        if hasattr(self, "status_lbl"):
            self._refresh_status()

    @staticmethod
    def _variant_preprocess_meta(variant: str) -> dict:
        try:
            data = _load(variant_path(variant))
        except Exception:
            return {}
        meta = data.get("variant")
        if not isinstance(meta, dict):
            return {}
        return {k: meta[k] for k in _GUI_PREPROCESS_KEYS if k in meta}

    def _set_target_res_widget(self, values) -> None:
        if values is None:
            selected = {1024}
        elif isinstance(values, (list, tuple, set)):
            selected = {int(v) for v in values}
        else:
            selected = {int(values)}
        for edge, checkbox in self.target_res_widget._boxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(edge in selected)
            checkbox.blockSignals(False)

    # ── Field labels & explain panel ───────────────────────────────

    def _field_label(self, key: str, text_str: str) -> ClickableLabel:
        """Build a ClickableLabel that shows this field's help when clicked."""
        lbl = ClickableLabel(text_str)
        lbl.setStyleSheet("color:#f0f0f0; text-decoration: underline dotted;")
        help_text = preprocess_field_help(key)
        lbl.clicked.connect(
            lambda _k=key, _h=help_text, _t=text_str: self._show_field_help(_t, _h)
        )
        return lbl

    def _show_default_explain(self) -> None:
        self._explain.setHtml(preprocess_guide())

    def _show_field_help(self, field_label: str, help_text: str | None) -> None:
        parts = [
            f"<h2 style='margin:0 0 10px 0; font-size:18px;'>"
            f"{html.escape(field_label)}</h2>"
        ]
        if help_text:
            parts.append(
                f"<p style='font-size:14px; line-height:1.6;'>"
                f"{html.escape(help_text)}</p>"
            )
        else:
            parts.append(
                f"<p style='color:#888; font-style:italic;'>"
                f"{html.escape(t('no_help_available'))}</p>"
            )
        self._explain.setHtml("".join(parts))

    # ── Status panel ───────────────────────────────────────────────

    def _refresh_status(self) -> None:
        snapshot = self.preprocess_config_snapshot()

        def _path(key: str, default: Path) -> Path:
            raw = snapshot.get(key)
            if not raw:
                return default
            p = Path(str(raw))
            return p if p.is_absolute() else ROOT / p

        preprocess_pattern = (
            self.preprocess_path_pattern_edit.text().strip()
            or DEFAULT_PREPROCESS_PATH_PATTERN
        )
        path_pattern = (
            preprocess_pattern
            if preprocess_pattern != DEFAULT_PREPROCESS_PATH_PATTERN
            else str(snapshot.get("path_pattern") or DEFAULT_PREPROCESS_PATH_PATTERN)
        )
        n_resized = _count_resized(
            _path("resized_image_dir", RESIZED_DIR),
            path_pattern,
        )
        caches = count_preprocess_caches(
            _path("lora_cache_dir", LORA_CACHE_DIR),
            path_pattern,
        )
        mask_n = _count_masks(_path("mask_dir", MASK_DIR), path_pattern)
        if n_resized == 0:
            self.status_lbl.setText(t("preprocess_status_no_resized"))
            return
        lines = [
            t("preprocess_status_resized", n=n_resized),
            t(
                "preprocess_status_caches",
                lat=caches["latents"],
                te=caches["te"],
                pe=caches["pe"],
            ),
            t("preprocess_status_masks", masks=mask_n),
        ]
        self.status_lbl.setText("  |  ".join(lines))

    # ── SAM rule cards ─────────────────────────────────────────────

    def _add_rule_card(self, rule: dict | None = None) -> None:
        card = _RuleCard(rule or {}, self._show_field_help)
        card.removed.connect(self._remove_rule_card)
        self._rule_cards.append(card)
        self._rules_layout.addWidget(card)
        self._update_remove_buttons()

    def _remove_rule_card(self, card: _RuleCard) -> None:
        if len(self._rule_cards) <= 1:
            return  # keep at least one rule
        self._rule_cards.remove(card)
        self._rules_layout.removeWidget(card)
        card.deleteLater()
        self._update_remove_buttons()

    def _update_remove_buttons(self) -> None:
        # A lone rule can't be removed (would leave an empty config).
        sole = len(self._rule_cards) <= 1
        for card in self._rule_cards:
            card.remove_btn.setEnabled(not sole)

    def _collect_rules(self) -> list[dict] | None:
        """Serialize every rule card, or None if one fails validation."""
        rules: list[dict] = []
        for card in self._rule_cards:
            try:
                rules.append(card.to_dict())
            except ValueError as bad_threshold:
                QMessageBox.warning(
                    self,
                    t("error"),
                    t(
                        "preprocess_invalid_float",
                        field=t("preprocess_sam_threshold"),
                        value=str(bad_threshold),
                    ),
                )
                return None
        return rules

    # ── Settings persistence ───────────────────────────────────────

    def _parse_float(self, text: str, field_label: str) -> float | None:
        try:
            return float(text)
        except ValueError:
            QMessageBox.warning(
                self,
                t("error"),
                t("preprocess_invalid_float", field=field_label, value=text),
            )
            return None

    def preprocess_env(self) -> dict[str, str]:
        """Environment values consumed by ``tasks.py preprocess``."""
        return {
            "CAPTION_SHUFFLE_VARIANTS": str(int(self.shuffle_spin.value())),
            "CAPTION_TAG_DROPOUT_RATE": self.dropout_edit.text().strip(),
            "PREPROCESS_PATH_PATTERN": (
                self.preprocess_path_pattern_edit.text().strip()
                or DEFAULT_PREPROCESS_PATH_PATTERN
            ),
        }

    def preprocess_overrides(self) -> dict[str, object]:
        """Flat config overrides that should be captured in preprocess snapshots."""
        return {
            "drop_lowres_images": self.drop_lowres_chk.isChecked(),
            "min_pixels": int(self.min_pixels_spin.value()),
            "target_res": self.target_res_widget.value(),
        }

    def preprocess_config_snapshot(self) -> dict[str, object]:
        """Full preprocess config snapshot captured at GUI submit time.

        The concrete paths come from the selected GUI method plus ``path_scope``.
        ``preprocess_path_pattern`` is not written into the flat config because
        training should not see that GUI-only preprocess filter; it is forwarded
        to tasks.py via ``PREPROCESS_PATH_PATTERN`` instead.
        """
        variant = self._variant or "lora"
        merged, _ = merged_gui_variant_preset(variant, "default")
        snapshot = ConfigTab._gui_scoped_paths(copy.deepcopy(merged))
        snapshot.update(self.preprocess_overrides())
        for key in (
            "base_config",
            "dataset_config",
            "variant",
            "method",
            "preset",
            "methods_subdir",
            "path_scope",
            "preprocess_path_pattern",
        ):
            snapshot.pop(key, None)

        def _clean(value):
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items() if v is not None}
            if isinstance(value, list):
                return [_clean(v) for v in value if v is not None]
            if isinstance(value, Path):
                return str(value)
            return value

        return _clean(snapshot)

    def persist_target_res(self) -> None:
        """Persist the tier selection to the current GUI variant.

        ConfigTab auto-chain/queue captures this value into the immutable job
        snapshot; plain CLI usage keeps using configs/preprocess.toml.
        """
        if not self._loading_variant:
            self._save_variant_preprocess_meta(validate_dropout=False)

    def persist_preprocess_inputs(self) -> bool:
        """Persist cache-building inputs used by ConfigTab's auto-chain/queue.

        This intentionally excludes mask-only settings so an invalid mask
        threshold cannot block a plain cache build.
        """
        return self._save_variant_preprocess_meta(validate_dropout=True)

    def _save_variant_preprocess_meta(self, *, validate_dropout: bool) -> bool:
        if not self._variant:
            return True
        dropout = (
            self._parse_float(
                self.dropout_edit.text().strip(),
                t("preprocess_caption_tag_dropout_rate"),
            )
            if validate_dropout
            else None
        )
        if validate_dropout and dropout is None:
            return False
        if dropout is None:
            try:
                dropout = float(self.dropout_edit.text().strip())
            except ValueError:
                dropout = DEFAULT_TE_TAG_DROPOUT

        path = variant_path(self._variant)
        data = _load(path)
        meta = data.get("variant")
        if not isinstance(meta, dict):
            meta = {}

        path_pattern = (
            self.preprocess_path_pattern_edit.text().strip()
            or DEFAULT_PREPROCESS_PATH_PATTERN
        )
        if path_pattern == DEFAULT_PREPROCESS_PATH_PATTERN:
            meta.pop("preprocess_path_pattern", None)
        else:
            meta["preprocess_path_pattern"] = path_pattern

        drop_lowres = self.drop_lowres_chk.isChecked()
        if drop_lowres == DEFAULT_DROP_LOWRES_IMAGES:
            meta.pop("drop_lowres_images", None)
        else:
            meta["drop_lowres_images"] = drop_lowres

        min_pixels = int(self.min_pixels_spin.value())
        if min_pixels == DEFAULT_MIN_PIXELS:
            meta.pop("min_pixels", None)
        else:
            meta["min_pixels"] = min_pixels

        target_res = self.target_res_widget.value()
        if target_res == DEFAULT_TARGET_RES:
            meta.pop("target_res", None)
        else:
            meta["target_res"] = target_res

        shuffle = int(self.shuffle_spin.value())
        if shuffle == DEFAULT_TE_SHUFFLE_VARIANTS:
            meta.pop("caption_shuffle_variants", None)
        else:
            meta["caption_shuffle_variants"] = shuffle

        if float(dropout) == float(DEFAULT_TE_TAG_DROPOUT):
            meta.pop("caption_tag_dropout_rate", None)
        else:
            meta["caption_tag_dropout_rate"] = float(dropout)

        if meta:
            data["variant"] = meta
        else:
            data.pop("variant", None)
        _save(path, data)
        return True

    def _save_all(self) -> bool:
        """Validate and persist every form value. Returns True on success."""
        dropout = self._parse_float(
            self.dropout_edit.text().strip(),
            t("preprocess_caption_tag_dropout_rate"),
        )
        if dropout is None:
            return False
        mit_threshold = self._parse_float(
            self.mit_threshold_edit.text().strip(),
            t("preprocess_mit_threshold"),
        )
        if mit_threshold is None:
            return False

        rules = self._collect_rules()
        if rules is None:
            return False

        mask_path_pattern = (
            self.mask_path_pattern_edit.text().strip() or DEFAULT_MASK_PATH_PATTERN
        )
        if not self._save_variant_preprocess_meta(validate_dropout=False):
            return False
        _save_sam_yaml(rules, mask_path_pattern)
        _save_settings(
            {
                "mit_text_threshold": mit_threshold,
                "mit_dilate": int(self.mit_dilate_spin.value()),
                "run_sam_mask": self.run_sam_mask_chk.isChecked(),
                "run_mit_mask": self.run_mit_mask_chk.isChecked(),
            }
        )
        return True

    def _save_all_clicked(self) -> None:
        if self._save_all():
            QMessageBox.information(self, t("saved"), t("preprocess_settings_saved"))

    # ── Daemon job actions ─────────────────────────────────────────

    def _is_running(self) -> bool:
        return self._job_id is not None

    def _make_run_button(self, label: str, style: str, run_cb) -> QToolButton:
        """Build a split Run button: main action runs now, dropdown queues it.

        ``run_cb`` is a ``_run_*`` handler taking a keyword-only ``queue`` flag;
        the dropdown calls it with ``queue=True`` (submit without attaching).
        """
        btn = QToolButton()
        # SplitButtonStyle (set before the stylesheet) widens the dropdown
        # indicator + paints its divider/tint, keeping the label centred in the
        # action segment. The style must outlive the button, so stash a ref.
        split_style = SplitButtonStyle()
        self._split_styles.append(split_style)
        btn.setStyle(split_style)
        btn.setText(label)
        btn.setStyleSheet(style)
        btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        btn.setPopupMode(QToolButton.MenuButtonPopup)
        btn.clicked.connect(lambda _checked=False: run_cb())
        menu = QMenu(btn)
        queue_action = menu.addAction(t("preprocess_add_to_queue"))
        queue_action.triggered.connect(lambda _checked=False: run_cb(queue=True))
        btn.setMenu(menu)
        self._run_buttons.append(btn)
        return btn

    def _run_te(self, *, queue: bool = False) -> None:
        # Unified "caching" step — runs `tasks.py preprocess`, which chains
        # resize → VAE-latent cache → text-embedding cache. Replaces the old
        # text-only path now that the ConfigTab's standalone Preprocess
        # button is gone and this tab owns the cache-build UI. The TE knobs
        # (shuffle / dropout) are still surfaced as env vars; resize and VAE
        # currently have no GUI-tunable parameters, so the form stays TE-only.
        if not self._save_all():
            return
        snapshot = self.preprocess_config_snapshot()
        self._submit(
            label="preprocess",
            argv=["tasks.py", "preprocess"],
            extra_env=self.preprocess_env(),
            config_snapshot=snapshot,
            attach=not queue,
        )

    def _run_mask(self, *, queue: bool = False) -> None:
        # Single-shot pipeline. ``tasks.py mask`` runs SAM and/or MIT into
        # a tempdir, merges the produced sources, and writes only the
        # merged result to ``post_image_dataset/masks/<rel>/``. SAM reads
        # ``configs/sam_mask.yaml`` directly; MIT picks up the
        # ``MIT_TEXT_THRESHOLD`` / ``MIT_DILATE`` env vars set below.
        # ``RUN_SAM_MASK`` / ``RUN_MIT_MASK`` gate each backend.
        if not self._save_all():
            return
        run_sam = self.run_sam_mask_chk.isChecked()
        run_mit = self.run_mit_mask_chk.isChecked()
        if not (run_sam or run_mit):
            QMessageBox.warning(self, t("error"), t("preprocess_mask_nothing_enabled"))
            return
        self._submit(
            label="mask",
            argv=["tasks.py", "mask"],
            extra_env={
                "MIT_TEXT_THRESHOLD": self.mit_threshold_edit.text().strip(),
                "MIT_DILATE": str(int(self.mit_dilate_spin.value())),
                "RUN_SAM_MASK": "1" if run_sam else "0",
                "RUN_MIT_MASK": "1" if run_mit else "0",
            },
            attach=not queue,
        )

    def _submit(
        self,
        *,
        label: str,
        argv: list[str],
        extra_env: dict,
        config_snapshot: dict | None = None,
        attach: bool = True,
    ) -> None:
        """Submit a preprocess/mask job to the daemon.

        The daemon spawns ``python <argv>`` detached and serializes it behind
        any running training job (single GPU). Pre-launch validation
        (``_save_all`` + per-step gating) is the caller's job.

        With ``attach=True`` (the main Run action) this tab takes over its
        log/bar and blocks the Run buttons until the job finishes. With
        ``attach=False`` (the "add to queue" dropdown) the job is submitted
        silently — the Run buttons stay live so the next step / variant can be
        queued, and the job is watched from the Queue tab."""
        if attach and self._is_running():
            QMessageBox.information(self, "", t("preprocess_already_running"))
            return
        if attach:
            # Busy UI + repaint before the submit so the tab feels responsive
            # while the daemon auto-start + /health wait completes on a cold
            # start.
            for btn in self._run_buttons:
                btn.setEnabled(False)
            self.save_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.log.clear()
            self._stdout_buf = ""
            self._progress_tracker.reset()
            self._progress_tracker.mark_starting(t("starting"))
            self.log.appendPlainText("> " + " ".join([sys.executable, *argv]))
            self.log.appendPlainText(t("daemon_submitting"))
            QApplication.processEvents()

        try:
            resp = gui_daemon.submit_command(
                label=label,
                argv=argv,
                extra_env=extra_env,
                config_snapshot=config_snapshot,
                # Main Run starts now; the "add to queue" dropdown holds it
                # paused until the Queue tab's "Start Queue".
                start=attach,
            )
        except Exception as e:  # noqa: BLE001 — daemon failed to start / submit
            QMessageBox.warning(self, t("error"), t("daemon_submit_failed", err=str(e)))
            if attach:
                self._restore_idle_ui()
            return
        job_id = resp.get("job_id") if isinstance(resp, dict) else None
        if not job_id:
            QMessageBox.warning(
                self, t("error"), t("daemon_submit_failed", err=str(resp))
            )
            if attach:
                self._restore_idle_ui()
            return
        if attach:
            self.log.appendPlainText(t("daemon_queued", job_id=job_id).rstrip("\n"))
            self._attach_to_job(job_id, replay_log=False)
        else:
            self.log.appendPlainText(t("preprocess_queued", label=label, job_id=job_id))

    def _try_reattach(self) -> None:
        """Bind to a preprocess/mask job still running when the tab first opens.

        Makes "close GUI mid-preprocess → reopen → re-attach" work. Skips a
        training job (that one belongs to the ConfigTab) and stays idle when the
        daemon is down."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable → nothing to attach
            return
        if not job_id or gui_daemon.read_job_kind(job_id) != "command":
            return
        # An auto-chain preprocess (tagged ANIMA_CHAIN_TRAIN) belongs to the
        # ConfigTab — it re-claims that one so the bar + Train-blocking + chain
        # into training stay on the training tab. Leave it alone here.
        if gui_daemon.read_job_chain_variant(job_id):
            return
        self.log.clear()
        self._stdout_buf = ""
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("starting"))
        self.log.appendPlainText(t("daemon_reattached", job_id=job_id).rstrip("\n"))
        self._attach_to_job(job_id, replay_log=True)

    def _attach_to_job(self, job_id: str, *, replay_log: bool) -> None:
        """Point the log + bar at a daemon job's on-disk files and start polling.

        ``replay_log`` reads ``stdout.log`` from the top (re-attach after a GUI
        restart); otherwise pre-existing output is skipped so a fresh launch
        shows only new lines."""
        self._job_id = job_id
        self._stdout_buf = ""
        self._stdout_tailer.watch(gui_daemon.stdout_path(job_id))
        if not replay_log:
            self._stdout_tailer.read_new()  # discard backlog
        for btn in self._run_buttons:
            btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._job_timer.start()

    def _poll_job(self) -> None:
        if not self._job_id:
            return
        self._drain_job_stdout()
        state = gui_daemon.read_job_state(self._job_id)
        if gui_daemon.is_terminal(state):
            self._on_job_finished(state)

    def _drain_job_stdout(self) -> None:
        """Append new stdout.log lines to the log (carriage-return aware); tqdm
        lines drive the bar instead of spamming the log (no progress.jsonl for
        preprocess/mask, so tqdm is the only progress signal)."""
        chunk = self._stdout_tailer.read_new()
        if not chunk:
            return
        parts = re.split(r"[\r\n]", self._stdout_buf + chunk)
        self._stdout_buf = parts[-1]  # incomplete trailing fragment
        for line in parts[:-1]:
            if self._progress_tracker.feed(line):
                continue
            if line:
                self.log.appendPlainText(line)

    def _on_job_finished(self, state: str | None) -> None:
        self._job_timer.stop()
        # Drain any trailing stdout before the finish banner. A half-written
        # tqdm fragment is dropped — the bar already reflected its state.
        self._drain_job_stdout()
        if self._stdout_buf and not TQDM_RE.search(self._stdout_buf):
            self.log.appendPlainText(self._stdout_buf)
        self._stdout_buf = ""
        job_id = self._job_id
        self._job_id = None
        self._stdout_tailer.reset()
        self._progress_tracker.reset()
        self.log.appendPlainText(gui_daemon.format_finish_banner(job_id, state))
        self._restore_idle_ui()
        self._refresh_status()

    def _restore_idle_ui(self) -> None:
        for btn in self._run_buttons:
            btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def _stop(self) -> None:
        # Abort the daemon job; the poll loop then observes the 'stopped' state
        # and restores the UI. The daemon stays up and advances its queue.
        if not self._job_id:
            return
        try:
            gui_daemon.stop_job(self._job_id)
        except Exception as e:  # noqa: BLE001
            self.log.appendPlainText(f"stop failed: {e}")

    def cleanup_subprocess(self) -> None:
        """App-shutdown hook. Stops observing but deliberately leaves the daemon
        job alive — it runs detached so a cache build / mask pass survives the
        GUI closing (re-attached on next launch)."""
        self._job_timer.stop()
