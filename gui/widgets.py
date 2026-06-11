"""Reusable Qt widgets + the config-form field factory.

Holds the lazy-tab mixin, the multi-scale ``target_res`` checkbox row, the
``_widget``/``_read`` pair that maps a config value to/from an editor widget,
and the aspect-preserving image label. Pulled out of the package root so the
widget code lives apart from the Qt-free config logic.
"""

from __future__ import annotations

import json
import re
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
    QVBoxLayout,
)

from gui.i18n import t

# flash4 is not supported yet (flash-attention-sm120 disabled)
_ATTN_MODES = ["flex", "flash"]


class LazyTabMixin:
    """Defer a tab's first expensive scan until the tab is actually opened.

    Several tabs walk dataset/checkpoint directories (and the Merge tab reads
    safetensors keys) during construction. Doing that for *every* tab up front
    is what made the window slow to appear, even though only the first tab is
    visible at launch. Mixing this in lets construction stay cheap: the heavy
    work runs on the first ``showEvent`` — i.e. when the user selects the tab —
    and exactly once thereafter. Subclasses override ``_lazy_init``.

    Mix in BEFORE ``QWidget`` so ``super().showEvent`` resolves to Qt's.
    """

    _lazy_done = False

    def showEvent(self, event):  # noqa: N802 — Qt event handler name
        super().showEvent(event)
        if not self._lazy_done:
            self._lazy_done = True
            self._lazy_init()

    def _lazy_init(self) -> None:
        """Run the tab's first directory scan / classification. Override."""


# Allowed multi-scale tiers — mirrors library.datasets.buckets.ALLOWED_TARGET_RES
# (hardcoded so the GUI import stays light / library-free).
_TARGET_RES_TIERS = (512, 768, 896, 1024, 1280, 1536)

# High-cost tiers: large per-image token counts + an extra compiled block
# graph each. Flagged in the GUI so users don't casually enable them.
_TARGET_RES_DANGER = {1280: 6300, 1536: 8640}


class _TargetResWidget(QWidget):
    """Horizontal row of tier checkboxes for the multi-scale ``target_res`` knob.

    Reads/writes a list of edge ints (e.g. ``[1024, 1536]``). Never returns an
    empty list — unchecking everything falls back to ``[1024]`` (the legacy
    single ~1MP tier) so preprocess/train always have a valid tier.

    The 1280/1536 tiers are visually flagged as "dangerous" (high token count
    + extra compile graph / VRAM) via colour + an i18n tooltip.
    """

    changed = Signal()

    def __init__(self, selected) -> None:
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        sel = {int(e) for e in selected} if selected else set()
        self._boxes: dict[int, QCheckBox] = {}
        for edge in _TARGET_RES_TIERS:
            cb = QCheckBox(str(edge))
            cb.setChecked(edge in sel)
            cb.toggled.connect(self.changed)
            if edge in _TARGET_RES_DANGER:
                cb.setStyleSheet("QCheckBox { color: #d9822b; font-weight: bold; }")
                cb.setToolTip(
                    t(
                        "target_res_danger_tooltip",
                        edge=edge,
                        tokens=_TARGET_RES_DANGER[edge],
                    )
                )
            lay.addWidget(cb)
            self._boxes[edge] = cb
        lay.addStretch(1)

    def value(self) -> list[int]:
        out = [e for e, cb in self._boxes.items() if cb.isChecked()]
        return out or [1024]


class _SamplePromptRow(QFrame):
    """One editable sample prompt while preserving train.py's one-line syntax."""

    changed = Signal()

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { border:1px solid #444; border-radius:4px; } "
            "QLabel, QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, "
            "QCheckBox { border:0; }"
        )

        lay = QGridLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setHorizontalSpacing(12)
        lay.setVerticalSpacing(6)

        self.select_box = QCheckBox(t("sample_prompt_select"))
        lay.addWidget(self.select_box, 0, 0, 1, 2)

        prompt_stack = QVBoxLayout()
        prompt_stack.setContentsMargins(0, 0, 0, 0)
        prompt_stack.setSpacing(4)
        lay.addLayout(prompt_stack, 1, 0)

        prompt_label = QLabel(t("sample_prompt_col_prompt"))
        prompt_stack.addWidget(prompt_label)
        self.prompt_edit = QPlainTextEdit(str(data.get("prompt", "")))
        self.prompt_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.prompt_edit.setMinimumHeight(118)
        self.prompt_edit.setPlaceholderText(t("sample_prompt_prompt_placeholder"))
        self.prompt_edit.textChanged.connect(self.changed.emit)
        prompt_stack.addWidget(self.prompt_edit)

        negative_label = QLabel(t("sample_prompt_col_negative"))
        negative_label.setToolTip(t("sample_prompt_tip_negative"))
        prompt_stack.addWidget(negative_label)
        self.negative = QPlainTextEdit(str(data.get("negative", "")))
        self.negative.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.negative.setMinimumHeight(72)
        self.negative.setMaximumHeight(96)
        self.negative.setPlaceholderText(t("sample_prompt_default_negative"))
        self.negative.setToolTip(t("sample_prompt_tip_negative"))
        self.negative.textChanged.connect(self.changed.emit)
        prompt_stack.addWidget(self.negative)

        opts = QGridLayout()
        opts.setContentsMargins(0, 0, 0, 0)
        opts.setHorizontalSpacing(8)
        opts.setVerticalSpacing(4)
        lay.addLayout(opts, 1, 1)

        self.width = self._int_spin(
            data.get("width", 0),
            default_label=t("sample_prompt_default_width"),
            tooltip=t("sample_prompt_tip_width"),
        )
        self.height = self._int_spin(
            data.get("height", 0),
            default_label=t("sample_prompt_default_height"),
            tooltip=t("sample_prompt_tip_height"),
        )
        self.steps = self._int_spin(
            data.get("steps", 0),
            maximum=1000,
            default_label=t("sample_prompt_default_steps"),
            tooltip=t("sample_prompt_tip_steps"),
        )
        self.seed = self._seed_spin(data.get("seed"))
        self.scale = self._float_spin(
            data.get("scale", 0.0),
            default_label=t("sample_prompt_default_cfg"),
            tooltip=t("sample_prompt_tip_cfg"),
        )
        self.guidance = self._float_spin(
            data.get("guidance", 0.0),
            default_label=t("sample_prompt_default_guidance"),
            tooltip=t("sample_prompt_tip_guidance"),
        )
        self.flow_shift = self._float_spin(
            data.get("flow_shift", 0.0),
            default_label=t("sample_prompt_default_shift"),
            tooltip=t("sample_prompt_tip_shift"),
        )
        self.extra = QLineEdit(str(data.get("extra", "")))
        self.extra.setToolTip(t("sample_prompt_tip_extra"))
        self.extra.textChanged.connect(lambda *_: self.changed.emit())

        for row, (label_text, widget, tooltip) in enumerate(
            (
                (
                    t("sample_prompt_col_width"),
                    self.width,
                    t("sample_prompt_tip_width"),
                ),
                (
                    t("sample_prompt_col_height"),
                    self.height,
                    t("sample_prompt_tip_height"),
                ),
                (
                    t("sample_prompt_col_steps"),
                    self.steps,
                    t("sample_prompt_tip_steps"),
                ),
                (t("sample_prompt_col_seed"), self.seed, t("sample_prompt_tip_seed")),
                (t("sample_prompt_col_cfg"), self.scale, t("sample_prompt_tip_cfg")),
                (
                    t("sample_prompt_col_guidance"),
                    self.guidance,
                    t("sample_prompt_tip_guidance"),
                ),
                (
                    t("sample_prompt_col_shift"),
                    self.flow_shift,
                    t("sample_prompt_tip_shift"),
                ),
                (t("sample_prompt_col_extra"), self.extra, t("sample_prompt_tip_extra")),
            )
        ):
            label = QLabel(label_text)
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            opts.addWidget(label, row, 0)
            opts.addWidget(widget, row, 1)

        lay.setColumnStretch(0, 4)
        lay.setColumnStretch(1, 2)
        self.setMinimumHeight(300)

    def _int_spin(
        self,
        value: int = 0,
        *,
        maximum: int = 8192,
        default_label: str,
        tooltip: str = "",
        unset: int = 0,
    ) -> QSpinBox:
        w = QSpinBox()
        w.setRange(unset, maximum)
        w.setSpecialValueText(default_label)
        w.setValue(int(value or unset))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        if tooltip:
            w.setToolTip(tooltip)
        return _no_wheel(w)

    def _seed_spin(self, value: int | None = None) -> QSpinBox:
        w = QSpinBox()
        w.setRange(-1, 2_147_483_647)
        w.setSpecialValueText(t("sample_prompt_default_seed"))
        w.setValue(-1 if value is None else int(value))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        w.setToolTip(t("sample_prompt_tip_seed"))
        return _no_wheel(w)

    def _float_spin(
        self, value: float = 0.0, *, default_label: str, tooltip: str = ""
    ) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(0.0, 100.0)
        w.setDecimals(2)
        w.setSingleStep(0.5)
        w.setSpecialValueText(default_label)
        w.setValue(float(value or 0.0))
        w.valueChanged.connect(lambda *_: self.changed.emit())
        w.setMinimumWidth(110)
        if tooltip:
            w.setToolTip(tooltip)
        return _no_wheel(w)

    @staticmethod
    def _single_line(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def value(self) -> str | None:
        prompt = self._single_line(self.prompt_edit.toPlainText())
        if not prompt:
            return None
        parts = [prompt]
        for widget, flag in (
            (self.width, "w"),
            (self.height, "h"),
            (self.steps, "s"),
        ):
            val = widget.value()
            if val > 0:
                parts.append(f"--{flag} {val}")
        seed = self.seed.value()
        if seed >= 0:
            parts.append(f"--d {seed}")
        for widget, flag in (
            (self.scale, "l"),
            (self.guidance, "g"),
            (self.flow_shift, "fs"),
        ):
            val = widget.value()
            if val > 0:
                parts.append(f"--{flag} {val:g}")
        negative = self._single_line(self.negative.toPlainText())
        if negative:
            parts.append(f"--n {negative}")
        extra = self._single_line(self.extra.text())
        if extra:
            parts.append(extra if extra.startswith("--") else "--" + extra)
        return " ".join(parts)


class _SamplePromptsWidget(QWidget):
    """Structured editor for train.py's one-line sample prompt syntax."""

    changed = Signal()

    _COLLAPSED_HEIGHT = 300
    _EXPANDED_HEIGHT = 650

    def __init__(self, prompts, fill: bool = False) -> None:
        super().__init__()
        # ``fill`` = expand to fill the host (used inside the editor dialog,
        # which is resizable). When false the widget self-clamps to a fixed
        # height and offers an expand/collapse toggle for inline embedding.
        self._fill = fill
        self._expanded = False
        self._rows: list[_SamplePromptRow] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        hint = QLabel(t("sample_prompt_hint"))
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#aaa;")
        lay.addWidget(hint)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding if fill else QSizePolicy.Fixed,
        )
        self.scroll.setStyleSheet(
            """
            QScrollBar:vertical {
                background: #242424;
                width: 14px;
                margin: 0;
            }
            QScrollBar:horizontal {
                background: #242424;
                height: 14px;
                margin: 0;
            }
            QScrollBar::handle {
                background: #7a7a7a;
                border-radius: 5px;
                min-height: 24px;
                min-width: 24px;
            }
            QScrollBar::handle:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line,
            QScrollBar::sub-line {
                width: 0;
                height: 0;
            }
            """
        )
        self._rows_widget = QWidget()
        self._row_layout = QVBoxLayout(self._rows_widget)
        self._row_layout.setContentsMargins(0, 0, 0, 0)
        self._row_layout.setSpacing(8)
        self._row_layout.addStretch(1)
        self.scroll.setWidget(self._rows_widget)
        lay.addWidget(self.scroll)

        row_lay = QHBoxLayout()
        row_lay.setContentsMargins(0, 0, 0, 0)
        add_btn = QPushButton(t("sample_prompt_add"))
        add_btn.clicked.connect(lambda: self._add_row({}))
        row_lay.addWidget(add_btn)
        select_all_btn = QPushButton(t("sample_prompt_select_all"))
        select_all_btn.clicked.connect(self._select_all)
        row_lay.addWidget(select_all_btn)
        remove_btn = QPushButton(t("sample_prompt_remove"))
        remove_btn.clicked.connect(self._remove_selected)
        row_lay.addWidget(remove_btn)
        self.expand_btn: QPushButton | None = None
        if not fill:
            self.expand_btn = QPushButton(t("sample_prompt_expand"))
            self.expand_btn.clicked.connect(self._toggle_expanded)
            row_lay.addWidget(self.expand_btn)
        row_lay.addStretch(1)
        lay.addLayout(row_lay)

        rows = self._parse_prompts(prompts)
        if rows:
            for row in rows:
                self._add_row(row)
        else:
            self._add_row({})
        self._apply_height()

    def _apply_height(self) -> None:
        if self._fill or self.expand_btn is None:
            return
        height = self._EXPANDED_HEIGHT if self._expanded else self._COLLAPSED_HEIGHT
        self.scroll.setMinimumHeight(height)
        self.scroll.setMaximumHeight(height)
        self.expand_btn.setText(
            t("sample_prompt_collapse")
            if self._expanded
            else t("sample_prompt_expand")
        )

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._apply_height()

    @staticmethod
    def _parse_prompts(prompts) -> list[dict[str, Any]]:
        if isinstance(prompts, (list, tuple)):
            lines = [str(p).strip() for p in prompts]
        elif prompts is None:
            lines = []
        else:
            lines = [ln.strip() for ln in str(prompts).splitlines()]
        return [
            _SamplePromptsWidget._parse_line(ln)
            for ln in lines
            if ln and not ln.startswith("#")
        ]

    @staticmethod
    def _parse_line(line: str) -> dict[str, Any]:
        parts = line.split(" --")
        out: dict[str, Any] = {"prompt": parts[0].strip()}
        extra: list[str] = []
        for part in parts[1:]:
            try:
                if m := re.match(r"w (\d+)$", part, re.IGNORECASE):
                    out["width"] = int(m.group(1))
                elif m := re.match(r"h (\d+)$", part, re.IGNORECASE):
                    out["height"] = int(m.group(1))
                elif m := re.match(r"s (\d+)$", part, re.IGNORECASE):
                    out["steps"] = int(m.group(1))
                elif m := re.match(r"d (\d+)$", part, re.IGNORECASE):
                    out["seed"] = int(m.group(1))
                elif m := re.match(r"l ([\d.]+)$", part, re.IGNORECASE):
                    out["scale"] = float(m.group(1))
                elif m := re.match(r"g ([\d.]+)$", part, re.IGNORECASE):
                    out["guidance"] = float(m.group(1))
                elif m := re.match(r"fs ([\d.]+)$", part, re.IGNORECASE):
                    out["flow_shift"] = float(m.group(1))
                elif m := re.match(r"n (.+)$", part, re.IGNORECASE):
                    out["negative"] = m.group(1).strip()
                else:
                    extra.append("--" + part.strip())
            except ValueError:
                extra.append("--" + part.strip())
        if extra:
            out["extra"] = " ".join(extra)
        return out

    def _add_row(self, data: dict[str, Any]) -> None:
        row = _SamplePromptRow(data)
        row.changed.connect(self.changed.emit)
        self._rows.append(row)
        self._row_layout.insertWidget(max(0, self._row_layout.count() - 1), row)
        self.changed.emit()

    def _select_all(self) -> None:
        for row in self._rows:
            row.select_box.setChecked(True)

    def _remove_selected(self) -> None:
        rows = [row for row in self._rows if row.select_box.isChecked()]
        if not rows:
            return
        answer = QMessageBox.question(
            self,
            t("sample_prompt_remove_confirm_title"),
            t("sample_prompt_remove_confirm_body", n=len(rows)),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        for row in rows:
            if row in self._rows:
                self._rows.remove(row)
                self._row_layout.removeWidget(row)
                row.setParent(None)
                row.deleteLater()
        if not self._rows:
            self._add_row({})
        self.changed.emit()

    def value(self) -> list[str]:
        lines: list[str] = []
        for row in self._rows:
            value = row.value()
            if value:
                lines.append(value)
        return lines


def _normalize_prompt_lines(prompts) -> list[str]:
    """Coerce a stored sample_prompts value (list or multiline string) into a
    clean list of non-empty, non-comment one-line prompts."""
    if isinstance(prompts, (list, tuple)):
        lines = [str(p).strip() for p in prompts]
    elif prompts is None:
        lines = []
    else:
        lines = [ln.strip() for ln in str(prompts).splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


class SamplePromptsDialog(QDialog):
    """Popup hosting the structured sample-prompt editor at full size."""

    def __init__(self, prompts, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(t("sample_prompt_dialog_title"))
        self.setModal(True)
        self.resize(960, 720)
        lay = QVBoxLayout(self)
        self._editor = _SamplePromptsWidget(prompts, fill=True)
        lay.addWidget(self._editor, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def value(self) -> list[str]:
        return self._editor.value()


class _SamplePromptsLauncher(QWidget):
    """Compact field stand-in: a button + summary that opens the editor popup.

    Replaces the tall inline editor in the config form. Holds the current
    prompt list and round-trips it through :class:`SamplePromptsDialog`.
    """

    changed = Signal()

    def __init__(self, prompts) -> None:
        super().__init__()
        self._prompts = _normalize_prompt_lines(prompts)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._edit_btn = QPushButton(t("sample_prompt_edit_button"))
        self._edit_btn.clicked.connect(self._open_dialog)
        lay.addWidget(self._edit_btn)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color:#aaa;")
        lay.addWidget(self._summary, 1)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        n = len(self._prompts)
        if not n:
            self._summary.setText(t("sample_prompt_summary_none"))
            self._summary.setToolTip("")
            return
        first = self._prompts[0].split(" --", 1)[0].strip()
        if len(first) > 60:
            first = first[:57] + "…"
        self._summary.setText(t("sample_prompt_summary_count", n=n, first=first))
        self._summary.setToolTip("\n".join(self._prompts))

    def _open_dialog(self) -> None:
        dlg = SamplePromptsDialog(self._prompts, self)
        if dlg.exec():
            self._prompts = dlg.value()
            self._refresh_summary()
            self.changed.emit()

    def value(self) -> list[str]:
        return list(self._prompts)


def _no_wheel(w: QWidget) -> QWidget:
    """Stop a hovered combo/spin from changing value (and stealing focus) on
    mouse-wheel scroll — otherwise scrolling the form silently edits whichever
    dropdown the cursor passes over. The widget still works via click + keys."""
    w.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    w.wheelEvent = lambda e: e.ignore()
    return w


def _widget(v: Any, key: str = "") -> QWidget:
    if key == "target_res":
        sel = v if isinstance(v, (list, tuple)) else ([v] if v else [1024])
        return _TargetResWidget(sel)
    if key == "sample_prompts":
        return _SamplePromptsLauncher(v)
    if key == "attn_mode":
        w = QComboBox()
        w.addItems(_ATTN_MODES)
        idx = w.findText(str(v))
        if idx >= 0:
            w.setCurrentIndex(idx)
        return _no_wheel(w)
    if key == "sample_decode_inline":
        # Tri-state: stored as the literal string "auto" / "true" / "false"
        # (all three are accepted by library.config.cli_args._optional_bool).
        # Must precede the bool branch below so a bool value gets the combo,
        # not a plain checkbox that can't express "auto".
        w = QComboBox()
        w.addItems(["auto", "true", "false"])
        if v is None:
            cur = "auto"
        elif isinstance(v, bool):
            cur = "true" if v else "false"
        else:
            cur = str(v).strip().lower()
            if cur in ("", "none"):
                cur = "auto"
        idx = w.findText(cur)
        if idx >= 0:
            w.setCurrentIndex(idx)
        return _no_wheel(w)
    if isinstance(v, bool):
        w = QCheckBox()
        w.setChecked(v)
        return w
    if isinstance(v, int):
        w = QSpinBox()
        # Per-key range overrides for fields that legitimately exceed the
        # default 10k cap (silently clips otherwise). Keep these explicit
        # rather than raising the global ceiling — most int fields are
        # small (epochs, ranks, expert counts) and a 10k cap keeps the
        # user from typoing a giant value into them.
        if key == "min_pixels":
            w.setRange(0, 100_000_000)  # 100MP — covers any real image
        else:
            w.setRange(0, 10000)
        w.setValue(v)
        return _no_wheel(w)
    if isinstance(v, float):
        return QLineEdit(f"{v:g}")
    if isinstance(v, list):
        return QLineEdit(json.dumps(v))
    return QLineEdit(str(v))


def _read(w: QWidget, orig: Any = None) -> Any:
    if isinstance(w, _TargetResWidget):
        return w.value()
    if isinstance(w, _SamplePromptsLauncher):
        return w.value()
    if isinstance(w, QPlainTextEdit):
        # sample_prompts box → list of non-empty, non-comment lines.
        return [
            ln.strip()
            for ln in w.toPlainText().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QSpinBox):
        return w.value()
    txt = w.text()
    if isinstance(orig, float):
        try:
            return float(txt)
        except ValueError:
            pass
    if isinstance(orig, list):
        try:
            return json.loads(txt)
        except (json.JSONDecodeError, ValueError):
            pass
    # Normalize Windows-style backslashes pasted into path/string fields.
    # Forward slashes are valid on every OS Python runs on, and avoid
    # downstream TOML escape errors (e.g. "C:\Users" → \U is not a valid
    # TOML escape).
    if "\\" in txt:
        txt = txt.replace("\\", "/")
    return txt


class ScaledImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._src: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)

    def set_source(self, pm: QPixmap):
        self._src = pm
        self._rescale()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._rescale()

    def _rescale(self):
        if self._src and not self._src.isNull():
            self.setPixmap(
                self._src.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
