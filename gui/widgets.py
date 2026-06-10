"""Reusable Qt widgets + the config-form field factory.

Holds the lazy-tab mixin, the multi-scale ``target_res`` checkbox row, the
``_widget``/``_read`` pair that maps a config value to/from an editor widget,
and the aspect-preserving image label. Pulled out of the package root so the
widget code lives apart from the Qt-free config logic.
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QWidget,
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
        # One preview prompt per line. Stored in TOML as a string (single line)
        # or an array (multi-line); train.py materializes either into the
        # sample_prompts.txt file the sampler reads.
        w = QPlainTextEdit()
        if isinstance(v, (list, tuple)):
            text = "\n".join(str(p) for p in v)
        elif v is None:
            text = ""
        else:
            text = str(v)
        w.setPlainText(text)
        w.setPlaceholderText("1girl, solo, ...\n(one preview prompt per line)")
        w.setMaximumHeight(120)
        return w
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
