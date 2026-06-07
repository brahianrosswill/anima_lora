"""TensorBoard server manager and runs panel widget for the Anima GUI."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from gui.i18n import t


class TensorBoardManager:
    """Manages a single tensorboard subprocess."""

    _PORT = 6006

    def __init__(self) -> None:
        self._process = None

    def start(self, log_dir: str) -> int:
        """Start TensorBoard pointed at *log_dir*. Returns the port.

        No-op (returns port) if already running. Raises ``RuntimeError`` when
        tensorboard is not importable."""
        if self.running:
            return self._PORT
        import subprocess

        try:
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tensorboard.main",
                    f"--logdir={log_dir}",
                    f"--port={self._PORT}",
                    "--bind_all",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            raise RuntimeError("tensorboard module not found")
        return self._PORT

    def stop(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                pass
            self._process = None

    @property
    def running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def open_browser(self) -> None:
        QDesktopServices.openUrl(QUrl(f"http://localhost:{self._PORT}"))


class _RunRow(QWidget):
    """One row: run name label + Remove button."""

    def __init__(self, run_dir: Path, is_current: bool, parent=None) -> None:
        super().__init__(parent)
        self.run_dir = run_dir

        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(6)

        self.label = QLabel(run_dir.name)
        self.label.setWordWrap(False)
        self._set_current_style(is_current)
        lay.addWidget(self.label, 1)

        self.remove_btn = QPushButton(t("tb_remove"))
        self.remove_btn.setFixedWidth(70)
        self.remove_btn.setStyleSheet("padding:2px 6px;")
        lay.addWidget(self.remove_btn)

    def _set_current_style(self, is_current: bool) -> None:
        if is_current:
            self.label.setStyleSheet("font-weight:bold;color:#27ae60;")
            suffix = t("tb_current_run_label")
            name = self.run_dir.name
            if not self.label.text().endswith(suffix):
                self.label.setText(name + suffix)
        else:
            self.label.setStyleSheet("")
            self.label.setText(self.run_dir.name)

    def mark_current(self, is_current: bool) -> None:
        self._set_current_style(is_current)


class TensorBoardPanel(QGroupBox):
    """A collapsible panel listing TensorBoard run dirs with Remove buttons
    and an "Open TensorBoard" launch button.

    Usage::

        panel = TensorBoardPanel()
        panel.set_log_dir("output/logs")        # on config load
        panel.set_current_run("output/logs/anima_20260607-0000")  # on run_start
    """

    def __init__(self, parent=None) -> None:
        super().__init__(t("tb_panel_title"), parent)
        self._manager = TensorBoardManager()
        self._log_dir: Optional[Path] = None
        self._current_run_name: Optional[str] = None
        self._rows: list[_RunRow] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        # Scrollable run list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMaximumHeight(140)
        self._scroll.setMinimumHeight(60)
        self._inner = QWidget()
        self._inner_lay = QVBoxLayout(self._inner)
        self._inner_lay.setContentsMargins(2, 2, 2, 2)
        self._inner_lay.setSpacing(1)
        self._inner_lay.addStretch()
        self._scroll.setWidget(self._inner)
        outer.addWidget(self._scroll)

        self._empty_label = QLabel(t("tb_no_runs"))
        self._empty_label.setStyleSheet("color:#888;font-size:11px;padding:4px;")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._inner_lay.insertWidget(0, self._empty_label)

        # Button bar
        btn_bar = QHBoxLayout()

        self._open_btn = QPushButton(t("tb_open"))
        self._open_btn.setStyleSheet(
            "background:#2471a3;color:white;font-weight:bold;padding:4px 14px;"
        )
        self._open_btn.clicked.connect(self._open_tensorboard)
        btn_bar.addWidget(self._open_btn)

        self._stop_btn = QPushButton(t("tb_stop"))
        self._stop_btn.clicked.connect(self._stop_tensorboard)
        self._stop_btn.setEnabled(False)
        btn_bar.addWidget(self._stop_btn)

        btn_bar.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color:#aaa;font-size:11px;")
        btn_bar.addWidget(self._status_label)

        outer.addLayout(btn_bar)

        # Periodically refresh the list while visible (catches new runs from
        # daemon-started training that the GUI never directly launched).
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(4000)
        self._refresh_timer.timeout.connect(self._refresh_runs)
        self._refresh_timer.start()

    # ── Public API ──────────────────────────────────────────────────────────

    def set_log_dir(self, log_dir: str) -> None:
        """Point the panel at a logging base directory and refresh the list."""
        self._log_dir = Path(log_dir)
        self._refresh_runs()

    def set_current_run(self, run_dir: str) -> None:
        """Mark a run (by full path or name) as the active training run.

        Called when a ``run_start`` progress event carries a ``log_dir`` field.
        The corresponding row is highlighted green and gains a "(current)"
        suffix to make it easy to spot in TensorBoard's sidebar.
        """
        self._current_run_name = Path(run_dir).name if run_dir else None
        self._refresh_runs()

    def cleanup(self) -> None:
        """Stop the TensorBoard server and halt background polling."""
        self._manager.stop()
        self._refresh_timer.stop()

    # ── Internal ────────────────────────────────────────────────────────────

    def _refresh_runs(self) -> None:
        if self._log_dir is None or not self._log_dir.exists():
            return

        try:
            dirs = sorted(
                [d for d in self._log_dir.iterdir() if d.is_dir()],
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return

        existing_names = {d.name for d in dirs}

        # Remove stale rows
        for row in list(self._rows):
            if row.run_dir.name not in existing_names:
                self._rows.remove(row)
                self._inner_lay.removeWidget(row)
                row.deleteLater()

        known_names = {row.run_dir.name for row in self._rows}

        # Add new rows (newest-first — insert before existing rows)
        insert_pos = 0
        for d in dirs:
            if d.name not in known_names:
                is_current = d.name == self._current_run_name
                row = _RunRow(d, is_current, self._inner)
                row.remove_btn.clicked.connect(lambda _, r=row: self._remove_run(r))
                self._rows.insert(insert_pos, row)
                # Insert before the stretch (last item) and any existing rows
                self._inner_lay.insertWidget(insert_pos, row)
                known_names.add(d.name)
                insert_pos += 1

        # Update current highlighting on existing rows
        for row in self._rows:
            row.mark_current(row.run_dir.name == self._current_run_name)

        # Toggle the empty-state label
        has_rows = bool(self._rows)
        self._empty_label.setVisible(not has_rows)

    def _remove_run(self, row: _RunRow) -> None:
        try:
            shutil.rmtree(row.run_dir)
        except OSError:
            pass
        if row in self._rows:
            self._rows.remove(row)
        self._inner_lay.removeWidget(row)
        row.deleteLater()
        has_rows = bool(self._rows)
        self._empty_label.setVisible(not has_rows)

    def _open_tensorboard(self) -> None:
        if self._log_dir is None:
            return
        try:
            port = self._manager.start(str(self._log_dir))
        except RuntimeError:
            self._status_label.setText(t("tb_not_installed"))
            return
        self._stop_btn.setEnabled(True)
        self._status_label.setText(t("tb_status_running", port=port))
        # Give TensorBoard ~1.5 s to bind its port before opening the browser.
        QTimer.singleShot(1500, self._manager.open_browser)

    def _stop_tensorboard(self) -> None:
        self._manager.stop()
        self._stop_btn.setEnabled(False)
        self._status_label.setText(t("tb_status_stopped"))
