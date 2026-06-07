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
        self._log_dir: Optional[str] = None

    def start(self, log_dir: str) -> int:
        """Start TensorBoard pointed at *log_dir*. Returns the port.

        If a server is already running against a *different* ``log_dir`` it is
        restarted so the view re-scopes (e.g. switching between "all runs" and
        the single current-run dir). Already running against the same dir is a
        no-op. Raises ``RuntimeError`` when tensorboard is not importable."""
        if self.running:
            if self._log_dir == log_dir:
                return self._PORT
            self.stop()
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
        self._log_dir = log_dir
        return self._PORT

    def stop(self) -> None:
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                pass
            self._process = None
        self._log_dir = None

    @property
    def running(self) -> bool:
        if self._process is None:
            return False
        return self._process.poll() is None

    def open_browser(self) -> None:
        QDesktopServices.openUrl(QUrl(f"http://localhost:{self._PORT}"))


class _RunRow(QWidget):
    """One row: run name label + View + Remove buttons."""

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

        self.view_btn = QPushButton(t("tb_view"))
        self.view_btn.setFixedWidth(60)
        self.view_btn.setStyleSheet("padding:2px 6px;")
        self.view_btn.setToolTip(t("tb_view_tip"))
        lay.addWidget(self.view_btn)

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

    def __init__(self, parent=None, expand: bool = False) -> None:
        super().__init__(t("tb_panel_title"), parent)
        self._manager = TensorBoardManager()
        self._log_dir: Optional[Path] = None
        self._current_run_name: Optional[str] = None
        self._current_run_path: Optional[Path] = None
        self._training_active: bool = False
        self._rows: list[_RunRow] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)

        # Scrollable run list
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        # As a bottom-of-form panel the list is capped so it doesn't crowd the
        # config; on its own dedicated tab (``expand=True``) it fills the page.
        if not expand:
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

        # Scoped to the single in-progress run — highlighted while training is
        # active so the user can jump straight to the current run's curves.
        self._current_btn = QPushButton(t("tb_open_current"))
        self._current_btn.clicked.connect(self._open_current_run)
        btn_bar.addWidget(self._current_btn)

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

        self._update_current_btn()

    # ── Public API ──────────────────────────────────────────────────────────

    def set_log_dir(self, log_dir: str) -> None:
        """Point the panel at a logging base directory and refresh the list."""
        self._log_dir = Path(log_dir)
        self._refresh_runs()

    def set_current_run(self, run_dir: str) -> None:
        """Mark a run (by full path or name) as the active training run.

        Called when a ``run_start`` progress event carries a ``log_dir`` field.
        The corresponding row is highlighted green and gains a "(current)"
        suffix to make it easy to spot in TensorBoard's sidebar. While a run is
        marked current the "현재 학습 조회" (view current run) button lights up.
        """
        if run_dir:
            self._current_run_path = Path(run_dir)
            self._current_run_name = self._current_run_path.name
            self._training_active = True
        else:
            self._current_run_path = None
            self._current_run_name = None
            self._training_active = False
        self._update_current_btn()
        self._refresh_runs()

    def clear_current_run(self) -> None:
        """Mark training as no longer active (dims the current-run button).

        Called when the run ends. The row highlight is kept so the just-finished
        run stays easy to spot, but the button stops advertising a live run.
        """
        self._training_active = False
        self._update_current_btn()

    def cleanup(self) -> None:
        """Stop the TensorBoard server and halt background polling."""
        self._manager.stop()
        self._refresh_timer.stop()

    # ── Internal ────────────────────────────────────────────────────────────

    def _update_current_btn(self) -> None:
        """Highlight + enable the current-run button only while a run is live."""
        live = self._training_active and self._current_run_path is not None
        self._current_btn.setEnabled(live)
        if live:
            # Pulsing-green primary look so it stands out during training.
            self._current_btn.setStyleSheet(
                "background:#27ae60;color:white;font-weight:bold;padding:4px 14px;"
            )
            self._current_btn.setToolTip(t("tb_open_current_tip"))
        else:
            self._current_btn.setStyleSheet("color:#888;padding:4px 14px;")
            self._current_btn.setToolTip(t("tb_open_current_idle_tip"))

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
                row.view_btn.clicked.connect(lambda _, r=row: self._launch(r.run_dir))
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
        self._launch(self._log_dir)

    def _open_current_run(self) -> None:
        """Launch TensorBoard scoped to only the current run's directory so the
        view shows just the in-progress training's curves."""
        self._launch(self._current_run_path)

    def _launch(self, log_dir: Optional[Path]) -> None:
        if log_dir is None:
            return
        try:
            port = self._manager.start(str(log_dir))
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


class TensorBoardTab(QWidget):
    """Dedicated tab hosting a :class:`TensorBoardPanel`.

    The panel used to sit permanently at the bottom of the ConfigTab; it now
    lives on its own tab so the training form isn't always crowded by it.
    ConfigTab keeps a reference to ``.panel`` for log-dir / current-run syncing.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        self.panel = TensorBoardPanel(self, expand=True)
        lay.addWidget(self.panel)

    def cleanup_subprocess(self) -> None:
        """App-shutdown hook (mirrors the other tabs) — stops the server."""
        self.panel.cleanup()
