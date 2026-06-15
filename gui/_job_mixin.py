"""DaemonJobMixin — shared submit + stdout-observer plumbing for the tabs.

Two independent pieces a host tab can use à la carte:

* :meth:`_submit_job` — the submit → error-check → job-id-extract dance every
  launch site repeats. A daemon-down exception and a response carrying no
  ``job_id`` both warn + run the caller's rollback. Used by all four
  config-style tabs (ConfigTab's four launch sites, EasyControl, distill,
  preprocess).
* the stdout *observer* (:meth:`_init_job_observer`, :meth:`_watch_job`,
  :meth:`_drain_job_stdout`, :meth:`_poll_job`, :meth:`_stop_job`) — a 400 ms
  ``QTimer`` that tails a job's ``stdout.log``, routes tqdm lines to the
  progress bar and the rest to the log, and calls ``_on_job_finished(state)``
  once the daemon reports a terminal state. Used by the standalone observers
  (distill, preprocess). ConfigTab keeps its own richer observer (progress.jsonl
  + live sample preview + preprocess→train chain) and only borrows
  :meth:`_submit_job`.

Host requirements for the observer: a ``self.log`` widget, a
``self._progress_tracker``, and an ``_on_job_finished(state)`` method. The log
sink for stdout lines is :meth:`_emit_log_line` (default ``appendPlainText``);
override it when the host writes the log differently (distill uses its ``_log``).
"""

from __future__ import annotations

import re

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QMessageBox

from gui import daemon as gui_daemon
from gui.i18n import t


class DaemonJobMixin:
    _job_id: str | None = None

    # ── submit ────────────────────────────────────────────────────────────
    def _submit_job(self, submit_fn, *, on_fail=None) -> str | None:
        """Run ``submit_fn`` (a daemon submit_* call) and return its job id.

        Returns ``None`` — after warning the user and invoking ``on_fail`` (the
        busy-UI rollback) — when the daemon raised or handed back no job id.
        """
        try:
            resp = submit_fn()
        except Exception as e:  # noqa: BLE001 — daemon down / submit failed
            QMessageBox.warning(self, t("error"), t("daemon_submit_failed", err=str(e)))
            if on_fail is not None:
                on_fail()
            return None
        job_id = resp.get("job_id") if isinstance(resp, dict) else None
        if not job_id:
            QMessageBox.warning(
                self, t("error"), t("daemon_submit_failed", err=str(resp))
            )
            if on_fail is not None:
                on_fail()
            return None
        return job_id

    # ── stdout observer ───────────────────────────────────────────────────
    def _init_job_observer(self, interval_ms: int = 400) -> None:
        """Set up the job-observer state. Call once from ``__init__``."""
        self._job_id = None
        self._stdout_buf = ""
        self._stdout_tailer = gui_daemon.FileTailer()
        self._job_timer = QTimer(self)
        self._job_timer.setInterval(interval_ms)
        self._job_timer.timeout.connect(self._poll_job)

    def _watch_job(self, job_id: str, *, replay_log: bool) -> None:
        """Point the tailer at ``job_id``'s stdout.log and start the poll timer.

        ``replay_log`` reads the log from the top (re-attach after a GUI
        restart); otherwise pre-existing output is discarded so a fresh launch
        shows only new lines. The host sets its own busy UI around this call.
        """
        self._job_id = job_id
        self._stdout_buf = ""
        self._stdout_tailer.watch(gui_daemon.stdout_path(job_id))
        if not replay_log:
            self._stdout_tailer.read_new()  # discard backlog from a fresh launch
        self._job_timer.start()

    def _emit_log_line(self, line: str) -> None:
        """Append one stdout line to the log. Default ``appendPlainText`` adds its
        own newline; override when the host's log sink differs."""
        self.log.appendPlainText(line)

    def _drain_job_stdout(self) -> None:
        """Append new stdout.log lines to the log (carriage-return aware).

        tqdm progress lines drive the bar via ``_progress_tracker.feed`` instead
        of spamming the log (preprocess/mask + the distill scripts emit no
        progress.jsonl, so tqdm is the only progress signal)."""
        chunk = self._stdout_tailer.read_new()
        if not chunk:
            return
        parts = re.split(r"[\r\n]", self._stdout_buf + chunk)
        self._stdout_buf = parts[-1]  # incomplete trailing fragment
        for line in parts[:-1]:
            if self._progress_tracker.feed(line):
                continue
            if line:
                self._emit_log_line(line)

    def _poll_job(self) -> None:
        if not self._job_id:
            return
        self._drain_job_stdout()
        state = gui_daemon.read_job_state(self._job_id)
        if gui_daemon.is_terminal(state):
            self._on_job_finished(state)

    def _stop_job(self) -> None:
        """Abort the attached daemon job; the poll loop then observes the
        terminal state and restores the UI. The daemon stays up."""
        if not self._job_id:
            return
        try:
            gui_daemon.stop_job(self._job_id)
        except Exception as e:  # noqa: BLE001
            self._emit_log_line(f"stop failed: {e}")
