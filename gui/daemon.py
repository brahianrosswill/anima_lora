"""GUI-side bridge to the local training daemon (``scripts/daemon``).

Phase 2 turns the GUI into a daemon *client*: the Train button submits a job to
the daemon (so training survives the GUI closing) and the tab then *observes*
that job by polling the per-job files the daemon already writes to local disk —
``job.json`` for state, ``progress.jsonl`` for the bar, ``stdout.log`` for the
log. Everything is poll-driven off the tab's existing ``QTimer``; there is
deliberately **no background thread / SSE consumer**, because the daemon is
localhost-only (a non-goal forbids remote) so the files are right there to read.

This keeps the heavy ``library.*`` / torch imports out of the GUI: the daemon
client is pure ``urllib`` and ``config`` is pure ``pathlib``.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from scripts.daemon import client as _client
from scripts.daemon import config as _cfg
from scripts.daemon.jobs import TERMINAL_STATES

# Re-export so callers don't reach into scripts.daemon themselves.
ensure_daemon = _client.ensure_daemon
is_running = _client.is_running


def submit_training(
    *,
    method: str,
    preset: str,
    methods_subdir: Optional[str],
    extra: Optional[list[str]] = None,
) -> dict:
    """Auto-start the daemon if needed and enqueue a training job.

    Mirrors what ``tasks.py lora-gui <variant>`` would have launched inline:
    ``method`` is the gui-methods variant stem and ``methods_subdir`` is
    ``"gui-methods"``. Returns the daemon's ``{job_id, state}`` response.
    """
    cl = ensure_daemon()
    return cl.submit(
        method=method,
        preset=preset,
        methods_subdir=methods_subdir,
        extra=extra or [],
    )


def stop_job(job_id: str) -> dict:
    """Abort a running/queued job (daemon stays up, advances the queue)."""
    return _client.DaemonClient().stop(job_id)


def active_job_id() -> Optional[str]:
    """The daemon's currently-running job id, or ``None`` (daemon down/idle).

    Used on tab construction to re-attach the UI to a job that's still running
    from a previous GUI session (or that the ComfyUI node / CLI submitted).
    """
    health = _client.DaemonClient().health()
    return health.get("active_job") if health else None


def progress_path(job_id: str) -> str:
    return str(_cfg.job_dir(job_id) / "progress.jsonl")


def stdout_path(job_id: str) -> str:
    return str(_cfg.job_dir(job_id) / "stdout.log")


def read_job_state(job_id: str) -> Optional[str]:
    """Read the persisted job state straight from ``job.json``.

    Cheaper and hang-proof vs. an HTTP round-trip every poll tick: the daemon
    writes ``job.json`` atomically on every state transition, so a local read is
    always a complete, current record. Returns ``None`` if the file isn't there
    yet (job dir not created) or is mid-rewrite.
    """
    try:
        data = json.loads(
            (_cfg.job_dir(job_id) / "job.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return data.get("state")


def is_terminal(state: Optional[str]) -> bool:
    return state in TERMINAL_STATES


class FileTailer:
    """Tail a growing text file by byte offset — thread-free, poll-driven.

    The daemon captures the training subprocess's stdout+stderr to
    ``stdout.log``; this reads whatever's been appended since the last call. A
    fresh :meth:`watch` (pos 0) replays the whole file, which is what re-attach
    relies on to repopulate the log widget after a GUI restart.
    """

    def __init__(self) -> None:
        self._path: Optional[str] = None
        self._pos = 0

    def watch(self, path: Optional[str]) -> None:
        self._path = path
        self._pos = 0

    def reset(self) -> None:
        self.watch(None)

    def read_new(self) -> str:
        """Return text appended since the last read (``""`` if none/absent)."""
        if not self._path or not os.path.exists(self._path):
            return ""
        try:
            with open(self._path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(self._pos)
                chunk = f.read()
                self._pos = f.tell()
        except OSError:
            return ""
        return chunk
