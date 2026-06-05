"""EasyControlTab — config editor for the EasyControl adapter family.

Reuses ConfigTab's grouped form + field-explanation panel + Save UI (so the tab
reads like the LoRA / Methods tabs — config on the left, explanation on the
right). Training rides ConfigTab's daemon path (the variant stem *is* the
train.py method — ``EASYADAPTER`` is only read at task dispatch, never deeper —
so it survives the GUI closing). Preprocess stays a bespoke QProcess
(``easycontrol-preprocess``, mangafy for colorize). The selected variant
implies an ``EASYADAPTER`` value for the preprocess route:

  • EasyControl (ref == target) — EASYADAPTER unset → configs trained from
    gui-methods/easycontrol.toml.
  • Colorize (B&W manga → color) — EASYADAPTER=colorize → mangafy preprocess
    (easycontrol_adapters/colorization/prep.py) + gui-methods/colorize.toml.

Variants are the ``[variant] family = "easycontrol"`` gui-methods files. Custom
variants don't map onto the easycontrol method selection, so the "+ New"
button and custom entries are suppressed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment
from PySide6.QtWidgets import QPushButton

from gui import (
    ROOT,
    confirm_existing_caches,
    confirm_resumable_checkpoint,
    list_gui_variants,
    merged_gui_variant_preset,
)
from gui import daemon as gui_daemon
from gui.i18n import t
from gui.tabs.config_tab import ConfigTab


class EasyControlTab(ConfigTab):
    # variant stem → EASYADAPTER value (absent → default ref==target EasyControl).
    _VARIANT_ENV = {"colorize": "colorize"}
    # variant stem → cache dir the preprocess-reuse prompt inspects. Colorize's
    # condition latents land in easycontrol/colorize/cond/; default EasyControl in
    # its own cache. (text/cond split for colorize is covered well enough by cond.)
    _CACHE_DIRS = {
        "easycontrol": "post_image_dataset/easycontrol",
        "colorize": "post_image_dataset/easycontrol/colorize/cond",
    }

    def __init__(self):
        # Created BEFORE super().__init__ because ConfigTab.__init__ ends by
        # calling _try_reattach → _attach_to_job, which disables preprocess_btn.
        # On a GUI reopen mid-train that reattach fires during super().__init__,
        # before the post-super setup below would otherwise create the button.
        self.preprocess_btn = QPushButton(t("preprocess"))

        super().__init__(methods=["easycontrol"])

        # The easycontrol route trains exactly the shipped family variants
        # (EASYADAPTER picks the method name), so custom variants can't map onto
        # it — hide the "+ New" button. _refresh_variant_row also drops customs.
        self.new_variant_btn.setVisible(False)
        # test-easycontrol needs a REF_IMAGE the plain Test button can't
        # supply, so hide Test on this tab.
        self.test_btn.setVisible(False)

        # "How to build your own EasyControl adapter" — opens ADAPTER_GUIDE.md in
        # the in-app markdown viewer (same dialog as the top-bar Guidebook). The
        # guide is the build-your-own-control-task reference (colorize worked
        # through in detail).
        self.adapter_guide_btn = QPushButton(t("adapter_guide"))
        self.adapter_guide_btn.setToolTip(t("adapter_guide_tooltip"))
        self.adapter_guide_btn.setStyleSheet(
            "background:#16a085;color:white;font-weight:bold;padding:4px 12px;"
        )
        self.adapter_guide_btn.clicked.connect(self._open_adapter_guide)
        self._top_bar.insertWidget(
            self._top_bar.indexOf(self.train_btn), self.adapter_guide_btn
        )

        # ConfigTab has no Preprocess button (it auto-chains a daemon preprocess).
        # EasyControl preprocessing is the bespoke easycontrol-preprocess
        # (mangafy for colorize), so it gets an explicit button before Train.
        # (preprocess_btn itself is built above the super() call.)
        self.preprocess_btn.setStyleSheet(
            "background:#2980b9;color:white;font-weight:bold;padding:4px 16px;"
        )
        self.preprocess_btn.clicked.connect(self._ec_start_preprocess)
        self._top_bar.insertWidget(
            self._top_bar.indexOf(self.train_btn), self.preprocess_btn
        )

        # Train still rides ConfigTab's daemon path (so it survives the GUI
        # closing), but through a slimmer handler: EasyControl has its own
        # explicit Preprocess button + bespoke caches, so it skips ConfigTab's
        # cache-existence check / auto-chain-preprocess logic and submits the
        # variant straight to the daemon.
        self.train_btn.clicked.disconnect()
        self.train_btn.clicked.connect(self._ec_start_train)

    # ── Adapter guide ──────────────────────────────────────────────

    def _open_adapter_guide(self) -> None:
        # Lazy import: gui.app imports this tab, so a top-level import would be
        # circular. The dialog is a generic markdown viewer (takes any path).
        from gui.app import GuidebookDialog
        from gui.i18n import current_language

        # Localized guide: ADAPTER_GUIDE.<lang>.md, English (ADAPTER_GUIDE.md) as
        # the fallback for `en` and any language whose translation is missing.
        base = ROOT / "easycontrol_adapters"
        path = base / f"ADAPTER_GUIDE.{current_language()}.md"
        if not path.exists():
            path = base / "ADAPTER_GUIDE.md"
        GuidebookDialog(path, self).exec()

    # ── Variant list: family built-ins only (no customs) ───────────

    def _refresh_variant_row(self, method: str) -> None:
        variants = [v for v in list_gui_variants(method) if not v.startswith("custom/")]
        current = [
            self.variant_combo.itemText(i) for i in range(self.variant_combo.count())
        ]
        if current != variants:
            self.variant_combo.blockSignals(True)
            self.variant_combo.clear()
            if variants:
                self.variant_combo.addItems(variants)
            self.variant_combo.blockSignals(False)

    # ── EASYADAPTER routing ────────────────────────────────────────

    def _ec_adapter(self) -> str | None:
        return self._VARIANT_ENV.get(self._current_variant())

    def _ec_proc_env(self) -> QProcessEnvironment:
        """System env with EASYADAPTER set (or cleared) for the active variant.
        Rebuilt each launch so a stale value can't leak across runs (the QProcess
        is reused)."""
        env = QProcessEnvironment.systemEnvironment()
        adapter = self._ec_adapter()
        if adapter:
            env.insert("EASYADAPTER", adapter)
        else:
            env.remove("EASYADAPTER")
        return env

    def _ec_cache_dir(self) -> Path:
        rel = self._CACHE_DIRS.get(
            self._current_variant(), "post_image_dataset/easycontrol"
        )
        return ROOT / rel

    def _ec_launch(self, argv: list[str], mode: str) -> None:
        if self._proc.state() != QProcess.NotRunning:
            return
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._running_mode = mode
        adapter = self._ec_adapter()
        prefix = f"EASYADAPTER={adapter} " if adapter else ""
        self._log(f"> {prefix}python {' '.join(argv)}\n")
        self._proc.setProcessEnvironment(self._ec_proc_env())
        self._ec_set_busy(True)
        self._proc.start(sys.executable, argv)

    def _ec_start_preprocess(self) -> None:
        if not confirm_existing_caches(self, self._ec_cache_dir()):
            return
        self._ec_launch(["tasks.py", "easycontrol-preprocess"], "ec_preprocess")

    def _ec_start_train(self) -> None:
        # Flush form edits so train.py reads the same gui-methods/<variant>.toml.
        if self._dirty:
            self._save_preset(silent=True)
        variant = self._current_variant()
        merged, _ = merged_gui_variant_preset(variant, self._IMPLICIT_PRESET)
        if not confirm_resumable_checkpoint(self, merged):
            return
        # The variant stem ("easycontrol" / "colorize") IS the train.py method
        # — EASYADAPTER is read only at task dispatch, never deeper — so submit
        # it to the daemon exactly like the LoRA tab. ConfigTab._launch_training
        # POSTs {method=variant, preset, methods_subdir="gui-methods"} and binds
        # the bar/log to the detached job (which then survives the GUI closing).
        self._launch_training(variant)

    # Daemon training disables Train/Test/combos via ConfigTab; also gray out the
    # EasyControl-only Preprocess button. Overriding _attach_to_job (not just the
    # launch site) covers re-attach on GUI reopen too. _restore_idle_ui re-enables
    # it on finish.
    def _attach_to_job(
        self, job_id: str, *, replay_log: bool, kind: str = "train"
    ) -> None:
        super()._attach_to_job(job_id, replay_log=replay_log, kind=kind)
        self.preprocess_btn.setEnabled(False)

    # ── Busy / idle state ──────────────────────────────────────────

    def _ec_set_busy(self, busy: bool) -> None:
        self.preprocess_btn.setEnabled(not busy)
        self.train_btn.setEnabled(not busy)
        self.stop_btn.setEnabled(busy)
        self.method_combo.setEnabled(not busy)
        self.variant_combo.setEnabled(not busy)

    def _restore_idle_ui(self):
        super()._restore_idle_ui()
        # ConfigTab's restore handles Train/Test/combos; re-enable Preprocess too.
        self.preprocess_btn.setEnabled(True)

    def _try_reattach(self) -> None:
        """Re-bind to an easycontrol/colorize daemon training job still running
        when this tab is constructed (close GUI mid-train → reopen → re-attach).

        Discriminate by method so we don't hijack another tab's job (e.g. a
        LoRA-tab training run): the daemon's single active job is shared across
        the ConfigTab subclasses, and this tab only owns its own family's
        variants. EasyControl preprocess runs as a QProcess (not a daemon
        command job), so there's nothing of ours to re-attach but the train."""
        try:
            job_id = gui_daemon.active_job_id()
        except Exception:  # noqa: BLE001 — daemon unreachable → nothing to attach
            return
        if not job_id or gui_daemon.read_job_kind(job_id) != "train":
            return
        family = {
            v for v in list_gui_variants("easycontrol") if not v.startswith("custom/")
        }
        if gui_daemon.read_job_label(job_id) not in family:
            return
        self.log.clear()
        self._reset_progress()
        self._progress_tracker.mark_starting(t("starting"))
        self._log(t("daemon_reattached", job_id=job_id))
        self._attach_to_job(job_id, replay_log=True, kind="train")
