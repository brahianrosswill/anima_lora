"""Anima LoRA GUI — main window, dark theme, and entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import toml
from PySide6.QtCore import QEvent, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import (
    DEFAULT_AUTOTAG_CONFIDENCE,
    get_setting,
    set_setting,
)
from gui import daemon as gui_daemon
from gui import theme as gui_theme
from gui.i18n import (
    available_languages,
    current_language,
    load_language,
    save_language,
    t,
)
from gui.tabs.config_tab import ConfigTab
from gui.tabs.easycontrol_tab import EasyControlTab
from gui.tabs.image_tab import ImageViewerTab
from gui.tabs.merge_tab import MergeTab
from gui.tabs.methods_tab import MethodsTab
from gui.tabs.preprocess_tab import PreprocessingTab
from gui.tabs.queue_tab import QueueTab
from gui.tensorboard import TensorBoardTab
from gui.system_dialog import (
    GITHUB_ISSUES_URL,
    GITHUB_REPO_URL,
    check_for_update_async,
    open_models_dialog,
    open_update_dialog,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GUIDELINES = _REPO_ROOT / "docs" / "guidelines"
_GUIDEBOOK_BY_LANG: dict[str, Path] = {
    "en": _GUIDELINES / "guidebook.md",
    "ko": _GUIDELINES / "가이드북.md",
    "cn": _GUIDELINES / "指南书.md",
    "ja": _GUIDELINES / "ガイドブック.md",
}
_GUIDEBOOK_FALLBACK = _GUIDEBOOK_BY_LANG["en"]
ICON_PATH = Path(__file__).resolve().parent / "icon.ico"


def _guidebook_path() -> Path:
    return _GUIDEBOOK_BY_LANG.get(current_language(), _GUIDEBOOK_FALLBACK)


LANG_NAMES = {"en": "English", "ko": "한국어", "cn": "简体中文", "ja": "日本語"}

# Keeps the live MainWindow alive across the in-place rebuild that applies a
# language change (main() seeds it; MainWindow._reload_ui swaps it).
_WINDOW: MainWindow | None = None


def _dark(app: QApplication):
    """Apply the user's chosen named theme (Dark / Light / Sepia).

    Thin wrapper kept for call-site stability; the palette + stylesheet now live
    in ``gui.theme`` so individual widgets can share the same tokens."""
    gui_theme.apply_theme(app)


class GuidebookDialog(QDialog):
    """In-app markdown viewer for the guidebook."""

    def __init__(self, md_path: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("guidebook"))
        self.resize(900, 720)
        self._md_path = md_path

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        # Resolve relative links/images against the markdown file's directory.
        self.browser.setSearchPaths([str(md_path.parent)])
        self.browser.document().setBaseUrl(
            QUrl.fromLocalFile(str(md_path.parent) + "/")
        )
        # Default anchor color is pure blue — illegible on the dark bg.
        self.browser.document().setDefaultStyleSheet(
            "a { color: #ffb86b; text-decoration: underline; }"
            "a:visited { color: #e6944e; }"
            "code { background:#2a2a2a; padding:1px 4px; border-radius:3px; }"
            "pre { background:#2a2a2a; padding:8px; border-radius:4px; }"
        )
        self.browser.setStyleSheet(
            "QTextBrowser { background:#1e1e1e; color:#dcdcdc; "
            "border:1px solid #444; padding:12px; }"
        )
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError as e:
            text = f"# Error\n\nCould not read `{md_path}`:\n\n`{e}`"
        self.browser.setMarkdown(text)
        lay.addWidget(self.browser)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        open_ext = QPushButton(t("guidebook_open_external"))
        open_ext.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._md_path)))
        )
        close = QPushButton(t("guidebook_close"))
        close.clicked.connect(self.close)
        btn_bar.addWidget(open_ext)
        btn_bar.addWidget(close)
        lay.addLayout(btn_bar)


def _mcp_paths() -> tuple[Path, Path]:
    """(venv python, MCP bridge script) for THIS checkout — real absolute
    paths, not the <repo> placeholder the docs use (scripts/daemon/README.md)."""
    venv_python = (
        _REPO_ROOT
        / ".venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    return venv_python, _REPO_ROOT / "scripts" / "daemon" / "mcp.py"


def _mcp_add_command() -> str:
    """The `claude mcp add` one-liner for Claude Code."""

    def q(p: Path) -> str:
        return f'"{p}"' if " " in str(p) else str(p)

    venv_python, bridge = _mcp_paths()
    return f"claude mcp add anima-daemon -- {q(venv_python)} {q(bridge)}"


def _mcp_json_config() -> str:
    """The client-agnostic mcpServers JSON block (Claude Desktop, OpenClaw, …).
    json.dumps so Windows backslashes come out escaped and paste-able."""
    import json

    venv_python, bridge = _mcp_paths()
    cfg = {
        "mcpServers": {
            "anima-daemon": {"command": str(venv_python), "args": [str(bridge)]}
        }
    }
    return json.dumps(cfg, indent=2)


class SettingsDialog(QDialog):
    """App settings: language + MCP server registration for agent clients."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("settings_title"))
        self.setMinimumWidth(560)
        # Set when the user picks a new language and opts into an immediate
        # reload; MainWindow checks it after exec() and rebuilds itself.
        self.reload_requested = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        lang_row = QHBoxLayout()
        lang_row.addWidget(QLabel(t("language")))
        self.lang_combo = QComboBox()
        for code in available_languages():
            self.lang_combo.addItem(LANG_NAMES.get(code, code), code)
        self.lang_combo.setCurrentIndex(available_languages().index(current_language()))
        self.lang_combo.currentIndexChanged.connect(self._change_lang)
        self.lang_combo.setFixedWidth(120)
        lang_row.addWidget(self.lang_combo)
        lang_row.addStretch()
        lay.addLayout(lang_row)

        prefs_group = QGroupBox(t("settings_prefs_header"))
        prefs_lay = QVBoxLayout(prefs_group)

        # Autotagger confidence floor (applied on top of the model's per-tag F1
        # thresholds; see AnimaTagger.predict_caption min_confidence).
        conf_row = QHBoxLayout()
        conf_label = QLabel(t("settings_autotag_confidence"))
        conf_label.setToolTip(t("settings_autotag_confidence_tooltip"))
        conf_row.addWidget(conf_label)
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2)
        self.conf_spin.setToolTip(t("settings_autotag_confidence_tooltip"))
        self.conf_spin.setValue(
            float(get_setting("autotag_confidence", DEFAULT_AUTOTAG_CONFIDENCE))
        )
        self.conf_spin.valueChanged.connect(
            lambda v: set_setting("autotag_confidence", round(float(v), 2))
        )
        self.conf_spin.setFixedWidth(80)
        conf_row.addWidget(self.conf_spin)
        conf_row.addStretch()
        prefs_lay.addLayout(conf_row)

        # Theme selector — Dark / Light / Sepia. Applied live across the app
        # palette + stylesheet; closing the dialog rebuilds the window so each
        # tab's per-widget tokens (gui.theme.tok) repaint in the new theme.
        theme_row = QHBoxLayout()
        theme_label = QLabel(t("settings_theme"))
        theme_label.setToolTip(t("settings_theme_tooltip"))
        theme_row.addWidget(theme_label)
        self.theme_combo = QComboBox()
        for code in gui_theme.THEME_ORDER:
            self.theme_combo.addItem(t(gui_theme.THEME_LABEL_KEYS[code]), code)
        cur = gui_theme.current_theme_name()
        self.theme_combo.setCurrentIndex(gui_theme.THEME_ORDER.index(cur))
        self.theme_combo.setToolTip(t("settings_theme_tooltip"))
        self.theme_combo.currentIndexChanged.connect(self._change_theme)
        self.theme_combo.setFixedWidth(140)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()
        prefs_lay.addLayout(theme_row)

        lay.addWidget(prefs_group)

        mcp_group = QGroupBox(t("settings_mcp_header"))
        mcp_lay = QVBoxLayout(mcp_group)
        self._add_command_block(
            mcp_lay, t("settings_mcp_desc"), _mcp_add_command(), height=64
        )
        self._add_command_block(
            mcp_lay, t("settings_mcp_desc_json"), _mcp_json_config(), height=140
        )
        lay.addWidget(mcp_group)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        close = QPushButton(t("settings_close"))
        close.clicked.connect(self.close)
        btn_bar.addWidget(close)
        lay.addLayout(btn_bar)

    def _add_command_block(
        self, layout: QVBoxLayout, desc: str, text: str, height: int
    ) -> None:
        """A word-wrapped description, a read-only monospace box, and a copy
        button that flashes confirmation."""
        label = QLabel(desc)
        label.setWordWrap(True)
        layout.addWidget(label)

        edit = QPlainTextEdit(text)
        edit.setReadOnly(True)
        mono = QFont("Consolas", 9)
        mono.setStyleHint(QFont.Monospace)
        edit.setFont(mono)
        edit.setFixedHeight(height)
        layout.addWidget(edit)

        copy_row = QHBoxLayout()
        copy_row.addStretch()
        btn = QPushButton(t("settings_mcp_copy"))
        btn.clicked.connect(lambda: self._copy(text, btn))
        copy_row.addWidget(btn)
        layout.addLayout(copy_row)

    def _copy(self, text: str, btn: QPushButton) -> None:
        QApplication.clipboard().setText(text)
        btn.setText(t("settings_mcp_copied"))
        QTimer.singleShot(1500, lambda: btn.setText(t("settings_mcp_copy")))

    def _change_lang(self, idx: int):
        lang = self.lang_combo.itemData(idx)
        # save_language also flips the in-process language, so the prompt
        # below (and a rebuilt MainWindow) already render in the new language.
        save_language(lang)
        choice = QMessageBox.question(
            self,
            t("settings_lang_apply_title"),
            t("settings_lang_apply_question"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if choice == QMessageBox.Yes:
            self.reload_requested = True
            self.accept()

    def _change_theme(self, idx: int) -> None:
        """Persist + apply the chosen theme live, then request a window rebuild.

        ``apply_theme`` restyles app-level chrome immediately; the rebuild on
        close makes per-widget ``tok()`` lookups (log boxes, previews, …) repaint
        too. Unlike a language change there's no confirm prompt — it's cheap and
        reversible."""
        name = self.theme_combo.itemData(idx)
        gui_theme.set_theme(name)
        app = QApplication.instance()
        if app is not None:
            gui_theme.apply_theme(app, name)
        self.reload_requested = True


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("window_title"))
        self.resize(1100, 750)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        # Right-click anywhere — including over text widgets that would otherwise
        # show Qt's default copy/select-all menu (explanation panels, log
        # consoles) — opens our app menu. An app-wide event filter catches the
        # ContextMenu event before it reaches the target widget so the menu is
        # uniform everywhere. Removed in closeEvent so a _reload_ui rebuild
        # doesn't stack filters.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        central = QWidget()
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        # Top button bar (guidebook / models / update / overlays / settings)
        lang_bar = QHBoxLayout()
        if ICON_PATH.exists():
            icon_label = QLabel()
            pix = QPixmap(str(ICON_PATH))
            if not pix.isNull():
                icon_label.setPixmap(
                    pix.scaled(
                        QSize(28, 28),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
            icon_label.setContentsMargins(4, 0, 6, 0)
            lang_bar.addWidget(icon_label)
        self.guide_btn = QPushButton(t("guidebook"))
        self.guide_btn.setToolTip(t("guidebook_tooltip"))
        self.guide_btn.setStyleSheet(
            "background:#16a085;color:white;font-weight:bold;padding:4px 12px;"
        )
        self.guide_btn.clicked.connect(self._open_guidebook)
        lang_bar.addWidget(self.guide_btn)

        self.models_btn = QPushButton(t("models_btn"))
        self.models_btn.setToolTip(t("models_btn_tooltip"))
        self.models_btn.clicked.connect(lambda: open_models_dialog(self))
        lang_bar.addWidget(self.models_btn)

        self.update_btn = QPushButton(t("update_btn"))
        self.update_btn.setToolTip(t("update_btn_tooltip"))
        self.update_btn.clicked.connect(lambda: open_update_dialog(self))
        lang_bar.addWidget(self.update_btn)
        # Background check — paints the button amber + "Update ●" when a newer
        # release exists. Skips silently when .anima_release.json is missing
        # (no baseline → can't tell if user is already current) and reuses the
        # 6h gui_settings.json cache so launches don't hit GitHub each time.
        self._update_check_thread = check_for_update_async(
            self, self._show_update_available
        )

        self.issues_btn = QPushButton(t("report_issue"))
        self.issues_btn.setToolTip(t("report_issue_tooltip"))
        self.issues_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_ISSUES_URL))
        )
        lang_bar.addWidget(self.issues_btn)

        # The Queue view is a top-bar toggle because the daemon job queue is
        # global — it spans every method, so it lives as an overlay over the
        # tab set rather than a tab inside it. Like TensorBoard; the two
        # overlays are mutually exclusive.
        self.queue_btn = QPushButton(t("tab_queue"))
        self.queue_btn.setCheckable(True)
        self._queue_idle_style = (
            "QPushButton { background:#5d6d7e; color:white; "
            "font-weight:bold; padding:4px 12px; border:1px solid #5d6d7e; "
            "border-radius:3px; }"
            "QPushButton:hover { background:#6b7c8c; }"
        )
        self._queue_active_style = (
            "QPushButton { background:#34495e; color:white; "
            "font-weight:bold; padding:4px 12px; border:1px solid #34495e; "
            "border-radius:3px; }"
            "QPushButton:hover { background:#3d566e; }"
        )
        self.queue_btn.toggled.connect(self._toggle_queue_view)
        lang_bar.addWidget(self.queue_btn)

        # TensorBoard is a top-bar toggle because the run list is shared across
        # every method — a single global view rather than a per-tab duplicate.
        # Toggling it on swaps the whole tab area for the TensorBoard panel;
        # toggling off returns to the tab set.
        self.tensorboard_btn = QPushButton(t("tab_tensorboard"))
        self.tensorboard_btn.setCheckable(True)
        self._tensorboard_idle_style = (
            "QPushButton { background:#2471a3; color:white; "
            "font-weight:bold; padding:4px 12px; border:1px solid #2471a3; "
            "border-radius:3px; }"
            "QPushButton:hover { background:#2e86c1; }"
        )
        self._tensorboard_active_style = (
            "QPushButton { background:#117864; color:white; "
            "font-weight:bold; padding:4px 12px; border:1px solid #117864; "
            "border-radius:3px; }"
            "QPushButton:hover { background:#148f77; }"
        )
        self.tensorboard_btn.toggled.connect(self._toggle_tensorboard)
        lang_bar.addWidget(self.tensorboard_btn)

        lang_bar.addStretch()
        self.settings_btn = QPushButton(t("settings_btn"))
        self.settings_btn.setToolTip(t("settings_btn_tooltip"))
        self.settings_btn.clicked.connect(self._open_settings)
        lang_bar.addWidget(self.settings_btn)
        main_lay.addLayout(lang_bar)

        # One tab set holds everything; the TensorBoard / Queue overlays share
        # a QStackedWidget with it so toggling swaps the visible view in place
        # — same window, no popup. All widgets stay alive across switches so
        # subprocess state and log buffers survive toggling.
        # The TensorBoard runs panel is a single shared instance (the run list
        # is method-agnostic). It's reached via the top-bar TensorBoard toggle
        # rather than a tab. Both ConfigTab and MethodsTab keep a reference to
        # its `.panel` so either can sync the log dir on variant switch /
        # launch / run_start.
        self._tb_tab = TensorBoardTab()

        # Built before ConfigTab so the Train auto-chain can flush this tab's
        # GUI preprocess settings to the selected method before it preprocesses.
        self._preprocess_tab = PreprocessingTab()

        self.tabs = QTabWidget()
        self.tabs.addTab(
            ConfigTab(
                methods=["lora", "tlora", "hydralora"],
                tb_panel=self._tb_tab.panel,
                preprocess_tab=self._preprocess_tab,
            ),
            t("tab_config"),
        )
        self.tabs.addTab(self._preprocess_tab, t("tab_preprocess"))
        self.tabs.addTab(ImageViewerTab(), t("tab_images"))
        self.tabs.addTab(MergeTab(), t("tab_merge"))
        # Experimental tabs sit at the end of the same set (the old top-bar
        # toggle that swapped a separate tab bar is gone). MethodsTab folds
        # every trainable experimental method behind one dropdown — FeRA /
        # ChimeraHydra / Soft Tokens (flat train.py methods) plus SPD / Turbo
        # (bespoke distill loops) — so they don't need a tab each. EasyControl
        # has its own preprocess/dataset lifecycle, so it keeps a dedicated tab.
        self.tabs.addTab(MethodsTab(tb_panel=self._tb_tab.panel), t("tab_experimental"))
        self.tabs.addTab(EasyControlTab(), t("tab_easycontrol"))

        # The Queue view is a global overlay reached via the top-bar toggle
        # (like TensorBoard), not a tab — the daemon queue spans every method.
        # Lives in the stack below.
        self._queue_tab = QueueTab()

        self.tab_stack = QStackedWidget()
        self.tab_stack.addWidget(self.tabs)
        self.tab_stack.addWidget(self._tb_tab)
        self.tab_stack.addWidget(self._queue_tab)
        main_lay.addWidget(self.tab_stack)
        self.setCentralWidget(central)

        self._update_tensorboard_btn_style(False)
        self._update_queue_btn_style(False)

    def closeEvent(self, event):
        # Drop the app-wide context-menu filter so a _reload_ui rebuild (which
        # closes this window and constructs a fresh one) doesn't leave a dead
        # window filtering events.
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        # Without this, closing the window leaves training subprocesses
        # (accelerate → train.py) orphaned and still holding VRAM.
        for i in range(self.tabs.count()):
            cleanup = getattr(self.tabs.widget(i), "cleanup_subprocess", None)
            if callable(cleanup):
                cleanup()
        # The shared TensorBoard + Queue views live in the stack, not a tab set.
        self._tb_tab.cleanup_subprocess()
        self._queue_tab.cleanup_subprocess()
        super().closeEvent(event)

    def _show_update_available(self, latest_tag: str) -> None:
        self.update_btn.setText(t("update_btn_available"))
        self.update_btn.setToolTip(t("update_btn_available_tooltip", v=latest_tag))
        self.update_btn.setStyleSheet(
            "QPushButton { background:#b45309; color:white; font-weight:bold; "
            "padding:4px 12px; border:1px solid #b45309; border-radius:3px; }"
            "QPushButton:hover { background:#d97706; }"
        )

    def _clear_overlay_toggle(self, btn: QPushButton, style_fn) -> None:
        """Silently un-check an overlay toggle (TensorBoard / Queue) and repaint
        it, without re-firing its toggled handler."""
        if btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)
            style_fn(False)

    def _toggle_tensorboard(self, on: bool):
        if on:
            # TensorBoard and Queue are mutually-exclusive overlays.
            self._clear_overlay_toggle(self.queue_btn, self._update_queue_btn_style)
            self.tab_stack.setCurrentWidget(self._tb_tab)
        else:
            self.tab_stack.setCurrentWidget(self.tabs)
        self._update_tensorboard_btn_style(on)

    def _update_tensorboard_btn_style(self, on: bool):
        self.tensorboard_btn.setStyleSheet(
            self._tensorboard_active_style if on else self._tensorboard_idle_style
        )

    def _toggle_queue_view(self, on: bool):
        if on:
            # Queue and TensorBoard are mutually-exclusive overlays.
            self._clear_overlay_toggle(
                self.tensorboard_btn, self._update_tensorboard_btn_style
            )
            self.tab_stack.setCurrentWidget(self._queue_tab)
        else:
            self.tab_stack.setCurrentWidget(self.tabs)
        self._update_queue_btn_style(on)

    def _update_queue_btn_style(self, on: bool):
        self.queue_btn.setStyleSheet(
            self._queue_active_style if on else self._queue_idle_style
        )

    def eventFilter(self, obj, event):  # noqa: N802 — Qt event handler name
        """Intercept every right-click in the app and show our menu instead of
        the target widget's default one (text widgets ship a copy/select-all
        menu we want to override). Returning True consumes the event."""
        if event.type() == QEvent.ContextMenu:
            self._show_context_menu(event.globalPos())
            return True
        return super().eventFilter(obj, event)

    def _show_context_menu(self, global_pos):
        """Walk up from the widget under the cursor; the first ancestor exposing
        a callable ``app_context_menu(target, global_pos)`` gets to supply the
        menu (e.g. the dataset image view → open-in-system-viewer). If none does
        — or it declines by returning None — fall back to the app default."""
        target = QApplication.widgetAt(global_pos)
        w = target
        while w is not None:
            provider = getattr(w, "app_context_menu", None)
            if callable(provider):
                menu = provider(target, global_pos)
                if menu is not None:
                    menu.exec(global_pos)
                    return
                break
            w = w.parentWidget()
        self._show_app_menu(global_pos)

    def _show_app_menu(self, global_pos):
        """The default right-click menu — currently just a link to the repo."""
        menu = QMenu(self)
        visit = menu.addAction(t("visit_github"))
        visit.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(GITHUB_REPO_URL))
        )
        menu.exec(global_pos)

    def _open_guidebook(self):
        path = _guidebook_path()
        if not path.exists():
            QMessageBox.warning(
                self, t("guidebook"), t("guidebook_missing", path=str(path))
            )
            return
        dlg = GuidebookDialog(path, self)
        dlg.show()

    def _open_settings(self):
        dlg = SettingsDialog(self)
        dlg.exec()
        if dlg.reload_requested:
            self._reload_ui()

    def _reload_ui(self):
        """Rebuild the main window in place to apply a language or theme change —
        every string and per-widget theme token is resolved at construction, so a
        fresh window is the cleanest way to re-render. The daemon owns running
        jobs, so only local UI
        state (unsaved edits, overlay subprocesses) resets; closeEvent reaps
        the old window's TensorBoard/Queue children as usual. New window is
        shown before the old closes so quitOnLastWindowClosed never fires."""
        global _WINDOW
        new = MainWindow()
        new.setGeometry(self.geometry())
        new.show()
        _WINDOW = new
        self.close()


def _ensure_source_image_dir() -> None:
    """Create the training source dir on launch so first-time users hit an
    empty folder rather than a confusing "no images found" error from the
    preprocess pipeline. Path comes from `source_image_dir` (now in
    configs/preprocess.toml; a legacy copy in base.toml still wins, mirroring
    load_path_overrides); falls back to `image_dataset/`.
    """
    src = "image_dataset"
    # preprocess.toml supplies the default; a legacy key in base.toml overrides
    # it (read second), matching load_path_overrides' precedence.
    for fname in ("preprocess.toml", "base.toml"):
        cfg_path = _REPO_ROOT / "configs" / fname
        try:
            if cfg_path.exists():
                raw = toml.loads(cfg_path.read_text(encoding="utf-8"))
                cfg_src = raw.get("source_image_dir")
                if isinstance(cfg_src, str) and cfg_src.strip():
                    src = cfg_src
        except (OSError, toml.TomlDecodeError):
            pass
    src_path = Path(src)
    if not src_path.is_absolute():
        src_path = _REPO_ROOT / src_path
    try:
        src_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"warn: could not create {src_path}: {e}", file=sys.stderr)


def _prefer_cleartype_font_engine() -> None:
    """Use Qt's GDI font engine on Windows so UI text gets ClearType.

    Qt 6 defaults to the DirectWrite font engine on Windows, which rasterizes
    small UI text with *grayscale* antialiasing — it reads soft/blurry next to
    native apps, and the effect is worse on lightly-hinted modern faces like the
    bundled Pretendard. The GDI engine uses ClearType subpixel rendering (what
    native Windows controls use), which snaps small text crisp. Selected via the
    ``windows:fontengine=gdi`` platform option, which must be set *before*
    ``QApplication`` is constructed. Skipped if the user already pinned
    ``QT_QPA_PLATFORM`` (e.g. a HiDPI user who prefers DirectWrite, or offscreen
    in tests) so we never clobber an explicit choice."""
    if sys.platform == "win32" and "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "windows:fontengine=gdi"


def main():
    load_language()
    _ensure_source_image_dir()
    _prefer_cleartype_font_engine()
    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    _dark(app)
    # Bring the local training daemon up at launch (idempotent — reuses one
    # already started by the CLI / a previous session, spawns one otherwise) so
    # the queue, the Train button, and re-attach are ready immediately. Best-
    # effort: a failure here never blocks the GUI from opening.
    gui_daemon.ensure_daemon_quietly()
    global _WINDOW
    _WINDOW = MainWindow()
    _WINDOW.show()
    sys.exit(app.exec())
