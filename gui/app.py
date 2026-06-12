"""Anima LoRA GUI — main window, dark theme, and entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import toml
from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from gui import daemon as gui_daemon
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


def _dark(app: QApplication):
    # Use a font that supports Korean glyphs on Windows
    font = QFont("Malgun Gothic", 9)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    p = QPalette()
    for role, color in [
        (QPalette.Window, QColor(30, 30, 30)),
        (QPalette.WindowText, QColor(220, 220, 220)),
        (QPalette.Base, QColor(25, 25, 25)),
        (QPalette.AlternateBase, QColor(35, 35, 35)),
        (QPalette.ToolTipBase, QColor(50, 50, 50)),
        (QPalette.ToolTipText, QColor(220, 220, 220)),
        (QPalette.Text, QColor(220, 220, 220)),
        (QPalette.Button, QColor(45, 45, 45)),
        (QPalette.ButtonText, QColor(220, 220, 220)),
        (QPalette.Highlight, QColor(60, 120, 200)),
        (QPalette.HighlightedText, QColor(255, 255, 255)),
        # Default Qt link blue (#0000ff-ish) is unreadable on dark bg.
        (QPalette.Link, QColor(0xFF, 0xB8, 0x6B)),
        (QPalette.LinkVisited, QColor(0xE6, 0x94, 0x4E)),
    ]:
        p.setColor(role, color)
    app.setPalette(p)
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold; border: 1px solid #444;
            border-radius: 4px; margin-top: 8px; padding-top: 16px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QPushButton { padding: 4px 12px; border: 1px solid #555; border-radius: 3px; }
        QPushButton:hover { background: #555; }
        QScrollArea { border: none; }
        QSplitter::handle { background: #444; }
        QLineEdit, QSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QListWidget {
            background: #2a2a2a; color: #dcdcdc; border: 1px solid #555; border-radius: 3px;
            padding: 2px 4px;
        }
        QComboBox QAbstractItemView {
            background: #2a2a2a; color: #dcdcdc; selection-background-color: #3c78c8;
        }
        QTabWidget::pane { border: 1px solid #444; }
        QTabBar::tab {
            background: #2a2a2a; color: #dcdcdc; border: 1px solid #444;
            padding: 6px 14px;
            font-size: 12.5px; font-weight: 500;
            border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
        }
        QTabBar::tab:selected { background: #1e1e1e; color: #ffffff; }
        QTabBar::tab:hover { background: #3a3a3a; }
        QToolTip { max-width: 400px; }
        QMenu {
            background: #2a2a2a; color: #dcdcdc; border: 1px solid #555;
        }
        QMenu::item { padding: 4px 20px; background: transparent; color: #dcdcdc; }
        QMenu::item:selected { background: #3c78c8; color: #ffffff; }
        QMenu::item:disabled { color: #777; }
        QMenu::separator { height: 1px; background: #444; margin: 4px 8px; }
    """)


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(t("window_title"))
        self.resize(1100, 750)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))

        central = QWidget()
        main_lay = QVBoxLayout(central)
        main_lay.setContentsMargins(0, 0, 0, 0)

        # Language selector bar
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
        lang_bar.addWidget(QLabel(t("language")))
        self.lang_combo = QComboBox()
        for code in available_languages():
            self.lang_combo.addItem(LANG_NAMES.get(code, code), code)
        self.lang_combo.setCurrentIndex(available_languages().index(current_language()))
        self.lang_combo.currentIndexChanged.connect(self._change_lang)
        self.lang_combo.setFixedWidth(100)
        lang_bar.addWidget(self.lang_combo)
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

    def _open_guidebook(self):
        path = _guidebook_path()
        if not path.exists():
            QMessageBox.warning(
                self, t("guidebook"), t("guidebook_missing", path=str(path))
            )
            return
        dlg = GuidebookDialog(path, self)
        dlg.show()

    def _change_lang(self, idx: int):
        lang = self.lang_combo.itemData(idx)
        save_language(lang)
        QMessageBox.information(
            self,
            "Language" if lang == "en" else "언어",
            "Please restart the application to apply the language change."
            if current_language() == "en"
            else "언어 변경을 적용하려면 앱을 다시 시작해주세요.",
        )


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


def main():
    load_language()
    _ensure_source_image_dir()
    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    _dark(app)
    # Bring the local training daemon up at launch (idempotent — reuses one
    # already started by the CLI / a previous session, spawns one otherwise) so
    # the queue, the Train button, and re-attach are ready immediately. Best-
    # effort: a failure here never blocks the GUI from opening.
    gui_daemon.ensure_daemon_quietly()
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
