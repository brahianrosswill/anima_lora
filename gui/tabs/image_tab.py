"""ImageViewerTab — dataset image browser with caption editor + history."""

from __future__ import annotations

import difflib
import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path

from PySide6.QtCore import (
    QElapsedTimer,
    QEvent,
    QProcess,
    QProcessEnvironment,
    QRect,
    Qt,
    QTimer,
    QUrl,
)
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import (
    DEFAULT_AUTOTAG_CONFIDENCE,
    ROOT,
    LazyTabMixin,
    ScaledImageLabel,
    _image_dirs,
    _imgs,
    get_setting,
)
from gui import daemon as gui_daemon
from gui._job_mixin import DaemonJobMixin
from gui.i18n import t
from gui.progress import TqdmProgressTracker, make_progress_bar
from gui.theme import tok

# Stdio protocol sentinels of the resident autotag worker (kept in sync with
# ``scripts/anima_tagger/autotag_server.py``). Hardcoded rather than imported
# because that module pulls in torch, which the GUI must stay free of for fast
# startup (see gui/CLAUDE.md).
_AUTOTAG_READY = "ANIMA_AUTOTAG_READY"
_AUTOTAG_RESULT_PREFIX = "ANIMA_AUTOTAG_RESULT\t"
_AUTOTAG_ERROR_PREFIX = "ANIMA_AUTOTAG_ERROR\t"

# Free the resident tagger (VRAM) after this many ms with no autotag request.
_AUTOTAG_IDLE_MS = 10 * 60 * 1000
# Poll cadence (ms) for "did some other GPU job start?" while resident.
_AUTOTAG_GPU_WATCH_MS = 700


# Mask overlay tint — translucent red on top of the *masked-out* region (the
# inverted mask: where the trainer ignores pixels). 55% opacity is strong
# enough to see the masked region clearly without burying source detail.
# We pre-multiply on the fly by filling an opaque red and then driving alpha
# from the mask + QPainter.setOpacity rather than baking alpha into the color.
_MASK_OVERLAY_COLOR_OPAQUE = QColor(255, 60, 60, 255)
_MASK_OVERLAY_OPACITY = 0.55

# Foreground tint for images marked for deletion (toggled with the Delete key,
# removed by the 삭제 button). Bright red so marks stand out in both views.
_DELETE_MARK_COLOR = QColor("#e74c3c")


def _resolve_mask_path(image_path: Path, current_dir: Path | None) -> Path | None:
    """Locate the merged mask PNG for ``image_path``.

    Mirrors the trainer's mask layout: ``post_image_dataset/masks/<rel>/<stem>_mask.png``
    where ``rel`` is the image's parent relative to ``current_dir``. Falls back
    to the legacy ``masks/merged/...`` tree before giving up.
    """
    if current_dir is None:
        return None
    try:
        rel = image_path.relative_to(current_dir)
    except ValueError:
        return None
    rel_parent = rel.parent
    name = f"{image_path.stem}_mask.png"
    for root in (ROOT / "post_image_dataset" / "masks", ROOT / "masks" / "merged"):
        candidate = root / rel_parent / name
        if candidate.is_file():
            return candidate
    return None


def _compose_mask_overlay(source: QPixmap, mask_path: Path) -> QPixmap:
    """Return ``source`` with a red translucent tint over the masked-out region.

    Convention from ``scripts/preprocess/merge_masks.py``: **white = "train here",
    black = ignored (text bubble / artifact)**. We invert so the tint lands
    on the *ignored* region — that's the half users want to see at a glance
    ("did the detector catch every bubble?").

    Implementation note: ``convertToFormat(Alpha8)`` does **not** repurpose a
    grayscale channel as alpha — Qt fills it with the source's actual alpha
    (which is opaque-255 for Grayscale8), giving a uniform tint. Use
    ``setAlphaChannel`` instead: when given a grayscale image, it copies the
    luminance into the alpha channel of an ARGB32 layer.

    Alignment: masks are generated at the **bucket** resolution
    (``post_image_dataset/resized/`` = scale-to-cover + center-crop of the
    original in ``image_dataset/``). A plain ``IgnoreAspectRatio`` rescale
    onto the source would (a) stretch non-uniformly when ARs differ and
    (b) ignore the cropped-out margins — both contribute visible drift on
    the original-image view. Invert the bucket transform: scale the mask
    uniformly to match the appropriate axis, then letterbox the other axis
    so masked features land where the trainer actually saw them.
    """
    mask_img = QImage(str(mask_path))
    if mask_img.isNull():
        return source
    gray = mask_img.convertToFormat(QImage.Format_Grayscale8)
    gray.invertPixels()  # bubble (was 0) → 255, train-here (was 255) → 0

    src_w, src_h = source.width(), source.height()
    mask_w, mask_h = gray.width(), gray.height()
    if (src_w, src_h) == (mask_w, mask_h):
        aligned = gray
    elif src_w * mask_h >= src_h * mask_w:
        # ar_src >= ar_mask: bucket cropped left/right of the original.
        # Match height, letterbox width.
        scaled_w = max(1, round(mask_w * src_h / mask_h))
        scaled = gray.scaled(
            scaled_w, src_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        aligned = QImage(src_w, src_h, QImage.Format_Grayscale8)
        aligned.fill(0)  # 0 = no tint on the cropped-out bars
        offset_x = max(0, (src_w - scaled_w) // 2)
        painter = QPainter(aligned)
        try:
            painter.drawImage(offset_x, 0, scaled)
        finally:
            painter.end()
    else:
        # ar_src < ar_mask: bucket cropped top/bottom of the original.
        # Match width, letterbox height.
        scaled_h = max(1, round(mask_h * src_w / mask_w))
        scaled = gray.scaled(
            src_w, scaled_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        aligned = QImage(src_w, src_h, QImage.Format_Grayscale8)
        aligned.fill(0)
        offset_y = max(0, (src_h - scaled_h) // 2)
        painter = QPainter(aligned)
        try:
            painter.drawImage(0, offset_y, scaled)
        finally:
            painter.end()

    layer = QImage(source.size(), QImage.Format_ARGB32)
    layer.fill(_MASK_OVERLAY_COLOR_OPAQUE)
    layer.setAlphaChannel(aligned)

    result = QPixmap(source)
    p = QPainter(result)
    try:
        p.setOpacity(_MASK_OVERLAY_OPACITY)
        p.drawImage(0, 0, layer)
    finally:
        p.end()
    return result


# Inline-highlight palette for the editor: a translucent green for inserted
# spans (visible on the dark theme without overpowering the text). We don't
# render deletions inline — the user already removed those characters, so we
# surface them via the (+X / −Y) summary in the caption header instead.
_ADD_BG = QColor(60, 130, 70, 120)


def _add_format() -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setBackground(_ADD_BG)
    return fmt


def _diff_spans(old: str, new: str) -> tuple[list[tuple[int, int]], int, int]:
    """Char-level diff between old and new.

    Returns (insert_spans_in_new, total_added_chars, total_removed_chars).
    insert_spans are (j1, j2) ranges in `new` that should be highlighted.
    """
    if old == new:
        return [], 0, 0
    sm = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
    spans: list[tuple[int, int]] = []
    add_total = 0
    rem_total = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            spans.append((j1, j2))
            add_total += j2 - j1
        elif tag == "replace":
            spans.append((j1, j2))
            add_total += j2 - j1
            rem_total += i2 - i1
        elif tag == "delete":
            rem_total += i2 - i1
    return spans, add_total, rem_total


def _history_path(caption_path: Path) -> Path:
    return caption_path.with_suffix(caption_path.suffix + ".history.jsonl")


def _read_history(caption_path: Path) -> list[dict]:
    """Return history entries (oldest first). Skips malformed lines."""
    hp = _history_path(caption_path)
    if not hp.exists():
        return []
    out: list[dict] = []
    for line in hp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict) and "ts" in entry and "text" in entry:
            out.append(entry)
    return out


def _append_history(caption_path: Path, prev_text: str) -> None:
    """Append the previous on-disk text as a history entry."""
    hp = _history_path(caption_path)
    entry = {"ts": datetime.now().isoformat(timespec="seconds"), "text": prev_text}
    with hp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# Border colors for inline tag boxes. Plain tags get a near-white border per
# the user's request — clearly distinct from the dark editor background and
# the light text. @artist boundary and "On the …" / "In the …" section
# headers keep their warm/cool tints so the trainer's split rules
# (anima_smart_shuffle in library/anima/training.py) stay visible.
_BOX_BORDER_PLAIN = QColor("#e0e0e0")
_BOX_BORDER_ARTIST = QColor("#c9a227")
_BOX_BORDER_SECTION = QColor("#5e8eb0")


def _tag_ranges(text: str):
    """Yield ``(start, end, tag_text)`` for each comma-separated, trimmed tag.

    Whitespace around each tag is excluded from the range so the painted box
    hugs the visible characters, not the surrounding spaces.
    """
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\n":
            i += 1
        start = i
        while i < n and text[i] != ",":
            i += 1
        end = i
        while end > start and text[end - 1] in " \t\n":
            end -= 1
        if end > start:
            yield (start, end, text[start:end])
        if i < n and text[i] == ",":
            i += 1


def _tag_border_color(tag: str) -> QColor:
    # Mirror library.anima.training._is_artist_tag: `@<non-space>` is an
    # artist handle (`@sincos`, `@no-artist` placeholder), while `@ @`
    # (booru `@_@` eye-shape, space-form) is a general-category tag and
    # must not steal the warm artist tint. Kept inline so this module
    # stays free of heavy library/* imports at GUI startup.
    if len(tag) >= 2 and tag[0] == "@" and not tag[1].isspace():
        return _BOX_BORDER_ARTIST
    if (
        tag.startswith("On the ")
        or tag.startswith("In the ")
        or ". On the " in tag
        or ". In the " in tag
    ):
        return _BOX_BORDER_SECTION
    return _BOX_BORDER_PLAIN


class BoxedCaptionEdit(QTextEdit):
    """QTextEdit that paints thin border boxes inline around each
    comma-separated tag.

    Uses ``viewportEvent`` rather than ``QTextCharFormat`` because Qt's
    text framework can set per-character backgrounds and foregrounds but
    not borders. We let Qt render the text normally, then overlay
    rectangles on the viewport by walking ``cursorRect()`` across each
    tag's character range. Boxes follow scroll, wrap, and live edits
    automatically because ``cursorRect()`` is always queried in current
    viewport coordinates.

    The font is configured with extra letter spacing and the document with
    a roomier line height so tag boxes have visible breathing room both
    horizontally (the comma+space between tags is wider) and vertically
    (wrapped lines don't crowd their box borders together).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        font = self.font()
        font.setPixelSize(14)
        # 115% letter spacing widens the natural gap between adjacent boxes
        # (the comma+space stretches with the rest of the text), which is
        # cheaper than fiddling with per-box geometry to manufacture gaps.
        font.setLetterSpacing(QFont.PercentageSpacing, 115)
        self.setFont(font)
        self._apply_block_format()

    def setPlainText(self, text: str) -> None:  # noqa: N802 — Qt API
        # setPlainText replaces the document, so the line-height format we
        # applied earlier gets reset. Reapply after every full replacement.
        super().setPlainText(text)
        self._apply_block_format()

    def _apply_block_format(self) -> None:
        cursor = QTextCursor(self.document())
        cursor.select(QTextCursor.Document)
        fmt = QTextBlockFormat()
        # ProportionalHeight = 1 (Qt's QTextBlockFormat.LineHeightTypes).
        # 140% gives clear vertical separation between wrapped lines without
        # making the editor feel stretched.
        fmt.setLineHeight(
            140, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value
        )
        cursor.mergeBlockFormat(fmt)

    def viewportEvent(self, event) -> bool:  # noqa: N802 — Qt API
        result = super().viewportEvent(event)
        if event.type() == QEvent.Paint:
            self._paint_boxes()
        return result

    def _paint_boxes(self) -> None:
        text = self.toPlainText()
        if not text.strip():
            return
        painter = QPainter(self.viewport())
        try:
            painter.setBrush(Qt.NoBrush)
            for start, end, tag in _tag_ranges(text):
                pen = QPen(_tag_border_color(tag))
                pen.setWidth(1)
                painter.setPen(pen)
                for r in self._tag_rects(start, end):
                    if r.width() > 0:
                        painter.drawRoundedRect(r, 2, 2)
        finally:
            painter.end()

    def _tag_rects(self, start: int, end: int) -> list[QRect]:
        """Per-line bounding rectangles for char range ``[start, end)``.

        Walks character-by-character so soft wraps (visual line breaks
        without an actual ``\\n``) get their own rectangle. For a typical
        caption (~500 chars) this is a few hundred ``cursorRect`` calls
        per paint — well under the budget for live editing.
        """
        if end <= start:
            return []
        cur = QTextCursor(self.document())
        cur.setPosition(start)
        cr = self.cursorRect(cur)
        line_left = cr.left()
        line_right = cr.left()
        line_top = cr.top()
        line_height = cr.height()
        rects: list[QRect] = []

        # Box pads slightly OUTWARD from the text so the glyphs sit inside
        # with a 1px halo. Negative pad → outward extension. Keeping the
        # outward extension small (1px instead of 2px) leaves more of the
        # comma+space between tags untouched, so adjacent boxes have a
        # visibly wider gap. Going to 0 would put glyph edges right on the
        # border line, which reads as "text escaping the box."
        pad_x = -1
        pad_y = -1

        def _emit() -> None:
            w = line_right - line_left - 2 * pad_x
            h = line_height - 2 * pad_y
            if w > 0 and h > 0:
                rects.append(QRect(line_left + pad_x, line_top + pad_y, w, h))

        for pos in range(start + 1, end + 1):
            cur.setPosition(pos)
            cr = self.cursorRect(cur)
            if cr.top() != line_top:
                _emit()
                line_left = cr.left()
                line_right = cr.left()
                line_top = cr.top()
                line_height = cr.height()
            else:
                line_right = cr.left()
        _emit()
        return rects


def _unified_diff_html(old: str, new: str) -> str:
    """Tiny unified diff with red-/green+ coloring; empty string means no changes."""
    if old == new:
        return ""
    diff = difflib.unified_diff(
        old.splitlines(),
        new.splitlines(),
        lineterm="",
        n=3,
    )
    rows: list[str] = []
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue  # filenames are noise here
        if line.startswith("@@"):
            rows.append(f'<span style="color:{tok("link")};">{escape(line)}</span>')
        elif line.startswith("+"):
            rows.append(f'<span style="color:#9ad17a;">{escape(line)}</span>')
        elif line.startswith("-"):
            rows.append(f'<span style="color:#e07a7a;">{escape(line)}</span>')
        else:
            rows.append(f'<span style="color:{tok("text_dim")};">{escape(line)}</span>')
    if not rows:
        return ""
    return (
        '<pre style="font-family:monospace;font-size:11px;margin:0;">'
        + "\n".join(rows)
        + "</pre>"
    )


class CaptionVersionsDialog(QDialog):
    """Browse prior versions of a caption and restore one in-place."""

    def __init__(self, caption_path: Path, current_disk_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("caption_versions_title", name=caption_path.stem))
        self.resize(820, 520)
        self._caption_path = caption_path
        self._current = current_disk_text
        self._restored: str | None = None  # set on Restore

        history = _read_history(caption_path)
        # Newest first — that's what users want to see at the top.
        self._history = list(reversed(history))

        lay = QVBoxLayout(self)

        sp = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        if not self._history:
            self.list.addItem(t("caption_versions_empty"))
            self.list.setEnabled(False)
        else:
            for entry in self._history:
                self.list.addItem(entry["ts"])
        self.list.currentRowChanged.connect(self._show_diff)
        sp.addWidget(self.list)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.diff = QTextBrowser()
        self.diff.setStyleSheet(
            f"QTextBrowser {{ background:{tok('base')}; color:{tok('text')}; "
            f"border:1px solid {tok('border_dim')}; padding:6px; }}"
        )
        rl.addWidget(self.diff, 1)
        sp.addWidget(right)
        sp.setSizes([220, 600])
        lay.addWidget(sp, 1)

        btns = QDialogButtonBox()
        self.restore_btn = btns.addButton(
            t("caption_versions_restore"), QDialogButtonBox.AcceptRole
        )
        self.restore_btn.setEnabled(False)
        self.restore_btn.clicked.connect(self._restore)
        close_btn = btns.addButton(
            t("caption_versions_close"), QDialogButtonBox.RejectRole
        )
        close_btn.clicked.connect(self.reject)
        lay.addWidget(btns)

        if self._history:
            self.list.setCurrentRow(0)

    def _show_diff(self, row: int) -> None:
        if not (0 <= row < len(self._history)):
            self.restore_btn.setEnabled(False)
            self.diff.setHtml("")
            return
        prev = self._history[row]["text"]
        html = _unified_diff_html(prev, self._current)
        if not html:
            self.diff.setHtml(
                f'<i style="color:{tok("text_dim")};">{t("caption_diff_clean")}</i>'
            )
        else:
            self.diff.setHtml(html)
        self.restore_btn.setEnabled(True)

    def _restore(self) -> None:
        row = self.list.currentRow()
        if not (0 <= row < len(self._history)):
            return
        self._restored = self._history[row]["text"]
        self.accept()

    def restored_text(self) -> str | None:
        return self._restored


class ImageViewerTab(DaemonJobMixin, LazyTabMixin, QWidget):
    def __init__(self):
        super().__init__()
        # Daemon job observer (curate-group runs as a command job) so the
        # grouping progress bar lives in this tab, not only the Queue tab.
        self._init_job_observer()
        self._all_images: list[Path] = []  # unfiltered, alphabetical (from _imgs)
        self._images: list[Path] = []  # currently displayed (filter + sort applied)
        self._dirs = _image_dirs()
        self._current_dir: Path | None = (
            None  # base of the loaded directory (for relative labels)
        )
        self._current_caption_path: Path | None = None
        self._disk_text: str = ""  # last value seen on disk (for diff baseline)
        self._suspend_dirty = False  # while we set text programmatically
        # Resident autotag worker (a torch subprocess holding the tagger model
        # so consecutive Autotag clicks skip the reload). GUI-owned via QProcess
        # — spawned on first use, kept alive ("loaded, waiting"), torn down
        # before any other GPU work (grouping/training/preprocess) frees the
        # card. See _run_autotag / _kill_tagger_worker.
        self._tagger_proc: QProcess | None = None
        self._tagger_ready = False
        self._tagger_buf = ""  # partial-line buffer for the worker's stdout
        self._autotag_inflight_image: Path | None = None  # image awaiting a result
        self._autotag_idle = QElapsedTimer()
        # Polls "did another GPU job start?" only while the worker is resident.
        self._gpu_watch_timer = QTimer(self)
        self._gpu_watch_timer.setInterval(_AUTOTAG_GPU_WATCH_MS)
        self._gpu_watch_timer.timeout.connect(self._autotag_gpu_watch_tick)
        # Make sure the resident worker (and its VRAM) dies with the app.
        _app = QApplication.instance()
        if _app is not None:
            _app.aboutToQuit.connect(self._kill_tagger_worker)
        self._search_text: str = ""
        self._sort_desc: bool = False
        # Similarity-group curation (make curate-group). _groups is the manifest's
        # group list; the tree view folds images under green per-group nodes (the
        # old dropdown filter was replaced by this always-on tree grouping).
        # Grouping is stem-keyed so it works whether this tab views image_dataset/
        # or post_image_dataset/resized/ (stems are unique + shared).
        self._groups: list[dict] = []
        # Images marked for deletion (Delete key toggles the current one red; the
        # 삭제 button moves the whole set to the OS trash). Keyed by full path so a
        # mark survives filter/sort/view rebuilds; cleared when the dir changes.
        self._marked: set[Path] = set()
        # Source pixmap + resolved mask for the currently shown image.
        # _overlay_pm is lazily composed on first toggle and cached so flipping
        # the checkbox doesn't re-run the QPainter pipeline.
        self._source_pm: QPixmap | None = None
        self._mask_path: Path | None = None
        self._overlay_pm: QPixmap | None = None
        lay = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel(t("directory")))
        self.dc = QComboBox()
        self.dc.addItems(self._dirs)
        self.dc.currentTextChanged.connect(self._load_dir)
        top.addWidget(self.dc, 1)
        self.reload_btn = QPushButton("↻")
        self.reload_btn.setFixedWidth(28)
        self.reload_btn.setToolTip(t("dataset_reload_tooltip"))
        self.reload_btn.clicked.connect(self._reload_current_dir)
        top.addWidget(self.reload_btn)
        self.open_dir_btn = QPushButton(t("dataset_open_dir"))
        self.open_dir_btn.setToolTip(t("dataset_open_dir_tooltip"))
        self.open_dir_btn.clicked.connect(self._open_current_dir)
        top.addWidget(self.open_dir_btn)
        # Group button, accented blue like the preprocess "run" buttons. Submits
        # `make curate-group`; results surface as green folds in the tree view.
        self.group_btn = QPushButton(t("dataset_group_rebuild"))
        self.group_btn.setToolTip(t("dataset_group_rebuild_tooltip"))
        self.group_btn.setStyleSheet(
            "QPushButton{background:#2980b9;color:white;font-weight:bold;"
            "padding:4px 16px;}"
        )
        self.group_btn.clicked.connect(self._rebuild_groups)
        top.addWidget(self.group_btn)
        self.add_dir_btn = QPushButton(t("dataset_add_dir"))
        self.add_dir_btn.setToolTip(t("dataset_add_dir_tooltip"))
        self.add_dir_btn.clicked.connect(self._add_dir)
        top.addWidget(self.add_dir_btn)
        self.cnt = QLabel()
        top.addWidget(self.cnt)
        lay.addLayout(top)

        # Grouping progress bar (curate-group). Hidden until a run starts; the
        # tracker drives it from the job's tqdm stdout, then hides it on finish.
        self.group_progress = make_progress_bar()
        self._progress_tracker = TqdmProgressTracker(self.group_progress)
        lay.addWidget(self.group_progress)

        sp = QSplitter(Qt.Horizontal)

        # Left panel: search + sort row, then the tree of images. The tree folds
        # images under their folder + green per-similarity-group nodes; selecting
        # a leaf routes through _show(index) via the _images array.
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)
        search_row = QHBoxLayout()
        search_row.setContentsMargins(0, 0, 0, 0)
        self.search = QLineEdit()
        self.search.setPlaceholderText(t("dataset_search_placeholder"))
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search, 1)
        self.sort_btn = QPushButton("↑")
        self.sort_btn.setFixedWidth(28)
        self.sort_btn.setToolTip(t("dataset_sort_asc_tooltip"))
        self.sort_btn.clicked.connect(self._toggle_sort)
        search_row.addWidget(self.sort_btn)
        ll.addLayout(search_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self._on_tree_item_changed)
        # Map item → image index so tree selections route through _show(index).
        self._tree_item_to_index: dict[QTreeWidgetItem, int] = {}
        ll.addWidget(self.tree, 1)
        sp.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        # Mask-overlay toggle. Disabled when the current image has no merged
        # mask under post_image_dataset/masks/; the checked state is preserved
        # across image navigation so it acts as a sticky "show overlay when
        # available" preference.
        img_head = QHBoxLayout()
        img_head.setContentsMargins(0, 0, 0, 0)
        self.overlay_cb = QCheckBox(t("dataset_mask_overlay"))
        self.overlay_cb.setEnabled(False)
        self.overlay_cb.toggled.connect(self._on_overlay_toggled)
        img_head.addWidget(self.overlay_cb)
        # Delete button: removes the images marked red via the Delete key. Red
        # accent to match the marks; disabled until at least one is marked.
        self.delete_btn = QPushButton(t("dataset_delete"))
        self.delete_btn.setToolTip(t("dataset_delete_tooltip"))
        self.delete_btn.setStyleSheet(
            "QPushButton{background:#c0392b;color:white;font-weight:bold;"
            "padding:4px 16px;}QPushButton:disabled{background:#5a3a37;color:#aaa;}"
        )
        self.delete_btn.clicked.connect(self._delete_marked)
        img_head.addWidget(self.delete_btn)
        # Cancel button: clears all deletion marks (same as pressing Esc).
        self.cancel_mark_btn = QPushButton(t("dataset_delete_clear"))
        self.cancel_mark_btn.setToolTip(t("dataset_delete_clear_tooltip"))
        self.cancel_mark_btn.clicked.connect(self._clear_marks)
        img_head.addWidget(self.cancel_mark_btn)
        img_head.addStretch()
        rl.addLayout(img_head)

        self.img = ScaledImageLabel()
        self.img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img.setMinimumSize(400, 400)
        rl.addWidget(self.img, 1)

        # Caption header: label + buttons
        cap_head = QHBoxLayout()
        self.cap_label = QLabel(t("caption"))
        cap_head.addWidget(self.cap_label)
        # Resident-tagger status ("loading…" / "loaded, waiting"). Hidden until
        # the worker is spawned; updated from the worker's stdout sentinels.
        self.autotag_status = QLabel()
        self.autotag_status.setStyleSheet(
            f"QLabel{{color:{tok('link')};font-size:11px;}}"
        )
        self.autotag_status.setVisible(False)
        cap_head.addWidget(self.autotag_status)
        cap_head.addStretch()
        self.save_btn = QPushButton(t("caption_save"))
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        self.revert_btn = QPushButton(t("caption_revert"))
        self.revert_btn.setEnabled(False)
        self.revert_btn.clicked.connect(self._revert)
        # Autotag: run the Anima Tagger on the current image and append its
        # predicted tags into the editor (review, then Save writes the .txt).
        # Accented blue like the other "run a model" actions.
        self.autotag_btn = QPushButton(t("caption_autotag"))
        self.autotag_btn.setToolTip(t("caption_autotag_tooltip"))
        self.autotag_btn.setStyleSheet(
            "QPushButton{background:#2980b9;color:white;font-weight:bold;}"
            "QPushButton:disabled{background:#2a4763;color:#aaa;}"
        )
        self.autotag_btn.clicked.connect(self._run_autotag)
        self.versions_btn = QPushButton(t("caption_versions"))
        self.versions_btn.clicked.connect(self._open_versions)
        cap_head.addWidget(self.save_btn)
        cap_head.addWidget(self.revert_btn)
        cap_head.addWidget(self.autotag_btn)
        cap_head.addWidget(self.versions_btn)
        rl.addLayout(cap_head)

        # Caption editor with inline tag-box overlay. Each comma-separated
        # tag is outlined by a thin rectangle painted on the viewport;
        # @artist and section headers use accent colors so the trainer's
        # split rules (anima_smart_shuffle in library/anima/training.py)
        # stay visible without a separate preview pane.
        self.cap = BoxedCaptionEdit()
        self.cap.setMaximumHeight(180)
        self.cap.textChanged.connect(self._on_text_changed)
        rl.addWidget(self.cap)

        # One-line grammar reminder, mirrors anima_smart_shuffle's split rules.
        self.guide = QLabel(t("caption_guideline_html"))
        self.guide.setWordWrap(True)
        self.guide.setTextFormat(Qt.RichText)
        self.guide.setStyleSheet(
            f"QLabel {{ color:{tok('text_dim')}; font-size:11px; padding:2px 4px; }}"
        )
        rl.addWidget(self.guide)

        sp.addWidget(right)
        sp.setSizes([340, 700])
        lay.addWidget(sp)

        QShortcut(QKeySequence("Right"), self, lambda: self._nav(1))
        QShortcut(QKeySequence("Left"), self, lambda: self._nav(-1))
        QShortcut(QKeySequence.Save, self, self._save)
        # Delete toggles the deletion mark on the current image; Esc un-marks it.
        # Both act per-current-image and are scoped to the tree (WidgetShortcut)
        # so they don't hijack the caption editor on focus.
        _del = QShortcut(QKeySequence.Delete, self.tree, self._toggle_mark_current)
        _del.setContext(Qt.WidgetShortcut)
        _esc = QShortcut(QKeySequence(Qt.Key_Escape), self.tree, self._unmark_current)
        _esc.setContext(Qt.WidgetShortcut)
        self._refresh_delete_button()

    def _lazy_init(self) -> None:
        # Walking the image dir + building the tree is deferred to first show.
        if self._dirs:
            self._load_dir(self.dc.currentText())

    # ── data loading ──────────────────────────────────────────

    def _open_current_dir(self):
        """Open the currently loaded dataset directory in the OS file manager."""
        if self._current_dir is None or not self._current_dir.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_dir)))

    # ── similarity groups (curate-group) ──────────────────────

    def _groups_manifest_path(self) -> Path:
        return ROOT / "post_image_dataset" / "groups" / "groups.json"

    def _load_groups(self) -> None:
        """Read groups.json (if present) into ``self._groups``.

        Pure JSON — keeps the GUI torch-free. The tree view folds images under
        green per-group nodes built from this list (ordered as written, largest
        first). A missing/unreadable manifest just leaves the plain folder tree.
        """
        self._groups = []
        path = self._groups_manifest_path()
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                groups = data.get("groups", [])
                if isinstance(groups, list):
                    self._groups = groups
            except (json.JSONDecodeError, OSError):
                self._groups = []

    def _rebuild_groups(self) -> None:
        """Submit `make curate-group` and observe it in-tab (progress bar here)."""
        if self._job_id:  # a grouping run is already attached
            QMessageBox.information(
                self, "", t("dataset_group_queued", job_id=self._job_id)
            )
            return
        # Grouping is GPU work (PE-Spatial) — free the resident tagger first so
        # the two don't fight over VRAM.
        self._kill_tagger_worker()
        # Busy UI + indeterminate bar before the submit so a cold-start daemon
        # spin-up still feels responsive.
        self.group_btn.setEnabled(False)
        self._progress_tracker.reset()
        self._progress_tracker.mark_starting(t("dataset_group_rebuild"))
        job_id = self._submit_job(
            lambda: gui_daemon.submit_command(
                label="curate-group", argv=["tasks.py", "curate-group"], start=True
            ),
            on_fail=self._restore_group_idle_ui,
        )
        if not job_id:
            return
        self._watch_job(job_id, replay_log=False)

    def _emit_log_line(self, line: str) -> None:
        """No log widget on this tab — non-progress stdout lines are dropped.

        The progress bar (tqdm) and the finish banner carry the user-facing
        signal; a full log belongs to the Queue tab.
        """

    def _on_job_finished(self, state: str | None) -> None:
        self._job_timer.stop()
        self._drain_job_stdout()
        self._stdout_buf = ""
        job_id = self._job_id
        self._job_id = None
        self._stdout_tailer.reset()
        self._progress_tracker.reset()
        self._restore_group_idle_ui()
        if gui_daemon.is_success(state):
            # Reload the manifest + re-render both views so new groups show now.
            prev = (
                self._current_caption_path.stem
                if self._current_caption_path is not None
                else None
            )
            self._load_groups()
            self._apply_filter_and_sort(prev_stem=prev)
        else:
            QMessageBox.warning(
                self, t("error"), gui_daemon.format_finish_banner(job_id, state)
            )

    def _restore_group_idle_ui(self) -> None:
        self.group_btn.setEnabled(True)

    # ── autotag (resident tagger worker) ──────────────────────

    def _run_autotag(self) -> None:
        """Tag the current image with the resident Anima Tagger.

        First click spawns a torch subprocess that loads the model once; it then
        stays alive so later clicks just stream an image path to it. The
        predicted tags are appended into the editor when the result comes back —
        the user reviews, then Save writes the ``.txt`` (creating it if absent).
        The worker is freed before any other GPU work (see _kill_tagger_worker).
        """
        idx = self._current_index()
        if not 0 <= idx < len(self._images):
            return
        # Don't grab the card while a daemon job (train/preprocess/group) holds
        # it — those take priority; tagging can wait until it's idle.
        if gui_daemon.active_job_id():
            QMessageBox.information(self, "", t("caption_autotag_busy"))
            return
        if self._autotag_inflight_image is not None:
            return  # a request is already in flight; ignore the double-click
        image_path = self._images[idx]
        self._autotag_inflight_image = image_path
        self._autotag_idle.restart()
        self.autotag_btn.setEnabled(False)
        if self._tagger_proc is None:
            self._spawn_tagger_worker()
            self._set_autotag_status(t("caption_autotag_loading"))
        elif self._tagger_ready:
            self._set_autotag_status(t("caption_autotag_running"))
            self._send_autotag_request(image_path)
        # else: worker still loading — _on_tagger_stdout sends it on READY.

    def _spawn_tagger_worker(self) -> None:
        """Launch the resident worker subprocess (torch lives here, not the GUI)."""
        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(["-m", "scripts.anima_tagger.autotag_server"])
        proc.setWorkingDirectory(str(ROOT))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # stream sentinel lines live
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_tagger_stdout)
        proc.finished.connect(self._on_tagger_finished)
        proc.errorOccurred.connect(lambda _e: self._on_tagger_finished(-1, None))
        self._tagger_proc = proc
        self._tagger_ready = False
        self._tagger_buf = ""
        self.autotag_status.setVisible(True)
        self._gpu_watch_timer.start()
        proc.start()

    def _send_autotag_request(self, image_path: Path) -> None:
        if self._tagger_proc is None:
            return
        # Prefix the per-request confidence floor (Settings → Autotag
        # confidence); read fresh each request so a settings change applies
        # without respawning the resident worker.
        try:
            conf = float(get_setting("autotag_confidence", DEFAULT_AUTOTAG_CONFIDENCE))
        except (TypeError, ValueError):
            conf = DEFAULT_AUTOTAG_CONFIDENCE
        conf = max(0.0, min(1.0, conf))
        self._tagger_proc.write(f"{conf}\t{image_path}\n".encode("utf-8"))

    def _on_tagger_stdout(self) -> None:
        if self._tagger_proc is None:
            return
        self._tagger_buf += bytes(self._tagger_proc.readAllStandardOutput()).decode(
            "utf-8", "replace"
        )
        *lines, self._tagger_buf = self._tagger_buf.split("\n")
        for line in lines:
            line = line.rstrip("\r")
            if not line:
                continue
            if line == _AUTOTAG_READY:
                self._tagger_ready = True
                self._set_autotag_status(t("caption_autotag_ready"))
                # Send the click that spawned the worker, now that it's loaded.
                if self._autotag_inflight_image is not None:
                    self._set_autotag_status(t("caption_autotag_running"))
                    self._send_autotag_request(self._autotag_inflight_image)
            elif line.startswith(_AUTOTAG_RESULT_PREFIX):
                self._apply_autotag_result(line[len(_AUTOTAG_RESULT_PREFIX) :])
            elif line.startswith(_AUTOTAG_ERROR_PREFIX):
                self._finish_autotag_request()
                QMessageBox.warning(
                    self,
                    t("error"),
                    t("caption_autotag_error", err=line[len(_AUTOTAG_ERROR_PREFIX) :]),
                )

    def _apply_autotag_result(self, caption: str) -> None:
        """Append the predicted caption into the editor (dirty → Save lights up)."""
        image = self._autotag_inflight_image
        self._finish_autotag_request()
        caption = caption.strip()
        if not caption:
            QMessageBox.information(self, "", t("caption_autotag_empty"))
            return
        # The user may have navigated away while the worker ran — only apply the
        # result if it still belongs to the caption currently on screen.
        if image is None or self._current_caption_path != image.with_suffix(".txt"):
            return
        existing = self.cap.toPlainText().strip()
        if existing:
            combined = existing.rstrip().rstrip(",").rstrip() + ", " + caption
        else:
            combined = caption
        # Set programmatically, then refresh manually so the diff highlight +
        # Save/Revert dirty state pick up the appended span (the suspend-dirty
        # guard otherwise swallows the textChanged signal).
        self._set_caption_text(combined)
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _finish_autotag_request(self) -> None:
        """Clear the in-flight state and re-arm the button after a reply."""
        self._autotag_inflight_image = None
        self.autotag_btn.setEnabled(True)
        self._autotag_idle.restart()
        if self._tagger_ready:
            self._set_autotag_status(t("caption_autotag_ready"))

    def _set_autotag_status(self, text: str) -> None:
        self.autotag_status.setText(text)
        self.autotag_status.setVisible(bool(text))

    def _autotag_gpu_watch_tick(self) -> None:
        """While the worker is resident: free it on any other GPU job or idle."""
        if self._tagger_proc is None:
            self._gpu_watch_timer.stop()
            return
        if gui_daemon.active_job_id():  # train/preprocess/group grabbed the card
            self._kill_tagger_worker()
            return
        if (
            self._autotag_inflight_image is None
            and self._autotag_idle.isValid()
            and self._autotag_idle.hasExpired(_AUTOTAG_IDLE_MS)
        ):
            self._kill_tagger_worker()

    def _kill_tagger_worker(self) -> None:
        """Tear down the resident worker and free its VRAM. Idempotent."""
        self._gpu_watch_timer.stop()
        proc = self._tagger_proc
        self._tagger_proc = None
        self._tagger_ready = False
        self._tagger_buf = ""
        self._autotag_inflight_image = None
        self.autotag_btn.setEnabled(True)
        self._set_autotag_status("")
        if proc is None:
            return
        try:
            proc.readyReadStandardOutput.disconnect()
            proc.finished.disconnect()
            proc.errorOccurred.disconnect()
        except (RuntimeError, TypeError):
            pass
        if proc.state() != QProcess.NotRunning:
            proc.closeWriteChannel()  # EOF on stdin → worker exits cleanly
            proc.kill()
            proc.waitForFinished(2000)
        proc.deleteLater()

    def _on_tagger_finished(self, _code, _status) -> None:
        """Worker exited (crash, kill, or EOF) — reset to the no-worker state."""
        was_inflight = self._autotag_inflight_image is not None
        self._gpu_watch_timer.stop()
        self._tagger_proc = None
        self._tagger_ready = False
        self._tagger_buf = ""
        self._autotag_inflight_image = None
        self.autotag_btn.setEnabled(True)
        self._set_autotag_status("")
        if was_inflight:
            QMessageBox.warning(
                self, t("error"), t("caption_autotag_error", err="exit")
            )

    def _load_dir(self, name: str, *, preserve_selection: bool = False):
        if not self._confirm_discard_if_dirty():
            # Roll the combo back without re-firing _load_dir.
            return
        d = self._dirs.get(name)
        if not d:
            return
        prev_stem: str | None = None
        if preserve_selection and self._current_caption_path is not None:
            prev_stem = self._current_caption_path.stem
        if d != self._current_dir:
            # Deletion marks are path-scoped to one dir; drop them on a switch.
            self._marked.clear()
            self._refresh_delete_button()
        self._current_dir = d
        self._load_groups()  # reload the group manifest for the tree folds
        self._all_images = _imgs(d)
        had_match = self._apply_filter_and_sort(prev_stem=prev_stem)
        if not self._images:
            self._current_caption_path = None
            self._set_caption_text("")
            self._disk_text = ""
            self._refresh_buttons()
            self._refresh_inline_diff()
        elif not had_match:
            # Fresh dir, no prior selection to restore — pick the first image.
            self._select_tree_index(0)

    def _display_label(self, p: Path) -> str:
        """``stem`` for top-level images, ``parent/stem`` for nested ones.

        Lets users tell apart shards organized by character/series subfolder
        in ``image_dataset/`` (the trainer enforces unique stems across the
        tree, so the stem itself is still a valid unique key — the prefix is
        purely a display affordance).
        """
        if self._current_dir is None:
            return p.stem
        try:
            rel = p.relative_to(self._current_dir)
        except ValueError:
            return p.stem
        if rel.parent == Path("."):
            return p.stem
        return f"{rel.parent.as_posix()}/{p.stem}"

    def _apply_filter_and_sort(self, *, prev_stem: str | None = None) -> bool:
        """Rebuild the visible tree from ``_all_images`` using the current
        search text and sort direction.

        Returns True if a row matching ``prev_stem`` was selected, False
        otherwise. Block-signals while rebuilding so search keystrokes don't
        trigger ``_on_tree_item_changed`` (which would prompt to save unsaved
        caption edits on every keystroke).
        """
        q = self._search_text.strip().lower()
        if q:
            visible = [
                p for p in self._all_images if q in self._display_label(p).lower()
            ]
        else:
            visible = list(self._all_images)
        if self._sort_desc:
            visible.reverse()
        self._images = visible

        # Try to keep the current selection visible after refilter/resort.
        # Falls back to ``prev_stem`` when called from _load_dir.
        target_stem: str | None = prev_stem
        if target_stem is None and self._current_caption_path is not None:
            target_stem = self._current_caption_path.stem

        target_row = -1
        for i, p in enumerate(visible):
            if p.stem == target_stem:
                target_row = i
                break

        self.tree.blockSignals(True)
        try:
            self._rebuild_tree(visible)
            if target_row >= 0:
                self._select_tree_index(target_row)
            else:
                self.tree.setCurrentItem(None)
        finally:
            self.tree.blockSignals(False)

        # Re-apply deletion marks to the freshly rebuilt items.
        self._refresh_mark_styles()

        total = len(self._all_images)
        shown = len(visible)
        if shown != total:  # narrowed by search
            self.cnt.setText(t("n_images_filtered", shown=shown, total=total))
        else:
            self.cnt.setText(t("n_images", n=total))
        return target_row >= 0

    def _rebuild_tree(self, visible: list[Path]) -> None:
        """Rebuild the tree widget from ``visible``.

        The folder structure under ``self._current_dir`` is primary. Within a
        folder, images that belong to a similarity group (from
        ``make curate-group``) nest one level deeper under a green per-group
        node; ungrouped images sit directly in the folder. Group nodes are
        per-folder, so a group spanning folders shows up once under each. Leaves
        are image stems; everything auto-expands so the hierarchy is visible
        without an extra click."""
        self.tree.clear()
        self._tree_item_to_index.clear()
        if not visible:
            return
        # Map each member stem → group index so grouped images get routed under
        # a green sub-node within their folder; the rest stay flat in the folder.
        stem_to_group: dict[str, int] = {}
        for gi, g in enumerate(self._groups):
            for m in g.get("members", []):
                stem_to_group[Path(m).stem] = gi

        # Cache folder QTreeWidgetItems by their relative parent path so
        # sibling images in the same folder share one parent node. Group
        # sub-nodes are keyed by (folder, group index) so each folder gets its
        # own green node per group.
        folder_items: dict[Path, QTreeWidgetItem] = {}
        group_nodes: dict[tuple[Path, int], QTreeWidgetItem] = {}
        group_counts: dict[tuple[Path, int], int] = {}
        for idx, p in enumerate(visible):
            rel: Path
            if self._current_dir is None:
                rel = Path(p.name)
            else:
                try:
                    rel = p.relative_to(self._current_dir)
                except ValueError:
                    rel = Path(p.name)
            folder = self._ensure_tree_folder(rel.parent, folder_items)
            gi = stem_to_group.get(p.stem)
            if gi is not None:
                key = (rel.parent, gi)
                parent = self._ensure_group_node(folder, key, group_nodes)
                group_counts[key] = group_counts.get(key, 0) + 1
            else:
                parent = folder
            leaf = QTreeWidgetItem(parent, [p.stem])
            self._tree_item_to_index[leaf] = idx
        # Label group nodes once their per-folder visible member count is known.
        for key, node in group_nodes.items():
            node.setText(
                0, t("dataset_group_label", n=key[1] + 1, size=group_counts[key])
            )
        self._float_groups_to_top(folder_items, group_nodes)
        self.tree.expandAll()

    def _float_groups_to_top(
        self,
        folder_items: dict[Path, QTreeWidgetItem],
        group_nodes: dict[tuple[Path, int], QTreeWidgetItem],
    ) -> None:
        """Reorder each folder's children so green group nodes sit above the
        ungrouped files at the *same* level (groups never cross into another
        folder — they stay children of their own folder, so they can't rise
        above a higher tree level). Within the group block and within the rest,
        the original filename order is preserved.
        """
        group_set = set(group_nodes.values())
        # Every parent that can hold a mix: each folder node + the invisible root
        # (top-level images that live directly under the viewed directory).
        parents = [*folder_items.values(), self.tree.invisibleRootItem()]
        for parent in parents:
            children = parent.takeChildren()
            grouped = [c for c in children if c in group_set]
            rest = [c for c in children if c not in group_set]
            if grouped:  # only touch folders that actually hold a group node
                parent.addChildren(grouped + rest)
            else:
                parent.addChildren(children)

    def _ensure_group_node(
        self,
        folder: QTreeWidget | QTreeWidgetItem,
        key: tuple[Path, int],
        group_nodes: dict[tuple[Path, int], QTreeWidgetItem],
    ) -> QTreeWidgetItem:
        """Lazily create the green similarity-group node under ``folder``.

        ``key`` is (folder rel-path, group index). Text is set later (in
        ``_rebuild_tree``) once the per-folder visible member count is known;
        only created for groups with a visible member in that folder."""
        cached = group_nodes.get(key)
        if cached is not None:
            return cached
        node = QTreeWidgetItem(folder, [""])
        node.setForeground(0, QColor("#27ae60"))
        font = node.font(0)
        font.setBold(True)
        node.setFont(0, font)
        group_nodes[key] = node
        return node

    def _ensure_tree_folder(
        self, rel_parent: Path, folder_items: dict[Path, QTreeWidgetItem]
    ) -> QTreeWidget | QTreeWidgetItem:
        """Resolve (and lazily create) the QTreeWidgetItem for ``rel_parent``.

        Returns ``self.tree`` for the root (Path('.')) so callers can pass it
        as the parent of a leaf item directly — QTreeWidgetItem(parent, …)
        accepts either the tree widget or another item.
        """
        if rel_parent in (Path("."), Path("")):
            return self.tree
        cached = folder_items.get(rel_parent)
        if cached is not None:
            return cached
        grandparent = self._ensure_tree_folder(rel_parent.parent, folder_items)
        item = QTreeWidgetItem(grandparent, [rel_parent.name])
        folder_items[rel_parent] = item
        return item

    def _select_tree_index(self, idx: int) -> None:
        """Highlight the tree leaf corresponding to image index ``idx``."""
        for item, i in self._tree_item_to_index.items():
            if i == idx:
                self.tree.setCurrentItem(item)
                return
        self.tree.setCurrentItem(None)

    def _on_search_changed(self, text: str) -> None:
        self._search_text = text
        self._apply_filter_and_sort()

    def _toggle_sort(self) -> None:
        self._sort_desc = not self._sort_desc
        self.sort_btn.setText("↓" if self._sort_desc else "↑")
        self.sort_btn.setToolTip(
            t("dataset_sort_desc_tooltip")
            if self._sort_desc
            else t("dataset_sort_asc_tooltip")
        )
        self._apply_filter_and_sort()

    def _reload_current_dir(self) -> None:
        """Re-scan the currently selected directory (for new/changed images)."""
        name = self.dc.currentText()
        if name:
            self._load_dir(name, preserve_selection=True)

    def _add_dir(self) -> None:
        """Pick a directory and add it to the combo for this session."""
        if not self._confirm_discard_if_dirty():
            return
        start = str(self._dirs.get(self.dc.currentText(), Path.home()))
        chosen = QFileDialog.getExistingDirectory(
            self, t("dataset_add_dir_picker"), start
        )
        if not chosen:
            return
        path = Path(chosen)
        # Use the absolute path string as the display key — unambiguous and
        # avoids collisions with the built-in short labels (image_dataset, …).
        label = str(path)
        for existing in self._dirs.values():
            if existing == path:
                QMessageBox.information(
                    self, t("directory"), t("dataset_add_dir_already", name=label)
                )
                # Switch to it so the user lands on the dir they tried to add.
                for k, v in self._dirs.items():
                    if v == path:
                        idx = self.dc.findText(k)
                        if idx >= 0:
                            self.dc.setCurrentIndex(idx)
                        break
                return
        self._dirs[label] = path
        self.dc.addItem(label)
        self.dc.setCurrentText(label)

    def _on_tree_item_changed(self, current, _previous) -> None:
        """Show the image for the newly selected tree leaf.

        Folder rows (no index) are non-selectable in the data sense; only
        leaves correspond to an image. We confirm-discard before switching so
        the unsaved-edit prompt fires on navigation.
        """
        if current is None:
            return
        idx = self._tree_item_to_index.get(current)
        if idx is None:
            return
        if not self._confirm_discard_if_dirty():
            prev = self._row_for_path(self._current_caption_path)
            if prev is not None and prev != idx:
                self.tree.blockSignals(True)
                try:
                    self._select_tree_index(prev)
                finally:
                    self.tree.blockSignals(False)
            return
        self._show(idx)

    def _show(self, row: int):
        if not 0 <= row < len(self._images):
            return
        p = self._images[row]
        pm = QPixmap(str(p))
        if not pm.isNull():
            self._set_image(p, pm)
        else:
            self._set_image(p, None)
        cp = p.with_suffix(".txt")
        self._current_caption_path = cp
        if cp.exists():
            text = cp.read_text(encoding="utf-8")
        else:
            text = ""
        self._disk_text = text
        self._set_caption_text(text if text else "")
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _set_image(self, p: Path, source: QPixmap | None) -> None:
        """Bind a new source pixmap + its (possibly absent) mask, then refresh."""
        self._source_pm = source
        self._mask_path = (
            _resolve_mask_path(p, self._current_dir) if source is not None else None
        )
        self._overlay_pm = None  # compose lazily in _apply_image_view
        self.overlay_cb.setEnabled(self._mask_path is not None)
        self._apply_image_view()

    def _apply_image_view(self) -> None:
        """Push the right pixmap onto ``self.img`` based on overlay state."""
        if self._source_pm is None:
            return
        if self.overlay_cb.isChecked() and self._mask_path is not None:
            if self._overlay_pm is None:
                self._overlay_pm = _compose_mask_overlay(
                    self._source_pm, self._mask_path
                )
            self.img.set_source(self._overlay_pm)
        else:
            self.img.set_source(self._source_pm)

    def _on_overlay_toggled(self, _checked: bool) -> None:
        self._apply_image_view()

    # ── deletion marking ──────────────────────────────────────

    def _current_index(self) -> int:
        """Index into ``self._images`` of the currently selected image.

        Reads the selected tree leaf; -1 if nothing is on an image (e.g. a
        folder/group node is selected)."""
        item = self.tree.currentItem()
        if item is not None:
            idx = self._tree_item_to_index.get(item)
            if idx is not None:
                return idx
        return -1

    def app_context_menu(self, target, global_pos):
        """Right-click hook (called by MainWindow's global filter). Over an
        image — the preview or a tree leaf — offer "open in system viewer"
        instead of the app default; return None elsewhere so the default shows."""
        path = self._image_under_cursor(target, global_pos)
        if path is None:
            return None
        menu = QMenu(self)
        act = menu.addAction(t("open_in_system_viewer"))
        act.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        )
        return menu

    def _image_under_cursor(self, target, global_pos) -> Path | None:
        """The image path the right-click landed on: the tree leaf under the
        cursor, or the currently shown preview. None if neither applies."""
        if target is self.tree or self.tree.isAncestorOf(target):
            pos = self.tree.viewport().mapFromGlobal(global_pos)
            item = self.tree.itemAt(pos)
            idx = self._tree_item_to_index.get(item) if item is not None else None
            if idx is not None and 0 <= idx < len(self._images):
                return self._images[idx]
            return None
        if target is self.img or self.img.isAncestorOf(target):
            idx = self._current_index()
            if 0 <= idx < len(self._images):
                return self._images[idx]
        return None

    def _toggle_mark_current(self) -> None:
        """Toggle the deletion mark on the currently selected image."""
        idx = self._current_index()
        if not 0 <= idx < len(self._images):
            return
        p = self._images[idx]
        if p in self._marked:
            self._marked.discard(p)
        else:
            self._marked.add(p)
        self._refresh_mark_styles()
        self._refresh_delete_button()

    def _refresh_mark_styles(self) -> None:
        """Repaint tree leaves red iff marked for deletion.

        Unmarked items clear the ForegroundRole entirely (rather than setting a
        default QBrush, which paints black — invisible on the dark theme) so
        they fall back to the palette text color."""
        for leaf, idx in self._tree_item_to_index.items():
            marked = idx < len(self._images) and self._images[idx] in self._marked
            if marked:
                leaf.setForeground(0, _DELETE_MARK_COLOR)
            else:
                leaf.setData(0, Qt.ForegroundRole, None)

    def _unmark_current(self) -> None:
        """Remove the deletion mark from the currently selected image (Esc)."""
        idx = self._current_index()
        if not 0 <= idx < len(self._images):
            return
        p = self._images[idx]
        if p not in self._marked:
            return
        self._marked.discard(p)
        self._refresh_mark_styles()
        self._refresh_delete_button()

    def _clear_marks(self) -> None:
        """Deselect every deletion target (취소 button)."""
        if not self._marked:
            return
        self._marked.clear()
        self._refresh_mark_styles()
        self._refresh_delete_button()

    def _refresh_delete_button(self) -> None:
        n = len(self._marked)
        self.delete_btn.setEnabled(n > 0)
        self.delete_btn.setText(t("dataset_delete") + (f" ({n})" if n else ""))
        self.cancel_mark_btn.setEnabled(n > 0)

    def _delete_marked(self) -> None:
        """Move every marked image (+ its caption sidecars) to the OS trash."""
        targets = sorted(self._marked)
        if not targets:
            return
        reply = QMessageBox.question(
            self,
            t("dataset_delete_confirm_title"),
            t("dataset_delete_confirm_body", n=len(targets)),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        # Remember where the user was so the rebuilt tree doesn't snap back to
        # the top: prefer the image they currently have open; if that one is
        # itself being deleted, anchor on its position so we can land on the
        # nearest surviving neighbour instead.
        open_stem = (
            self._current_caption_path.stem
            if self._current_caption_path is not None
            else None
        )
        old_images = list(self._images)
        anchor_row = self._current_index()
        targets_set = set(targets)

        from send2trash import send2trash  # lazy: keeps GUI startup light

        errors: list[str] = []
        for p in targets:
            try:
                for f in self._deletion_files(p):
                    if f.exists():
                        send2trash(str(f))
            except OSError as e:
                errors.append(f"{p.name}: {e}")
        self._marked.clear()
        self._refresh_delete_button()
        # Drop the editor context so the post-delete reload doesn't prompt about
        # a caption whose image we just removed, then re-scan from disk.
        self._current_caption_path = None
        self._disk_text = ""
        self._set_caption_text("")
        if self._current_dir is not None:
            self._all_images = _imgs(self._current_dir)
        self._apply_filter_and_sort()
        if self._images:
            self._select_tree_index(
                self._post_delete_row(open_stem, old_images, anchor_row, targets_set)
            )
        else:
            self._set_image_none()
            self._refresh_buttons()
            self._refresh_inline_diff()
        if errors:
            QMessageBox.warning(
                self, t("error"), t("dataset_delete_failed", err="\n".join(errors))
            )

    def _post_delete_row(
        self,
        open_stem: str | None,
        old_images: list[Path],
        anchor_row: int,
        deleted: set[Path],
    ) -> int:
        """Pick which row to reselect after a delete so the view stays put.

        If the image the user had open survived, return its new row. Otherwise
        walk outward from ``anchor_row`` (forward first, then backward) over the
        pre-delete order to find the nearest surviving neighbour, and return its
        new row. Falls back to the top only when nothing else matches.
        """
        new_row = {p.stem: i for i, p in enumerate(self._images)}
        if open_stem is not None and open_stem in new_row:
            return new_row[open_stem]
        if old_images:
            start = anchor_row if anchor_row >= 0 else 0
            order = list(range(start, len(old_images))) + list(range(start - 1, -1, -1))
            for j in order:
                if old_images[j] not in deleted and old_images[j].stem in new_row:
                    return new_row[old_images[j].stem]
        return 0

    def _deletion_files(self, image: Path) -> list[Path]:
        """Image + its caption sidecar + caption history (all trashed together)."""
        cap = image.with_suffix(".txt")
        return [image, cap, _history_path(cap)]

    def _set_image_none(self) -> None:
        """Clear the preview pane (used after deleting the last image)."""
        self._source_pm = None
        self._mask_path = None
        self._overlay_pm = None
        self.overlay_cb.setEnabled(False)
        self.img.clear()

    # ── caption editing ───────────────────────────────────────

    def _set_caption_text(self, text: str) -> None:
        self._suspend_dirty = True
        try:
            self.cap.setPlainText(text)
        finally:
            self._suspend_dirty = False

    def _on_text_changed(self) -> None:
        if self._suspend_dirty:
            return
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _is_dirty(self) -> bool:
        if self._current_caption_path is None:
            return False
        return self.cap.toPlainText() != self._disk_text

    def _refresh_buttons(self) -> None:
        dirty = self._is_dirty()
        self.save_btn.setEnabled(dirty)
        self.revert_btn.setEnabled(dirty)
        marker = t("caption_dirty_marker") if dirty else ""
        label = t("caption") + marker
        if dirty:
            _, add, rem = _diff_spans(self._disk_text, self.cap.toPlainText())
            if add or rem:
                label += "  " + t("caption_diff_stats", add=add, rem=rem)
        self.cap_label.setText(label)
        # Versions button is enabled whenever there's a caption file context;
        # the dialog itself shows "(no prior versions)" when empty.
        self.versions_btn.setEnabled(self._current_caption_path is not None)

    def _refresh_inline_diff(self) -> None:
        """Highlight inserted spans (vs disk) directly in the editor."""
        if self._current_caption_path is None:
            self.cap.setExtraSelections([])
            return
        spans, _, _ = _diff_spans(self._disk_text, self.cap.toPlainText())
        if not spans:
            self.cap.setExtraSelections([])
            return
        fmt = _add_format()
        sels: list[QTextEdit.ExtraSelection] = []
        doc = self.cap.document()
        for j1, j2 in spans:
            cur = QTextCursor(doc)
            cur.setPosition(j1)
            cur.setPosition(j2, QTextCursor.KeepAnchor)
            es = QTextEdit.ExtraSelection()
            es.cursor = cur
            es.format = fmt
            sels.append(es)
        self.cap.setExtraSelections(sels)

    def _save(self) -> None:
        cp = self._current_caption_path
        if cp is None or not self._is_dirty():
            return
        new_text = self.cap.toPlainText()
        try:
            # Snapshot the prior on-disk version into history before overwriting.
            # Skip when the previous file didn't exist (nothing to preserve).
            if cp.exists():
                _append_history(cp, self._disk_text)
            cp.write_text(new_text, encoding="utf-8")
        except OSError as e:
            QMessageBox.warning(self, t("error"), t("caption_save_failed", err=str(e)))
            return
        self._disk_text = new_text
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _revert(self) -> None:
        if self._current_caption_path is None:
            return
        self._set_caption_text(self._disk_text)
        self._refresh_buttons()
        self._refresh_inline_diff()

    def _open_versions(self) -> None:
        cp = self._current_caption_path
        if cp is None:
            return
        # Diff inside the dialog compares against the on-disk text, so save
        # any pending edits or warn? We keep it simple: dialog always uses
        # disk as the comparison baseline. If user restores a version, it
        # replaces *editor* contents (becomes a pending edit until they Save).
        dlg = CaptionVersionsDialog(cp, self._disk_text, self)
        if dlg.exec() == QDialog.Accepted:
            restored = dlg.restored_text()
            if restored is not None:
                self._set_caption_text(restored)
                self._refresh_buttons()
                self._refresh_inline_diff()

    # ── navigation helpers ────────────────────────────────────

    def _row_for_path(self, cp: Path | None) -> int | None:
        if cp is None:
            return None
        for i, p in enumerate(self._images):
            if p.with_suffix(".txt") == cp:
                return i
        return None

    def _confirm_discard_if_dirty(self) -> bool:
        """Prompt to save if dirty. Returns False if the user cancels."""
        if not self._is_dirty():
            return True
        reply = QMessageBox.question(
            self,
            t("caption_unsaved_title"),
            t("caption_unsaved_body"),
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Save:
            self._save()
            # If the save failed, _is_dirty() is still True — abort the switch.
            return not self._is_dirty()
        # Discard: drop edits silently.
        return True

    def _nav(self, d: int):
        r = self._current_index() + d
        if 0 <= r < len(self._images):
            self._select_tree_index(r)
