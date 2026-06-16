"""Named visual themes for the GUI — Dark / Light / Sepia.

Replaces the old single dark palette + accent-color picker. Each theme is a flat
set of **semantic color tokens** (background, panel, text, border, accent, …);
``apply_theme`` turns the active theme into a ``QPalette`` + global stylesheet,
and ``tok()`` lets individual widgets pull the same tokens instead of hardcoding
hex literals — so a neutral surface follows the theme instead of staying a dark
island on a light window.

Design notes
------------
* **Neutral surfaces/text route through tokens; saturated *action* buttons do
  not.** A guide button (teal), update button (amber), or danger button (red)
  is white-on-saturated-color and reads fine on any background, so those stay
  hardcoded at their call sites. Only the neutral chrome (window/panel/input/
  text/border) varies by theme.
* Widgets read tokens at *build* time. A theme switch re-applies the palette +
  global stylesheet live (instant for app-level styling) and the caller rebuilds
  the window so per-widget ``tok()`` lookups pick up the new values.

This module may import Qt (it is only ever imported from the Qt side); keep
``_paths.py`` / ``config_io.py`` Qt-free as before.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QPainter,
    QPalette,
    QPixmap,
    QPolygon,
)
from PySide6.QtWidgets import QApplication

from gui._paths import DEFAULT_THEME, get_setting, set_setting

# Bundled UI font (Pretendard, OFL) — the design system's primary sans. Three
# static weights live next to this module; see gui/fonts/README.md.
_FONT_DIR = Path(__file__).parent / "fonts"
_PRETENDARD_FILES = (
    "Pretendard-Regular.ttf",
    "Pretendard-Medium.ttf",
    "Pretendard-Bold.ttf",
)
# Resolved once: the loaded family name ("Pretendard") or None if the files are
# missing / Qt refused them, in which case we fall back to OS fonts only.
_bundled_family: str | None = None
_fonts_loaded = False


@dataclass(frozen=True)
class Theme:
    """A flat palette of semantic color tokens (hex strings).

    Naming is by *role*, not appearance, so the same key means the same thing in
    every theme:

    * ``window`` deepest app background; ``base`` text-entry/list canvas;
      ``panel`` a slightly raised neutral surface (help/log/preview boxes).
    * ``input_bg`` / ``input_hover`` text-field fill; ``surface`` / ``surface_hover``
      neutral button & tab fill.
    * ``text`` primary; ``text_bright`` max-contrast; ``text_dim`` secondary/hint.
    * ``border`` / ``border_dim`` strong / subtle separators.
    * ``accent`` selection/highlight; ``accent_text`` text on accent.
    * ``link`` / ``link_visited``; status ``ok`` / ``warn`` / ``err``.
    * ``scroll_bg`` / ``scroll_handle`` / ``scroll_handle_hover`` scrollbars.
    * ``is_dark`` whether the theme reads as dark (lets callers branch when a
      token isn't enough).
    """

    name: str
    is_dark: bool
    window: str
    base: str
    panel: str
    input_bg: str
    input_hover: str
    surface: str
    surface_hover: str
    text: str
    text_bright: str
    text_dim: str
    border: str
    border_dim: str
    accent: str
    accent_text: str
    tab_selected: str
    tooltip_bg: str
    link: str
    link_visited: str
    ok: str
    warn: str
    err: str
    scroll_bg: str
    scroll_handle: str
    scroll_handle_hover: str


_DARK = Theme(
    name="dark",
    is_dark=True,
    window="#1e1e1e",
    base="#191919",
    panel="#2b2b2b",
    input_bg="#2a2a2a",
    input_hover="#2c2c2c",
    surface="#3a3a3a",
    surface_hover="#4a4a4a",
    text="#dcdcdc",
    text_bright="#ffffff",
    text_dim="#888888",
    border="#555555",
    border_dim="#444444",
    accent="#3c78c8",
    accent_text="#ffffff",
    tab_selected="#1e1e1e",
    tooltip_bg="#323232",
    link="#ffb86b",
    link_visited="#e6944e",
    ok="#4ade80",
    warn="#fbbf24",
    err="#f87171",
    scroll_bg="#242424",
    scroll_handle="#7a7a7a",
    scroll_handle_hover="#a0a0a0",
)

_LIGHT = Theme(
    name="light",
    is_dark=False,
    window="#f4f4f4",
    base="#ffffff",
    panel="#ececec",
    input_bg="#ffffff",
    input_hover="#f0f0f0",
    surface="#e4e4e4",
    surface_hover="#d6d6d6",
    text="#1f1f1f",
    text_bright="#000000",
    text_dim="#6a6a6a",
    border="#b6b6b6",
    border_dim="#d2d2d2",
    accent="#2f6fb0",
    accent_text="#ffffff",
    tab_selected="#ffffff",
    tooltip_bg="#fafafa",
    link="#1a5fb4",
    link_visited="#7a3fb0",
    ok="#1a7f37",
    warn="#9a6700",
    err="#c01c28",
    scroll_bg="#e0e0e0",
    scroll_handle="#b4b4b4",
    scroll_handle_hover="#909090",
)

_SEPIA = Theme(
    name="sepia",
    is_dark=False,
    window="#f4ecd8",
    base="#fffaf0",
    panel="#ebe0c8",
    input_bg="#fffaf0",
    input_hover="#f7efdd",
    surface="#e6dabe",
    surface_hover="#dccfae",
    text="#4a4036",
    text_bright="#2a231b",
    text_dim="#8a7d68",
    border="#c8b896",
    border_dim="#d8cbae",
    accent="#b4691f",
    accent_text="#ffffff",
    tab_selected="#fffaf0",
    tooltip_bg="#f0e6cf",
    link="#9a4f12",
    link_visited="#7a3f0f",
    ok="#5a7d2a",
    warn="#9a6700",
    err="#b03028",
    scroll_bg="#e6dabe",
    scroll_handle="#c0ad88",
    scroll_handle_hover="#a89468",
)

THEMES: dict[str, Theme] = {t.name: t for t in (_DARK, _LIGHT, _SEPIA)}

# Display order + i18n label keys for the Settings selector.
THEME_ORDER = ("dark", "light", "sepia")
THEME_LABEL_KEYS = {
    "dark": "settings_theme_dark",
    "light": "settings_theme_light",
    "sepia": "settings_theme_sepia",
}

# The live theme, resolved once per apply_theme() so tok() stays cheap and never
# re-reads the settings file per widget.
_active: Theme = THEMES[DEFAULT_THEME]


def current_theme_name() -> str:
    """The persisted theme name (falls back to the default)."""
    name = get_setting("theme", DEFAULT_THEME)
    return name if name in THEMES else DEFAULT_THEME


def active_theme() -> Theme:
    """The Theme last passed to apply_theme() (the live look)."""
    return _active


def tok(key: str) -> str:
    """One color token from the active theme, e.g. ``tok("panel")``.

    Use this at widget build time instead of a hardcoded neutral hex so the
    widget follows the theme. Raises if the key is unknown (typo guard)."""
    return getattr(_active, key)


def set_theme(name: str) -> None:
    """Persist the chosen theme name (does not re-apply — caller does that)."""
    if name in THEMES:
        set_setting("theme", name)


# App font size (point size handed to app.setFont). 10pt is the design default —
# see the note in apply_theme(). The slider in SettingsDialog persists an override
# here; clamped so a stray value can't shrink the UI to nothing or blow it up.
DEFAULT_FONT_SIZE = 10
FONT_SIZE_MIN = 8
FONT_SIZE_MAX = 18


def current_font_size() -> int:
    """The persisted app font point size (clamped, falls back to the default)."""
    try:
        size = int(get_setting("font_size", DEFAULT_FONT_SIZE))
    except (TypeError, ValueError):
        return DEFAULT_FONT_SIZE
    return max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, size))


def set_font_size(size: int) -> None:
    """Persist the chosen app font size (does not re-apply — caller does that)."""
    set_setting("font_size", max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, int(size))))


def rich_text_pt(px_at_default: float) -> str:
    """A rich-text ``font-size`` (in pt) that scales with the app font knob.

    The explanation-panel QTextDocument default font already tracks the Settings
    font size, but inline ``font-size:<n>px`` in Qt rich text is *absolute* — Qt
    honours only ``px``/``pt`` there (``em``/``%`` are silently ignored), so a px
    literal pins that text regardless of the knob. Convert a design value
    expressed "in px at the 10pt default" into a pt size scaled by the current
    app font so headings/body track the knob. 10pt ≈ 13.33px at 96dpi."""
    return f"{px_at_default * current_font_size() / 13.333:.1f}pt"


def _build_palette(t: Theme) -> QPalette:
    p = QPalette()
    for role, color in [
        (QPalette.Window, QColor(t.window)),
        (QPalette.WindowText, QColor(t.text)),
        (QPalette.Base, QColor(t.base)),
        (QPalette.AlternateBase, QColor(t.panel)),
        (QPalette.ToolTipBase, QColor(t.tooltip_bg)),
        (QPalette.ToolTipText, QColor(t.text)),
        (QPalette.Text, QColor(t.text)),
        (QPalette.Button, QColor(t.surface)),
        (QPalette.ButtonText, QColor(t.text)),
        (QPalette.Highlight, QColor(t.accent)),
        (QPalette.HighlightedText, QColor(t.accent_text)),
        (QPalette.Link, QColor(t.link)),
        (QPalette.LinkVisited, QColor(t.link_visited)),
        # Disabled text needs an explicit dim or it inherits full-contrast text.
        (QPalette.PlaceholderText, QColor(t.text_dim)),
    ]:
        p.setColor(role, color)
    disabled = QColor(t.text_dim)
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, disabled)
    return p


_ARROW_DIR = Path(tempfile.gettempdir()) / "anima_gui_arrows"
_arrow_cache: dict[tuple[str, str], str] = {}


def _arrow_icon(color: str, direction: str) -> str:
    """Render a small filled-triangle arrow PNG in ``color`` and return its path.

    Qt's stylesheet engine doesn't support the CSS border-triangle trick (it
    paints a filled box — the 'white tofu'), and the native spin-arrow primitive
    renders dark-on-dark in the dark theme. So we paint our own theme-coloured
    arrow with QPainter and feed it to ``image: url(...)``. Cached per
    (color, direction); files live in a temp dir keyed by colour so a theme
    switch regenerates them."""
    key = (color, direction)
    cached = _arrow_cache.get(key)
    if cached:
        return cached
    _ARROW_DIR.mkdir(parents=True, exist_ok=True)
    # 2x size for crisp rendering on HiDPI; the stylesheet draws it at ~11px.
    w, h = 22, 14
    pm = QPixmap(w, h)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    pad_x, pad_y = 4, 3
    if direction == "up":
        tri = QPolygon([QPoint(pad_x, h - pad_y), QPoint(w - pad_x, h - pad_y), QPoint(w // 2, pad_y)])
    else:
        tri = QPolygon([QPoint(pad_x, pad_y), QPoint(w - pad_x, pad_y), QPoint(w // 2, h - pad_y)])
    p.drawPolygon(tri)
    p.end()
    safe = color.lstrip("#")
    path = _ARROW_DIR / f"{direction}_{safe}.png"
    pm.save(str(path), "PNG")
    # Qt CSS wants forward slashes even on Windows.
    url = path.as_posix()
    _arrow_cache[key] = url
    return url


def _build_stylesheet(t: Theme, font_family: str = "", font_size: int = DEFAULT_FONT_SIZE) -> str:
    # Enforce the UI font family AND size through the stylesheet, not just
    # app.setFont(). On Windows the native style paints many controls with the
    # system font and ignores the application QFont (QFontInfo still *reports*
    # the app font, so this is invisible to code — only the pixels are wrong),
    # and for any widget Qt resolves through the stylesheet (e.g. labels that
    # carry their own colour stylesheet) the point size from app.setFont() does
    # not reliably propagate. A stylesheet rule has higher precedence, so naming
    # both family and size on the base QWidget forces every widget to render in
    # the bundled font at the chosen size — so the Settings font-size knob scales
    # tab titles and field labels too. Per-widget overrides (monospace log views
    # set their own family + px size) are more specific and still win.
    font_rule = (
        f"* {{ font-family: {font_family}; font-size: {font_size}pt; }}\n"
        if font_family
        else f"* {{ font-size: {font_size}pt; }}\n"
    )
    return f"""
        {font_rule}
        QGroupBox {{
            font-weight: bold; border: 1px solid {t.border_dim};
            border-radius: 4px; margin-top: 8px; padding-top: 16px;
        }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
        QPushButton {{ padding: 4px 12px; border: 1px solid {t.border}; border-radius: 3px; }}
        QPushButton:hover {{ background: {t.surface_hover}; }}
        QScrollArea {{ border: none; }}
        QSplitter::handle {{ background: {t.border_dim}; }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit, QListWidget {{
            background: {t.input_bg}; color: {t.text}; border: 1px solid {t.border};
            border-radius: 3px; padding: 2px 4px;
        }}
        /* Spin buttons: both steppers sit side by side on the right edge (down
           then up) instead of the cramped native vertical stack, and the arrows
           are enlarged so they're legible. */
        QSpinBox, QDoubleSpinBox {{ padding-right: 34px; }}
        /* height is intentionally oversized: Qt clamps the spin-button to the
           field height, so this fills it edge-to-edge at any font size (no
           top/bottom gap) instead of the native ~15px centred box. */
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border; subcontrol-position: right;
            width: 16px; height: 200px; border-left: 1px solid {t.border}; background: {t.surface};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border; subcontrol-position: right; margin-right: 16px;
            width: 16px; height: 200px; border-left: 1px solid {t.border}; background: {t.surface};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background: {t.surface_hover};
        }}
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
            background: {t.accent};
        }}
        /* Theme-coloured arrow icons painted by _arrow_icon — Qt can't render a
           CSS border-triangle (paints a white box) and the native arrow is
           dark-on-dark in the dark theme. */
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: url({_arrow_icon(t.text, "up")}); width: 11px; height: 7px;
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: url({_arrow_icon(t.text, "down")}); width: 11px; height: 7px;
        }}
        QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
            image: url({_arrow_icon(t.text_bright, "up")});
        }}
        QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
            image: url({_arrow_icon(t.text_bright, "down")});
        }}
        QComboBox QAbstractItemView {{
            background: {t.input_bg}; color: {t.text}; selection-background-color: {t.accent};
        }}
        QTabWidget::pane {{ border: 1px solid {t.border_dim}; }}
        QTabBar::tab {{
            background: {t.input_bg}; color: {t.text}; border: 1px solid {t.border_dim};
            padding: 6px 14px;
            font-size: {font_size + 1}pt; font-weight: 500;
            border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{ background: {t.tab_selected}; color: {t.text_bright}; }}
        QTabBar::tab:hover {{ background: {t.surface}; }}
        QToolTip {{ max-width: 400px; background: {t.tooltip_bg}; color: {t.text};
            border: 1px solid {t.border}; }}
        QMenu {{
            background: {t.input_bg}; color: {t.text}; border: 1px solid {t.border};
        }}
        QMenu::item {{ padding: 4px 20px; background: transparent; color: {t.text}; }}
        QMenu::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}
        QMenu::item:disabled {{ color: {t.text_dim}; }}
        QMenu::separator {{ height: 1px; background: {t.border_dim}; margin: 4px 8px; }}
    """


def _load_bundled_fonts() -> str | None:
    """Register the bundled Pretendard weights with Qt (once per process).

    Returns the family name to put at the head of the UI font stack, or ``None``
    if no bundled file loaded (then we render in the OS font as before). All
    three weights register under one Qt family, so a plain ``QFont(family)`` +
    ``setWeight`` resolves the right instance."""
    global _bundled_family, _fonts_loaded
    if _fonts_loaded:
        return _bundled_family
    _fonts_loaded = True
    for fname in _PRETENDARD_FILES:
        path = _FONT_DIR / fname
        if not path.exists():
            continue
        fid = QFontDatabase.addApplicationFont(str(path))
        if fid == -1:
            continue
        fams = QFontDatabase.applicationFontFamilies(fid)
        if fams:
            _bundled_family = fams[0]  # "Pretendard"
    return _bundled_family


def apply_theme(app: QApplication, name: str | None = None) -> Theme:
    """Resolve + apply a theme to the whole app (palette + global stylesheet).

    ``name`` defaults to the persisted theme. Updates the module-global active
    theme so subsequent ``tok()`` lookups (and rebuilt widgets) use it. Returns
    the applied Theme. The font is set here too (kept from the old ``_dark``)."""
    global _active
    resolved = name if (name in THEMES) else current_theme_name()
    t = THEMES[resolved]
    _active = t

    # App font: the bundled Pretendard (design-system primary sans) leads the
    # stack. Pretendard covers Latin + Korean Hangul + Japanese kana, but NOT
    # Han ideographs (kanji/hanzi) or emoji — so the only fallbacks that remain
    # load-bearing are the CJK font (kanji/hanzi for the JA/ZH UI strings) and
    # the color-emoji font (📖 ⚙ 🔍 … wayfinding markers). Naming a CJK family
    # alone doesn't cascade to an emoji font, so emoji would render as tofu;
    # listing both explicitly fixes that. (A plain Latin fallback would be
    # redundant with Pretendard, so it's omitted.)
    if sys.platform == "win32":
        families = ["Malgun Gothic", "Segoe UI Emoji"]
    elif sys.platform == "darwin":
        families = ["Apple SD Gothic Neo", "Apple Color Emoji"]
    else:  # Linux: Noto Sans CJK + Noto Color Emoji ship with most distros.
        families = ["Noto Sans CJK KR", "Noto Color Emoji"]
    bundled = _load_bundled_fonts()
    if bundled:
        families.insert(0, bundled)
    font = QFont()
    font.setFamilies(families)
    # 10pt, not the OS-native 9pt: Pretendard has a smaller x-height / apparent
    # size than Windows' Segoe UI at the same point size, so matching 9pt makes
    # every label read a notch smaller than native apps. 10pt brings Pretendard
    # back in line with Segoe UI 9pt visually. The user can override this from
    # Settings (persisted as "font_size"); current_font_size() clamps it.
    font.setPointSize(current_font_size())
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    # Same family stack handed to the stylesheet so the family is enforced even
    # for native-styled controls that ignore app.setFont() on Windows. Quote
    # each name (Qt CSS needs quotes for multi-word families like "Malgun Gothic").
    family_css = ", ".join(f'"{f}"' for f in families)

    app.setPalette(_build_palette(t))
    app.setStyleSheet(_build_stylesheet(t, family_css, current_font_size()))
    return t
