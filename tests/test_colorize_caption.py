"""Color-only caption filter for the colorization EasyControl task.

Guards :func:`color_caption.filter_to_colors` — the transform that reduces a
full Anima caption to its color tags before re-encoding into the colorize
text_cache_dir. A regression here would silently change what the colorize
adapter learns to steer on.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COLORIZE_DIR = REPO_ROOT / "easycontrol_adapters" / "colorization"
if str(COLORIZE_DIR) not in sys.path:
    sys.path.insert(0, str(COLORIZE_DIR))

from color_caption import (  # noqa: E402
    filter_to_colors,
    filter_to_colors_and_copyright,
    is_color_tag,
    is_copyright_tag,
)


def test_keeps_hair_eye_skin_colors():
    assert is_color_tag("blue hair")
    assert is_color_tag("aqua eyes")
    assert is_color_tag("light brown hair")
    assert is_color_tag("grey eyes")
    assert is_color_tag("dark skin")
    # multi-color escapes
    assert is_color_tag("two-tone hair")
    assert is_color_tag("heterochromia")


def test_keeps_color_noun_tags():
    assert is_color_tag("yellow shirt")
    assert is_color_tag("red ribbon")
    assert is_color_tag("white background")
    assert is_color_tag("dark green dress")  # modifier + color
    assert is_color_tag("monochrome")


def test_drops_non_color_tags():
    for tag in (
        "1girl",
        "looking at viewer",
        "long hair",  # length, not color
        "closed eyes",  # state, not color
        "open mouth",
        "smile",
        "@eufoniuz",
        "ponytail",
    ):
        assert not is_color_tag(tag), tag


def test_filter_extracts_colors_in_order():
    caption = (
        "explicit, 1boy, 1girl, suou yuki, @eufoniuz, brown eyes, brown hair, "
        "looking at viewer, ponytail, smile, yellow shirt, green shorts"
    )
    out = filter_to_colors(caption)
    assert out == "brown eyes, brown hair, yellow shirt, green shorts"


def test_filter_empty_when_no_color():
    # No color tag → empty caption (trains the unconditional auto-color mode).
    assert filter_to_colors("1girl, solo, looking at viewer, smile") == ""
    assert filter_to_colors("") == ""


def test_long_hair_not_kept_but_blue_hair_is():
    # 'long hair' shares the 'hair' suffix but has no color word → dropped.
    assert filter_to_colors("long hair, blue hair") == "blue hair"


# ----- color + copyright variant ------------------------------------------

_COPYRIGHT = frozenset({"genshin impact", "fate/grand order"})


def test_is_copyright_tag_case_insensitive():
    assert is_copyright_tag("genshin impact", _COPYRIGHT)
    assert is_copyright_tag("Genshin Impact", _COPYRIGHT)
    assert not is_copyright_tag("1girl", _COPYRIGHT)
    assert not is_copyright_tag("blue hair", _COPYRIGHT)


def test_copyright_kept_first_then_colors():
    caption = (
        "1girl, nilou (genshin impact), genshin impact, @sincos, blue hair, "
        "looking at viewer, red dress"
    )
    out = filter_to_colors_and_copyright(caption, _COPYRIGHT)
    # copyright leads, color tags follow in original order; non-color dropped.
    assert out == "genshin impact, blue hair, red dress"


def test_copyright_absent_falls_back_to_colors():
    caption = "1girl, solo, blue hair, yellow shirt"
    out = filter_to_colors_and_copyright(caption, _COPYRIGHT)
    assert out == "blue hair, yellow shirt"


def test_copyright_protected_from_dropout():
    # The copyright tag must survive even at dropout_rate=1.0, color tags drop.
    import random

    from library.preprocess import generate_caption_variants

    random.seed(0)
    caption = filter_to_colors_and_copyright(
        "genshin impact, blue hair, red dress, green eyes", _COPYRIGHT
    )
    variants = generate_caption_variants(
        caption,
        num_variants=4,
        tag_dropout_rate=1.0,
        protect_fn=lambda t: is_copyright_tag(t, _COPYRIGHT),
    )
    # v0 pristine keeps everything; v1+ drop all color tags but keep copyright.
    assert variants[0] == caption
    for v in variants[1:]:
        assert "genshin impact" in v
        assert "blue hair" not in v and "red dress" not in v and "green eyes" not in v
