#!/usr/bin/env python3
"""Color-only caption filter for the colorization EasyControl task.

The colorization condition (mangafied B&W lineart + screentone) already encodes
*everything spatial* — composition, poses, objects, layout. The one variable it
cannot encode is **hue/chroma**. So the text channel for colorization should
carry *only* color information: every surviving token is then a color fact the
model can't get from structure, which gives a strong text→color coupling and
makes color prompts actually steer at inference.

:func:`filter_to_colors` reduces a full Anima caption to its color tags only:

  * hair / eye / skin color tags ("blue hair", "aqua eyes", "dark skin", plus
    the multi-color escapes: "two-tone hair", "heterochromia", …)
  * any ``<color> <noun>`` tag — clothing, objects, background
    ("yellow shirt", "red ribbon", "white background", "blue sky")
  * standalone palette descriptors ("monochrome", "colorful", "pastel colors")

Everything else is dropped. Images whose caption has no color tag collapse to an
empty caption (→ unconditional / auto-color sample), which is fine — that's the
empty-prompt colorization mode.

Tag order is preserved (the slot order is irrelevant once non-color tags are
gone). Pure stdlib so it stays importable from the preprocess path and unit
tests without pulling torch.

:func:`filter_to_colors_and_copyright` is the variant used by the colorize prep
when copyright tags should ride along with the color tags ("genshin impact,
pink hair, blue eyes"). The manga cond can't encode *which series* a page is
from, so copyright is genuinely-ambiguous text the model can bind to — and
unlike a hue it shouldn't be tag-dropped, so it's emitted as a protected prefix
(see ``protect_fn`` in :func:`library.preprocess.generate_caption_variants`).
Copyright tags are identified against the corpus copyright vocab in the caption
index (``post_image_dataset/captions/caption_index.json`` ``groups.copyright``).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

# Repo root: easycontrol_adapters/colorization/color_caption.py → ../../..
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTION_INDEX = (
    _REPO_ROOT / "post_image_dataset" / "captions" / "caption_index.json"
)

# Single color words that head a "<color> <noun>" tag or end a "<x> hair/eyes".
# Booru's stable color vocab + a few common extended hues. Kept deliberately
# broad on the permissive side: a false-keep ("blue rose") is still color info,
# a false-drop loses a hue the model needs.
COLOR_WORDS: frozenset[str] = frozenset(
    {
        "aqua", "black", "blonde", "blue", "brown", "green", "grey", "gray",
        "orange", "pink", "purple", "red", "silver", "white", "yellow",
        "tan", "teal", "cyan", "magenta", "maroon", "navy", "olive", "violet",
        "beige", "gold", "golden", "cream", "crimson", "scarlet", "turquoise",
        "lavender", "peach", "azure", "ivory", "bronze",
    }
)

# Leading modifiers that precede a color word ("light blue hair", "dark green
# dress", "pale ...").
COLOR_MODIFIERS: frozenset[str] = frozenset(
    {"light", "dark", "pale", "deep", "bright", "pastel"}
)

# Multi-word / non-"<color> X" tags that are still pure color information and
# should be kept whole.
COLOR_PHRASES: frozenset[str] = frozenset(
    {
        "monochrome", "greyscale", "grayscale", "colorful", "muted color",
        "pastel colors", "limited palette", "spot color", "rainbow order",
        # hair/eye multi-color escapes (booru groups.yaml escapes)
        "two-tone hair", "multicolored hair", "gradient hair", "streaked hair",
        "split-color hair", "rainbow hair", "colored inner hair",
        "multicolored eyes", "heterochromia", "two-tone eyes",
        # skin tone
        "dark skin", "pale skin", "tan", "sun tan", "tanlines", "tan lines",
        "dark-skinned female", "dark-skinned male",
    }
)


def is_color_tag(tag: str) -> bool:
    """True if ``tag`` carries color/hue information worth keeping."""
    t = tag.strip().lower()
    if not t:
        return False
    if t in COLOR_PHRASES:
        return True
    words = t.split()
    # "<...> hair/eyes/skin" with a color word anywhere → colored body feature.
    if words[-1] in ("hair", "eyes", "skin") and any(
        w in COLOR_WORDS for w in words
    ):
        return True
    # Leading color word: "<color> <noun>" (clothing, object, background, …).
    if words[0] in COLOR_WORDS:
        return True
    # Modifier + color word: "light blue dress", "dark green skirt".
    if (
        words[0] in COLOR_MODIFIERS
        and len(words) >= 2
        and words[1] in COLOR_WORDS
    ):
        return True
    return False


def filter_to_colors(caption: str) -> str:
    """Reduce a comma-separated Anima caption to its color tags only.

    Returns a comma-joined string of the kept tags (original order), or ``""``
    when the caption has no color tags.
    """
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    kept = [t for t in tags if is_color_tag(t)]
    return ", ".join(kept)


@functools.lru_cache(maxsize=4)
def load_copyright_tags(index_path: str | Path = DEFAULT_CAPTION_INDEX) -> frozenset[str]:
    """Lowercased set of every copyright tag name from the caption index.

    Reads ``groups.copyright`` (a ``{copyright_name: [image_ids]}`` map) from the
    method-agnostic typed-tag index. Returns an empty set if the index is missing
    so callers degrade to "no copyright kept" rather than crashing. Cached because
    the index is a couple-thousand-image JSON and this runs once per caption.
    """
    p = Path(index_path)
    if not p.exists():
        return frozenset()
    data = json.loads(p.read_text(encoding="utf-8"))
    groups = data.get("groups", {})
    copyright_group = groups.get("copyright", {})
    return frozenset(name.strip().lower() for name in copyright_group if name.strip())


def is_copyright_tag(tag: str, copyright_tags: frozenset[str]) -> bool:
    """True if ``tag`` is a known copyright/series name (case-insensitive)."""
    return tag.strip().lower() in copyright_tags


def filter_to_colors_and_copyright(
    caption: str, copyright_tags: frozenset[str] | None = None
) -> str:
    """Reduce a caption to its copyright tags (first) followed by its color tags.

    Copyright tags are emitted *before* the color tags so they form a contiguous
    leading run, mirroring the ``@artist``-prefix convention the variant
    generator already protects. A color prompt then reads naturally — "genshin
    impact, pink hair, blue eyes". Pass ``copyright_tags`` to avoid re-reading the
    caption index per call; defaults to :func:`load_copyright_tags`. A tag that is
    both a copyright name and color-shaped is kept once, in the copyright slot.
    """
    if copyright_tags is None:
        copyright_tags = load_copyright_tags()
    tags = [t.strip() for t in caption.split(",") if t.strip()]
    copyrights = [t for t in tags if is_copyright_tag(t, copyright_tags)]
    seen = {t.lower() for t in copyrights}
    colors = [t for t in tags if t.lower() not in seen and is_color_tag(t)]
    return ", ".join(copyrights + colors)
