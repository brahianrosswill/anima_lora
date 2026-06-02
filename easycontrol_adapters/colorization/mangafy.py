"""Mangafication: color illustration → synthetic B&W manga (lineart + screentone).

v0 — pure ``cv2`` / ``numpy``, **no model downloads**. Produces a 3-channel
grayscale RGB image whose distribution approximates a real screentoned manga
page, so a colorization adapter trained with these as the *condition* (target =
the original color image) generalizes to real B&W manga at inference.

Pipeline
--------
1. **XDoG lineart** (Winnemöller extended difference-of-Gaussians) → crisp ink
   contours. More manga-like than Canny (clean, anti-aliased strokes).
2. **Screentone tone** — luminance is band-limited; mid/dark regions are filled
   with an algorithmic screen thresholded against luminance. Real manga isn't all
   clustered dots, and a page picks its tone *by value* — so we split the toned range
   into a few luminance bands (by 명암 / brightness) and give each band its own
   *pattern* (seeded): ``dot`` (clustered-dot halftone), ``line`` (parallel-line /
   hatch tone), or ``cross`` (cross-hatch), at its own period/angle. The texture
   therefore changes exactly where the value changes — darks, mids, and lights read as
   distinct tones. Brightest stays white, darkest stays solid black.
3. **Composite** line over tone, emit as 3-channel RGB (the Qwen VAE is RGB-only).

Per-image jitter (seeded by stem) varies the XDoG sigma, the band count + each
band's screen period / angle / *pattern* (dot/line/cross), and the luma weights, so
the model keys on *structure*, not one fixed screen operator — and so it's robust to
whatever real screentone it meets at inference.

Why this is Phase A: learned lineart (Anime2Sketch / sketchKeras) and ScreenVAE
screentone synthesis sit closer to the real-manga manifold but need model
downloads — see ``README.md`` Phase B. This v0 gets a training run today.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Default knobs (overridable; jittered per-stem when ``seed`` is given) ─────
_XDOG_SIGMA = 0.8  # base blur scale (px); lines thicken with the image size below
_XDOG_K = 1.6  # second-Gaussian scale ratio
_XDOG_SHARP = 18.0  # p — DoG sharpening; higher = thinner/harder lines
_XDOG_EPS = 0.10  # soft-threshold midpoint
_XDOG_PHI = 12.0  # soft-threshold slope
_TONE_PERIOD = 2.3  # dot screen period (px) — fine manga screentone pitch at ~900px
_HATCH_PERIOD = 3  # line/cross period (px) — a touch coarser than dots, fine hatch tone
# The screen is rendered at this supersample factor, hard-thresholded, then resolved by
# an **area-average** downscale (cv2.INTER_AREA = exact ss×ss box mean, the correct
# supersample resolve), so fine and rotated screens anti-alias into smooth gray instead
# of folding into coarse low-frequency moiré (a rotated line/cross at ~3px is near Nyquist
# on the native grid → beat stripes; resolved at NxN it downsamples cleanly). ss sets the
# number of gray coverage levels (ss² → here 16), so it's the tone-smoothness knob; area
# beats Gaussian/Lanczos/soft-threshold resolves, which wash the texture or ring. Output
# tone is anti-aliased gray in [0,1], not 1-bit — also closer to a real scan.
_TONE_SS = 4
_TONE_ANGLE = 45.0  # screen rotation (deg)
_TONE_WHITE_CUT = 0.90  # luminance ≥ this → pure white (highlights, no tone)
_TONE_BLACK_CUT = 0.14  # luminance ≤ this → solid black (deep shadow)
# Screen patterns: real manga uses clustered dots, parallel-line ("sand"/hatch)
# tone, and cross-hatch. We assign a *different* pattern to each luminance band (see
# below), so the screen texture changes where the value (명암) changes — the manga
# convention — and the colorizer keys on tonal structure, not one fixed dot operator.
_TONE_KINDS = ("dot", "line", "cross")
# Per-image the toned range (between the white/black cuts) is split into this many
# luminance bands by value; each band gets its own pattern/angle/period so darks,
# mids, and lights carry distinct tones like a real page. 1 = single tone whole image.
_TONE_BAND_COUNTS = (1, 2, 3, 4)
_TONE_BAND_WEIGHTS = (0.10, 0.40, 0.35, 0.15)


def _luminance(img_rgb: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    """RGB uint8 → float32 luminance in [0, 1] with (jitterable) channel weights."""
    rgb = img_rgb.astype(np.float32) / 255.0
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    return np.clip(rgb @ w, 0.0, 1.0)


def _xdog(
    gray: np.ndarray,
    *,
    sigma: float,
    k: float,
    sharp: float,
    eps: float,
    phi: float,
) -> np.ndarray:
    """Winnemöller XDoG. Returns float32 in [0, 1]; ink ≈ 0, paper ≈ 1."""
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = (1.0 + sharp) * g1 - sharp * g2
    out = np.ones_like(dog)
    below = dog < eps
    out[below] = 1.0 + np.tanh(phi * (dog[below] - eps))
    return np.clip(out, 0.0, 1.0)


def _screen_field(
    h: int, w: int, *, period: float, angle: float, kind: str
) -> np.ndarray:
    """Periodic screen field in [0, 1] used as a spatially-varying ink threshold.

    ``dot`` = rotated clustered-dot screen (sin × sin, peaks at cell centres);
    ``line`` = a single rotated sinusoid → parallel-line / hatch tone; ``cross`` =
    two perpendicular line screens unioned → cross-hatch. In every case darker
    luminance falls below the field's peaks more often, so ink coverage grows with
    darkness — only the *texture* of that ink changes."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle)
    xr = xs * np.cos(a) - ys * np.sin(a)
    yr = xs * np.sin(a) + ys * np.cos(a)
    wx = 2.0 * np.pi * xr / period
    wy = 2.0 * np.pi * yr / period
    if kind == "line":
        return (np.sin(wx) + 1.0) * 0.5
    if kind == "cross":
        # ink where *either* stripe set is dark → union of two orthogonal screens
        return np.maximum((np.sin(wx) + 1.0) * 0.5, (np.sin(wy) + 1.0) * 0.5)
    return (np.sin(wx) * np.sin(wy) + 1.0) * 0.5  # "dot" (default)


def _pick_band_count(rng: np.random.Generator) -> int:
    """Seeded number of luminance bands (``_TONE_BAND_COUNTS``)."""
    return int(rng.choice(_TONE_BAND_COUNTS, p=_TONE_BAND_WEIGHTS))


def _band_kinds(n: int, rng: np.random.Generator) -> list[str]:
    """``n`` screen patterns, one per luminance band, distinct between neighbours.

    A shuffled cycle of ``_TONE_KINDS`` so adjacent value bands never share a pattern
    (the value→tone change stays legible); *which* pattern lands on which band is
    random per image, but it's fixed within the image."""
    perm = list(rng.permutation(_TONE_KINDS))
    return [str(perm[i % len(perm)]) for i in range(n)]


def _luma_band_labels(
    gray: np.ndarray, *, black_cut: float, white_cut: float, n: int
) -> np.ndarray:
    """Per-pixel luminance-band index in ``[0, n)`` (0 = darkest toned band).

    Band edges are quantiles of the *toned* pixels (those between the cuts), so every
    band is populated regardless of the image's histogram and the split tracks the
    actual 명암 distribution: darker value → lower band → its own screen. Pixels past
    the cuts are clamped here (they're overwritten by the solid-black / white flatten
    in :func:`_render_tone`)."""
    toned = gray[(gray > black_cut) & (gray < white_cut)]
    if n <= 1 or toned.size == 0:
        return np.zeros(gray.shape, np.int32)
    edges = np.percentile(toned, np.linspace(0.0, 100.0, n + 1)[1:-1])
    return np.clip(np.digitize(gray, edges), 0, n - 1).astype(np.int32)


def _band_plan(
    rng: np.random.Generator,
    *,
    n_bands: int | None,
    kind: str | None,
    angle: float | None,
    period: float,
    hatch_period: float,
) -> tuple[int, list[tuple[str, float, float]]]:
    """Seeded per-band screen plan: ``(n, [(kind, angle_deg, base_period), …])``.

    Consumes ``rng`` in the exact order the rendering loop needs (band count →
    per-band patterns → per-band angle then period), so a CPU and a GPU renderer
    fed the same seed pick the *same* structure — only the pixel math differs."""
    n = max(1, n_bands if n_bands is not None else _pick_band_count(rng))
    kinds = [kind] * n if kind is not None else _band_kinds(n, rng)
    plan: list[tuple[str, float, float]] = []
    for i in range(n):
        ki = kinds[i]
        base = period if ki == "dot" else hatch_period
        ra = angle if angle is not None else float(rng.uniform(0.0, 90.0))
        # jitter the period per band (only with >1) so the bands differ in scale too;
        # range kept tight so a band can't balloon back to a coarse screen
        rp = base if n == 1 else base * float(rng.uniform(0.85, 1.3))
        plan.append((str(ki), ra, rp))
    return n, plan


def _render_tone(
    gray: np.ndarray,
    *,
    rng: np.random.Generator,
    period: float,
    white_cut: float,
    black_cut: float,
    hatch_period: float | None = None,
    ss: int = _TONE_SS,
    kind: str | None = None,
    angle: float | None = None,
    n_bands: int | None = None,
) -> np.ndarray:
    """Luminance-band screentone. float32 in [0,1]; ink = 0, paper = 1.

    Splits the toned value range into ``n_bands`` luminance bands (random count unless
    pinned) and screens each band with its own pattern (dot/line/cross), angle, and
    period — so the texture changes where the value (명암) changes, the way a manga
    artist picks a different tone for darks, mids, and lights. Ink coverage still grows
    with darkness everywhere (the screen is a luminance threshold); the per-band
    pattern only changes *which* texture fills it. Highlights/shadows past the cuts are
    flattened (manga convention).

    Dots use ``period``; line/cross use the coarser ``hatch_period`` (defaults to 2× the
    dot period) — fine dots + clean coarse hatch, the real-page split. The screen is
    rendered at ``ss``× resolution, thresholded, then **area-downscaled**, so fine and
    rotated screens anti-alias into smooth gray rather than aliasing into coarse moiré;
    the returned tone is therefore continuous gray in [0,1], not 1-bit.

    ``kind`` / ``angle`` / ``n_bands`` pin those choices when given (QA / one fixed
    operator); otherwise they're drawn from the seeded ``rng``."""
    if hatch_period is None:
        hatch_period = period * 2.0
    h, w = gray.shape
    ss = max(1, int(ss))
    gs = (
        cv2.resize(gray, (w * ss, h * ss), interpolation=cv2.INTER_LINEAR)
        if ss > 1
        else gray
    )
    hs, ws = gs.shape
    n, plan = _band_plan(
        rng,
        n_bands=n_bands,
        kind=kind,
        angle=angle,
        period=period,
        hatch_period=hatch_period,
    )
    labels = _luma_band_labels(gs, black_cut=black_cut, white_cut=white_cut, n=n)
    tone = np.ones_like(gs)  # 1 = paper, 0 = ink
    for i, (ki, ra, rp) in enumerate(plan):
        screen = _screen_field(hs, ws, period=rp * ss, angle=ra, kind=ki)
        sel = labels == i
        tone[sel] = (gs[sel] >= screen[sel]).astype(np.float32)
    tone[gs >= white_cut] = 1.0
    tone[gs <= black_cut] = 0.0
    if ss > 1:  # area-average downscale → anti-aliased gray tone (kills moiré)
        tone = cv2.resize(tone, (w, h), interpolation=cv2.INTER_AREA)
    return tone


def resolve_params(
    img_rgb: np.ndarray, rng: np.random.Generator, overrides: dict
) -> dict:
    """Resolve all seeded/overridable knobs into a flat dict (backend-agnostic).

    Draws the per-image jitter from ``rng`` so any backend (numpy or torch) that
    is handed the *same* generator picks identical knobs — the GPU port reuses
    this verbatim and only swaps out the pixel math. Returns the scalars plus the
    raw ``overrides`` passthrough for ``ss`` / ``tone_kind`` / ``angle`` /
    ``n_bands`` (consumed later by the band plan)."""
    # Scale the line/dot size with the image's short side so a fixed period
    # doesn't turn into mush on small images or vanish on large ones.
    short = float(min(img_rgb.shape[:2]))
    scale = max(0.5, short / 900.0)
    return dict(
        sigma=overrides.get(
            "sigma", _XDOG_SIGMA * scale * float(rng.uniform(0.8, 1.25))
        ),
        k=overrides.get("k", _XDOG_K),
        sharp=overrides.get("sharp", _XDOG_SHARP * float(rng.uniform(0.8, 1.2))),
        eps=overrides.get("eps", _XDOG_EPS),
        phi=overrides.get("phi", _XDOG_PHI),
        period=overrides.get(
            "period", _TONE_PERIOD * scale * float(rng.uniform(0.9, 1.2))
        ),
        hatch_period=overrides.get(
            "hatch_period", _HATCH_PERIOD * scale * float(rng.uniform(0.9, 1.2))
        ),
        white_cut=overrides.get("white_cut", _TONE_WHITE_CUT),
        black_cut=overrides.get("black_cut", _TONE_BLACK_CUT),
        # jitter luma weights around BT.601 so the cond isn't a single fixed operator
        luma_weights=overrides.get(
            "luma_weights",
            (
                0.299 * float(rng.uniform(0.85, 1.15)),
                0.587 * float(rng.uniform(0.9, 1.1)),
                0.114 * float(rng.uniform(0.7, 1.3)),
            ),
        ),
        ss=int(overrides.get("ss", _TONE_SS)),
        tone_kind=overrides.get("tone_kind"),
        angle=overrides.get("angle"),
        n_bands=overrides.get("n_bands"),
    )


def mangafy_array(
    img_rgb: np.ndarray,
    *,
    seed: int | None = None,
    **overrides,
) -> np.ndarray:
    """Color RGB uint8 ``(H, W, 3)`` → mangafied B&W RGB uint8 ``(H, W, 3)``.

    ``seed`` (e.g. a stable hash of the image stem) drives reproducible per-image
    jitter of the XDoG/screen knobs. Any knob can be pinned via ``overrides``
    (``sigma``, ``k``, ``sharp``, ``eps``, ``phi``, ``period``, ``hatch_period``,
    ``ss``, ``angle``, ``white_cut``, ``black_cut``, ``tone_kind``, ``n_bands``,
    ``luma_weights``).
    Pinning ``tone_kind`` / ``angle`` / ``n_bands`` forces a single fixed screen
    (e.g. ``n_bands=1, tone_kind="line"`` for QA)."""
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    img_rgb = img_rgb[:, :, :3]

    rng = np.random.default_rng(seed)
    p = resolve_params(img_rgb, rng, overrides)

    gray = _luminance(img_rgb, p["luma_weights"])
    line = _xdog(
        gray, sigma=p["sigma"], k=p["k"], sharp=p["sharp"], eps=p["eps"], phi=p["phi"]
    )
    tone = _render_tone(
        gray,
        rng=rng,
        period=p["period"],
        hatch_period=p["hatch_period"],
        ss=p["ss"],
        white_cut=p["white_cut"],
        black_cut=p["black_cut"],
        kind=p["tone_kind"],
        angle=p["angle"],
        n_bands=p["n_bands"],
    )
    # ink wherever either the line or the screen is dark
    manga = np.minimum(line, tone)
    out = (np.clip(manga, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.stack([out] * 3, axis=-1)


def mangafy_pil(pil_image, *, seed: int | None = None, **overrides):
    """PIL convenience wrapper around :func:`mangafy_array`."""
    from PIL import Image

    arr = np.array(pil_image.convert("RGB"))
    return Image.fromarray(mangafy_array(arr, seed=seed, **overrides))
