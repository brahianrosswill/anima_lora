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
   with an algorithmic clustered-dot halftone (a rotated dot screen thresholded
   against luminance). Brightest band stays white, darkest stays solid black —
   matching how manga keeps highlights white and applies tone only to shadows.
3. **Composite** line over tone, emit as 3-channel RGB (the Qwen VAE is RGB-only).

Per-image jitter (seeded by stem) varies the XDoG sigma, screen period, dot
angle, and luma weights so the model keys on *structure*, not one fixed halftone
operator — and so it's robust to whatever real screentone it meets at inference.

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
_TONE_PERIOD = 5.0  # dot screen period (px) — ~ manga screentone LPI at ~900px
_TONE_ANGLE = 45.0  # dot screen rotation (deg)
_TONE_WHITE_CUT = 0.86  # luminance ≥ this → pure white (highlights, no tone)
_TONE_BLACK_CUT = 0.14  # luminance ≤ this → solid black (deep shadow)


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


def _screentone(
    gray: np.ndarray,
    *,
    period: float,
    angle: float,
    white_cut: float,
    black_cut: float,
) -> np.ndarray:
    """Clustered-dot halftone. Returns float32 in {0,1}-ish; ink = 0, paper = 1.

    A rotated dot field acts as a spatially-varying threshold: darker luminance
    falls below the dot peaks more often → denser black dots; lighter luminance
    stays mostly white. Highlights/shadows past the cuts are flattened so faces
    stay clean and deep shadows go solid (manga convention)."""
    h, w = gray.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    a = np.deg2rad(angle)
    xr = xs * np.cos(a) - ys * np.sin(a)
    yr = xs * np.sin(a) + ys * np.cos(a)
    # dot field in [0,1], peaks at cell centres
    screen = (np.sin(2.0 * np.pi * xr / period) * np.sin(2.0 * np.pi * yr / period) + 1.0) * 0.5
    tone = (gray >= screen).astype(np.float32)  # 1 = paper, 0 = ink
    tone[gray >= white_cut] = 1.0
    tone[gray <= black_cut] = 0.0
    return tone


def mangafy_array(
    img_rgb: np.ndarray,
    *,
    seed: int | None = None,
    **overrides,
) -> np.ndarray:
    """Color RGB uint8 ``(H, W, 3)`` → mangafied B&W RGB uint8 ``(H, W, 3)``.

    ``seed`` (e.g. a stable hash of the image stem) drives reproducible per-image
    jitter of the XDoG/screen knobs. Any knob can be pinned via ``overrides``
    (``sigma``, ``k``, ``sharp``, ``eps``, ``phi``, ``period``, ``angle``,
    ``white_cut``, ``black_cut``, ``luma_weights``)."""
    if img_rgb.ndim == 2:
        img_rgb = np.stack([img_rgb] * 3, axis=-1)
    img_rgb = img_rgb[:, :, :3]

    rng = np.random.default_rng(seed)

    # Scale the line/dot size with the image's short side so a fixed period
    # doesn't turn into mush on small images or vanish on large ones.
    short = float(min(img_rgb.shape[:2]))
    scale = max(0.5, short / 900.0)

    sigma = overrides.get("sigma", _XDOG_SIGMA * scale * float(rng.uniform(0.8, 1.25)))
    k = overrides.get("k", _XDOG_K)
    sharp = overrides.get("sharp", _XDOG_SHARP * float(rng.uniform(0.8, 1.2)))
    eps = overrides.get("eps", _XDOG_EPS)
    phi = overrides.get("phi", _XDOG_PHI)
    period = overrides.get("period", _TONE_PERIOD * scale * float(rng.uniform(0.85, 1.3)))
    angle = overrides.get("angle", float(rng.uniform(0.0, 90.0)))
    white_cut = overrides.get("white_cut", _TONE_WHITE_CUT)
    black_cut = overrides.get("black_cut", _TONE_BLACK_CUT)
    # jitter luma weights around BT.601 so the cond isn't a single fixed operator
    luma_weights = overrides.get(
        "luma_weights",
        (
            0.299 * float(rng.uniform(0.85, 1.15)),
            0.587 * float(rng.uniform(0.9, 1.1)),
            0.114 * float(rng.uniform(0.7, 1.3)),
        ),
    )

    gray = _luminance(img_rgb, luma_weights)
    line = _xdog(gray, sigma=sigma, k=k, sharp=sharp, eps=eps, phi=phi)
    tone = _screentone(
        gray, period=period, angle=angle, white_cut=white_cut, black_cut=black_cut
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
