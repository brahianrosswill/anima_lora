"""FreeText Stage-2 — Spectral-Modulated Glyph Injection (SGMI, paper §3.2).

The *pure* injection machinery (PIL + torch.fft, no model dependency), mirroring
``stage1.py``'s split. The driver (``stage2_sgmi.py``) loads the DiT/VAE, derives
the writing mask ``R`` from Stage-1, renders + VAE-encodes the glyph reference,
and installs an :class:`SGMIInjector` into the sampler boundary.

The three sub-stages, with their paper equations:

  3.2.1  Noise-aligned latent projection   z_ref^(t) = α_t z_ref + σ_t ε   (Eq 7-8)
  3.2.2  Log-Gabor spectral modulation      Ĝ·F(z_ref^(t)) → IFFT          (Eq 9-11)
  3.2.3  Annealed spatiotemporal injection  z̃ = (1-λR)⊙z + λR⊙z_sgmi        (Eq 12-14)

Anima specifics (the load-bearing adaptations vs. the paper's DDPM notation):

* **Flow-matching noise alignment.** Anima is a rectified-flow model: the
  sampler's latent at sigma σ is ``z_σ = (1-σ)·z0 + σ·ε`` (verified from the
  Euler step ``denoised = z - σ·v`` in ``library/inference/sampling.py``). So the
  paper's ``(α_t, σ_t)`` maps to ``(1-σ, σ)`` and Eq 8 becomes
  ``z_ref^(σ) = (1-σ)·z_ref + σ·ε``. σ runs 1→0 over the trajectory, so the
  injection window ``t∈[0.8T, 0.6T]`` is ``σ∈[0.6, 0.8]`` (planted hard at the
  high-σ edge, annealed to zero by σ=0.6 — *before* Anima's detail-resolving tail
  at σ≲0.45, leaving the denoiser room to develop the strokes it was handed).

* **Same normalized latent space.** ``vae.encode_pixels_to_latents`` returns the
  mean/std-normalized latent the sampler operates in (decode is its inverse), so
  the glyph latent is on-scale for the masked blend — no extra rescaling.

This module is the honest test of the paper's own caveat (§4.2.3): SGMI is "most
effective when the base model already has a usable representation for the target
characters; it strengthens glyph structure rather than enabling unseen characters
from scratch." Korean-on-Anima (squiggles, never cleanly drawn) is exactly the
out-of-distribution case the paper did *not* claim to solve.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Noto Sans CJK KR (covers Hangul + Latin). fc-match resolves the .ttc face index.
_DEFAULT_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_DEFAULT_FONT_INDEX = 1  # KR face inside the .ttc


def resolve_font(family: str = "Noto Sans CJK KR") -> tuple[str, int]:
    """Resolve a font family to ``(path, ttc_index)`` via fontconfig, with a
    hard-coded Noto CJK fallback so the bench runs on a bare box."""
    try:
        out = subprocess.run(
            ["fc-match", "-f", "%{file}:%{index}", family],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        path, idx = out.rsplit(":", 1)
        if Path(path).exists():
            return path, int(idx)
    except Exception:
        pass
    return _DEFAULT_FONT, _DEFAULT_FONT_INDEX


# ---------------------------------------------------------------------------
# 3.2.1 (part)  Glyph rasterization — render the target string into region R.
# ---------------------------------------------------------------------------
def _fit_font(text: str, box_w: int, box_h: int, font_path: str, font_index: int,
              max_lines: int = 3) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Largest font size (and line wrap) that fits ``text`` in ``box_w × box_h``.

    Short strings stay on one line; long phrases wrap to ≤``max_lines`` so the
    glyphs do not shrink to illegible specks inside a tall, narrow region.
    """
    def wrap(n_lines: int) -> list[str]:
        if n_lines <= 1 or " " not in text:
            return [text]
        words = text.split(" ")
        per = max(1, len(words) // n_lines + (1 if len(words) % n_lines else 0))
        return [" ".join(words[i:i + per]) for i in range(0, len(words), per)]

    best = None
    for n_lines in range(1, max_lines + 1):
        lines = wrap(n_lines)
        if len(lines) > n_lines:
            continue
        # Binary-search the point size that fits both width and height.
        lo, hi = 6, max(8, box_h)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            font = ImageFont.truetype(font_path, mid, index=font_index)
            widths, heights = [], []
            for ln in lines:
                bb = font.getbbox(ln)
                widths.append(bb[2] - bb[0])
                heights.append(bb[3] - bb[1])
            total_h = int(sum(heights) * 1.25)
            if max(widths) <= box_w and total_h <= box_h:
                lo = mid
            else:
                hi = mid - 1
        font = ImageFont.truetype(font_path, lo, index=font_index)
        # Prefer the wrap that yields the largest size.
        if best is None or lo > best[2]:
            best = (font, lines, lo)
    return best[0], best[1]


def render_glyph_image(
    text: str,
    region_box_px: tuple[int, int, int, int],
    canvas_hw: tuple[int, int],
    *,
    font_path: str | None = None,
    font_index: int | None = None,
    fg: float = 1.0,
    bg: float = 0.0,
    pad_frac: float = 0.08,
) -> np.ndarray:
    """Rasterize ``text`` into the pixel-space ``region_box_px`` (x0,y0,x1,y1) of
    a ``canvas_hw`` (H,W) image. Returns float32 RGB in [0,1] (default white-on-
    black). The glyphs are centered and sized to fill the box (minus ``pad_frac``).
    """
    if font_path is None:
        font_path, font_index = resolve_font()
    elif font_index is None:
        font_index = 0
    H, W = canvas_hw
    x0, y0, x1, y1 = region_box_px
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
    pad_w, pad_h = int(box_w * pad_frac), int(box_h * pad_frac)
    fit_w, fit_h = max(1, box_w - 2 * pad_w), max(1, box_h - 2 * pad_h)

    font, lines = _fit_font(text, fit_w, fit_h, font_path, font_index)

    img = Image.new("L", (W, H), int(bg * 255))
    draw = ImageDraw.Draw(img)
    line_h = [font.getbbox(ln)[3] - font.getbbox(ln)[1] for ln in lines]
    gap = int(max(line_h) * 0.25) if lines else 0
    total_h = sum(line_h) + gap * (len(lines) - 1)
    cy = (y0 + y1) / 2 - total_h / 2
    yy = cy
    for ln, lh in zip(lines, line_h):
        bb = font.getbbox(ln)
        lw = bb[2] - bb[0]
        xx = (x0 + x1) / 2 - lw / 2 - bb[0]
        draw.text((xx, yy - bb[1]), ln, fill=int(fg * 255), font=font)
        yy += lh + gap

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.repeat(arr[:, :, None], 3, axis=2)


# ---------------------------------------------------------------------------
# 3.2.2  Log-Gabor spectral modulation (Eq 9-11)
# ---------------------------------------------------------------------------
def log_gabor_radial(
    shape: tuple[int, int], f0: float = 0.25, sigma_ratio: float = 0.65,
    device=None, dtype=torch.float32,
) -> torch.Tensor:
    """Radial (orientation-agnostic) Log-Gabor band-pass filter on the 2D DFT grid.

    ``G(ρ) = exp(-ln(ρ/f0)^2 / (2 ln(σ_ratio)^2))`` (Field 1987), DC zeroed. ``ρ``
    is normalized so Nyquist = 1.0. ``f0`` = center frequency (mid-band carries
    glyph strokes), ``sigma_ratio`` = bandwidth (σ_f/f0; smaller = narrower band).
    Orientation-agnostic because glyph stroke energy spans all orientations; the
    paper's ``G(ρ,θ)`` reduces to this when summed over an even orientation bank.
    """
    H, W = shape
    fy = torch.fft.fftfreq(H, device=device, dtype=dtype).view(H, 1)
    fx = torch.fft.fftfreq(W, device=device, dtype=dtype).view(1, W)
    rho = torch.sqrt(fy * fy + fx * fx) / 0.5  # normalize Nyquist (0.5 cyc/px) → 1
    rho = rho.clamp_min(1e-6)
    G = torch.exp(-(torch.log(rho / f0) ** 2) / (2.0 * (np.log(sigma_ratio) ** 2)))
    G[0, 0] = 0.0  # kill DC (low-freq background / "semantic leakage")
    return G


def apply_log_gabor(z: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
    """Per-channel 2D-FFT band-pass: z [C,H,W] → real(IFFT(G·FFT(z))). (Eq 9-11)"""
    Z = torch.fft.fft2(z.float())
    Zf = Z * filt.to(Z.device)
    return torch.fft.ifft2(Zf).real.to(z.dtype)


# ---------------------------------------------------------------------------
# 3.2.1 (part)  Flow-matching noise alignment (Eq 8)
# ---------------------------------------------------------------------------
def noise_align(z0: torch.Tensor, sigma: float, generator=None) -> torch.Tensor:
    """``z_σ = (1-σ)·z0 + σ·ε`` — Anima's rectified-flow forward to noise level σ
    (the flow-matching form of the paper's ``α_t z_ref + σ_t ε``)."""
    eps = torch.randn(z0.shape, dtype=torch.float32, device=z0.device,
                      generator=generator)
    return (1.0 - sigma) * z0.float() + sigma * eps


# ---------------------------------------------------------------------------
# 3.2.3  Annealed spatiotemporal injection (Eq 12-14)
# ---------------------------------------------------------------------------
def cosine_lambda(sigma: float, sigma_start: float, sigma_end: float) -> float:
    """Eq 13 — cosine-annealed weight. λ=1 at the high-σ window edge (``sigma_start``,
    hard plant) → λ=0 at ``sigma_end`` (free develop). 0 outside [sigma_end, sigma_start]."""
    if not (sigma_end <= sigma <= sigma_start):
        return 0.0
    frac = (sigma_start - sigma) / max(sigma_start - sigma_end, 1e-9)
    return 0.5 * (1.0 + np.cos(np.pi * frac))


@dataclass
class SGMIInjector:
    """Sampler-boundary masked-replace injector (Eq 14).

    Pre-computes the glyph latent ``z_ref`` (optionally Log-Gabor pre-filtered in
    the clean domain) once; per step it noise-aligns to the current σ, optionally
    Log-Gabor filters the noisy latent (paper-faithful), and blends into ``R`` with
    the cosine-annealed weight. ``apply`` is called from the monkeypatched Euler
    step with the *next* latent ``z^(σ_next)`` so the following forward denoises it.
    """

    z_ref: torch.Tensor          # clean glyph latent [1,C,H,W] (sampler-normalized)
    mask: torch.Tensor           # writing mask R [1,1,H,W] float {0,1}
    sigma_start: float = 0.8
    sigma_end: float = 0.6
    lam_scale: float = 1.0       # global multiplier on λ (1.0 = paper)
    anneal: str = "cosine"       # "cosine" (Eq 13) | "flat" (λ=lam_scale in window)
    use_log_gabor: bool = True
    lg_order: str = "post"       # "post" = filter noisy z_σ (paper Eq 9-11);
                                 # "pre"  = filter clean z_ref once, then noise-align
    f0: float = 0.25
    sigma_ratio: float = 0.65
    seed: int = 1234
    _filt: torch.Tensor = field(default=None, repr=False)
    _gen: torch.Generator = field(default=None, repr=False)
    log: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.z_ref.ndim != 4:
            raise ValueError(f"z_ref must be [B,C,H,W], got {tuple(self.z_ref.shape)}")
        if self.mask.ndim != 4:
            raise ValueError(f"mask must be [B,1,H,W], got {tuple(self.mask.shape)}")
        if self.z_ref.shape[-2:] != self.mask.shape[-2:]:
            raise ValueError(
                f"z_ref/mask spatial mismatch: {tuple(self.z_ref.shape)} vs {tuple(self.mask.shape)}"
            )
        dev = self.z_ref.device
        self._gen = torch.Generator(device=dev).manual_seed(self.seed)
        if self.use_log_gabor:
            self._filt = log_gabor_radial(
                self.z_ref.shape[-2:], f0=self.f0, sigma_ratio=self.sigma_ratio,
                device=dev,
            )
            if self.lg_order == "pre":
                # Filter the clean glyph once; per-step we only noise-align.
                self.z_ref = torch.stack(
                    [apply_log_gabor(self.z_ref[0], self._filt)]
                )

    def lam(self, sigma: float) -> float:
        if not (self.sigma_end <= sigma <= self.sigma_start):
            return 0.0
        if self.anneal == "flat":
            return self.lam_scale
        return self.lam_scale * cosine_lambda(sigma, self.sigma_start, self.sigma_end)

    @staticmethod
    def _match_latent_ndim(x: torch.Tensor, z_next: torch.Tensor, name: str) -> torch.Tensor:
        if x.ndim == z_next.ndim:
            return x
        if z_next.ndim == 5 and x.ndim == 4:
            return x.unsqueeze(2)
        raise ValueError(
            f"{name} rank {x.ndim} is incompatible with sampler latent rank {z_next.ndim}: "
            f"{tuple(x.shape)} vs {tuple(z_next.shape)}"
        )

    def apply(self, z_next: torch.Tensor, sigma_next: float) -> torch.Tensor:
        """Eq 14 masked replacement at noise level ``sigma_next``. No-op outside
        the window. Returns same dtype as ``z_next``."""
        lam = self.lam(sigma_next)
        if lam <= 0.0:
            return z_next
        z_sigma = noise_align(self.z_ref[0], sigma_next, self._gen).unsqueeze(0)
        if self.use_log_gabor and self.lg_order == "post":
            z_sigma = torch.stack([apply_log_gabor(z_sigma[0], self._filt)])
        z_sigma = self._match_latent_ndim(z_sigma, z_next, "z_sigma")
        m = self._match_latent_ndim(self.mask.to(z_next.device), z_next, "mask") * lam
        out_shape = torch.broadcast_shapes(z_next.shape, z_sigma.shape, m.shape)
        if out_shape != z_next.shape:
            raise ValueError(
                "SGMI broadcast would change latent shape: "
                f"z_next={tuple(z_next.shape)} z_sigma={tuple(z_sigma.shape)} mask={tuple(m.shape)}"
            )
        out = (1.0 - m) * z_next.float() + m * z_sigma.float()
        self.log.append({"sigma": round(float(sigma_next), 4), "lam": round(float(lam), 4)})
        return out.to(z_next.dtype)


# ---------------------------------------------------------------------------
# Mask geometry helpers (latent ↔ pixel).
# ---------------------------------------------------------------------------
def mask_bbox(mask_hw: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box (x0,y0,x1,y1) of a boolean/uint mask, or None if empty."""
    ys, xs = np.nonzero(np.asarray(mask_hw) > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def latent_box_to_pixel(box, h_lat, w_lat, H, W):
    """Scale a latent-grid bbox to pixel coordinates."""
    x0, y0, x1, y1 = box
    sx, sy = W / w_lat, H / h_lat
    return int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)
