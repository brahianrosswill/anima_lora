"""Forward operators ``A`` for FLAIR inverse problems (pixel space, [-1, 1]).

FLAIR's data term is ``‖y − A(D(μ))‖²`` where ``D`` is the VAE decoder, so every
operator here works on **decoded pixels** in the VAE's [-1, 1] range, NOT on
latents. Each operator exposes:

  - ``degrade(x)``  — the forward map ``A`` (differentiable; used by the HDC
                      projection, which backprops through ``A ∘ D``).
  - ``adjoint_init(y)`` — an approximate ``A^T`` lifting the observation back to
                      the target resolution, used ONLY for the variational
                      mean's adjoint initialization ``μ = E(A^T y)``.
  - ``measure(x_gt)`` — synthesize a noisy observation ``y = A(x_gt) + ν`` for
                      bench / observation evaluation (real deployments receive
                      ``y`` directly).

``sr`` is the Phase-0 SR operator (promoted from the bench unchanged).
``inpaint`` is the binary-mask operator that powers FLAIR-edit
(``docs/proposal/flair_edit.md``): ``A = m ⊙ ·`` is self-adjoint, so the adjoint
init is just the masked observation (a flat / mid-gray hole in [-1, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SROperator:
    """Bicubic ×``scale`` super-resolution degradation.

    ``A`` = antialiased bicubic downsample (the paper's setting). The HDC step
    differentiates through this; the explicit adjoint is only the upsample used
    to seed ``μ`` — an *approximate* transpose, which is all the adjoint init
    needs (FLAIR refines it; it is not assumed exact).
    """

    scale: int = 8
    sigma_nu: float = 0.01  # measurement-noise std in [-1, 1] space (0.5% of range)

    def degrade(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, H, W] in [-1, 1]; H, W divisible by scale.
        return F.interpolate(
            x, scale_factor=1.0 / self.scale, mode="bicubic", antialias=True
        )

    def adjoint_init(
        self, y: torch.Tensor, *, target_hw: tuple[int, int]
    ) -> torch.Tensor:
        # Lift the low-res observation back to the target grid (approx A^T).
        return F.interpolate(y, size=target_hw, mode="bicubic", align_corners=False)

    def measure(
        self, x_gt: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        y = self.degrade(x_gt)
        if self.sigma_nu > 0:
            noise = torch.randn(
                y.shape, generator=generator, device=y.device, dtype=y.dtype
            )
            y = y + self.sigma_nu * noise
        return y.clamp(-1.0, 1.0)


@dataclass
class InpaintOperator:
    """Binary-mask inpainting — the operator behind FLAIR-edit.

    ``A(x) = m ⊙ x`` where ``m`` is the **keep mask** (1 = observed/locked pixel,
    0 = region to fill). The mask is its own transpose (a diagonal 0/1 projection
    is symmetric and idempotent), so ``A = A^T`` and the adjoint init is simply
    the masked observation — the fill region reads as the flat value 0, i.e. a
    mid-gray hole in [-1, 1] (matching the proposal's "gray hole encodes flat").

    ``mask`` is ``[1, 1, H, W]`` (or broadcastable to ``[B, 3, H, W]``) in
    ``{0, 1}``; it broadcasts across the 3 colour channels.
    """

    mask: torch.Tensor  # [1, 1, H, W] keep-mask in {0, 1}
    sigma_nu: float = 0.0  # editing receives a clean observation by default

    def _m(self, x: torch.Tensor) -> torch.Tensor:
        return self.mask.to(device=x.device, dtype=x.dtype)

    def degrade(self, x: torch.Tensor) -> torch.Tensor:
        # m ⊙ x — zero out the fill region, keep the rest bit-for-bit.
        return self._m(x) * x

    def adjoint_init(
        self, y: torch.Tensor, *, target_hw: tuple[int, int]
    ) -> torch.Tensor:
        # Self-adjoint: A^T y = m ⊙ y = y (y is already masked & full-res). The
        # masked region is 0 → flat mid-gray hole the VAE encodes as a neutral
        # latent for μ to refine. target_hw is accepted for a uniform call site.
        return self._m(y) * y

    def measure(
        self, x_gt: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        y = self.degrade(x_gt)
        if self.sigma_nu > 0:
            noise = self._m(y) * torch.randn(
                y.shape, generator=generator, device=y.device, dtype=y.dtype
            )
            y = y + self.sigma_nu * noise
        return y.clamp(-1.0, 1.0)


def build_operator(
    task: str,
    *,
    scale: int = 8,
    sigma_nu: float = 0.01,
    mask: torch.Tensor | None = None,
):
    """Construct a FLAIR forward operator by task name.

    ``sr`` → :class:`SROperator` (``scale`` / ``sigma_nu``); ``inpaint`` →
    :class:`InpaintOperator` (requires ``mask``, a keep-mask in ``{0, 1}``).
    ``colorize`` / ``deblur`` are the remaining Phase-4 pilots and slot in here.
    """
    if task == "sr":
        return SROperator(scale=scale, sigma_nu=sigma_nu)
    if task == "inpaint":
        if mask is None:
            raise ValueError("build_operator('inpaint') requires a keep-mask tensor.")
        return InpaintOperator(mask=mask, sigma_nu=sigma_nu)
    raise ValueError(
        f"Unknown FLAIR task {task!r}; implemented: 'sr', 'inpaint'. "
        "colorize/deblur are Phase 4 (docs/proposal/flair_inverse.md)."
    )
