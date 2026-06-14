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
                      bench evaluation (real deployments receive ``y`` directly).

Phase 0 ships ``sr`` only. ``inpaint`` / ``colorize`` / ``deblur`` are the
Phase 3/4 pilots (proposal) and slot in as siblings here.
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

    def adjoint_init(self, y: torch.Tensor, *, target_hw: tuple[int, int]) -> torch.Tensor:
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


def build_operator(task: str, *, scale: int, sigma_nu: float):
    if task == "sr":
        return SROperator(scale=scale, sigma_nu=sigma_nu)
    raise ValueError(
        f"Phase 0 implements only task='sr'; got {task!r}. "
        "inpaint/colorize/deblur are Phase 3/4 (docs/proposal/flair_inverse.md)."
    )
