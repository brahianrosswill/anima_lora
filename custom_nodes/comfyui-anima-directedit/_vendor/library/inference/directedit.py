"""DirectEdit (Yang & Ye, arXiv:2605.02417v1) — flow-based image editing primitive.

Two-pass training-free editor for flow-matching DiTs:

1. **Inversion** (clean -> noise): step backward through the same Euler ODE
   the generator runs forward, querying v_θ at each step's input. Record
   per-step residuals ``ΔZ_i = Z_inv[i+1] − Z_inv[i]`` — these are the
   "anchor" the paper uses to make reconstruction bit-exact instead of
   trying to rectify the inversion path itself.

2. **Editing** (noise -> clean): standard generation loop, but every model
   call is queried at ``Z[i] + ΔZ[i]`` instead of ``Z[i]``. The cross-attn
   prompt is the edit target ψ_tar; the residual ΔZ pins the trajectory to
   the source. Run a parallel src-stream with ψ_src for V-injection / mask
   blending (both deferred to v2 — left as hookable args here).

Anima conventions used:
  * sigmas[0] = 1 (pure noise), sigmas[T] = 0 (clean), per
    ``library/inference/sampling.py::get_timesteps_sigmas``.
  * Latents: 5D ``[B, C, 1, H/8, W/8]`` (frame dim of 1 — image, not video).
  * The model's call signature matches what ``generate_body`` uses:
    ``anima(latents, t_expand, embed, padding_mask=...)`` where ``embed`` is
    already-preprocessed crossattn (post-T5, 512-padded).

This module is self-contained: ``invert`` and ``edit_forward`` accept the
already-loaded ``Anima`` model and pre-encoded ``ψ_src`` / ``ψ_tar`` embeds,
so the calling script (``scripts/edit.py``) can reuse the existing TE/VAE/DiT
loaders from ``library.inference.{models,text}``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from library.anima import models as anima_models

logger = logging.getLogger(__name__)


def _padding_mask_for(latents: torch.Tensor) -> torch.Tensor:
    """Anima expects a (B, 1, H_lat, W_lat) zero mask for non-padded inputs."""
    bs = latents.shape[0]
    h_lat = latents.shape[-2]
    w_lat = latents.shape[-1]
    return torch.zeros(bs, 1, h_lat, w_lat, dtype=torch.bfloat16, device=latents.device)


@torch.no_grad()
def _v_pred(
    anima: anima_models.Anima,
    latents: torch.Tensor,
    sigma: torch.Tensor,
    embed: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    guidance_scale: float,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """One model forward (with optional CFG). Returns velocity prediction."""
    t_expand = sigma.expand(latents.shape[0]).to(latents.device, dtype=torch.bfloat16)
    noise_pred = anima(latents, t_expand, embed, padding_mask=padding_mask)
    if guidance_scale != 1.0 and embed_neg is not None:
        uncond = anima(latents, t_expand, embed_neg, padding_mask=padding_mask)
        noise_pred = uncond + guidance_scale * (noise_pred - uncond)
    return noise_pred


@torch.no_grad()
def invert(
    anima: anima_models.Anima,
    z_clean: torch.Tensor,
    embed_src: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    sigmas: torch.Tensor,
    guidance_scale: float = 1.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Invert ``z_clean`` (= VAE-encoded source image) along the Anima ODE.

    Returns ``(z_inv, delta_z)``:
      * ``z_inv``: list of length T+1, where ``z_inv[T] == z_clean`` (cast/dtype
        match the input) and ``z_inv[0]`` is the maximally-noised inversion.
      * ``delta_z``: list of length T, with ``delta_z[i] = z_inv[i+1] − z_inv[i]``
        — the anchor residuals consumed by ``edit_forward``.

    Inversion convention (paper §3.2 in our index):
        ``z_inv[i] = z_inv[i+1] + (sigmas[i] − sigmas[i+1]) · v_θ(z_inv[i+1], σ=sigmas[i+1])``
    iterated for ``i = T-1 .. 0``.

    CFG during inversion is usually a wash (the source has no negative concept
    to push away from). Default ``guidance_scale=1.0`` skips it. Pass >1.0
    only if you want the inverted noise to land where re-generation with the
    same CFG would put it.
    """
    device = z_clean.device
    T = sigmas.shape[0] - 1
    padding_mask = _padding_mask_for(z_clean)

    z_inv: List[torch.Tensor] = [None] * (T + 1)  # type: ignore[list-item]
    z_inv[T] = z_clean.to(torch.bfloat16)

    delta_z: List[torch.Tensor] = [None] * T  # type: ignore[list-item]

    iterator = tqdm(range(T - 1, -1, -1), desc="DirectEdit inversion", total=T)
    for i in iterator:
        sigma_in = sigmas[i + 1].to(device)  # σ of the input we feed v_θ
        v = _v_pred(
            anima,
            z_inv[i + 1],
            sigma_in,
            embed_src,
            embed_neg,
            guidance_scale,
            padding_mask,
        )
        # z_inv[i] is at higher noise; (sigmas[i] - sigmas[i+1]) > 0 in our index.
        coeff = (sigmas[i] - sigmas[i + 1]).to(device, dtype=torch.float32)
        z_inv[i] = (z_inv[i + 1].float() + coeff * v.float()).to(torch.bfloat16)
        delta_z[i] = (z_inv[i + 1].float() - z_inv[i].float()).to(torch.bfloat16)

    return z_inv, delta_z


@torch.no_grad()
def edit_forward(
    anima: anima_models.Anima,
    z_init: torch.Tensor,
    delta_z: List[torch.Tensor],
    embed_tar: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    sigmas: torch.Tensor,
    guidance_scale: float = 4.0,
    embed_src: Optional[torch.Tensor] = None,  # noqa: ARG001 — V-injection hook (v2)
    t_inj: int = 0,  # noqa: ARG001 — number of early steps to inject src V (v2)
    mask: Optional[torch.Tensor] = None,  # noqa: ARG001 — background-lock (v2)
) -> torch.Tensor:
    """Forward (noise -> clean) edit pass anchored to the inversion residuals.

    Step rule (paper §3.2 in our index):
        ``ẑ_i = z[i] + delta_z[i]                        # anchor``
        ``v_i = v_θ(ẑ_i, σ=sigmas[i], ψ_tar)             # query at anchored pt``
        ``z[i+1] = z[i] − (sigmas[i] − sigmas[i+1]) · v_i # standard Euler step``

    Notes:
      * ``z_init`` should be ``z_inv[0]`` from ``invert(...)`` for the
        residual trick to fire correctly.
      * ``embed_src``/``t_inj``/``mask`` are accepted (and silenced for now via
        ``noqa: ARG001``) so the v2 V-injection + background-lock can be
        wired without changing the call signature. v1 behavior matches the
        paper at ``t_inj=0, mask=None``: pure ΔZ-anchored edit, no parallel
        src stream.
    """
    device = z_init.device
    T = sigmas.shape[0] - 1
    if len(delta_z) != T:
        raise ValueError(
            f"delta_z has length {len(delta_z)} but sigmas implies T={T} steps "
            "— inversion and editing must use the same sigma schedule."
        )
    padding_mask = _padding_mask_for(z_init)

    z = z_init.to(torch.bfloat16)
    iterator = tqdm(range(T), desc="DirectEdit editing", total=T)
    for i in iterator:
        z_hat = (z.float() + delta_z[i].to(device).float()).to(torch.bfloat16)
        sigma_in = sigmas[i].to(device)
        v = _v_pred(
            anima,
            z_hat,
            sigma_in,
            embed_tar,
            embed_neg,
            guidance_scale,
            padding_mask,
        )
        coeff = (sigmas[i] - sigmas[i + 1]).to(device, dtype=torch.float32)
        z = (z.float() - coeff * v.float()).to(torch.bfloat16)

    return z
