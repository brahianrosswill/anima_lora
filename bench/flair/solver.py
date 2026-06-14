"""FLAIR Algorithm 1 — variational posterior solve on the Anima flow prior.

Self-contained port of *Solving Inverse Problems with FLAIR* (Erbach et al.,
NeurIPS 2025), Algorithm 1, mapped onto Anima's flow-matching DiT + Qwen VAE.
Lives in ``bench/`` (not yet wired into the inference engine) so Phase 0 can
validate the port before promotion — see ``docs/proposal/flair_inverse.md``.

Convention mapping (verified against ``library/inference/sampling.py`` and the
``generate_body`` DiT call):

  - FLAIR's interpolation parameter ``t`` IS Anima's flow-matching ``σ`` — same
    ``x_t = (1−t)·x0 + t·ε`` identity, evaluated on Anima's flow-shifted σ grid.
  - Anima's ``anima(...)`` output is the **velocity** ``v = ε − x0`` (the engine
    computes ``denoised = x_t − σ·v``), which is exactly FLAIR's ``v_θ`` — no ε/v
    reparam needed. The reference velocity ``u_t = (ε̂ − x_t)/(1−t) = ε̂ − μ`` is
    in the same convention, so the regularizer pull ``v_θ − u_t`` and its sign
    line up with Anima's ``noise_pred`` directly.
  - The DiT takes a 5D ``(B,C,1,H,W)`` latent (singleton at **dim 2**); μ is held
    4D ``(B,C,H,W)`` and ``unsqueeze(2)`` / ``squeeze(2)`` only at the boundary —
    targeting dim 2 explicitly per the CLAUDE.md invariant.

Phase 0 uses the **uncalibrated** regularizer weight ``λ_R(t) = reg_scale·t``
(the paper's CRW-ablation baseline); the calibrated ``λ_R(t)`` table is Phase 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas

DEFAULT_LAMBDA_PATH = "networks/calibration/flair_lambda_r.npz"


@dataclass
class FlairConfig:
    infer_steps: int = 50  # σ-grid resolution (1→0); loop stops at t_stop
    flow_shift: float = 3.0  # Anima's canonical schedule; Phase 1 may revisit
    t_stop: float = 0.2  # stop refining below this σ (paper: SD3 unreliable < 0.2)
    guidance_scale: float = 1.0  # CFG on the regularizer velocity
    reg_scale: float = 1.0  # λ_R base; λ_R(t) = reg_scale·t (uncalibrated, Phase 0)
    hdc_steps: int = 3  # SGD steps for hard data consistency per σ-step
    hdc_lr: float = 0.05  # Adam lr for the HDC projection
    alpha: float = 0.5  # DTA: deterministic↔stochastic re-noise tradeoff
    seed: int = 0
    # Phase 1 — calibrated regularizer weight (CRW). When ``lambda_values`` is set
    # (via :func:`load_lambda_table`), ``λ_R(t)`` is read from the table by interp
    # instead of the linear ``reg_scale·t`` baseline; ``reg_scale`` then scales the
    # (peak-normalized) calibrated curve, so it stays the same comparable knob.
    # Both arrays are σ-ascending; ``cutoff_sigma`` zeroes λ_R below it.
    lambda_sigmas: Optional[np.ndarray] = None
    lambda_values: Optional[np.ndarray] = None
    cutoff_sigma: float = 0.0


def load_lambda_table(
    path: str = "auto",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a calibrated λ_R table → ``(sigmas_asc, lambda_norm_asc, cutoff_sigma)``.

    Mirrors the CNS ``from_path("auto")`` pattern: ``"auto"`` resolves the shipped
    default under the repo home. The raw npz stores ``lambda_r = 1/error`` (Eq. 14)
    whose absolute scale is arbitrary (units of 1/velocity²); we **peak-normalize
    over the active region** (σ ≥ cutoff) so the table is an O(1) *shape* and the
    solver's ``reg_scale`` stays directly comparable to the Phase-0 ``λ_R=t`` arm.
    """
    p = DEFAULT_LAMBDA_PATH if path == "auto" else path
    resolved = resolve_under_home(p)
    if not Path(resolved).exists():
        raise FileNotFoundError(
            f"FLAIR calibration not found: {resolved}. Generate it with "
            "`python bench/flair/calibrate_lambda.py` (Phase 1)."
        )
    d = np.load(resolved)
    sig = np.asarray(d["sigmas"], dtype=np.float64)
    lam = np.asarray(d["lambda_r"], dtype=np.float64)
    cutoff = float(d["cutoff_sigma"])
    order = np.argsort(sig)  # np.interp needs ascending xp
    sig, lam = sig[order], lam[order]
    active = sig >= cutoff
    peak = float(lam[active].max()) if active.any() else float(lam.max())
    lam = lam / max(peak, 1e-12)
    return sig, lam, cutoff


def _lambda_of_t(t: float, cfg: FlairConfig) -> float:
    """λ_R at noise level ``t`` — calibrated table when present, else linear."""
    if cfg.lambda_values is None:
        return cfg.reg_scale * t  # uncalibrated CRW baseline (Phase 0)
    if t < cfg.cutoff_sigma:
        return 0.0  # CRW zeroing below the Anima low-noise cutoff
    return cfg.reg_scale * float(np.interp(t, cfg.lambda_sigmas, cfg.lambda_values))


def _velocity(
    anima,
    x_t: torch.Tensor,
    t: float,
    embed: torch.Tensor,
    neg_embed: torch.Tensor,
    guidance_scale: float,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """v_θ(x_t, t) with optional CFG. x_t is 4D; returns 4D fp32 velocity."""
    x5 = x_t.unsqueeze(2).to(torch.bfloat16)  # → (B,C,1,H,W), singleton at dim 2
    te = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.bfloat16)
    with torch.no_grad():
        v = anima(x5, te, embed, padding_mask=padding_mask).squeeze(2).float()
        if guidance_scale != 1.0:
            v_u = anima(x5, te, neg_embed, padding_mask=padding_mask).squeeze(2).float()
            v = v_u + guidance_scale * (v - v_u)
    return v


def _hdc_project(mu, vae, operator, y, *, steps: int, lr: float) -> torch.Tensor:
    """Hard data consistency: μ ← argmin ‖y − A(D(μ))‖², SGD through the decoder.

    The one place FLAIR backprops — and only through the VAE decoder ``D`` and
    the (cheap, linear) forward operator ``A``, never through the DiT. μ is
    optimized in fp32; ``decode_to_pixels`` casts to the VAE dtype internally
    (a differentiable cast, so grad flows back to the fp32 leaf).
    """
    if steps <= 0:
        return mu.detach()
    mu = mu.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([mu], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        img = vae.decode_to_pixels(mu)  # 4D [B,3,H,W] in [-1,1]
        loss = F.mse_loss(operator.degrade(img.float()), y)
        loss.backward()
        opt.step()
    return mu.detach()


@torch.no_grad()
def _init_mu(vae, operator, y, *, target_hw, device) -> torch.Tensor:
    """Adjoint init μ = E(A^T y): lift y to target res, VAE-encode."""
    up = operator.adjoint_init(y, target_hw=target_hw)
    return vae.encode_pixels_to_latents(up.to(torch.bfloat16)).float()


def flair_solve(
    anima,
    vae,
    *,
    embed: torch.Tensor,
    neg_embed: torch.Tensor,
    y: torch.Tensor,
    operator,
    target_hw: tuple[int, int],
    cfg: FlairConfig,
    device: torch.device,
    log=None,
) -> torch.Tensor:
    """Run Algorithm 1 and return the variational mean μ (4D latent [1,C,h,w]).

    ``y`` is the pixel-space observation ([-1,1], low-res for SR). ``target_hw``
    is the full-resolution pixel size the reconstruction is decoded at. ``embed``
    / ``neg_embed`` are the bf16 text embeddings from ``prepare_text_inputs``.
    Base-DiT prior only (no adapter), so the hydra/step-expert side-channel
    setters in ``generate_body`` are intentionally skipped.
    """
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    mu = _init_mu(vae, operator, y, target_hw=target_hw, device=device)  # [1,C,h,w]
    h_lat, w_lat = mu.shape[-2], mu.shape[-1]
    padding_mask = torch.zeros(
        mu.shape[0], 1, h_lat, w_lat, dtype=torch.bfloat16, device=device
    )
    eps_hat = torch.randn(mu.shape, generator=gen, device=device, dtype=torch.float32)

    timesteps, _sigmas = get_timesteps_sigmas(cfg.infer_steps, cfg.flow_shift, device)

    n_nfe = 0
    for i, t_tensor in enumerate(timesteps):
        t = float(t_tensor)
        if t < cfg.t_stop:
            break
        one_minus_t = max(1.0 - t, 1e-4)

        # Noisy latent of the variational distribution + its reference velocity.
        x_t = one_minus_t * mu + t * eps_hat
        u_t = eps_hat - mu  # = (ε̂ − x_t)/(1−t)

        # (i) flow-matching regularizer pull (forward eval — no backprop through DiT)
        v = _velocity(anima, x_t, t, embed, neg_embed, cfg.guidance_scale, padding_mask)
        n_nfe += 2 if cfg.guidance_scale != 1.0 else 1
        lam = _lambda_of_t(t, cfg)  # linear (Phase 0) or calibrated table (Phase 1)
        mu = mu - lam * (v - u_t)

        # (ii) hard data consistency — exact projection onto the measurement
        mu = _hdc_project(mu, vae, operator, y, steps=cfg.hdc_steps, lr=cfg.hdc_lr)

        # (iii) deterministic trajectory adjustment — re-noise toward the
        # one-step predicted endpoint x̂_1 instead of fresh Gaussian.
        x_hat1 = x_t + one_minus_t * v
        eps = torch.randn(mu.shape, generator=gen, device=device, dtype=torch.float32)
        eps_hat = cfg.alpha * x_hat1 + math.sqrt(max(1.0 - cfg.alpha**2, 0.0)) * eps

        if log is not None:
            log(f"  step {i:02d}  t={t:.3f}  λ={lam:.3f}  |μ|={mu.float().std():.3f}")

    if log is not None:
        log(f"  FLAIR solve done — {n_nfe} DiT NFE")
    return mu
