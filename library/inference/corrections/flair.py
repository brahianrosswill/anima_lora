"""FLAIR Algorithm 1 — variational posterior solve on the Anima flow prior.

Port of *Solving Inverse Problems with FLAIR* (Erbach et al., NeurIPS 2025),
Algorithm 1, mapped onto Anima's flow-matching DiT + Qwen VAE. Phase 0/1
validated this in ``bench/flair/`` (SR ×8, λ_R calibration); it is promoted here
so the editing application can ship it from the inference engine — see
``docs/proposal/flair_inverse.md`` (method) and ``docs/proposal/flair_edit.md``
(the FLAIR-edit application). ``bench/flair/{solver,operators}.py`` are thin
re-export shims so the Phase-0/1 benches keep running unchanged.

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
(the paper's CRW-ablation baseline); the calibrated ``λ_R(t)`` table is Phase 1
(``networks/calibration/flair_lambda_r.npz``, loaded via ``load_lambda_table``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from library.env import resolve_under_home
from library.inference.sampling import get_timesteps_sigmas

logger = logging.getLogger(__name__)

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
    alpha: float = 1.0  # DTA: scale on the α=1−t re-noise schedule (1.0 = paper `inv_alpha: 1-t`)
    dta_fixed: bool = False  # if True, use a constant α (=alpha) instead of the α·(1−t) schedule
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
    progress: bool = False,
    debug_dir: Optional[Path] = None,
) -> torch.Tensor:
    """Run Algorithm 1 and return the variational mean μ (4D latent [1,C,h,w]).

    ``y`` is the pixel-space observation ([-1,1]; low-res for SR, masked for
    inpaint). ``target_hw`` is the full-resolution pixel size the reconstruction
    is decoded at. ``embed`` / ``neg_embed`` are the bf16 text embeddings from
    ``prepare_text_inputs``. Base-DiT prior only (no adapter), so the
    hydra/step-expert side-channel setters in ``generate_body`` are intentionally
    skipped. ``progress`` shows a tqdm bar over the active σ-steps (each step is a
    DiT forward + ``hdc_steps`` decoder backprops, so it's slow — the bar matters).
    """
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    # --- optional σ-checkpoint decode instrumentation -----------------------
    # Decode the running variational mean μ at a few σ levels so we can watch the
    # hole evolve (does structure form early then average out to the mean, or
    # never form?). Pure observation — no effect on the solve.
    dbg_targets = [0.7, 0.45, 0.2, 0.11] if debug_dir is not None else []
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def _dbg_save(tag: str, lat: torch.Tensor) -> None:
        if debug_dir is None:
            return
        from PIL import Image

        img = vae.decode_to_pixels(lat.to(torch.bfloat16)).float()  # 4D [-1,1]
        arr = ((img[0].clamp(-1, 1) + 1.0) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr).save(debug_dir / f"mu_{tag}.png")

    mu = _init_mu(vae, operator, y, target_hw=target_hw, device=device)  # [1,C,h,w]
    _dbg_save("init", mu)
    h_lat, w_lat = mu.shape[-2], mu.shape[-1]
    padding_mask = torch.zeros(
        mu.shape[0], 1, h_lat, w_lat, dtype=torch.bfloat16, device=device
    )
    # Carried DTA state = the previous step's one-step endpoint prediction x̂_1
    # (raw, unmixed). Seeded with pure Gaussian so the first (high-σ) step's mix
    # is ~fully random — see the DTA block in the loop.
    endpoint = torch.randn(mu.shape, generator=gen, device=device, dtype=torch.float32)

    timesteps, _sigmas = get_timesteps_sigmas(cfg.infer_steps, cfg.flow_shift, device)
    # Only the σ ≥ t_stop steps actually run — count them up front so the bar (and
    # the NFE estimate) reflect real work, not the full grid.
    active = [tt for tt in timesteps if float(tt) >= cfg.t_stop]
    step_iter = enumerate(active)
    if progress:
        from tqdm import tqdm

        step_iter = tqdm(step_iter, total=len(active), desc="FLAIR solve", unit="step")

    n_nfe = 0
    for i, t_tensor in step_iter:
        t = float(t_tensor)
        one_minus_t = max(1.0 - t, 1e-4)

        # (iii') deterministic trajectory adjustment — mix the carried endpoint
        # prediction with fresh Gaussian using the *t-dependent* coefficient
        # α=1−t (FLAIR ``inv_alpha: 1-t``). High-σ steps (α→0) re-noise almost
        # fully randomly so the prior can synthesize content in unconstrained
        # regions (the inpaint hole); low-σ steps (α→1) lock onto the endpoint.
        # A fixed α here collapses the hole to the flat init's mean — the bug
        # that made inpaint awful while SR (every pixel constrained) survived.
        a = cfg.alpha if cfg.dta_fixed else cfg.alpha * (1.0 - t)
        a = min(max(a, 0.0), 1.0)
        eps = torch.randn(mu.shape, generator=gen, device=device, dtype=torch.float32)
        eps_hat = a * endpoint + math.sqrt(max(1.0 - a * a, 0.0)) * eps

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

        # σ-checkpoint decode: fire when t first drops to/below a target level.
        while dbg_targets and t <= dbg_targets[0]:
            _dbg_save(f"t{dbg_targets.pop(0):.3f}_step{i:02d}", mu)

        # Carry the raw one-step endpoint prediction x̂_1 (unmixed) — next step's
        # DTA mixes it with fresh noise at that step's α.
        endpoint = x_t + one_minus_t * v

        if log is not None:
            log(f"  step {i:02d}  t={t:.3f}  λ={lam:.3f}  |μ|={mu.float().std():.3f}")
        if progress and hasattr(step_iter, "set_postfix"):
            step_iter.set_postfix(t=f"{t:.3f}", lam=f"{lam:.2f}")

    if log is not None:
        log(f"  FLAIR solve done — {n_nfe} DiT NFE")
    return mu


# --------------------------------------------------------------------------- #
# Engine orchestration — the FLAIR-edit branch driven from ``generate()``.
# --------------------------------------------------------------------------- #

_IMG_MEAN = None  # placeholder kept for symmetry; pixels are plain [-1,1]


def _load_pixels(path: str, hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Load an image → ``[1, 3, H, W]`` in [-1, 1], resized to ``hw`` (H, W)."""
    from PIL import Image

    resolved = str(resolve_under_home(path)) if not Path(path).is_file() else path
    img = Image.open(resolved).convert("RGB")
    h, w = hw
    img = img.resize((w, h), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)  # HWC [0,1]
    x = arr.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return x * 2.0 - 1.0  # → [-1,1]


def _load_keep_mask(
    path: str, hw: tuple[int, int], device: torch.device, *, invert: bool
) -> torch.Tensor:
    """Load an **edit** mask PNG → a ``[1,1,H,W]`` *keep* mask in {0, 1}.

    Convention: white (>0.5) marks the region to **edit/fill**; the returned keep
    mask is its complement (1 = locked/observed, 0 = fill). ``invert=True`` flips
    this for callers that already supply keep-masks (white = keep).
    """
    from PIL import Image

    resolved = str(resolve_under_home(path)) if not Path(path).is_file() else path
    h, w = hw
    m = Image.open(resolved).convert("L").resize((w, h), Image.NEAREST)
    edit = (torch.from_numpy(np.asarray(m, dtype=np.float32) / 255.0) > 0.5).float()
    if invert:
        edit = 1.0 - edit
    keep = 1.0 - edit  # 1 = keep/observed, 0 = fill
    return keep.view(1, 1, h, w).to(device)


def _keep_mask_from_concept(
    image_path: str,
    phrase: str,
    hw: tuple[int, int],
    device: torch.device,
    *,
    dilate: int,
) -> torch.Tensor:
    """SAM3 text→mask → ``[1,1,H,W]`` keep mask (1 = keep, 0 = the concept region)."""
    from PIL import Image

    from library.vision.mask_from_concept import mask_from_concept

    resolved = (
        str(resolve_under_home(image_path))
        if not Path(image_path).is_file()
        else image_path
    )
    pil = Image.open(resolved).convert("RGB")
    edit_np = mask_from_concept(
        pil, phrase, dilate=dilate, device=str(device)
    )  # HxW {0,1}
    edit = torch.from_numpy(edit_np.astype(np.float32))
    h, w = hw
    edit = F.interpolate(
        edit.view(1, 1, *edit.shape), size=(h, w), mode="nearest"
    ).view(1, 1, h, w)
    keep = (1.0 - edit).to(device)
    return keep


def _resolve_keep_mask(args, hw, device) -> torch.Tensor:
    """Resolve the FLAIR-edit keep mask from ``--flair_mask`` or ``--flair_mask_prompt``."""
    mask_path = getattr(args, "flair_mask", None)
    mask_prompt = getattr(args, "flair_mask_prompt", None)
    invert = bool(getattr(args, "flair_mask_invert", False))
    if mask_path:
        return _load_keep_mask(mask_path, hw, device, invert=invert)
    if mask_prompt:
        dilate = int(getattr(args, "flair_mask_dilate", 0) or 0)
        return _keep_mask_from_concept(
            args.flair_edit_image, mask_prompt, hw, device, dilate=dilate
        )
    raise ValueError(
        "FLAIR-edit needs a mask: pass --flair_mask <png> or --flair_mask_prompt "
        "<concept>. (The argparse layer validates this; reaching here means a "
        "GenerationRequest bypassed build_default_args.)"
    )


def run_flair_edit(
    args,
    anima,
    context,
    context_null,
    device: torch.device,
    seed: int,
    shared_models: Optional[dict] = None,
) -> torch.Tensor:
    """Engine branch for ``--flair_task edit``: solve Algorithm 1, return a latent.

    Replaces the standard Euler/er-SDE loop (FLAIR optimizes ``μ``, it does not
    denoise a fixed trajectory). Returns a **5D** latent ``(B,C,1,h,w)`` so the
    normal ``save_output`` decode path consumes it unchanged.

    The VAE is taken from ``shared_models["vae"]`` when present, else loaded here
    (and freed before return) — HDC backprops through the decoder every σ-step, so
    the VAE must be resident on ``device`` for the whole solve.
    """
    from library.inference.corrections.flair_operators import build_operator
    from library.inference.output import check_inputs

    height, width = check_inputs(args)
    hw = (height, width)

    # --- VAE (resident for the HDC decode backprop) ------------------------
    vae = (shared_models or {}).get("vae")
    own_vae = vae is None
    if own_vae:
        from library.models.qwen_vae import load_vae

        vae = load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=getattr(args, "vae_chunk_size", None),
            disable_cache=getattr(args, "vae_disable_cache", False),
            dtype=torch.bfloat16,
            eval=True,
        )
    vae.to(device)

    # --- source + mask + observation --------------------------------------
    src = _load_pixels(args.flair_edit_image, hw, device)  # [1,3,H,W] in [-1,1]
    keep = _resolve_keep_mask(args, hw, device)  # [1,1,H,W] keep-mask
    operator = build_operator("inpaint", mask=keep, sigma_nu=0.0)
    y = operator.degrade(src)  # observation = visible pixels (masked source)

    # --- text embeddings (delta-only prompt → context already prepared) ----
    embed = context["embed"][0].to(device, dtype=torch.bfloat16)
    if context_null is None:
        context_null = context
    neg_embed = context_null["embed"][0].to(device, dtype=torch.bfloat16)

    # --- λ_R: linear Phase-0 baseline (default) or the calibrated table -----
    # Default is the linear λ_R=reg_scale·σ (→0 as σ→0). The calibrated table is
    # opt-in (``--flair_calib auto``) because it peaks λ_R in the σ<0.45 band and
    # DIVERGES μ in an unconstrained inpaint hole: the hole feeds the prior an
    # off-distribution latent → large velocity → with high λ_R, |μ| runs away
    # (observed 0.9→2.0) and decodes to RGB static. It is calibrated on *valid*
    # noised images, so it's only sound for fully-constrained tasks (SR/colorize).
    lam_sig = lam_val = None
    cutoff = 0.0
    t_stop = float(getattr(args, "flair_t_stop", 0.2))
    calib = getattr(args, "flair_calib", "off")
    if calib and str(calib).lower() not in ("", "none", "off", "linear"):
        try:
            lam_sig, lam_val, cutoff = load_lambda_table(calib)
            t_stop = cutoff
            logger.info(
                "FLAIR-edit: calibrated λ_R (%s) — cutoff σ=%.3f, loop t_stop=%.3f",
                calib,
                cutoff,
                t_stop,
            )
        except FileNotFoundError:
            logger.warning(
                "FLAIR-edit: calibration table %r missing — falling back to the "
                "linear λ_R=reg_scale·σ baseline (t_stop=%.3f).",
                calib,
                t_stop,
            )

    cfg = FlairConfig(
        infer_steps=int(args.infer_steps),
        flow_shift=float(args.flow_shift),
        t_stop=t_stop,
        guidance_scale=float(getattr(args, "guidance_scale", 1.0)),
        reg_scale=float(getattr(args, "flair_reg_scale", 0.5)),
        hdc_steps=int(getattr(args, "flair_hdc_steps", 8)),
        hdc_lr=float(getattr(args, "flair_hdc_lr", 0.1)),
        alpha=float(getattr(args, "flair_alpha", 1.0)),
        dta_fixed=bool(getattr(args, "flair_dta_fixed", False)),
        seed=seed if isinstance(seed, int) else int(seed[0]),
        lambda_sigmas=lam_sig,
        lambda_values=lam_val,
        cutoff_sigma=cutoff,
    )

    keep_frac = float(keep.mean().item())
    logger.info(
        "FLAIR-edit: %dx%d, keep %.1f%% / fill %.1f%%, %d steps × %d HDC, "
        "α=%.2f reg=%.2f prompt=%r",
        height,
        width,
        100.0 * keep_frac,
        100.0 * (1.0 - keep_frac),
        cfg.infer_steps,
        cfg.hdc_steps,
        cfg.alpha,
        cfg.reg_scale,
        context.get("prompt"),
    )

    mu = flair_solve(
        anima,
        vae,
        embed=embed,
        neg_embed=neg_embed,
        y=y,
        operator=operator,
        target_hw=hw,
        cfg=cfg,
        device=device,
        log=logger.info if getattr(args, "flair_verbose", False) else None,
        progress=True,
        debug_dir=(
            Path(args.save_path) / "_flair_debug"
            if getattr(args, "flair_verbose", False)
            else None
        ),
    )

    latent5d = mu.unsqueeze(2).to(torch.bfloat16)  # (B,C,1,h,w) for save_output

    if own_vae:
        del vae
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return latent5d
