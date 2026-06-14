#!/usr/bin/env python
"""FLAIR Phase 1 — calibrate the regularizer weight λ_R(σ) on the Anima prior.

Implements paper Eq. 14 (Erbach et al., NeurIPS 2025), the Calibrated Regularizer
Weight (CRW): weight the flow-matching regularizer by the *inverse expected model
error* at each noise level, so FLAIR trusts the prior pull where the DiT predicts
velocity accurately and backs off where it doesn't.

    λ_R(σ) = ( (1/N) Σ_i ‖v_θ(x_σ^i, σ) − u_σ(x_σ^i | ε)‖² )⁻¹

with x_σ = (1−σ)·x0 + σ·ε,  ε ~ N(0, I),  and the conditional-FM target velocity
u_σ = ε − x0 (Anima's ``anima(...)`` output IS v = ε − x0, the same convention the
solver uses — see ``solver.py`` docstring). x0 are real calibration latents
VAE-encoded from ``post_image_dataset``.

Two deliberate Anima-specific deviations from the paper:

  - **Calibrate on the deployed σ grid, not linear t.** The paper sweeps 100
    linearly spaced t∈[0,1]; we sweep ``get_timesteps_sigmas(n, flow_shift)`` so the
    table lands on the *flow-shifted* (shift=3) σ values the solver actually
    traverses (proposal note #1). ``--n_timesteps`` keeps the paper's resolution.
  - **Re-measure the low-noise cutoff.** The paper zeroes λ_R for t<0.2 (an SD3
    fact). Anima resolves x0 by σ≈0.45 (project memory
    ``sigma_signal_resolves_by_045``), so the cutoff is re-derived from the error
    curve here — pre-registered hypothesis: it lands near σ≈0.45, not 0.2.

Output ``networks/calibration/flair_lambda_r.npz`` (keys ``sigmas``, ``lambda_r``,
``error``, ``cutoff_sigma``) is consumed by ``solver.load_lambda_table("auto")``.

    uv run python bench/flair/calibrate_lambda.py                      # quick smoke (N=8)
    uv run python bench/flair/calibrate_lambda.py --n 100 --res 512    # the proposal run
    uv run python bench/flair/calibrate_lambda.py --cutoff 0.45        # force the cutoff

This is forward-only (no HDC, no VAE backprop), so it is far lighter than the
Phase-0 SR bench — the VAE is touched once per image for the x0 encode, then the
sweep is pure DiT forwards.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    prepare_text_inputs,
)
from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.flair.solver import _velocity  # noqa: E402
from library.env import resolve_under_home  # noqa: E402
from library.inference.sampling import get_timesteps_sigmas  # noqa: E402
from library.runtime.harness import build_inference_bundle  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _discover_images(src: Path, n: int) -> list[Path]:
    """Sorted first-N images under ``src`` (follows symlinks — post_image_dataset
    is a symlink to nested artist dirs, so a plain rglob would miss them)."""
    found: list[Path] = []
    for root, _dirs, files in os.walk(src, followlinks=True):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMG_EXTS:
                found.append(Path(root) / f)
    found.sort()
    return found[:n]


def _load_image(path: Path, res: int, device: torch.device) -> torch.Tensor:
    """Center-crop to square, resize to ``res``, return [1,3,res,res] in [-1,1]."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((res, res), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)  # HWC [0,1]
    x = arr.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return x * 2.0 - 1.0  # → [-1,1]


def _caption(path: Path, fallback: str) -> str:
    txt = path.with_suffix(".txt")
    if txt.exists():
        cap = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if cap:
            return cap
    return fallback


def _detect_cutoff(
    sig_desc: np.ndarray, err_desc: np.ndarray, *, knee_mult: float
) -> float:
    """Anima's low-noise cutoff: the σ in the tail below which the FM error blows up.

    The error-vs-σ curve is U-shaped — high at the σ→1 noise end (hard to predict
    from near-pure noise), minimal at the *resolve* σ (Anima ≈0.43, where x0 is
    recovered), then rising again into the σ→0 tail where the prior degrades. The
    cutoff is about that **low-σ tail only**, so we anchor the reliability baseline
    to the error *minimum* (the resolve point), NOT a mid/high band that would be
    inflated by the noisy σ→1 end. Cutoff = the highest σ *below* the minimum where
    error first exceeds ``knee_mult ×`` min-error; λ_R is zeroed below it. Returns
    0.0 if the tail never degrades past the knee (prior stays reliable to σ→0).
    """
    sig = np.asarray(sig_desc, dtype=np.float64)
    err = np.asarray(err_desc, dtype=np.float64)
    i_min = int(err.argmin())
    sig_min, err_min = float(sig[i_min]), float(err[i_min])
    thr = knee_mult * err_min
    low_bad = sig[(sig < sig_min) & (err > thr)]
    return float(low_bad.max()) if low_bad.size else 0.0


def _save_plot(
    run_dir: Path, sig: np.ndarray, err: np.ndarray, lam: np.ndarray, cutoff: float
) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(sig, err, ".-")
    ax[0].set(xlabel="σ", ylabel="mean FM error  ‖v_θ−u‖²", title="model error vs σ")
    ax[0].axvline(cutoff, color="r", ls="--", label=f"cutoff σ={cutoff:.3f}")
    ax[0].legend()
    ax[1].plot(sig, lam / max(lam.max(), 1e-12), ".-")
    ax[1].plot(sig, sig, ":", color="gray", label="λ_R=σ (Phase 0)")
    ax[1].axvline(cutoff, color="r", ls="--")
    ax[1].set(xlabel="σ", ylabel="λ_R (peak-normalized)", title="calibrated λ_R(σ)")
    ax[1].legend()
    fig.tight_layout()
    out = run_dir / "lambda_r_curve.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out.name


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--src", default="post_image_dataset/resized", help="image dir")
    p.add_argument(
        "--n", type=int, default=8, help="calibration images (proposal: 100)"
    )
    p.add_argument(
        "--res", type=int, default=512, help="square resolution to encode at"
    )
    p.add_argument(
        "--n_timesteps",
        type=int,
        default=100,
        help="σ-grid resolution (paper uses 100)",
    )
    p.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="Anima's canonical schedule (calibrate on the deployed grid)",
    )
    p.add_argument("--batch", type=int, default=4, help="images per DiT forward")
    p.add_argument(
        "--cutoff",
        default="auto",
        help="'auto' (knee-detect) or a float σ to zero λ_R below",
    )
    p.add_argument(
        "--knee_mult",
        type=float,
        default=2.0,
        help="auto cutoff: error > knee_mult×baseline ⇒ unreliable",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="fallback if no .txt sidecar"
    )
    p.add_argument("--label", default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help="compile_blocks() the DiT — the token family AND batch are constant "
        "across the whole sweep (when batch divides N), so this traces one graph "
        "reused for every σ-step. Big win at 768px / high N.",
    )
    p.add_argument(
        "--no_save",
        action="store_true",
        help="skip writing the canonical npz (only the run-dir copy)",
    )
    p.add_argument("--verbose", action="store_true")
    opts = p.parse_args()

    if opts.res % 8 != 0:
        raise SystemExit(f"--res {opts.res} must be divisible by 8 (VAE ×8).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = Path(opts.src)
    if not src.is_absolute():
        src = Path(__file__).resolve().parents[2] / src
    images = _discover_images(src, opts.n)
    if not images:
        raise SystemExit(f"no images found under {src}")

    ckpts = default_checkpoints()
    req = GenerationRequest(
        dit=ckpts.dit,
        vae=ckpts.vae,
        text_encoder=ckpts.text_encoder,
        prompt=opts.prompt,
        negative_prompt=DEFAULT_NEG,
        save_path="output/tests/flair/_unused.png",
        infer_steps=opts.n_timesteps,
        guidance_scale=1.0,
        image_size=(opts.res, opts.res),
        sampler="er_sde",
        flow_shift=opts.flow_shift,
        seed=opts.seed,
    )
    args = req.to_args()
    args.device = str(device)
    args.compile = False

    print(
        f"[flair-calib] loading DiT + VAE + TE …  ({len(images)} imgs, {opts.res}px, "
        f"{opts.n_timesteps} σ-steps, shift={opts.flow_shift})"
    )
    bundle = build_inference_bundle(args, device=device)
    anima, vae, shared = bundle.model, bundle.vae, bundle.shared_models
    vae.to(device)

    if opts.compile:
        if len(images) % opts.batch != 0:
            print(
                f"[flair-calib] warning: --batch {opts.batch} does not divide "
                f"N={len(images)} — the partial last batch traces a 2nd graph."
            )
        print("[flair-calib] compile_blocks() — first forward pays ~30-60s")
        anima.compile_blocks()

    # Encode x0 latents + per-image text embeddings up front (forward-only sweep).
    latents, embeds = [], []
    for img_path in images:
        gt = _load_image(img_path, opts.res, device)
        with torch.no_grad():
            z = vae.encode_pixels_to_latents(gt.to(torch.bfloat16)).float()  # [1,C,h,w]
        cap = _caption(img_path, opts.prompt)
        context, _null = prepare_text_inputs(
            args, device, anima, shared, prompt=cap, negative_prompt=DEFAULT_NEG
        )
        latents.append(z)
        embeds.append(context["embed"][0].to(device, torch.bfloat16))
        if opts.verbose:
            print(f"  encoded {img_path.name}  cap='{cap[:48]}'")
    x0 = torch.cat(latents, dim=0)  # [N,C,h,w]
    N = x0.shape[0]

    # Deployed σ grid (descending, excludes 0). One fixed ε̂ per image (seeded).
    timesteps, _sig = get_timesteps_sigmas(opts.n_timesteps, opts.flow_shift, device)
    gen = torch.Generator(device=device).manual_seed(opts.seed)
    eps = torch.randn(x0.shape, generator=gen, device=device, dtype=torch.float32)
    u = eps - x0  # conditional-FM target velocity, σ-independent

    h_lat, w_lat = x0.shape[-2], x0.shape[-1]
    pad = torch.zeros(opts.batch, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)

    sig_list, err_list = [], []
    for si, t_tensor in enumerate(timesteps):
        t = float(t_tensor)
        x_t = (1.0 - t) * x0 + t * eps  # [N,C,h,w]
        sq_sum = 0.0
        for b0 in range(0, N, opts.batch):
            b1 = min(b0 + opts.batch, N)
            xb = x_t[b0:b1]
            eb = embeds[b0] if b1 - b0 == 1 else torch.cat(embeds[b0:b1], dim=0)
            v = _velocity(anima, xb, t, eb, eb, 1.0, pad[: b1 - b0])  # [b,C,h,w]
            sq_sum += (v - u[b0:b1]).pow(2).sum().item()
        err = sq_sum / (N * x0.shape[1] * h_lat * w_lat)  # mean over images & elems
        sig_list.append(t)
        err_list.append(err)
        if opts.verbose or si % 10 == 0:
            print(f"  σ={t:.3f}  err={err:.5f}  λ_R≈{1.0 / max(err, 1e-12):.2f}")

    sig = np.asarray(sig_list, dtype=np.float64)  # descending
    err = np.asarray(err_list, dtype=np.float64)
    lam = 1.0 / np.clip(err, 1e-12, None)  # raw Eq. 14 (arbitrary scale)

    if opts.cutoff == "auto":
        cutoff = _detect_cutoff(sig, err, knee_mult=opts.knee_mult)
    else:
        cutoff = float(opts.cutoff)

    run_dir = make_run_dir("flair", label=opts.label or f"calib-{N}img-{opts.res}px")
    np.savez(
        run_dir / "flair_lambda_r.npz",
        sigmas=sig,
        lambda_r=lam,
        error=err,
        cutoff_sigma=cutoff,
    )
    plot_name = _save_plot(run_dir, sig, err, lam, cutoff)

    if not opts.no_save:
        dst = Path(resolve_under_home("networks/calibration/flair_lambda_r.npz"))
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, sigmas=sig, lambda_r=lam, error=err, cutoff_sigma=cutoff)
        print(f"[flair-calib] wrote canonical table → {dst}")

    resolve_sigma = float(sig[int(err.argmin())])  # error-min = where the prior is best
    metrics = {
        "n_images": N,
        "n_timesteps": opts.n_timesteps,
        "flow_shift": opts.flow_shift,
        "cutoff_sigma": cutoff,
        "cutoff_mode": opts.cutoff,
        "knee_mult": opts.knee_mult,
        "min_error": float(err.min()),
        # the "break"/resolve point — the σ≈0.45 hypothesis is really ABOUT this,
        # not the (deeper) zeroing cutoff.
        "resolve_sigma": resolve_sigma,
        "max_error": float(err.max()),
        "max_error_sigma": float(sig[int(err.argmax())]),
        # pre-registered (sigma_signal_resolves_by_045): the prior's most-reliable
        # σ should land near 0.45.
        "resolve_near_045": bool(abs(resolve_sigma - 0.45) <= 0.1),
        "error_by_sigma": [
            {"sigma": float(s), "error": float(e), "lambda_r": float(lv)}
            for s, e, lv in zip(sig, err, lam)
        ],
    }
    artifacts = ["flair_lambda_r.npz"] + ([plot_name] if plot_name else [])
    write_result(
        run_dir,
        script=__file__,
        args=opts,
        metrics=metrics,
        label=opts.label,
        artifacts=artifacts,
        device=device,
    )
    print(
        f"\n[flair-calib] {run_dir}\n"
        f"  resolve σ (error min, λ_R peak) = {resolve_sigma:.3f}  "
        f"(hypothesis ≈0.45 → {'HIT' if metrics['resolve_near_045'] else 'MISS'})\n"
        f"  min error {metrics['min_error']:.5f}  |  max error {metrics['max_error']:.5f} "
        f"at σ={metrics['max_error_sigma']:.3f}\n"
        f"  λ_R-zeroing cutoff σ = {cutoff:.3f}  (loop t_stop in the calibrated arm)\n"
        f"  A/B next: bench/flair/sanity_sr.py --calib auto  vs  the λ_R=σ Phase-0 arm."
    )


if __name__ == "__main__":
    main()
