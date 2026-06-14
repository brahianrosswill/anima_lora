#!/usr/bin/env python
"""FLAIR Phase 0 — port-validation sanity bench (SR ×8, uncalibrated λ_R=t).

This is the GATE in ``docs/proposal/flair_inverse.md``: it does NOT measure the
method, it validates the **port**. It runs FLAIR's Algorithm 1 on the base Anima
prior to recover high-res images from bicubic-downsampled observations, with the
*uncalibrated* regularizer weight ``λ_R(t)=reg_scale·t`` (the paper's CRW-ablation
baseline — calibration is Phase 1). If the reconstruction is sharp and
data-consistent (PSNR well above the bicubic-upsample baseline, no black frames,
no layout corruption), the σ/t mapping, the v_θ sign, the adjoint init, the
VAE-decode HDC projection, and the dim-2 handling are all correct → proceed to
Phase 1. If it's garbage/black, the PORT is broken (fix that before reading any
quality number); if it's merely soft, the prior may not drive SR — a cheap close.

    uv run python bench/flair/sanity_sr.py                       # quick smoke (N=4, 512px)
    uv run python bench/flair/sanity_sr.py --n 100 --res 768     # the proposal's full gate
    uv run python bench/flair/sanity_sr.py --cfg 4.0 --reg_scale 0.5 --label cfg4

Memory note: HDC backprops through the VAE decoder, so the DiT and VAE are both
resident on the GPU. Default res=512 keeps that comfortable on 24GB; bump --res
to 768 on a bigger card, or drop --hdc_steps / --res if you OOM.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import (  # noqa: E402
    GenerationRequest,
    default_checkpoints,
    prepare_text_inputs,
)
from bench._anima import DEFAULT_NEG, DEFAULT_PROMPT  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from bench.flair.operators import build_operator  # noqa: E402
from bench.flair.solver import FlairConfig, flair_solve, load_lambda_table  # noqa: E402
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


def _to01(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1.0, 1.0) + 1.0) / 2.0


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(_to01(a), _to01(b)).item()
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def _ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    na = _to01(a)[0].permute(1, 2, 0).cpu().numpy()
    nb = _to01(b)[0].permute(1, 2, 0).cpu().numpy()
    return float(structural_similarity(na, nb, channel_axis=2, data_range=1.0))


def _save_triptych(path: Path, gt, baseline, recon) -> None:
    """GT | bicubic-upsample(y) | FLAIR — side by side, the visual gate."""
    panels = [
        (_to01(t)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        for t in (gt, baseline, recon)
    ]
    strip = np.concatenate(panels, axis=1)
    Image.fromarray(strip).save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="post_image_dataset/resized", help="image dir")
    p.add_argument(
        "--n", type=int, default=4, help="number of images (proposal gate: 100)"
    )
    p.add_argument(
        "--res", type=int, default=512, help="target HR resolution (÷ 8·scale)"
    )
    p.add_argument("--scale", type=int, default=8, help="SR factor")
    p.add_argument(
        "--sigma_nu", type=float, default=0.01, help="meas. noise std (0.5%)"
    )
    # FLAIR knobs
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--t_stop", type=float, default=0.2)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--reg_scale", type=float, default=1.0)
    p.add_argument(
        "--calib",
        default=None,
        help="Phase-1 calibrated λ_R table: 'auto' (shipped npz) or a path. "
        "When set, λ_R is read from the table and the loop stops at the "
        "calibrated cutoff; omit for the Phase-0 λ_R=reg_scale·σ baseline.",
    )
    p.add_argument("--hdc_steps", type=int, default=3)
    p.add_argument("--hdc_lr", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--prompt", default=DEFAULT_PROMPT, help="fallback if no .txt sidecar"
    )
    p.add_argument("--label", default=None)
    p.add_argument(
        "--compile",
        action="store_true",
        help="compile_blocks() the DiT (one graph — latent shape is fixed across the solve)",
    )
    p.add_argument("--verbose", action="store_true")
    opts = p.parse_args()

    if opts.res % (8 * opts.scale) != 0:
        raise SystemExit(
            f"--res {opts.res} must be divisible by 8·scale={8 * opts.scale} "
            "(VAE ×8 + clean ÷scale downsample)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = Path(opts.src)
    if not src.is_absolute():
        src = Path(__file__).resolve().parents[2] / src
    images = _discover_images(src, opts.n)
    if not images:
        raise SystemExit(f"no images found under {src}")

    # Fully-defaulted args namespace + the loaded TE / DiT / VAE (one load).
    ckpts = default_checkpoints()
    req = GenerationRequest(
        dit=ckpts.dit,
        vae=ckpts.vae,
        text_encoder=ckpts.text_encoder,
        prompt=opts.prompt,
        negative_prompt=DEFAULT_NEG,
        save_path="output/tests/flair/_unused.png",
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=(opts.res, opts.res),
        sampler="er_sde",
        flow_shift=opts.flow_shift,
        seed=opts.seed,
    )
    args = req.to_args()
    args.device = str(device)
    args.compile = False

    print(
        f"[flair] loading DiT + VAE + TE …  ({len(images)} images, {opts.res}px ×{opts.scale})"
    )
    bundle = build_inference_bundle(args, device=device)
    anima, vae, shared = bundle.model, bundle.vae, bundle.shared_models
    vae.to(device)  # resident: HDC backprops through the decoder every σ-step

    if opts.compile:
        # The FLAIR latent shape is constant across the whole solve (one token
        # family), so this traces a single block graph reused every σ-step. No
        # adapter here → compile after load is the full compile-after-apply order.
        print("[flair] compile_blocks() — first forward pays ~30-60s")
        anima.compile_blocks()

    operator = build_operator("sr", scale=opts.scale, sigma_nu=opts.sigma_nu)
    # Phase-1 calibrated CRW: load the λ_R(σ) table and stop the loop AT its cutoff
    # (paper convention: λ_R is zeroed below the cutoff AND the loop stops there, so
    # t_stop == cutoff). The calibrated cutoff is the whole point of re-measuring the
    # Anima break — Anima stays reliable to σ≈0.11, deeper than the SD3 default 0.2 —
    # so the calibrated arm refines deeper than the linear arm's --t_stop. The table
    # governs the stop here; --t_stop only applies to the linear arm.
    lam_sig = lam_val = None
    cutoff = 0.0
    t_stop = opts.t_stop
    if opts.calib is not None:
        lam_sig, lam_val, cutoff = load_lambda_table(opts.calib)
        t_stop = cutoff
        print(
            f"[flair] calibrated λ_R ({opts.calib}) — cutoff σ={cutoff:.3f}, "
            f"loop t_stop={t_stop:.3f} (--t_stop {opts.t_stop} ignored in calibrated arm)"
        )
    cfg = FlairConfig(
        infer_steps=opts.steps,
        flow_shift=opts.flow_shift,
        t_stop=t_stop,
        guidance_scale=opts.cfg,
        reg_scale=opts.reg_scale,
        hdc_steps=opts.hdc_steps,
        hdc_lr=opts.hdc_lr,
        alpha=opts.alpha,
        seed=opts.seed,
        lambda_sigmas=lam_sig,
        lambda_values=lam_val,
        cutoff_sigma=cutoff,
    )

    arm = "calib" if opts.calib is not None else "lin"
    run_dir = make_run_dir(
        "flair", label=opts.label or f"sr{opts.scale}-{opts.res}px-{arm}"
    )
    meas_gen = torch.Generator(device=device).manual_seed(opts.seed)

    rows = []
    for idx, img_path in enumerate(images):
        gt = _load_image(img_path, opts.res, device)
        y = operator.measure(gt, generator=meas_gen)
        baseline = operator.adjoint_init(
            y, target_hw=(opts.res, opts.res)
        )  # bicubic up

        cap = _caption(img_path, opts.prompt)
        context, context_null = prepare_text_inputs(
            args, device, anima, shared, prompt=cap, negative_prompt=DEFAULT_NEG
        )
        embed = context["embed"][0].to(device, torch.bfloat16)
        neg_embed = context_null["embed"][0].to(device, torch.bfloat16)

        print(f"[flair] {idx + 1}/{len(images)}  {img_path.name}  cap='{cap[:48]}'")
        mu = flair_solve(
            anima,
            vae,
            embed=embed,
            neg_embed=neg_embed,
            y=y,
            operator=operator,
            target_hw=(opts.res, opts.res),
            cfg=cfg,
            device=device,
            log=(print if opts.verbose else None),
        )
        with torch.no_grad():
            recon = vae.decode_to_pixels(mu.to(torch.bfloat16)).float()

        m = {
            "image": img_path.name,
            "flair_psnr": _psnr(recon, gt),
            "flair_ssim": _ssim(recon, gt),
            "baseline_psnr": _psnr(baseline, gt),
            "baseline_ssim": _ssim(baseline, gt),
        }
        m["psnr_gain"] = m["flair_psnr"] - m["baseline_psnr"]
        rows.append(m)
        _save_triptych(run_dir / f"{idx:03d}_{img_path.stem}.png", gt, baseline, recon)
        print(
            f"        PSNR flair={m['flair_psnr']:.2f} vs bicubic={m['baseline_psnr']:.2f} "
            f"(Δ{m['psnr_gain']:+.2f})  SSIM flair={m['flair_ssim']:.3f}"
        )
        del mu, recon
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _agg(key: str) -> dict:
        vals = [r[key] for r in rows]
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    metrics = {
        "n_images": len(rows),
        "flair_psnr": _agg("flair_psnr"),
        "flair_ssim": _agg("flair_ssim"),
        "baseline_psnr": _agg("baseline_psnr"),
        "baseline_ssim": _agg("baseline_ssim"),
        "psnr_gain": _agg("psnr_gain"),
        "per_image": rows,
        # The GATE: FLAIR must clear the bicubic baseline on PSNR (and look sharp
        # in the triptychs). A non-positive mean gain ⇒ port broken or prior weak.
        "gate_psnr_gain_positive": bool(np.mean([r["psnr_gain"] for r in rows]) > 0),
    }
    artifacts = [f"{i:03d}_{img.stem}.png" for i, img in enumerate(images)]
    write_result(
        run_dir,
        script=__file__,
        args=opts,
        metrics=metrics,
        label=opts.label,
        artifacts=artifacts,
        device=device,
    )
    g = metrics["psnr_gain"]["mean"]
    print(
        f"\n[flair] {run_dir}\n"
        f"  FLAIR  PSNR {metrics['flair_psnr']['mean']:.2f}  SSIM {metrics['flair_ssim']['mean']:.3f}\n"
        f"  bicubic PSNR {metrics['baseline_psnr']['mean']:.2f}  SSIM {metrics['baseline_ssim']['mean']:.3f}\n"
        f"  mean PSNR gain {g:+.2f}  →  GATE {'PASS' if g > 0 else 'FAIL'} "
        f"(also eyeball the triptychs for sharpness/black-frame/layout)."
    )


if __name__ == "__main__":
    main()
