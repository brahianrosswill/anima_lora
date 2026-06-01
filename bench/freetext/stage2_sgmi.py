"""FreeText Stage-2 driver — Spectral-Modulated Glyph Injection (SGMI, §3.2).

Stage-1 answered *where* to write (the mask ``R``). This answers *what*: it
renders the target string into ``R`` with a real font, VAE-encodes it to a glyph
latent, and injects it at the sampler boundary over the mid-early window
``σ∈[0.6,0.8]`` (paper ``t∈[0.8T,0.6T]``) via the annealed masked replacement
``z̃ = (1-λR)⊙z + λR⊙z_sgmi`` (Eq 14). The glyph structure is supplied *by us*
(an external raster), not the model — so this is the decisive test of the open
question: **can SGMI drive glyphs the base has never cleanly drawn (Korean)?**

The injection is wired without touching production inference: ``library.inference``
runs the real Euler loop (real forward, real CFG); we monkeypatch the one-line
Euler step ``inference_utils.step`` to masked-replace after each step (the CNS
precedent). A module-global :class:`~stage2.SGMIInjector` (or ``None`` for the
base run) is consulted per step, so base vs. SGMI differ *only* by the injection.

Run (reuses a cached Stage-1 map dump for R; no re-capture):

    python bench/freetext/stage2_sgmi.py \
        --prompt 'a girl ... reads "안녕하세요"' --target 안녕하세요 \
        --maps_npz bench/freetext/results/20260601-1539-korean-annyeong/maps_raw.npz \
        --label sgmi-annyeong
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_SIBLING = str(Path(__file__).resolve().parent)
if _SIBLING not in sys.path:
    sys.path.insert(0, _SIBLING)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import anima_lora  # noqa: E402
import stage1 as s1  # noqa: E402
import stage2 as s2  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from inference import resolve_seed  # noqa: E402
from library.inference import generate, get_generation_settings  # noqa: E402
from library.inference import sampling as inference_utils  # noqa: E402
from library.inference.models import load_dit_model, load_shared_models  # noqa: E402
from probe_localization import DIT, QWEN3, VAE, build_args  # noqa: E402
from stage1_localize import load_cached_maps  # noqa: E402

# ---------------------------------------------------------------------------
# Monkeypatch the Euler step so SGMI sees every sampler boundary. The real step
# is `latents - (σ_i - σ_{i+1})·v`; we run it, then masked-replace the result
# (the latent at σ_{i+1}) so the *next* forward denoises the injected glyph.
# ---------------------------------------------------------------------------
_REAL_STEP = inference_utils.step
_INJECTOR: s2.SGMIInjector | None = None
_DEBUG = __import__("os").environ.get("STAGE2_DEBUG")
_TIMER = {"t": None}


def _patched_step(latents, noise_pred, sigmas, step_i):
    if _DEBUG:
        import time
        now = time.time()
        dt = None if _TIMER["t"] is None else now - _TIMER["t"]
        _TIMER["t"] = now
        fin = bool(torch.isfinite(latents).all())
        print(f"[dbg] step {step_i} dt={dt} latents dtype={latents.dtype} "
              f"finite={fin} absmax={float(latents.abs().max()):.3g}", flush=True)
    out = _REAL_STEP(latents, noise_pred, sigmas, step_i)
    inj = _INJECTOR
    if inj is not None and step_i + 1 < len(sigmas):
        out = inj.apply(out, float(sigmas[step_i + 1]))
        if _DEBUG and float(sigmas[step_i + 1]) <= inj.sigma_start:
            print(f"[dbg]   injected@σ={float(sigmas[step_i+1]):.3f} "
                  f"out finite={bool(torch.isfinite(out).all())} "
                  f"absmax={float(out.abs().max()):.3g} "
                  f"dtype={out.dtype} contig={out.is_contiguous()}", flush=True)
    return out


inference_utils.step = _patched_step


# ---------------------------------------------------------------------------
# Variants — the ablation ladder.
# ---------------------------------------------------------------------------
def variant_specs(a) -> dict:
    """Injector kwargs per variant name (None = base, no injection)."""
    win = dict(sigma_start=a.sigma_start, sigma_end=a.sigma_end)
    lg = dict(f0=a.f0, sigma_ratio=a.sigma_ratio)
    return {
        "base": None,
        # paper-faithful: cosine anneal + Log-Gabor on the noise-aligned latent.
        "sgmi": dict(anneal="cosine", lam_scale=a.lam_scale, use_log_gabor=True,
                     lg_order="post", **win, **lg),
        # strongest possible raw injection: flat hard-replace, no spectral filter.
        # "does the denoiser keep an injected glyph at all?"
        "sgmi_hard": dict(anneal="flat", lam_scale=1.0, use_log_gabor=False, **win),
        # cosine but no Log-Gabor — isolates what the band-pass buys.
        "sgmi_nolg": dict(anneal="cosine", lam_scale=a.lam_scale,
                          use_log_gabor=False, **win),
        # --- OOD-Korean improvement ladder (diagnostic; per-variant windows) ---
        # Lever A: extend the flat hard-replace DOWN into Anima's detail-resolving
        # tail (x0/strokes resolve at σ≲0.45) so glyphs lock instead of being
        # "cleaned up" by the free tail. Same engine as sgmi_hard, lower σ_end.
        "hard_deep": dict(anneal="flat", lam_scale=1.0, use_log_gabor=False,
                          sigma_start=a.sigma_start, sigma_end=0.35),
        "hard_deepest": dict(anneal="flat", lam_scale=1.0, use_log_gabor=False,
                             sigma_start=a.sigma_start, sigma_end=0.20),
        # Lever E: ink-shaped mask (only where the glyph has strokes), deep window.
        # Pins strokes without painting the whole sign dark; denoiser keeps the
        # sign's natural look between strokes. `_mask` is popped before construction.
        "hard_ink": dict(anneal="flat", lam_scale=1.0, use_log_gabor=False,
                         sigma_start=a.sigma_start, sigma_end=0.35, _mask="ink"),
        # A·E·slightly-soft: ink mask, deep, λ=0.85 so the denoiser reconciles.
        "hard_ink_soft": dict(anneal="flat", lam_scale=0.85, use_log_gabor=False,
                              sigma_start=a.sigma_start, sigma_end=0.30, _mask="ink"),
    }


def run_generation(args, gen_settings, shared_models, injector):
    """One full sampling run with `injector` (or None) active. Returns decoded PIL."""
    global _INJECTOR
    _INJECTOR = injector
    if injector is not None:
        injector.log.clear()
    try:
        latent = generate(args, gen_settings, shared_models)
    finally:
        _INJECTOR = None
    vae = shared_models["_vae"]
    img = anima_lora.decode_to_pil(vae, latent, gen_settings.device)
    return img[0] if isinstance(img, list) else img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default='a girl holding a sign that reads "안녕하세요"')
    p.add_argument("--target", default="안녕하세요", help="Stage-1 localize target.")
    p.add_argument("--glyph_text", default=None, help="String to rasterize (default: --target).")
    p.add_argument("--maps_npz", required=True, help="Cached Stage-1 map dump → R.")
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024])
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_weight", default=None)
    p.add_argument("--dit", default=DIT)
    p.add_argument("--text_encoder", default=QWEN3)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--label", default=None)
    p.add_argument("--variants", default="base,sgmi,sgmi_hard,sgmi_nolg")
    # SGMI knobs
    p.add_argument("--sigma_start", type=float, default=0.8)
    p.add_argument("--sigma_end", type=float, default=0.6)
    p.add_argument("--lam_scale", type=float, default=1.0)
    p.add_argument("--f0", type=float, default=0.25)
    p.add_argument("--sigma_ratio", type=float, default=0.65)
    p.add_argument("--fg", type=float, default=1.0, help="glyph foreground in [0,1]")
    p.add_argument("--bg", type=float, default=0.0, help="glyph background in [0,1]")
    # Stage-1 (gate-validated config; rarely overridden)
    p.add_argument("--thresh_mode", default="quantile")
    p.add_argument("--thr_q", type=float, default=0.85)
    p.add_argument("--region_select", default="centroid")
    p.add_argument("--grow_dilate", type=int, default=1)
    p.add_argument("--grow_min_frac", type=float, default=0.03)
    a = p.parse_args()

    glyph_text = a.glyph_text or a.target
    H, W = a.image_size
    h_lat, w_lat = H // 8, W // 8

    # --- Stage-1: derive R from the cached map dump (CPU) ---
    na = argparse.Namespace(maps_npz=a.maps_npz)
    _img, _rgb, maps_by_tl, groups, n_blocks, hp, wp, n_steps, _dev = load_cached_maps(na)
    res = s1.localize(
        maps_by_tl, hp=hp, wp=wp, h_lat=h_lat, w_lat=w_lat,
        anchor_mode="entity", select_mode="concentration", top_k=24, nbhd=3,
        thresh_mode=a.thresh_mode, thr_q=a.thr_q, region_select=a.region_select,
        grow_dilate=a.grow_dilate, grow_min_frac=a.grow_min_frac,
    )
    R = res.latent_mask  # (h_lat, w_lat) uint8
    cov = float(R.mean())
    box_lat = s2.mask_bbox(R)
    if box_lat is None:
        raise SystemExit("[stage2] Stage-1 produced an empty mask R; abstain.")
    box_px = s2.latent_box_to_pixel(box_lat, h_lat, w_lat, H, W)
    print(f"[stage2] R covers {cov*100:.1f}% of latent; latent bbox {box_lat}; "
          f"pixel bbox {box_px}; glyph_text={glyph_text!r}")

    # --- Load models (once) ---
    args = build_args(a)
    if getattr(args, "device", None) is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.seed = resolve_seed(args)
    gen_settings = get_generation_settings(args)
    device = gen_settings.device

    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}
    vae = anima_lora.load_vae(args.vae, device="cpu", disable_mmap=True,
                              spatial_chunk_size=64, disable_cache=True)
    vae.to(torch.bfloat16).eval().to(device)
    shared_models["_vae"] = vae
    anima = load_dit_model(args, device, torch.bfloat16)
    shared_models["model"] = anima

    # --- Render + VAE-encode the glyph reference ---
    glyph_rgb = s2.render_glyph_image(
        glyph_text, box_px, (H, W), fg=a.fg, bg=a.bg,
    )  # [H,W,3] float32 [0,1]
    glyph_t = torch.from_numpy(glyph_rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    glyph_t = glyph_t.to(device, torch.bfloat16)
    with torch.no_grad():
        z_ref = vae.encode_pixels_to_latents(glyph_t).float()  # [1,C,h_lat,w_lat]
    assert z_ref.shape[-2:] == (h_lat, w_lat), (z_ref.shape, h_lat, w_lat)
    mask_t = torch.from_numpy(R.astype(np.float32)).view(1, 1, h_lat, w_lat).to(device)
    # Ink-shaped mask: pool the glyph raster to latent res, keep cells that carry
    # stroke energy, AND with R. Pins strokes without painting the whole sign.
    ink_lat = glyph_rgb[:, :, 0].reshape(h_lat, 8, w_lat, 8).mean(axis=(1, 3))
    ink = ((ink_lat > 0.10).astype(np.float32)) * R.astype(np.float32)
    ink_t = torch.from_numpy(ink).view(1, 1, h_lat, w_lat).to(device)
    masks = {"region": mask_t, "ink": ink_t}
    print(f"[stage2] z_ref {tuple(z_ref.shape)}  region mask {float(R.mean())*100:.1f}%  "
          f"ink mask {float(ink.mean())*100:.1f}%")

    # --- Run the ladder ---
    specs = variant_specs(a)
    names = [v.strip() for v in a.variants.split(",") if v.strip()]
    run_dir = make_run_dir("freetext", a.label or f"sgmi-{a.target[:6]}")
    images, inj_logs = {}, {}
    for name in names:
        kw = specs.get(name, None)
        injector = None
        if kw is not None:
            kw = dict(kw)
            mask_for = masks[kw.pop("_mask", "region")]
            injector = s2.SGMIInjector(z_ref=z_ref.to(device), mask=mask_for, **kw)
        print(f"[stage2] generating variant {name!r} "
              f"({'base' if injector is None else 'SGMI'})...")
        img = run_generation(args, gen_settings, shared_models, injector)
        img.save(run_dir / f"{name}.png")
        images[name] = np.asarray(img.convert("RGB"))
        if injector is not None:
            inj_logs[name] = injector.log
            n_inj = len(injector.log)
            print(f"[stage2]   injected at {n_inj} steps; "
                  f"λ range [{min((s['lam'] for s in injector.log), default=0):.2f}, "
                  f"{max((s['lam'] for s in injector.log), default=0):.2f}]")

    # --- Save glyph raster + R overlay, and a comparison grid ---
    Image.fromarray((glyph_rgb * 255).astype(np.uint8)).save(run_dir / "glyph_raster.png")

    n = len(names) + 1  # +1 for the glyph/R panel
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.4))
    if n == 1:
        axes = [axes]
    # glyph + R overlay on the base decode
    base_rgb = images.get("base", next(iter(images.values())))
    axes[0].imshow(base_rgb)
    R_px = s1.to_latent_mask(res.grown_mask, H, W).astype(float)  # upsample for overlay
    axes[0].imshow(np.where(R_px > 0, 1.0, np.nan), cmap="spring", alpha=0.35,
                   interpolation="nearest", extent=(0, W, H, 0))
    axes[0].imshow(np.where(glyph_rgb[:, :, 0] > 0.3, 1.0, np.nan), cmap="cool",
                   alpha=0.7, interpolation="nearest", extent=(0, W, H, 0))
    axes[0].set_title(f"R ({cov*100:.1f}%) + glyph raster", fontsize=9)
    axes[0].axis("off")
    for ax, name in zip(axes[1:], names):
        ax.imshow(images[name])
        tag = "base" if specs.get(name) is None else "SGMI"
        ax.set_title(f"{name} [{tag}]", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"FreeText Stage-2 SGMI — {glyph_text!r}  (cfg {a.guidance_scale:g}, seed {a.seed})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(run_dir / "comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "target": a.target, "glyph_text": glyph_text,
        "prompt": a.prompt,
        "R_latent_frac": cov, "R_latent_bbox": list(box_lat),
        "R_pixel_bbox": list(box_px),
        "variants": names,
        "sgmi": dict(sigma_start=a.sigma_start, sigma_end=a.sigma_end,
                     lam_scale=a.lam_scale, f0=a.f0, sigma_ratio=a.sigma_ratio),
        "injection_log": inj_logs,
        "stage1": dict(thresh_mode=a.thresh_mode, thr_q=a.thr_q,
                       region_select=a.region_select, grow_dilate=a.grow_dilate,
                       grow_min_frac=a.grow_min_frac),
    }
    artifacts = ["comparison.png", "glyph_raster.png"] + [f"{n}.png" for n in names]
    write_result(run_dir, script=__file__, args=a, metrics=metrics,
                 label=a.label, artifacts=artifacts, device=device)
    print(f"[stage2] results -> {run_dir}")
    print(f"[stage2] READ THE GRIDS: {run_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
