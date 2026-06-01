"""FreeText Stage-1 driver — endogenous I2T attention -> binary writing mask R.

Phase-1 of FreeText (arXiv 2601.00535) on Anima. The Phase-0 probe
(``probe_localization.py``) established that base-Anima's image->text
cross-attention localizes the writing region (GO, 2-3.6x uniform). This driver
turns that raw signal into the actual Stage-1 product: the high-confidence
latent writing mask ``R`` that Stage-2 (SGMI) injects glyph priors into.

It reuses the probe's *faithful* attention capture verbatim (eager
``softmax(QK^T)`` recompute on every block's ``cross_attn`` — cross-attn has no
RoPE so this reproduces the fused kernel), then runs the pure pipeline in
``stage1.py``: anchor map (Eq 2) -> soft-IoU top-K timestep-layer selection
(Eq 3-4) -> neighborhood-denoise / Otsu / DBSCAN / topology score / latent
resize (Eq 5-6).

Run:
    python bench/freetext/stage1_localize.py --label anima-base
    python bench/freetext/stage1_localize.py \
        --prompt 'a poster that reads "HELLO"' --target HELLO \
        --gt_box 280 360 760 520        # optional pixel-space box -> IoU
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Sibling modules (`stage1`, `probe_localization`) live next to this file. Python
# auto-adds the script dir, but be explicit so `-m` / import-from-elsewhere works.
_SIBLING = str(Path(__file__).resolve().parent)
if _SIBLING not in sys.path:
    sys.path.insert(0, _SIBLING)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import anima_lora  # noqa: E402
import stage1 as s1  # noqa: E402  (sibling module)
from bench._common import make_run_dir, write_result  # noqa: E402
from inference import resolve_seed  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.inference import generate, get_generation_settings  # noqa: E402
from library.inference.models import load_dit_model, load_shared_models  # noqa: E402

# Reuse the probe's capture machinery verbatim — same eager-recompute hook,
# token grouping, cond-step bookkeeping, and shared capture state.
from probe_localization import (  # noqa: E402
    DEFAULT_NEG,
    DEFAULT_PROMPT,
    DIT,
    QWEN3,
    VAE,
    _STATE,
    _STORE,
    _install_hooks,
    build_args,
    compute_token_groups,
    cond_steps,
    patch_grid,
)


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------
def _overlay(ax, img_rgb, heat, title, alpha=0.5, cmap="jet", interp="bilinear"):
    H, W = img_rgb.shape[:2]
    ax.imshow(img_rgb, extent=(0, W, H, 0))
    h = np.asarray(heat, dtype=np.float64)
    h = h - h.min()
    if h.max() > 0:
        h = h / h.max()
    ax.imshow(h, extent=(0, W, H, 0), cmap=cmap, alpha=alpha, interpolation=interp)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def render_pipeline(img_rgb, res: s1.Stage1Result, run_dir, gt_grid=None):
    """Six-panel: decoded | aggregate M | denoised | Otsu B | DBSCAN | R."""
    H, W = img_rgb.shape[:2]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    ax = axes.ravel()

    ax[0].imshow(img_rgb, extent=(0, W, H, 0))
    ax[0].set_title("decoded", fontsize=9)
    ax[0].axis("off")
    if gt_grid is not None:
        _draw_grid_box(ax[0], gt_grid, W, H, color="lime")

    _overlay(ax[1], img_rgb, res.aggregate,
             f"aggregate M (top-{res.params['top_k']})", alpha=0.55)
    _overlay(ax[2], img_rgb, res.denoised, f"neighborhood-denoised\n(otsu={res.otsu_thr:.2f})",
             alpha=0.55)

    # Otsu binary
    ax[3].imshow(img_rgb, extent=(0, W, H, 0))
    ax[3].imshow(res.binary.astype(float), extent=(0, W, H, 0), cmap="Reds",
                 alpha=0.45, interpolation="nearest")
    ax[3].set_title(f"Otsu binary B ({int(res.binary.sum())} patches)", fontsize=9)
    ax[3].axis("off")

    # DBSCAN regions (colored), best outlined
    ax[4].imshow(img_rgb, extent=(0, W, H, 0))
    lm = res.label_map.astype(float)
    lm_show = np.where(res.label_map >= 0, lm, np.nan)
    ax[4].imshow(lm_show, extent=(0, W, H, 0), cmap="tab10", alpha=0.6,
                 interpolation="nearest")
    n_reg = len(res.regions)
    sel = res.selected_idx
    qtxt = ""
    if sel >= 0 and res.region_stats.get("scores"):
        qtxt = f"\nbest q={res.region_stats['scores'][sel]:.2f} size={res.region_stats['sizes'][sel]}"
    ax[4].set_title(f"DBSCAN regions ({n_reg}){qtxt}", fontsize=9)
    ax[4].axis("off")

    # Final writing mask R (grown extent) over the image; seed (peak region)
    # outlined inside it when a grow step expanded it.
    ax[5].imshow(img_rgb, extent=(0, W, H, 0))
    ax[5].imshow(res.grown_mask.astype(float), extent=(0, W, H, 0), cmap="spring",
                 alpha=0.45, interpolation="nearest")
    grow_note = ""
    if not np.array_equal(res.grown_mask, res.region_mask):
        seed_show = np.where(res.region_mask, 1.0, np.nan)
        ax[5].imshow(seed_show, extent=(0, W, H, 0), cmap="cool", alpha=0.85,
                     interpolation="nearest")
        pr = res.params
        grow_note = (f"\nseed {int(res.region_mask.sum())}p → grown "
                     f"{int(res.grown_mask.sum())}p "
                     f"(dil={pr['grow_dilate']} bbox={pr['grow_bbox']} "
                     f"minf={pr['grow_min_frac']:g})")
    ax[5].set_title(f"writing mask R ({res.latent_mask.shape[0]}x"
                    f"{res.latent_mask.shape[1]} latent){grow_note}", fontsize=9)
    ax[5].axis("off")
    if gt_grid is not None:
        _draw_grid_box(ax[5], gt_grid, W, H, color="lime")

    fig.suptitle("FreeText Stage-1 — attention-guided text-region localization", fontsize=12)
    fig.tight_layout()
    out = run_dir / "stage1_pipeline.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_grid_box(ax, grid_mask, W, H, color="lime"):
    ys, xs = np.nonzero(grid_mask)
    if len(xs) == 0:
        return
    hp, wp = grid_mask.shape
    x0 = xs.min() / wp * W
    x1 = (xs.max() + 1) / wp * W
    y0 = ys.min() / hp * H
    y1 = (ys.max() + 1) / hp * H
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=color, lw=1.6, ls="--"))


def render_selection_heatmap(img_rgb, res: s1.Stage1Result, n_steps, n_blocks, run_dir):
    """Stage-1 'timestep-layer selection' evidence (paper Fig 3 / §3.1.2):
    step x block soft-IoU matrix (top-K outlined), the per-map concentration
    matrix, and the reference Y the soft-IoU scored against."""
    H, W = img_rgb.shape[:2]
    score_mat = np.full((n_steps, n_blocks), np.nan)
    for (si, bp), sc in zip(res.keys, res.scores):
        score_mat[si, bp] = sc
    conc_mat = None
    if res.concentrations is not None:
        conc_mat = np.full((n_steps, n_blocks), np.nan)
        for (si, bp), c in zip(res.keys, res.concentrations):
            conc_mat[si, bp] = c

    fig, axx = plt.subplots(1, 3, figsize=(15, 4.6))
    sel = set(res.selected_tl)

    im0 = axx[0].imshow(score_mat, aspect="auto", cmap="viridis", origin="lower")
    for (si, bp) in sel:
        axx[0].add_patch(Rectangle((bp - 0.5, si - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=1.0))
    axx[0].set_title(f"selection score ({res.params['select_mode']}) — red=top-{res.params['top_k']}",
                     fontsize=9)
    axx[0].set_xlabel("block"); axx[0].set_ylabel("step")
    fig.colorbar(im0, ax=axx[0], fraction=0.046)

    if conc_mat is not None:
        im1 = axx[1].imshow(conc_mat, aspect="auto", cmap="magma", origin="lower")
        axx[1].set_title("concentration (top-5% mass) per (t,l)", fontsize=9)
        axx[1].set_xlabel("block"); axx[1].set_ylabel("step")
        fig.colorbar(im1, ax=axx[1], fraction=0.046)
    else:
        axx[1].axis("off")

    axx[2].imshow(img_rgb, extent=(0, W, H, 0))
    refh = res.reference - res.reference.min()
    if refh.max() > 0:
        refh = refh / refh.max()
    axx[2].imshow(refh, extent=(0, W, H, 0), cmap="jet", alpha=0.55,
                  interpolation="bilinear")
    axx[2].set_title(f"reference Y (conc_q={res.params.get('ref_conc_q')})", fontsize=9)
    axx[2].axis("off")

    fig.suptitle("Stage-1 timestep-layer selection", fontsize=12)
    fig.tight_layout()
    out = run_dir / "selection_heatmap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Capture (GPU) vs. cached-load (CPU, for fast offline param sweeps).
# ---------------------------------------------------------------------------
def capture_maps(a):
    """Run a generation with the probe's eager-recompute hook and reduce to
    per-(step, block) entity/sink maps. Returns the tuple main() unpacks."""
    args = build_args(a)
    if getattr(args, "device", None) is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.seed = resolve_seed(args)
    gen_settings = get_generation_settings(args)
    device = gen_settings.device
    print(f"[stage1] target={a.target!r} cfg={a.guidance_scale} steps={a.infer_steps} "
          f"size={a.image_size} device={device}")

    tokenizer = anima_utils.load_qwen3_tokenizer(args.text_encoder)
    groups = compute_token_groups(tokenizer, a.prompt, a.target, max_length=512)
    print(f"[stage1] target tokens {groups['target']} -> {groups['target_tokens']}; "
          f"#real={groups['n_real']} #pad={groups['n_pad']} #special={len(groups['special'])}")
    if not groups["target"]:
        print("[stage1] WARNING: target substring not tokenized; entity map empty.")

    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}
    vae = anima_lora.load_vae(args.vae, device="cpu", disable_mmap=True,
                              spatial_chunk_size=64, disable_cache=True)
    vae.to(torch.bfloat16).eval().to(device)

    anima = load_dit_model(args, device, torch.bfloat16)
    _, pre_h = _install_hooks(anima)
    shared_models["model"] = anima

    _STORE.clear()
    _STATE["groups"] = groups
    _STATE["fwd"] = -1
    _STATE["on"] = True
    latent = generate(args, gen_settings, shared_models)
    _STATE["on"] = False
    pre_h.remove()

    img = anima_lora.decode_to_pil(vae, latent, device)
    if isinstance(img, list):
        img = img[0]
    img_rgb = np.asarray(img.convert("RGB"))

    steps = cond_steps()
    n_blocks = len(anima.blocks)
    h_latent, w_latent = a.image_size[0] // 8, a.image_size[1] // 8
    any_rec = next(iter(_STORE.values()))
    hp, wp = patch_grid(h_latent, w_latent, any_rec["L_img"])
    print(f"[stage1] captured {len(_STORE)} maps; {len(steps)} cond steps; "
          f"grid {hp}x{wp}; blocks={n_blocks}")

    maps_by_tl = {}
    for si, fwd in enumerate(steps):
        for bp in range(n_blocks):
            rec = _STORE.get((fwd, bp))
            if rec is not None:
                maps_by_tl[(si, bp)] = {
                    "entity": np.asarray(rec["entity"], dtype=np.float32),
                    "sink": np.asarray(rec["pad"], dtype=np.float32),
                }
    return img, img_rgb, maps_by_tl, groups, n_blocks, hp, wp, len(steps), device


def dump_maps_npz(path, maps_by_tl, hp, wp, h_lat, w_lat, n_blocks, n_steps, groups):
    keys = sorted(maps_by_tl.keys())
    entity = np.stack([maps_by_tl[k]["entity"] for k in keys]).astype(np.float16)
    sink = np.stack([maps_by_tl[k]["sink"] for k in keys]).astype(np.float16)
    np.savez_compressed(
        path,
        entity=entity, sink=sink, keys=np.array(keys, dtype=np.int16),
        grid=np.array([hp, wp, h_lat, w_lat, n_blocks, n_steps], dtype=np.int32),
        target_tokens=np.array(groups["target_tokens"], dtype=object),
        gcounts=np.array([groups["n_real"], groups["n_pad"],
                          len(groups["special"])], dtype=np.int32),
    )
    print(f"[stage1] dumped captured maps -> {path}")


def load_cached_maps(a):
    """Load maps captured by a prior --dump_maps run; no GPU / no generation."""
    from PIL import Image

    npz_path = Path(a.maps_npz)
    d = np.load(npz_path, allow_pickle=True)
    hp, wp, h_lat, w_lat, n_blocks, n_steps = (int(x) for x in d["grid"])
    keys = [tuple(int(v) for v in k) for k in d["keys"]]
    entity, sink = d["entity"].astype(np.float32), d["sink"].astype(np.float32)
    maps_by_tl = {
        k: {"entity": entity[i], "sink": sink[i]} for i, k in enumerate(keys)
    }
    n_real, n_pad, n_special = (int(x) for x in d["gcounts"])
    groups = {
        "target_tokens": list(d["target_tokens"]),
        "n_real": n_real, "n_pad": n_pad,
        "special": [None] * n_special,
    }
    img_path = npz_path.parent / "decoded.png"
    img = Image.open(img_path).convert("RGB")
    img_rgb = np.asarray(img)
    print(f"[stage1] loaded {len(keys)} cached maps from {npz_path} "
          f"(grid {hp}x{wp}, {n_steps} steps, {n_blocks} blocks)")
    return img, img_rgb, maps_by_tl, groups, n_blocks, hp, wp, n_steps, "cpu"


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--target", default="ANIMA")
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="H W")
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
    # Stage-1 knobs (forwarded to stage1.localize)
    p.add_argument("--anchor_mode", default="entity",
                   choices=["entity", "sink", "entity_sink"])
    p.add_argument("--anchor_combine", default="normsum", choices=["normsum", "rawsum"])
    p.add_argument("--select_mode", default="concentration",
                   choices=["concentration", "softiou"],
                   help="concentration = direct peakiness ranking (Anima default); "
                        "softiou = paper Eq-3 soft-IoU vs reference Y.")
    p.add_argument("--ref_mode", default="concentration",
                   choices=["concentration", "consensus", "sink"],
                   help="soft-IoU reference Y (also rendered as a diagnostic).")
    p.add_argument("--ref_reduce", default="mean", choices=["mean", "median"])
    p.add_argument("--ref_conc_q", type=float, default=0.75,
                   help="ref_mode=concentration: keep top (1-q) peakiest maps for Y.")
    p.add_argument("--top_k", type=int, default=24)
    p.add_argument("--nbhd", type=int, default=3)
    p.add_argument("--thresh_mode", default="otsu",
                   choices=["otsu", "quantile", "peak_rel"],
                   help="binarization: otsu (contrast-sensitive); quantile/peak_rel "
                        "(contrast-invariant, pair with --region_select peak for "
                        "un-rendered text).")
    p.add_argument("--thr_q", type=float, default=0.85,
                   help="thresh_mode=quantile: keep the top (1-thr_q) patches.")
    p.add_argument("--thr_rel", type=float, default=0.55,
                   help="thresh_mode=peak_rel: keep patches >= thr_rel * peak.")
    p.add_argument("--dbscan_eps", type=float, default=1.5)
    p.add_argument("--dbscan_min_samples", type=int, default=4)
    p.add_argument("--tau_q", type=float, default=0.8)
    p.add_argument("--region_select", default="mass",
                   choices=["q", "qmass", "mass", "peak", "centroid"],
                   help="centroid = region at the mass-weighted attention centroid "
                        "(robust to sink peaks on soft maps); peak = region holding "
                        "the global argmax; mass = largest warm blob.")
    # Grow the placed-but-tight seed to a Stage-2 injection extent (no-op by default).
    p.add_argument("--grow_dilate", type=int, default=0,
                   help="dilate the selected region by N patches (isotropic).")
    p.add_argument("--grow_bbox", action="store_true",
                   help="rectangularize the grown region to its bounding box.")
    p.add_argument("--grow_min_frac", type=float, default=0.0,
                   help="coverage floor (frac of patch grid); dilate until met.")
    p.add_argument("--grow_max_dilate", type=int, default=8,
                   help="cap on extra dilation iters used to reach grow_min_frac.")
    p.add_argument("--gt_box", type=float, nargs=4, default=None,
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Optional pixel-space writing box for IoU validation.")
    # Fast offline iteration: capture once with --dump_maps, then sweep Stage-1
    # params against the cached maps with --maps_npz (no GPU / no generation).
    p.add_argument("--dump_maps", action="store_true",
                   help="Also save maps_raw.npz (captured per-(t,l) maps) for reuse.")
    p.add_argument("--maps_npz", default=None,
                   help="Load captured maps from a prior --dump_maps run; skip generation.")
    a = p.parse_args()

    H, W = a.image_size
    h_latent, w_latent = H // 8, W // 8

    if a.maps_npz is not None:
        img, img_rgb, maps_by_tl, groups, n_blocks, hp, wp, n_steps, device = (
            load_cached_maps(a)
        )
    else:
        img, img_rgb, maps_by_tl, groups, n_blocks, hp, wp, n_steps, device = (
            capture_maps(a)
        )

    res = s1.localize(
        maps_by_tl, hp=hp, wp=wp, h_lat=h_latent, w_lat=w_latent,
        anchor_mode=a.anchor_mode, anchor_combine=a.anchor_combine,
        select_mode=a.select_mode, ref_mode=a.ref_mode, ref_reduce=a.ref_reduce,
        ref_conc_q=a.ref_conc_q,
        top_k=a.top_k, nbhd=a.nbhd,
        thresh_mode=a.thresh_mode, thr_q=a.thr_q, thr_rel=a.thr_rel,
        dbscan_eps=a.dbscan_eps,
        dbscan_min_samples=a.dbscan_min_samples, tau_q=a.tau_q,
        region_select=a.region_select,
        grow_dilate=a.grow_dilate, grow_bbox=a.grow_bbox,
        grow_min_frac=a.grow_min_frac, grow_max_dilate=a.grow_max_dilate,
    )

    # Unsupervised quality metrics: attention lift inside R vs outside. Measured on
    # the grown mask (what Stage-2 actually injects into) — growing into cooler
    # margin lowers lift, which is the placement/extent tradeoff made visible.
    rm = res.grown_mask
    inside = float(res.aggregate[rm].mean()) if rm.any() else 0.0
    outside = float(res.aggregate[~rm].mean()) if (~rm).any() else 0.0
    lift = inside / (outside + 1e-9)

    gt_grid = None
    iou_metrics = {}
    if a.gt_box is not None:
        gt_grid = s1.box_to_grid_mask(a.gt_box, hp, wp, W, H)
        iou_metrics = {
            "gt_box": a.gt_box,
            "iou_region_vs_gt": s1.hard_iou(res.grown_mask, gt_grid),
            "iou_seed_vs_gt": s1.hard_iou(res.region_mask, gt_grid),
            "iou_otsu_vs_gt": s1.hard_iou(res.binary, gt_grid),
        }
        print(f"[stage1] IoU(region,gt)={iou_metrics['iou_region_vs_gt']:.3f}  "
              f"IoU(otsu,gt)={iou_metrics['iou_otsu_vs_gt']:.3f}")

    run_dir = make_run_dir("freetext", a.label or f"stage1-{a.target.lower()}-cfg{a.guidance_scale:g}")
    img.save(run_dir / "decoded.png")
    p1 = render_pipeline(img_rgb, res, run_dir, gt_grid=gt_grid)
    p2 = render_selection_heatmap(img_rgb, res, n_steps, n_blocks, run_dir)
    artifacts = ["decoded.png", p1.name, p2.name, "stage1.npz"]
    np.savez_compressed(
        run_dir / "stage1.npz",
        aggregate=res.aggregate.astype(np.float32),
        denoised=res.denoised.astype(np.float32),
        binary=res.binary.astype(np.uint8),
        region_mask=res.region_mask.astype(np.uint8),
        grown_mask=res.grown_mask.astype(np.uint8),
        latent_mask=res.latent_mask,
        label_map=res.label_map.astype(np.int16),
        reference=res.reference.astype(np.float32),
        scores=res.scores.astype(np.float32),
    )
    if a.dump_maps and a.maps_npz is None:
        dump_maps_npz(run_dir / "maps_raw.npz", maps_by_tl, hp, wp, h_latent,
                      w_latent, n_blocks, n_steps, groups)
        artifacts.append("maps_raw.npz")

    n_sel_mid = sum(1 for (si, bp) in res.selected_tl if 0.10 * n_steps <= si <= 0.70 * n_steps)
    metrics = {
        "target": a.target,
        "target_tokens": groups["target_tokens"],
        "n_real_tokens": groups["n_real"],
        "n_pad_tokens": groups["n_pad"],
        "n_special": len(groups["special"]),
        "patch_grid": [hp, wp],
        "latent_shape": [h_latent, w_latent],
        "n_cond_steps": n_steps,
        "n_blocks": n_blocks,
        "n_candidate_maps": len(res.keys),
        "n_regions": len(res.regions),
        "selected_region_idx": res.selected_idx,
        "region_stats": res.region_stats,
        "region_mask_patches": int(res.region_mask.sum()),
        "grown_mask_patches": int(res.grown_mask.sum()),
        "latent_mask_frac": float(res.latent_mask.mean()),
        "otsu_thr": res.otsu_thr,
        "agg_lift_in_vs_out": lift,
        "agg_mean_inside_R": inside,
        "agg_mean_outside_R": outside,
        "selected_tl_blocks": sorted({bp for (_, bp) in res.selected_tl}),
        "selected_tl_steps": sorted({si for (si, _) in res.selected_tl}),
        "n_selected_in_mid_window": n_sel_mid,
        "params": res.params,
        **iou_metrics,
    }
    write_result(run_dir, script=__file__, args=a, metrics=metrics, label=a.label,
                 artifacts=artifacts, device=device)

    print(f"[stage1] regions={len(res.regions)} best_idx={res.selected_idx} "
          f"R covers {metrics['latent_mask_frac']*100:.1f}% of latent; "
          f"attention lift in/out={lift:.2f}")
    print(f"[stage1] selected blocks={metrics['selected_tl_blocks']}")
    print(f"[stage1] results -> {run_dir}")


if __name__ == "__main__":
    main()
