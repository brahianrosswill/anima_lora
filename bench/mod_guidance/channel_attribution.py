"""Channel-attribution bench for the mod-guidance pooled-text path.

WHY THIS EXISTS
---------------
`docs/findings/mod_guidance_quality_tag_axis.md` measured everything in
`pooled_text_proj` GEOMETRY (cosines between projected pooled vectors) and never
sampled an image. Two leaps were never tested:

  1. "quality axis" — the axis it maps is really a *content-magnitude* axis
     (an arbitrary artist tag drives it 3-4x harder than `score_9`, which makes
     no sense for a quality lever). The doc half-admits this ("isn't pure
     quality ... correlates with strong, specific content").
  2. "double-drive degrades quality / DC-blowout" — inferred from cosines,
     never observed in image space.

A tag edit does not act through one channel. It enters the DiT through TWO
separable inputs:

    pooled / mod channel :  crossattn_emb.max(1) -> pooled_text_proj -> AdaLN
    cross-attn channel   :  the full crossattn_emb sequence -> cross-attention

The pooled channel is permutation-invariant (max over the sequence) and collapses
to a single vector; the cross-attn channel is order- and shape-sensitive. They
are *separable at the forward* via `pooled_text_override` (models.py:1643): we can
run cross-attn from prompt A while feeding the pooled vector of prompt B.

This bench answers the live, mechanistic questions in IMAGE / LATENT space:

    swap       Causal channel decomposition of a tag edit. For (base, base+tag)
               render the 2x2 of {cross=base|tag} x {pool=base|tag} and split the
               image movement into a cross-attn delta and a pooled delta; measure
               whether the two channels reinforce (preload), conflict (cancel), or
               are orthogonal. THE KEY EXPERIMENT.
    order      Pure cross-attn isolator. Permute the tag order: the pooled vector
               is provably identical (max is order-invariant), so 100% of any image
               movement is cross-attn. Compares against the same-prompt seed floor.
    intensity  Pooled-channel response curve. Sweep the mod-guidance steering
               weight w (the doc's actual double-drive mechanism) and measure
               off-baseline movement + the DC-blowout proxy (pixel spatial std /
               tone shift). Directly tests "does a hard push drift quality worse".

All three save image grids -- READ THE GRIDS, the scalar metrics are a guide, not
the verdict (cf. the pose-blind PE-cosine lesson elsewhere in this repo).

Outputs land in bench/mod_guidance/results/<ts>[-label]/ via bench/_common.py.

Run
---
    uv run python bench/mod_guidance/channel_attribution.py \
        --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
        --experiment all --label probe

    # smoke (1 prompt, 1 tag, fewer steps)
    uv run python bench/mod_guidance/channel_attribution.py \
        --pooled_text_proj output/ckpt/pooled_text_proj-0530.safetensors \
        --experiment swap --prompts "1girl, solo, outdoors" --tags "score_9" \
        --infer_steps 12 --label smoke
"""

from __future__ import annotations

import argparse
import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# bench/ is not an installed package -- bootstrap the repo root onto sys.path so
# `library` / `bench._common` import the same way the sibling benches do.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("mod_guidance.channel_attribution")

# A near-square on-distribution training bucket (W, H) from CONSTANT_TOKEN_BUCKETS.
DEFAULT_W, DEFAULT_H = 1024, 1008

DEFAULT_PROMPTS = [
    "1girl, solo, looking at viewer, outdoors, standing",
    "a red fox sitting in a snowy forest, soft morning light",
]
# A quality tag, a second quality word, and a content tag. Pass artist/character
# tags via --tags to probe the finding's "named entities dominate the axis" claim.
DEFAULT_TAGS = ["score_9", "holding a sword"]
DEFAULT_NEG = ""

EPS = 1e-8


# --------------------------------------------------------------------------- #
# Render job plumbing
# --------------------------------------------------------------------------- #
@dataclass
class RenderJob:
    """One image to render. `cross_prompt` feeds cross-attention; `pool_prompt`'s
    pooled vector feeds the AdaLN mod channel (None -> use cross_prompt's). When
    `mod_w` > 0 the steering buffers are armed (intensity experiment)."""

    key: str
    cross_prompt: str
    pool_prompt: Optional[str]
    seed: int
    mod_w: float = 0.0
    mod_pos: Optional[str] = None
    mod_neg: Optional[str] = None


@dataclass
class Rendered:
    latent: torch.Tensor = None  # (16, H_lat, W_lat) fp32 cpu
    pe: torch.Tensor = None  # (D,) unit fp32 cpu
    pixel_std: float = 0.0  # mean over channels of per-image spatial std
    tone: float = 0.0  # mean abs pixel value in [0,1] (DC / pink proxy)


# --------------------------------------------------------------------------- #
# Stage 1: text encoding (TE loaded, then freed)
# --------------------------------------------------------------------------- #
def encode_prompts(model, prompts: list[str], args, device) -> dict[str, torch.Tensor]:
    """Encode each unique prompt to a (1, 512, 1024) crossattn_emb (post-LLMAdapter,
    padded) -- the *exact* tensor the live mod-guidance path pools, via the real
    `_encode_prompt_for_mod` helper against the loaded DiT (TE + DiT briefly
    coexist, same as the shipped mod-guidance setup). Frees the TE after."""
    from library.inference.text import ensure_text_strategies
    from library.inference.models import load_text_encoder
    from library.inference.corrections.mod_guidance import _encode_prompt_for_mod
    from library.runtime.device import clean_memory_on_device

    ensure_text_strategies(args.text_encoder)
    te = load_text_encoder(text_encoder=args.text_encoder, device=device)
    te.eval()

    out: dict[str, torch.Tensor] = {}
    for p in prompts:
        if p not in out:
            out[p] = _encode_prompt_for_mod(p, model, te, device).to("cpu", dtype=torch.bfloat16)
            logger.info(f"  encoded: {p!r} -> {tuple(out[p].shape)}")

    del te
    gc.collect()
    clean_memory_on_device(device)
    return out


# --------------------------------------------------------------------------- #
# Stage 2: denoising (DiT loaded, then freed)
# --------------------------------------------------------------------------- #
def build_dit(args, device):
    from library.anima import weights as anima_utils

    model = anima_utils.load_anima_model(
        device,
        args.dit,
        attn_mode=args.attn_mode,
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima_utils.load_pooled_text_proj(model, args.pooled_text_proj, device)
    model.to(device)
    model.eval()
    return model


def _pool(crossattn: torch.Tensor) -> torch.Tensor:
    """pooled = max over the (padded) sequence -> (1, 1024). Matches the live path."""
    return crossattn.max(dim=1).values


def _set_mod_buffers(model, delta_unit, schedule):
    model._mod_guidance_delta.copy_(
        delta_unit.to(model._mod_guidance_delta.device, dtype=model._mod_guidance_delta.dtype)
    )
    model._mod_guidance_schedule.copy_(
        torch.tensor(schedule, device=model._mod_guidance_schedule.device, dtype=model._mod_guidance_schedule.dtype)
    )
    model._mod_guidance_final_w.fill_(0.0)


def _zero_mod_buffers(model):
    model._mod_guidance_delta.zero_()
    model._mod_guidance_schedule.zero_()
    model._mod_guidance_final_w.fill_(0.0)


def render_jobs(model, jobs, cross_cache, args, device) -> dict[str, torch.Tensor]:
    """Render every job to a clean latent (16, H_lat, W_lat) fp32 cpu."""
    from library.inference import sampling as inference_utils

    H_lat, W_lat = args.height // 8, args.width // 8
    dtype = torch.bfloat16
    proj_dtype = model.pooled_text_proj[0].weight.dtype

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    timesteps = timesteps.to(device, dtype=dtype)
    do_cfg = abs(args.guidance_scale - 1.0) > 1e-6

    neg_cross = cross_cache[args.negative].to(device, dtype=dtype)
    neg_pool = _pool(neg_cross).to(proj_dtype)
    padding_mask = torch.zeros(1, 1, H_lat, W_lat, dtype=dtype, device=device)

    out: dict[str, torch.Tensor] = {}
    for job in jobs:
        # Steering buffers (intensity experiment) or off.
        if job.mod_w > 0.0:
            from library.inference.corrections.mod_guidance import build_mod_schedule

            pos_c = cross_cache[job.mod_pos].to(device, dtype=dtype)
            negc = cross_cache[job.mod_neg].to(device, dtype=dtype)
            with torch.no_grad():
                d = model.pooled_text_proj(_pool(pos_c).to(proj_dtype)) - model.pooled_text_proj(
                    _pool(negc).to(proj_dtype)
                )
            sched_args = argparse.Namespace(
                mod_w=job.mod_w, mod_start_layer=8, mod_end_layer=27, mod_taper=0
            )
            _set_mod_buffers(model, d, build_mod_schedule(sched_args, len(model.blocks)))
        else:
            _zero_mod_buffers(model)

        cross_pos = cross_cache[job.cross_prompt].to(device, dtype=dtype)
        pool_src = job.pool_prompt if job.pool_prompt is not None else job.cross_prompt
        pool_override = _pool(cross_cache[pool_src].to(device, dtype=dtype)).to(proj_dtype)

        sampler = inference_utils.ERSDESampler(sigmas, seed=job.seed, device=device)
        gen = torch.Generator(device=device).manual_seed(job.seed)
        latents = torch.randn(
            (1, 16, 1, H_lat, W_lat), dtype=dtype, device=device, generator=gen
        )

        for i, t in enumerate(timesteps):
            t_exp = t.expand(latents.shape[0])
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                noise_pred = model.forward_mini_train_dit(
                    latents, t_exp, cross_pos,
                    padding_mask=padding_mask, pooled_text_override=pool_override,
                )
                if do_cfg:
                    uncond = model.forward_mini_train_dit(
                        latents, t_exp, neg_cross,
                        padding_mask=padding_mask, pooled_text_override=neg_pool,
                    )
                    noise_pred = uncond + args.guidance_scale * (noise_pred - uncond)
            denoised = latents.float() - sigmas[i] * noise_pred.float()
            latents = sampler.step(latents, denoised, i).to(latents.dtype)

        out[job.key] = latents.float().squeeze(2).squeeze(0).cpu()  # (16,H,W)
        logger.info(f"  rendered: {job.key}")

    _zero_mod_buffers(model)
    return out


# --------------------------------------------------------------------------- #
# Stage 3+4: decode (VAE) + perceptual features (PE)
# --------------------------------------------------------------------------- #
def decode_and_featurize(latents: dict, args, device) -> dict[str, Rendered]:
    from library.models.qwen_vae import load_vae
    from library.runtime.device import clean_memory_on_device

    vae = load_vae(args.vae, device=device, dtype=torch.bfloat16, eval=True)
    vae.to(device)

    pixels: dict[str, torch.Tensor] = {}  # [-1,1] CHW fp32 cpu
    rendered: dict[str, Rendered] = {}
    with torch.no_grad():
        for key, lat in latents.items():
            z = lat.unsqueeze(0).unsqueeze(2).to(device, dtype=vae.dtype)  # (1,16,1,H,W)
            px = vae.decode_to_pixels(z)
            if px.ndim == 5:
                px = px.squeeze(2)
            px = px.to("cpu", dtype=torch.float32)[0].clamp(-1, 1)  # (3,H,W)
            pixels[key] = px
            r = Rendered()
            # DC-blowout proxies: spatial std collapse + tone toward a flat fill.
            r.pixel_std = float(px.std(dim=(-2, -1)).mean())
            r.tone = float(((px + 1) / 2).mean())
            rendered[key] = r
    vae.to("cpu")
    del vae
    gc.collect()
    clean_memory_on_device(device)

    # PE-Core features (the repo's CMMD feature space; pose-blind -- a guide).
    from library.vision.encoder import encode_pe_from_imageminus1to1, load_pe_encoder
    from library.training.cmmd import pool_and_normalize

    bundle = load_pe_encoder(device)
    with torch.no_grad():
        for key, px in pixels.items():
            feats = encode_pe_from_imageminus1to1(bundle, px.unsqueeze(0).to(device))[0]
            rendered[key].pe = pool_and_normalize(feats).cpu()
    del bundle
    gc.collect()
    clean_memory_on_device(device)

    return rendered, pixels


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _flat(lat: torch.Tensor) -> torch.Tensor:
    return lat.reshape(-1).float()


def _norm(v: torch.Tensor) -> float:
    return float(v.norm())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na < EPS or nb < EPS:
        return 0.0
    return float((a @ b) / (na * nb))


# --------------------------------------------------------------------------- #
# Experiments -- each builds jobs, then consumes the rendered map.
# --------------------------------------------------------------------------- #
def exp_swap(prompts, tags, seeds, neg):
    """2x2 channel decomposition. Returns (jobs, lambda(rendered)->rows)."""
    jobs, specs = [], []
    for pi, base in enumerate(prompts):
        for ti, tag in enumerate(tags):
            tagged = f"{base}, {tag}"
            for s in seeds:
                kb = f"swap/p{pi}t{ti}s{s}"
                conf = {
                    "BB": (base, base), "TT": (tagged, tagged),
                    "TB": (tagged, base), "BT": (base, tagged),
                }
                for name, (cp, pp) in conf.items():
                    jobs.append(RenderJob(f"{kb}/{name}", cp, pp, s))
                specs.append((base, tag, s, kb))
    return jobs, lambda R: _swap_rows(R, specs)


def _swap_rows(R, specs):
    rows = []
    for base, tag, s, kb in specs:
        for space, get in (("latent", lambda r: _flat(r.latent)), ("pe", lambda r: r.pe.float())):
            bb, tt, tb, bt = (R[f"{kb}/{n}"] for n in ("BB", "TT", "TB", "BT"))
            d_full = get(tt) - get(bb)
            d_cross = get(tb) - get(bb)
            d_pool = get(bt) - get(bb)
            nf = _norm(d_full)
            rows.append({
                "experiment": "swap", "space": space, "base": base, "tag": tag, "seed": s,
                "norm_full": nf, "norm_cross": _norm(d_cross), "norm_pool": _norm(d_pool),
                "pool_share": _norm(d_pool) / (nf + EPS),
                "cross_share": _norm(d_cross) / (nf + EPS),
                "cos_cross_pool": _cos(d_cross, d_pool),
                "additivity_resid": _norm(d_full - (d_cross + d_pool)) / (nf + EPS),
            })
    return rows


def exp_order(prompts, seeds, neg, n_perm, rng):
    """Permute comma-tags: pooled identical (sanity-checked), movement = cross-attn."""
    jobs, specs = [], []
    for pi, base in enumerate(prompts):
        toks = [t.strip() for t in base.split(",") if t.strip()]
        perms = []
        for _ in range(n_perm):
            q = toks[:]
            rng.shuffle(q)
            perms.append(", ".join(q))
        canon = ", ".join(toks)
        for s in seeds:
            jobs.append(RenderJob(f"order/p{pi}s{s}/canon", canon, canon, s))
            jobs.append(RenderJob(f"order/p{pi}s{s}/canon2", canon, canon, s + 100000))
            for j, pm in enumerate(perms):
                jobs.append(RenderJob(f"order/p{pi}s{s}/perm{j}", pm, pm, s))
            specs.append((base, canon, perms, s, pi))
    return jobs, lambda R: _order_rows(R, specs)


def _order_rows(R, specs):
    rows = []
    for base, canon, perms, s, pi in specs:
        for space, get in (("latent", lambda r: _flat(r.latent)), ("pe", lambda r: r.pe.float())):
            c = get(R[f"order/p{pi}s{s}/canon"])
            seed_floor = _norm(get(R[f"order/p{pi}s{s}/canon2"]) - c)
            d_orders = [
                _norm(get(R[f"order/p{pi}s{s}/perm{j}"]) - c) for j in range(len(perms))
            ]
            rows.append({
                "experiment": "order", "space": space, "base": base, "seed": s,
                "order_dist_mean": float(np.mean(d_orders)),
                "order_dist_max": float(np.max(d_orders)),
                "seed_floor": seed_floor,
                "order_vs_seed": float(np.mean(d_orders)) / (seed_floor + EPS),
            })
    return rows


def exp_intensity(prompts, seeds, neg, w_points, steer_pos, steer_neg):
    """Sweep steering w; measure off-baseline movement + DC-blowout proxies."""
    jobs, specs = [], []
    for pi, base in enumerate(prompts):
        for s in seeds:
            for w in w_points:
                jobs.append(RenderJob(
                    f"intensity/p{pi}s{s}/w{w}", base, base, s,
                    mod_w=float(w), mod_pos=steer_pos, mod_neg=steer_neg,
                ))
            specs.append((base, s, pi))
    return jobs, lambda R: _intensity_rows(R, specs, w_points)


def _intensity_rows(R, specs, w_points):
    rows = []
    for base, s, pi in specs:
        r0 = R[f"intensity/p{pi}s{s}/w{w_points[0]}"]
        base_pe = r0.pe.float()
        for w in w_points:
            r = R[f"intensity/p{pi}s{s}/w{w}"]
            rows.append({
                "experiment": "intensity", "space": "image", "base": base, "seed": s, "w": float(w),
                "pe_move_from_w0": _norm(r.pe.float() - base_pe),
                "pixel_std": r.pixel_std, "tone": r.tone,
                "pixel_std_drop": (r0.pixel_std - r.pixel_std) / (r0.pixel_std + EPS),
            })
    return rows


# --------------------------------------------------------------------------- #
# Grid saving (read the grids!)
# --------------------------------------------------------------------------- #
def save_grids(pixels, run_dir, experiment):
    try:
        from PIL import Image, ImageDraw
    except Exception:
        logger.warning("PIL unavailable -- skipping grids")
        return []

    def to_img(px):
        x = ((px.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).numpy().transpose(1, 2, 0)
        return Image.fromarray(x)

    # Group keys by their grid prefix (everything before the last "/").
    groups: dict[str, list[str]] = {}
    for k in pixels:
        if not k.startswith(experiment):
            continue
        groups.setdefault(k.rsplit("/", 1)[0], []).append(k)
    artifacts = []
    for grp, keys in sorted(groups.items()):
        keys = sorted(keys)
        thumbs = [to_img(pixels[k]).resize((256, 256)) for k in keys]
        labels = [k.rsplit("/", 1)[1] for k in keys]
        w = sum(t.width for t in thumbs)
        canvas = Image.new("RGB", (w, 256 + 16), (16, 16, 16))
        x = 0
        d = ImageDraw.Draw(canvas)
        for t, lbl in zip(thumbs, labels):
            canvas.paste(t, (x, 16))
            d.text((x + 2, 2), lbl, fill=(230, 230, 230))
            x += t.width
        name = grp.replace("/", "_") + ".png"
        canvas.save(run_dir / name)
        artifacts.append(name)
    return artifacts


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit", default="models/diffusion_models/anima-base-v1.0.safetensors")
    p.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    p.add_argument("--text_encoder", default="models/text_encoders/qwen_3_06b_base.safetensors")
    p.add_argument("--pooled_text_proj", required=True, help="trained pooled_text_proj checkpoint")
    p.add_argument("--experiment", choices=["swap", "order", "intensity", "all"], default="all")
    p.add_argument("--prompts", type=str, default=None, help="';'-separated base prompts")
    p.add_argument("--tags", type=str, default=None, help="','-separated tags to splice (swap)")
    p.add_argument("--negative", type=str, default=DEFAULT_NEG)
    p.add_argument("--seeds", type=str, default="0", help="','-separated seeds")
    p.add_argument("--n_perm", type=int, default=3, help="order: permutations per prompt")
    p.add_argument("--w_points", type=str, default="0,2,3,5,8", help="intensity: steering w sweep")
    p.add_argument("--steer_pos", type=str, default="score_9, absurdres", help="intensity steering p+")
    p.add_argument("--steer_neg", type=str, default="", help="intensity steering p-")
    p.add_argument("--height", type=int, default=DEFAULT_H)
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--infer_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--attn_mode", type=str, default="torch")
    p.add_argument("--label", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = (
        [s.strip() for s in args.prompts.split(";") if s.strip()]
        if args.prompts else DEFAULT_PROMPTS
    )
    tags = (
        [s.strip() for s in args.tags.split(",") if s.strip()]
        if args.tags else DEFAULT_TAGS
    )
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    w_points = [float(w) for w in args.w_points.split(",") if w.strip()]
    rng = np.random.default_rng(1234)  # fixed seed -> reproducible permutations

    exps = ["swap", "order", "intensity"] if args.experiment == "all" else [args.experiment]

    # Build all jobs + their row-extractors.
    jobs, extractors = [], []
    if "swap" in exps:
        j, f = exp_swap(prompts, tags, seeds, args.negative)
        jobs += j
        extractors.append(f)
    if "order" in exps:
        j, f = exp_order(prompts, seeds, args.negative, args.n_perm, rng)
        jobs += j
        extractors.append(f)
    if "intensity" in exps:
        j, f = exp_intensity(prompts, seeds, args.negative, w_points, args.steer_pos, args.steer_neg)
        jobs += j
        extractors.append(f)

    # Collect every prompt that any job needs to encode.
    needed = {args.negative}
    for j in jobs:
        needed.add(j.cross_prompt)
        if j.pool_prompt:
            needed.add(j.pool_prompt)
        if j.mod_pos:
            needed.add(j.mod_pos)
        if j.mod_neg:
            needed.add(j.mod_neg)

    logger.info(f"Channel-attribution bench: {len(jobs)} renders, {len(needed)} prompts, exps={exps}")

    # ---- staged pipeline. DiT carries the LLM adapter + pooled_text_proj, so it
    # must be resident to encode faithfully (TE coexists briefly, like the live
    # mod-guidance setup). Then: encode (free TE) -> render -> free DiT -> VAE -> PE.
    logger.info("[1/4] building DiT + encoding prompts (TE)")
    model = build_dit(args, device)
    cross_cache = encode_prompts(model, sorted(needed), args, device)

    # Sanity: order experiment relies on pooled being permutation-invariant.
    if "order" in exps:
        _order_pool_sanity(jobs, cross_cache)

    logger.info("[2/4] denoising (DiT)")
    latents = render_jobs(model, jobs, cross_cache, args, device)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("[3/4] decode (VAE) + [4/4] features (PE)")
    rendered, pixels = decode_and_featurize(latents, args, device)

    # ---- metrics
    rows = []
    for f in extractors:
        rows += f(rendered)

    run_dir = make_run_dir("mod_guidance", label=args.label)
    artifacts = ["rows.csv"]
    _write_csv(run_dir / "rows.csv", rows)
    for e in exps:
        artifacts += save_grids(pixels, run_dir, e)

    metrics = _summarize(rows)
    _log_summary(metrics)

    write_result(
        run_dir, script=__file__, args=args,
        metrics=metrics, artifacts=artifacts,
    )
    logger.info(f"\nDone. Results -> {run_dir}")


def _order_pool_sanity(jobs, cross_cache):
    """Assert pooled(perm) == pooled(canon): the order experiment's whole premise."""
    worst = 0.0
    # Compare each order group's pooled vectors -- they share the same token bag.
    by_group: dict[str, list[str]] = {}
    for j in jobs:
        if j.key.startswith("order"):
            by_group.setdefault(j.key.rsplit("/", 1)[0], []).append(j.cross_prompt)
    for grp, prompts in by_group.items():
        pooled = [_pool(cross_cache[p].float()) for p in prompts]
        ref = pooled[0]
        for v in pooled[1:]:
            worst = max(worst, float((v - ref).abs().max()))
    logger.info(f"  order pooled-invariance check: max|Δpool| = {worst:.2e} (expect ~0)")
    if worst > 1e-2:
        logger.warning(
            "  pooled vectors are NOT permutation-identical -- order experiment "
            "attribution is contaminated (check tokenizer / padding)."
        )


def _write_csv(path, rows):
    import csv

    if not rows:
        path.write_text("")
        return
    keys = list({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _summarize(rows):
    def agg(pred, field):
        vals = [r[field] for r in rows if pred(r) and field in r]
        return float(np.mean(vals)) if vals else None

    def is_exp(experiment, space):
        return lambda r: r["experiment"] == experiment and r["space"] == space

    out = {}
    for space in ("latent", "pe"):
        sw = is_exp("swap", space)
        out[f"swap.{space}.pool_share_mean"] = agg(sw, "pool_share")
        out[f"swap.{space}.cross_share_mean"] = agg(sw, "cross_share")
        out[f"swap.{space}.cos_cross_pool_mean"] = agg(sw, "cos_cross_pool")
        out[f"swap.{space}.additivity_resid_mean"] = agg(sw, "additivity_resid")
        out[f"order.{space}.order_vs_seed_mean"] = agg(is_exp("order", space), "order_vs_seed")
    intensity_rows = [r for r in rows if r["experiment"] == "intensity"]
    if intensity_rows:
        out["intensity.max_pixel_std_drop"] = max(r["pixel_std_drop"] for r in intensity_rows)
        out["intensity.n_points"] = len({r["w"] for r in intensity_rows})
    out["n_rows"] = len(rows)
    return out


def _log_summary(m):
    logger.info("\n=== SUMMARY ===")
    for sp in ("latent", "pe"):
        ps = m.get(f"swap.{sp}.pool_share_mean")
        cs = m.get(f"swap.{sp}.cross_share_mean")
        cos = m.get(f"swap.{sp}.cos_cross_pool_mean")
        res = m.get(f"swap.{sp}.additivity_resid_mean")
        if ps is not None:
            logger.info(
                f"swap [{sp}]  pool_share={ps:.3f}  cross_share={cs:.3f}  "
                f"cos(cross,pool)={cos:+.3f}  additivity_resid={res:.3f}"
            )
        ovs = m.get(f"order.{sp}.order_vs_seed_mean")
        if ovs is not None:
            logger.info(f"order[{sp}]  order_dist / seed_floor = {ovs:.3f}")
    if "intensity.max_pixel_std_drop" in m:
        logger.info(f"intensity   max pixel-std drop vs w0 = {m['intensity.max_pixel_std_drop']:+.3f}")
    logger.info(
        "\nReads: pool_share≈0 -> the mod channel barely carries the edit (topic low-value). "
        "cos(cross,pool)>0 -> channels reinforce (preload/double-drive); <0 -> conflict/cancel. "
        "order/seed≪1 -> cross-attn weakly order-sensitive. "
        "Large pixel-std drop at high w -> DC-blowout is real in image space."
    )


if __name__ == "__main__":
    main()
