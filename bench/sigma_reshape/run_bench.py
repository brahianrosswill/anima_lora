#!/usr/bin/env python
"""σ-reshape phase-2 — does packing steps into the low-σ tail buy NFE efficiency?

Background
----------
``bench/dynamic_spectrum`` answered "is a fixed *uniform* schedule mistuned?":
yes — ~60% of local Euler-fattening-error mass sits in σ<0.45 (only 19% of the
steps), and σ alone ranks that error (within-traj ρ≈0.876), so no learned head /
no ‖v‖-gate is warranted. The cheapest actionable follow-up it pointed to is a
**static σ-reshaped step schedule**: bias ``get_timesteps_sigmas`` to place more
knots in σ<0.45, no model, no per-step decision.

This bench measures whether that reshape actually helps *image* quality at fixed
NFE — the thing the dynamic_spectrum probe explicitly did NOT measure (it scored
ODE truncation error, not output quality; "minimize fattening error" and "best
image at fixed NFE" can diverge, because flow_shift keeps steps at the noisy end
for a reason — early structure formation).

Knob
----
``get_timesteps_sigmas(..., tail_power=p)``: ``p=1.0`` is the canonical schedule
bit-for-bit; ``p>1`` warps the uniform grid toward σ=0 (denser tail), ``p<1`` the
opposite. Endpoints σ∈{0,1} fixed. (frac of steps with σ<0.45 at 12 NFE: p=1.0→
0.17, 1.5→0.33, 2.0→0.42.)

Method (``--sampler``, CFG=1)
-----------------------------
Reference = a deep converged solve (``--ref_steps``, p=1.0, SAME sampler) per
prompt/seed. The schedule's job is to land at that same endpoint with fewer steps,
so *distance-to-converged* isolates discretization error (the schedule's lever)
from model quality. For each (tail_power, NFE) we generate the low-NFE latent on
the SAME prompt/seed and report three distances to the converged ref:

- ``latent_endpoint`` — mean ‖x_lowNFE − x_ref‖ / ‖x_ref‖ (paired, free, the most
  direct discretization-error signal; not FM loss — it compares trajectory
  *endpoints*, and FM loss is structurally blind to the sampling schedule anyway).
- ``pe_cosine`` — mean 1 − cos(PE(x_lowNFE), PE(x_ref)) (paired, perceptual).
- ``cmmd`` — CMMD²(PE set of low-NFE, PE set of converged) (set-level; the repo's
  blessed quality metric, but noisy at small N).

Lower is better for all three; the converged ref scores ~0 by construction. The
best ``p`` at a given NFE reaches the converged image with the fewest steps.

**Decision metric depends on the sampler.** Euler is deterministic → the converged
solve is a fixed point, so the paired ``pe_cosine`` is clean discretization error
(the verdict reads it). er_sde is *stochastic* (higher-order + injected noise) →
there is no single endpoint; different step-counts draw different samples from the
same distribution, so paired metrics carry SDE-variance noise. For er_sde the
verdict auto-switches to ``cmmd`` (distribution-to-distribution, variance-robust),
which is the honest "does the reshape match the converged distribution" signal.

CFG>1 / non-square aspect stay out of scope for this read (the documented
follow-up if a single p wins, cf. DCW's CFG/aspect-dependent optimum).

Usage
-----
  # smoke (~2 min)
  uv run python -m bench.sigma_reshape.run_bench \
      --num_prompts 4 --ref_steps 40 --nfe 8 16 --tail_powers 1.0 2.0 --label smoke

  # real read — real captions + block-compile (backgroundable)
  uv run python -m bench.sigma_reshape.run_bench \
      --num_prompts 24 --ref_steps 100 --nfe 8 12 16 \
      --tail_powers 1.0 1.5 2.0 --compile --label reshape-realcap
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anima_lora import GenerationRequest, generate, get_generation_settings  # noqa: E402
from bench._anima import DEFAULT_DIT, DEFAULT_TEXT_ENCODER, DEFAULT_VAE  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402

BUILTIN_PROMPTS = [
    "a red fox curled asleep in a snowy pine forest, soft morning light",
    "portrait of an elderly fisherman, weathered face, golden hour, photorealistic",
    "a neon-lit cyberpunk alley in the rain, reflections on wet asphalt",
    "still life of lemons and a ceramic jug on a linen cloth, oil painting",
    "a sweeping mountain valley with a river, dramatic clouds, wide landscape",
    "a calico cat sitting on a windowsill beside a potted succulent",
    "an astronaut floating above a coral-colored planet, cinematic",
    "a bustling night market with paper lanterns and steam from food stalls",
]


def _load_dataset_prompts(dataset_dir: str, n: int, seed: int) -> list[str]:
    """Sample n real captions from .txt sidecars (the caption master).

    image_dataset/ is a symlink to nested artist dirs, so os.walk(followlinks=True)
    (rglob/plain find miss them — see project_image_dataset_symlink_nested).
    """
    root = Path(dataset_dir)
    if not root.exists():
        return []
    txts: list[str] = []
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for fn in files:
            if fn.endswith(".txt"):
                txts.append(os.path.join(dirpath, fn))
    txts.sort()
    if not txts:
        return []
    random.Random(seed).shuffle(txts)
    prompts: list[str] = []
    for path in txts:
        try:
            cap = " ".join(Path(path).read_text(errors="ignore").split())
        except OSError:
            continue
        if cap:
            prompts.append(cap)
        if len(prompts) >= n:
            break
    return prompts


def _to_bchw_latent(lat: torch.Tensor) -> torch.Tensor:
    """Normalize a generated latent to 4D [B, C, H, W] on CPU float."""
    lat = lat.detach()
    if lat.dim() == 5:  # [B, C, T=1, H, W] -> drop the singleton frame axis (dim 2)
        lat = lat.squeeze(2)
    return lat.float().cpu()


def _decode_to_image(vae, lat_bchw: torch.Tensor, device) -> torch.Tensor:
    """Decode a 4D latent to a single [3, H, W] pixel tensor in [-1, 1] (CPU float)."""
    with torch.no_grad():
        img = vae.decode_to_pixels(lat_bchw.to(device, vae.dtype))  # 4D in -> 4D out
    if img.dim() == 5:
        img = img.squeeze(2)
    return img[0].detach().float().cpu()


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dit", default=DEFAULT_DIT)
    p.add_argument("--vae", default=DEFAULT_VAE)
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--compile", action="store_true", help="block-compile the DiT")
    p.add_argument("--num_prompts", type=int, default=24)
    p.add_argument(
        "--ref_steps", type=int, default=100, help="converged reference NFE (p=1.0)"
    )
    p.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=[8, 12, 16],
        help="low-NFE budgets to test",
    )
    p.add_argument("--tail_powers", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="H W")
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument(
        "--sampler",
        default="euler",
        choices=["euler", "er_sde"],
        help="euler = deterministic (paired metrics are clean discretization error); "
        "er_sde = stochastic higher-order (no single endpoint, so read CMMD — the "
        "verdict auto-switches to CMMD for er_sde; paired metrics get SDE-variance noise).",
    )
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed0", type=int, default=1234)
    p.add_argument("--dataset_dir", default="image_dataset")
    p.add_argument("--prompts_file", default=None)
    p.add_argument(
        "--grid_prompts", type=int, default=6, help="rows in the eyeball montage"
    )
    p.add_argument(
        "--grid_nfe",
        type=int,
        default=None,
        help="which NFE the montage columns use (must be in --nfe; default = smallest)",
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    # ----------------------------------------------------------------- prompts
    prompt_source = "builtin"
    prompts = BUILTIN_PROMPTS
    if args.prompts_file:
        lines = [
            ln.strip()
            for ln in Path(args.prompts_file).read_text().splitlines()
            if ln.strip()
        ]
        if lines:
            prompts, prompt_source = lines, f"file:{args.prompts_file}"
    elif args.dataset_dir:
        sampled = _load_dataset_prompts(args.dataset_dir, args.num_prompts, args.seed0)
        if sampled:
            prompts, prompt_source = (
                sampled,
                f"dataset:{args.dataset_dir}({len(sampled)})",
            )
        else:
            print(
                f"[warn] no captions under {args.dataset_dir}; using built-in prompts"
            )
    prompts = prompts[: args.num_prompts]
    print(f"[prompts] source={prompt_source}  n={len(prompts)}")

    H, W = args.image_size

    def _gen_latent(
        prompt: str, seed: int, nfe: int, tail_power: float
    ) -> torch.Tensor:
        req = GenerationRequest(
            prompt=prompt,
            negative_prompt="",
            image_size=(H, W),
            infer_steps=nfe,
            guidance_scale=args.guidance_scale,
            flow_shift=args.flow_shift,
            sigma_tail_power=tail_power,
            sampler=args.sampler,
            seed=seed,
            dit=args.dit,
            vae=args.vae,
            text_encoder=args.text_encoder,
            device=args.device,
            attn_mode=args.attn_mode,
            extra_argv=("--compile_blocks",) if args.compile else (),
        )
        a = req.to_args()
        a.device = args.device
        with torch.no_grad():
            return generate(a, gen_settings, shared_models=shared_models)

    # ------------------------------------------------- warm-up load (model once)
    from library.inference.models import load_shared_models
    from library.models import qwen_vae

    warm = GenerationRequest(
        prompt=prompts[0],
        image_size=(H, W),
        infer_steps=4,
        guidance_scale=args.guidance_scale,
        flow_shift=args.flow_shift,
        sampler=args.sampler,
        seed=args.seed0,
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        device=args.device,
        attn_mode=args.attn_mode,
        extra_argv=("--compile_blocks",) if args.compile else (),
    )
    wa = warm.to_args()
    wa.device = args.device
    gen_settings = get_generation_settings(wa)
    shared_models: dict = load_shared_models(wa)
    with torch.no_grad():
        generate(wa, gen_settings, shared_models=shared_models)

    vae = qwen_vae.load_vae(args.vae, dtype=torch.bfloat16, device=args.device)
    vae.to(args.device, dtype=torch.bfloat16)
    vae.eval()

    configs = [(tp, nfe) for tp in args.tail_powers for nfe in args.nfe]

    # ---------------------------------------------- phase A: generate + decode
    # Keep DiT + VAE on GPU together (matches the validation path). Store latents
    # (for the endpoint distance) and decoded pixels (for PE) on CPU.
    ref_lat: list[torch.Tensor] = []
    ref_img: list[torch.Tensor] = []
    cfg_lat: dict[tuple, list[torch.Tensor]] = {c: [] for c in configs}
    cfg_img: dict[tuple, list[torch.Tensor]] = {c: [] for c in configs}

    for k, prompt in enumerate(prompts):
        seed = args.seed0 + k
        rlat = _to_bchw_latent(_gen_latent(prompt, seed, args.ref_steps, 1.0))
        ref_lat.append(rlat)
        ref_img.append(_decode_to_image(vae, rlat, args.device))
        for tp, nfe in configs:
            lat = _to_bchw_latent(_gen_latent(prompt, seed, nfe, tp))
            cfg_lat[(tp, nfe)].append(lat)
            cfg_img[(tp, nfe)].append(_decode_to_image(vae, lat, args.device))
        print(f"[prompt {k + 1}/{len(prompts)}] '{prompt[:42]}...'")

    # ------------------------------------------------------- phase B: PE encode
    # Park DiT + VAE on CPU, bring PE on (avoids 3-model VRAM pressure at 1024²).
    from library.training.cmmd import cmmd_from_pools, pool_and_normalize
    from library.vision.encoder import encode_pe_from_imageminus1to1, load_pe_encoder

    shared_models["model"].to("cpu")
    vae.to("cpu")
    if args.device == "cuda":
        torch.cuda.empty_cache()
    bundle = load_pe_encoder(torch.device(args.device))

    def _pe_pool(images: list[torch.Tensor]) -> torch.Tensor:
        """[3,H,W] list (all same bucket here) -> [N, D] unit-norm pooled feats."""
        batch = torch.stack(images, dim=0).to(args.device)
        feats_list = encode_pe_from_imageminus1to1(bundle, batch, same_bucket=True)
        return torch.stack([pool_and_normalize(f).cpu() for f in feats_list], dim=0)

    ref_pool = _pe_pool(ref_img)  # [N, D]
    cfg_pool = {c: _pe_pool(cfg_img[c]) for c in configs}

    # --------------------------------------------------------- phase C: metrics
    def _endpoint(c) -> float:
        ds = []
        for lo, ref in zip(cfg_lat[c], ref_lat):
            ds.append(float((lo - ref).norm() / ref.norm().clamp_min(1e-8)))
        return sum(ds) / len(ds)

    def _pe_cos(c) -> float:
        gp, rp = cfg_pool[c], ref_pool  # both unit-norm -> cos = dot
        cos = (gp * rp).sum(dim=1)
        return float((1.0 - cos).mean())

    table = {}
    for c in configs:
        tp, nfe = c
        table[f"p{tp}_nfe{nfe}"] = {
            "tail_power": tp,
            "nfe": nfe,
            "latent_endpoint": round(_endpoint(c), 5),
            "pe_cosine": round(_pe_cos(c), 5),
            "cmmd": round(cmmd_from_pools(ref_pool, cfg_pool[c]), 4),
        }

    # Decision metric: pe_cosine for deterministic euler (paired = clean discretization
    # error); cmmd for stochastic er_sde (no single endpoint → paired metrics carry
    # SDE-variance noise, so the distribution-to-distribution CMMD is the honest signal).
    dmetric = "cmmd" if args.sampler != "euler" else "pe_cosine"
    per_nfe_winner = {}
    baseline_beaten = 0
    for nfe in args.nfe:
        rows = [(tp, table[f"p{tp}_nfe{nfe}"][dmetric]) for tp in args.tail_powers]
        best_tp, best_v = min(rows, key=lambda r: r[1])
        base_v = table[f"p1.0_nfe{nfe}"][dmetric] if 1.0 in args.tail_powers else None
        per_nfe_winner[f"nfe{nfe}"] = {
            "decision_metric": dmetric,
            "best_tail_power": best_tp,
            f"best_{dmetric}": best_v,
            f"baseline_p1.0_{dmetric}": base_v,
            "improvement_vs_baseline": round(base_v - best_v, 5)
            if base_v is not None
            else None,
        }
        if base_v is not None and best_tp != 1.0 and best_v < base_v:
            baseline_beaten += 1

    if 1.0 in args.tail_powers and len(args.nfe):
        frac = baseline_beaten / len(args.nfe)
        if frac >= 0.5:
            verdict = (
                f"RESHAPE HELPS ({args.sampler}): a p>1 schedule beats canonical p=1.0 on "
                f"{dmetric} at {baseline_beaten}/{len(args.nfe)} NFE budgets. Proceed to a CFG×aspect sweep."
            )
        elif baseline_beaten == 0:
            verdict = (
                f"NO RESHAPE WIN ({args.sampler}): canonical p=1.0 is best (or tied) at every "
                f"NFE on {dmetric}; the low-σ tail-densify does not improve the converged-target match here."
            )
        else:
            verdict = (
                f"MIXED ({args.sampler}): p>1 wins at {baseline_beaten}/{len(args.nfe)} NFE "
                f"budgets on {dmetric}; weak/inconsistent — inspect the table + grid before building."
            )
    else:
        verdict = (
            "no p=1.0 baseline in --tail_powers; cannot judge reshape vs canonical."
        )

    metrics = {
        "config": {
            "num_prompts": len(prompts),
            "prompt_source": prompt_source,
            "ref_steps": args.ref_steps,
            "nfe": args.nfe,
            "tail_powers": args.tail_powers,
            "image_size": [H, W],
            "flow_shift": args.flow_shift,
            "guidance_scale": args.guidance_scale,
            "sampler": args.sampler,
        },
        "table": table,
        "per_nfe_winner": per_nfe_winner,
        "verdict": verdict,
    }

    # --------------------------------------------------------------- artifacts
    run_dir = make_run_dir("sigma_reshape", label=args.label)

    csv = run_dir / "results.csv"
    with csv.open("w") as f:
        f.write("tail_power,nfe,latent_endpoint,pe_cosine,cmmd\n")
        for c in configs:
            r = table[f"p{c[0]}_nfe{c[1]}"]
            f.write(
                f"{c[0]},{c[1]},{r['latent_endpoint']},{r['pe_cosine']},{r['cmmd']}\n"
            )
    artifacts = ["results.csv"]

    # Line plot: each metric vs NFE, one line per tail_power.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for mi, mname in enumerate(["latent_endpoint", "pe_cosine", "cmmd"]):
            for tp in args.tail_powers:
                ys = [table[f"p{tp}_nfe{nfe}"][mname] for nfe in args.nfe]
                ax[mi].plot(args.nfe, ys, "-o", ms=4, label=f"p={tp}")
            ax[mi].set_xlabel("NFE (Euler steps)")
            ax[mi].set_ylabel(mname)
            ax[mi].set_title(f"{mname} vs NFE (↓ = closer to converged)")
            ax[mi].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(run_dir / "metrics.png", dpi=130)
        artifacts.append("metrics.png")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot failed: {e}")

    # Eyeball montage: rows = first grid_prompts prompts; cols = [converged] +
    # [each tail_power] at the SMALLEST NFE (where schedule differences are largest).
    try:
        from PIL import Image

        nfe_lo = args.grid_nfe if args.grid_nfe in args.nfe else min(args.nfe)
        n_rows = min(args.grid_prompts, len(prompts))
        cols = [("ref", None)] + [(tp, nfe_lo) for tp in args.tail_powers]

        def _to_pil(img_chw: torch.Tensor) -> Image.Image:
            arr = (
                ((img_chw.clamp(-1, 1) + 1) * 127.5)
                .round()
                .byte()
                .permute(1, 2, 0)
                .numpy()
            )
            return Image.fromarray(arr)

        thumb = 256
        tiles = []
        for r in range(n_rows):
            row = []
            for tp, nfe in cols:
                img = ref_img[r] if tp == "ref" else cfg_img[(tp, nfe)][r]
                row.append(_to_pil(img).resize((thumb, thumb)))
            tiles.append(row)
        grid = Image.new("RGB", (thumb * len(cols), thumb * n_rows), "white")
        for r, row in enumerate(tiles):
            for c, im in enumerate(row):
                grid.paste(im, (c * thumb, r * thumb))
        grid.save(run_dir / "grid.png")
        # column legend printed to stdout (cols left→right)
        col_names = [f"converged({args.ref_steps})"] + [
            f"p={tp}@nfe{nfe_lo}" for tp in args.tail_powers
        ]
        print(f"[grid] columns L→R: {col_names}")
        artifacts.append("grid.png")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] grid failed: {e}")

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        device=torch.device(args.device),
        label=args.label,
    )

    print(f"\n=== σ-reshape ({args.sampler}, CFG={args.guidance_scale}) ===")
    print(
        f"prompts={len(prompts)}  ref={args.ref_steps}  nfe={args.nfe}  p={args.tail_powers}"
    )
    for nfe in args.nfe:
        cells = "  ".join(
            f"p{tp}:{table[f'p{tp}_nfe{nfe}'][dmetric]:.4f}" for tp in args.tail_powers
        )
        print(f"  NFE={nfe:<3} {dmetric}  {cells}")
    print(f"VERDICT: {verdict}")
    print(f"-> {run_dir}")


if __name__ == "__main__":
    main()
