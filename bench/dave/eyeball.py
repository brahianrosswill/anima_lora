#!/usr/bin/env python
"""DAVE eyeball sweep — render baseline vs DAVE settings across seeds.

Not a metrics bench: just generates the `make test` prompt under a few DAVE
configs (the model is loaded once and shared) and dumps labeled PNGs into
``output/tests/dave/`` so the diversity/quality tradeoff can be eyeballed.

    uv run python bench/dave/eyeball.py
    uv run python bench/dave/eyeball.py --seeds 1000 1001 1002
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anima_lora import (  # noqa: E402
    GenerationRequest,
    decode_to_pil,
    default_checkpoints,
    generate,
    get_generation_settings,
    load_dit_model,
    load_vae,
    prepare_text_inputs,
)

DEFAULT_PROMPT = (
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
    ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
    " background features some trees and blue sky with clouds."
)
DEFAULT_NEGATIVE = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
)

# (label, dave-on, strength, block_lo, block_hi). block_hi=-1 → last block.
CONFIGS = [
    ("baseline", False, 0.0, 0, -1),
    ("s0.05_all", True, 0.05, 0, -1),
    ("s0.10_all", True, 0.10, 0, -1),
    ("s0.10_mid9-18", True, 0.10, 9, 18),
    ("s0.20_mid9-18", True, 0.20, 9, 18),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1000, 1001])
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE)
    opts = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(__file__).resolve().parents[2] / "output" / "tests" / "dave"
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = default_checkpoints()
    req = GenerationRequest(
        dit=ckpts.dit,
        vae=ckpts.vae,
        text_encoder=ckpts.text_encoder,
        prompt=opts.prompt,
        negative_prompt=opts.negative_prompt,
        save_path="output/tests/dave/_unused.png",
        infer_steps=opts.steps,
        guidance_scale=opts.cfg,
        image_size=(1024, 1024),
        sampler="euler",
        flow_shift=3.0,
        seed=opts.seeds[0],
    )
    args = req.to_args()
    args.device = device
    args.compile = False
    args.compile_blocks = False

    gen_settings = get_generation_settings(args)
    print("[eyeball] loading DiT + VAE…")
    anima = load_dit_model(args, device, torch.bfloat16)
    # VAE stays on CPU; decode_latent moves it on/off device per decode, so the
    # shared DiT and the VAE never both sit on the GPU.
    vae = load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
        dtype=torch.bfloat16,
        eval=True,
    )
    context, context_null = prepare_text_inputs(args, device, anima)
    text_data = {"context": context, "context_null": context_null}
    shared = {"model": anima}

    for seed in opts.seeds:
        for label, on, strength, blo, bhi in CONFIGS:
            args.seed = seed
            args.dave = "auto" if on else None
            args.dave_strength = strength
            args.dave_block_lo = blo
            args.dave_block_hi = bhi
            args.dave_sigma_lo = 0.0
            args.dave_sigma_hi = 1.0
            print(f"[eyeball] seed={seed} {label}")
            latent = generate(
                args, gen_settings, shared_models=shared,
                precomputed_text_data=text_data,
            )
            img = decode_to_pil(vae, latent, device)
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"seed{seed}_{label}.png".replace(".", "p", label.count("."))
            img.save(out_dir / fname)

    print(f"\n→ {out_dir}")


if __name__ == "__main__":
    main()
