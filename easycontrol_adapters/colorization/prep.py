#!/usr/bin/env python3
"""Mangafy + cache condition latents for the colorization EasyControl task.

Two idempotent stages:

1. **Mangafy** — walk every color image under ``--src`` (the existing resized
   training images), screen each to a synthetic B&W manga page, and write the
   result to ``--staging`` mirroring the source subpath. Two engines via
   ``--engine``: ``sd`` (default, Phase B) runs the learned sketch2manga screening
   (:func:`screentone_sd.screentone_array`, on-manifold tones, needs
   ``make exp-easycontrol-download EASYADAPTER=colorize``); ``cv2`` runs the fast
   XDoG + algorithmic-screentone fallback (:func:`mangafy.mangafy_array`, no
   downloads). Skips already-staged PNGs unless ``--overwrite``.

2. **Encode** — VAE-encode the staged manga images into ``--cond_cache_dir`` via
   the existing ``library.preprocess.cache_latents`` (same ``{stem}_{WxH}_anima.npz``
   format as the target latent cache), at the image's **native size** so each
   cond latent shape matches its target latent exactly. Skips cached resolutions.

The cond cache is paired with the color target at train time by the loader's
``cond_cache_dir`` subset knob (stem-matched). Run from the repo root::

    python easycontrol_adapters/colorization/prep.py            # full dataset
    python easycontrol_adapters/colorization/prep.py --limit 8  # quick QA batch
    make exp-easycontrol-preprocess EASYADAPTER=colorize        # via task runner
"""

from __future__ import annotations

import argparse
import os
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from typing import Callable

import numpy as np
import torch
from PIL import Image

from library.preprocess import cache_latents, tqdm_progress
from library.preprocess._dataset import walk_images

# Screener: color RGB uint8 (H,W,3) + per-stem seed → B&W RGB uint8 (H,W,3) same size.
Screener = Callable[[np.ndarray, int], np.ndarray]


def _stable_seed(name: str) -> int:
    """Process-stable per-stem seed (Python's hash() is salted; crc32 isn't)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _make_screener(engine: str, sd_kwargs: dict) -> Screener:
    """Resolve ``engine`` to a ``(img, seed) → manga`` callable.

    Imports are deferred so the heavy ``sd`` stack (diffusers / controlnet_aux)
    is only loaded when actually selected, and ``cv2`` stays download-free. Takes
    plain (picklable) args rather than the argparse Namespace so it can be rebuilt
    inside a worker process."""
    if engine == "cv2":
        import cv2

        cv2.setNumThreads(
            1
        )  # one cv2 thread per worker — the pool gives the parallelism
        from mangafy import mangafy_array

        return lambda img, seed: mangafy_array(img, seed=seed)

    if engine == "gpu":
        # Same algorithmic screentone as cv2, but the trig screens + XDoG blurs run on
        # CUDA — one serial GPU pass, no ProcessPool. Structurally identical per seed.
        from mangafy_gpu import mangafy_array_gpu

        return lambda img, seed: mangafy_array_gpu(img, seed=seed)

    from screentone_sd import screentone_array

    return lambda img, seed: screentone_array(img, seed=seed, **sd_kwargs)


# ── Worker process state (cv2 engine only) ───────────────────────────────────
# The screener (a closure over a cv2/numpy module) isn't picklable, so each worker
# rebuilds it once in its initializer and stashes it — plus the constant paths — in
# a module global, then maps over image paths. The mangafy math is pure numpy/cv2,
# so this scales ~linearly with cores; the seeded per-stem jitter keeps every worker
# bit-identical to the serial path.
_WORKER: dict = {}


def _worker_init(
    engine: str, sd_kwargs: dict, src: Path, staging: Path, overwrite: bool
):
    _WORKER.update(
        screener=_make_screener(engine, sd_kwargs),
        src=src,
        staging=staging,
        overwrite=overwrite,
    )


def _worker_process(path_str: str) -> int:
    p = Path(path_str)
    rel = p.relative_to(_WORKER["src"])
    out = (_WORKER["staging"] / rel).with_suffix(".png")
    if out.exists() and not _WORKER["overwrite"]:
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    img = np.array(Image.open(p).convert("RGB"))
    manga = _WORKER["screener"](img, _stable_seed(rel.stem))
    Image.fromarray(manga).save(out)
    return 1


def stage_mangafy(
    src: Path,
    staging: Path,
    *,
    engine: str,
    sd_kwargs: dict,
    recursive: bool,
    overwrite: bool,
    limit: int | None,
    workers: int,
) -> tuple[int, int]:
    images = walk_images(src, recursive=recursive)
    if limit:
        images = images[:limit]
    progress = tqdm_progress("Mangafy")
    progress(0, total=len(images))

    # The sd engine holds a single GPU pipeline, so it stays serial; the cv2 engine
    # fans out across processes (``--workers``) when there's more than one to use.
    if engine == "cv2" and workers > 1 and len(images) > 1:
        written = 0
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(engine, sd_kwargs, src, staging, overwrite),
        ) as ex:
            # small chunks: each image is ~seconds of cv2 work, so IPC overhead is
            # negligible and fine-grained chunks load-balance variable image sizes well.
            for w in ex.map(_worker_process, [str(p) for p in images], chunksize=2):
                written += w
                progress(1)
        return len(images), written

    screener = _make_screener(engine, sd_kwargs)
    written = 0
    for p in images:
        rel = p.relative_to(src)
        out = (staging / rel).with_suffix(".png")
        if out.exists() and not overwrite:
            progress(1, detail=f"skip {p.name}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = np.array(Image.open(p).convert("RGB"))
        manga = screener(img, _stable_seed(rel.stem))
        Image.fromarray(manga).save(out)
        written += 1
        progress(1, detail=p.name)
    return len(images), written


def stage_encode(
    staging: Path,
    cond_cache_dir: Path,
    *,
    vae_path: str,
    batch_size: int,
    chunk_size: int,
    recursive: bool,
):
    from library.models import qwen_vae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading VAE from {vae_path} ...")
    vae = qwen_vae.load_vae(
        vae_path,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=chunk_size,
        disable_cache=True,
    )
    vae.to(device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()

    stats = cache_latents(
        staging,
        vae,
        cache_dir=cond_cache_dir,
        recursive=recursive,
        batch_size=batch_size,
        progress=tqdm_progress("Caching cond latents"),
    )

    vae.to("cpu")
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="post_image_dataset/resized")
    parser.add_argument("--staging", default="post_image_dataset/colorize_staging")
    parser.add_argument("--cond_cache_dir", default="post_image_dataset/colorize_cond")
    parser.add_argument("--vae", default="models/vae/qwen_image_vae.safetensors")
    parser.add_argument(
        "--engine",
        choices=["sd", "cv2", "gpu"],
        default="gpu",
        help="condition synthesizer: sd = learned sketch2manga screening (Phase B, "
        "needs `make exp-easycontrol-download EASYADAPTER=colorize`); "
        "cv2 = XDoG+halftone fallback (no downloads, CPU ProcessPool); "
        "gpu = same XDoG+halftone math on CUDA (no downloads, fast, serial)",
    )
    parser.add_argument(
        "--steps", type=int, default=40, help="sd engine: diffusion steps"
    )
    parser.add_argument(
        "--cfg", type=float, default=9.0, help="sd engine: guidance scale"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="sd engine: img2img denoise (lower = more faithful to source structure)",
    )
    parser.add_argument(
        "--tone_period",
        type=float,
        default=4.5,
        help="sd engine: halftone dot period (px at long_side); larger = coarser dots",
    )
    parser.add_argument(
        "--long_side", type=int, default=1024, help="sd engine: screening resolution"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="cv2 engine: parallel mangafy worker processes (1 = serial). The sd "
        "engine ignores this (single GPU pipeline).",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument(
        "--recursive", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="re-mangafy staged PNGs that exist"
    )
    parser.add_argument("--limit", type=int, default=None, help="cap #images (QA)")
    parser.add_argument("--skip_mangafy", action="store_true")
    parser.add_argument("--skip_encode", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    staging = Path(args.staging)
    cond_cache_dir = Path(args.cond_cache_dir)

    if not args.skip_mangafy:
        sd_kwargs = dict(
            steps=args.steps,
            cfg=args.cfg,
            long_side=args.long_side,
            strength=args.strength,
            tone_period=args.tone_period,
        )
        seen, written = stage_mangafy(
            src,
            staging,
            engine=args.engine,
            sd_kwargs=sd_kwargs,
            recursive=args.recursive,
            overwrite=args.overwrite,
            limit=args.limit,
            workers=args.workers,
        )
        print(
            f"\nMangafy ({args.engine}): {written} written, "
            f"{seen - written} skipped/seen={seen}"
        )
        if args.engine == "sd":
            # Free the SD pipeline's VRAM before the encode stage loads the Qwen VAE.
            from screentone_sd import unload

            unload()

    if not args.skip_encode:
        stats = stage_encode(
            staging,
            cond_cache_dir,
            vae_path=args.vae,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
            recursive=args.recursive,
        )
        print(
            f"\nCond latent caching complete: {stats.written} cached, "
            f"{stats.skipped} skipped (already existed)"
        )


if __name__ == "__main__":
    main()
