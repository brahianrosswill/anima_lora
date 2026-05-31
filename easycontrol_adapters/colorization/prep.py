#!/usr/bin/env python3
"""Mangafy + cache condition latents for the colorization EasyControl task.

Two idempotent stages:

1. **Mangafy** — walk every color image under ``--src`` (the existing resized
   training images), run :func:`mangafy.mangafy_array` (XDoG lineart + algorithmic
   screentone), and write the B&W result to ``--staging`` mirroring the source
   subpath. Skips already-staged PNGs unless ``--overwrite``.

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
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# When run as a file, this script's dir is sys.path[0] → import the sibling.
from mangafy import mangafy_array

from library.preprocess import cache_latents, tqdm_progress
from library.preprocess._dataset import walk_images


def _stable_seed(name: str) -> int:
    """Process-stable per-stem seed (Python's hash() is salted; crc32 isn't)."""
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def stage_mangafy(
    src: Path,
    staging: Path,
    *,
    recursive: bool,
    overwrite: bool,
    limit: int | None,
) -> tuple[int, int]:
    images = walk_images(src, recursive=recursive)
    if limit:
        images = images[:limit]
    written = 0
    progress = tqdm_progress("Mangafy")
    progress(0, total=len(images))
    for p in images:
        rel = p.relative_to(src)
        out = (staging / rel).with_suffix(".png")
        if out.exists() and not overwrite:
            progress(1, detail=f"skip {p.name}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        img = np.array(Image.open(p).convert("RGB"))
        manga = mangafy_array(img, seed=_stable_seed(rel.stem))
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
        seen, written = stage_mangafy(
            src,
            staging,
            recursive=args.recursive,
            overwrite=args.overwrite,
            limit=args.limit,
        )
        print(f"\nMangafy: {written} written, {seen - written} skipped/seen={seen}")

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
