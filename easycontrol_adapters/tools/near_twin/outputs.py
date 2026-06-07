#!/usr/bin/env python3
"""near_twin outputs — HTML contact sheet, TSV, pair-tree export, dataset blueprint.

The rendering/materialization layer of the near-twin miner. Pure I/O over the
``PairRecord``s produced by ``near_twin.engine`` — no matching logic lives here.
The two training-shaped artifacts are the ``_tags`` / ``_no_tags`` pair tree
(``export_pairs``) and the EasyControl dataset blueprint (``write_dataset_config``);
HTML + TSV are curation-only.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .engine import (
    REPO_ROOT,
    Member,
    PairRecord,
    caption_text,
    diff_bbox_norm,
    read_tags,
)


def _thumb_b64(
    member: Member, bbox: tuple[float, float, float, float] | None, size: int
) -> str:
    with Image.open(member.image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size), Image.BILINEAR)
        if bbox and bbox[2] > bbox[0]:
            d = ImageDraw.Draw(im)
            w, h = im.size
            d.rectangle(
                [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h],
                outline=(255, 40, 40),
                width=3,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_html(pairs: list[PairRecord], out_path: Path, thumb: int) -> None:
    rows = []
    for p in pairs:
        bbox = diff_bbox_norm(p.match.diff_cells, p.match.G)
        holder, clean = p.holder_member(), p.clean_member()
        added = sorted((read_tags(holder.txt_path) - read_tags(clean.txt_path)))
        removed = sorted((read_tags(clean.txt_path) - read_tags(holder.txt_path)))
        h_img = _thumb_b64(holder, bbox, thumb)
        c_img = _thumb_b64(clean, bbox, thumb)
        rows.append(f"""
        <div class="pair">
          <div class="meta">
            <b>{html.escape(p.artist)}</b> &nbsp; {html.escape(p.pair_id)}
            &nbsp; {p.a.wh[0]}×{p.a.wh[1]}
            &nbsp;|&nbsp; cos {p.cosine:.3f} &nbsp; match {p.match.match_frac:.2f}
            &nbsp; extra-diff {p.verdict.n_extra_diff}
            &nbsp; holder=<b>{html.escape(holder.stem)}</b>
          </div>
          <div class="imgs">
            <figure><img src="data:image/jpeg;base64,{h_img}"><figcaption>_tags ({html.escape(holder.stem)})</figcaption></figure>
            <figure><img src="data:image/jpeg;base64,{c_img}"><figcaption>_no_tags ({html.escape(clean.stem)})</figcaption></figure>
          </div>
          <div class="tags">
            <span class="add">+ {html.escape(", ".join(added) or "—")}</span><br>
            <span class="rem">− {html.escape(", ".join(removed) or "—")}</span>
          </div>
        </div>""")
    doc = f"""<!doctype html><meta charset="utf-8"><title>near-twin pairs</title>
<style>
body{{font:13px/1.4 system-ui,sans-serif;background:#111;color:#ddd;margin:0;padding:16px}}
h1{{font-size:18px}}
.pair{{border:1px solid #333;border-radius:8px;padding:10px;margin:10px 0;background:#181818}}
.meta{{color:#9cf;margin-bottom:6px}}
.imgs{{display:flex;gap:10px}}
figure{{margin:0}} img{{max-width:{thumb}px;border:1px solid #333;border-radius:4px}}
figcaption{{color:#888;font-size:11px}}
.tags{{margin-top:6px;font-size:12px}}
.add{{color:#7cdd7c}} .rem{{color:#dd7c7c}}
</style>
<h1>near-twin tag-gap pairs — {len(pairs)} accepted</h1>
{"".join(rows)}
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")


def write_tsv(pairs: list[PairRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "artist",
                "id_a",
                "id_b",
                "wh",
                "cosine",
                "match_frac",
                "gap_holder",
                "n_extra_diff",
                "extra_diff_tags",
                "diff_bbox",
            ]
        )
        for p in pairs:
            x, y = p.ids
            bbox = diff_bbox_norm(p.match.diff_cells, p.match.G)
            w.writerow(
                [
                    p.artist,
                    x,
                    y,
                    f"{p.a.wh[0]}x{p.a.wh[1]}",
                    f"{p.cosine:.4f}",
                    f"{p.match.match_frac:.3f}",
                    p.holder_member().stem,
                    p.verdict.n_extra_diff,
                    "|".join(p.verdict.extra_diff_tags),
                    ",".join(f"{v:.3f}" for v in bbox),
                ]
            )


def _materialize_one(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def export_pairs(
    pairs: list[PairRecord], export_dir: Path, copy: bool, emit_mask: bool
) -> int:
    """Materialize the ``_tags`` / ``_no_tags`` pair tree (the training-shaped output).

    Symlinks keep the source extension (``_tags.webp``); ``--copy`` re-saves as
    PNG. The EasyControl loader globs png+webp, so stem-matching is unaffected.
    """
    written = 0
    for p in pairs:
        holder, clean = p.holder_member(), p.clean_member()
        adir = export_dir / p.artist
        ext = ".png" if copy else holder.image_path.suffix
        for member, side in ((holder, "_tags"), (clean, "_no_tags")):
            stem_ext = ".png" if copy else member.image_path.suffix
            img_dst = adir / f"{p.pair_id}{side}{stem_ext}"
            if copy:
                with Image.open(member.image_path) as im:
                    img_dst.parent.mkdir(parents=True, exist_ok=True)
                    im.convert("RGB").save(img_dst)
            else:
                _materialize_one(member.image_path, img_dst, copy=False)
            txt_dst = adir / f"{p.pair_id}{side}.txt"
            txt_dst.write_text(caption_text(member.txt_path), encoding="utf-8")
        if emit_mask:
            _write_mask(p, adir / f"{p.pair_id}_mask.png")
        written += 1
        _ = ext  # holder ext informational only
    return written


def _write_mask(p: PairRecord, out: Path) -> None:
    G = p.match.G
    grid = np.zeros((G, G), dtype=np.uint8)
    for c in p.match.diff_cells:
        grid[c // G, c % G] = 255
    with Image.open(p.holder_member().image_path) as im:
        w, h = im.size
    Image.fromarray(grid, mode="L").resize((w, h), Image.NEAREST).save(out)


def _rel_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


# The miner rewrites everything BELOW this sentinel (the output dataset
# blueprint) on every run, and leaves everything above it — the user's
# [staging] / [preprocess] knobs + comments — untouched. So one file is both the
# run config (input, read via --config) and the generated subset blueprint (output).
_BLUEPRINT_SENTINEL = (
    "# === generated dataset blueprint (rewritten by the miner; do not edit below) ==="
)

_DEFAULT_STAGING_HEADER = """\
# near-twin tag-gap miner config. Edit the [staging] table, then run the two
# pipeline steps (knobs for each live in their own table here):
#   make easycontrol-staging    EASYADAPTER=near_twin   # [staging] → mine pair tree
#   make easycontrol-preprocess EASYADAPTER=near_twin   # [preprocess] → VAE/TE caches
# CLI flags override these. Paths support {CAPTION_CORPUS_DIR}/$VARS/~ expansion.

# Mining run knobs (legacy name [miner] still accepted). Any --flag dest works.
[staging]
# Discriminator — set exactly one of: tag_any / tag / region (bool) / signal.
tag_any = ["speech bubble", "thought bubble", "blank speech bubble"]
# Source trees (<dir>/<artist>/<id>.<ext>).
image_dirs = ["{CAPTION_CORPUS_DIR}/retrieved"]
# Stage/threshold knobs (any --flag dest works here):
sim_min = 0.85
match_frac_min = 0.66
cell_match_min = 0.9
max_extra_diff = 6

# Resize + VAE/TE caching pass over the mined pair tree (read by
# `make easycontrol-preprocess EASYADAPTER=near_twin`). Native-res staging is
# resized into constant-token buckets under resized_dir (the training image_dir),
# then VAE/TE-encoded into cache_dir.
[preprocess]
image_dir = 'post_image_dataset/easycontrol/near_twins/staging'    # resize source
resized_dir = 'post_image_dataset/easycontrol/near_twins/resized'  # → training image_dir
cache_dir = 'post_image_dataset/easycontrol/near_twins/cache'
cond_dir = 'post_image_dataset/easycontrol/near_twins/cond'  # paired _tags reference latents (symlinks)
target_res = [1024]  # bucket tiers (512 768 1024 1280 1536); must match training
min_pixels = 0  # 0 = never drop a member (would orphan its pair partner)
recursive = true
vae = 'models/vae/qwen_image_vae.safetensors'
batch_size = 4
chunk_size = 64
qwen3 = 'models/text_encoders/qwen_3_06b_base.safetensors'
dit = 'models/diffusion_models/anima-base-v1.0.safetensors'
caption_shuffle_variants = 4
caption_tag_dropout_rate = 0.1

# Training overrides for `make easycontrol EASYADAPTER=near_twin` (optional).
# Each key is folded in as a `--key value` CLI override on top of
# configs/methods/easycontrol.toml; `--dataset_config` is injected automatically.
# Omit the table to train with the plain easycontrol method defaults.
[training]
output_name = "anima_easycontrol_near_twins"
learning_rate = 3e-5
max_train_epochs = 8
# Removal task: the _tags reference is the structural source of truth, so keep
# the cond clean (no training-time perturbation) for faithful reconstruction.
easycontrol_cond_noise_max = 0.0
"""


def _blueprint_text(export_dir: Path) -> str:
    # Training reads the bucket-resized tree (the preprocess pass resizes the
    # native-res `staging/` export into `resized/` before VAE-encoding), so the
    # subset image_dir points at the resized sibling, not the export dir itself.
    img_dir = _rel_to_repo(export_dir.parent / "resized")
    cache_dir = _rel_to_repo(export_dir.parent / "cache")
    cond_dir = _rel_to_repo(export_dir.parent / "cond")
    return f"""{_BLUEPRINT_SENTINEL}
# Near-twin pair tree wired as an EasyControl *text-removal* control task. The
# mined staging tree is resized into constant-token buckets under `resized/`,
# then VAE/TE-encoded into `cache/` (nested <artist>/ mirror of the resized tree).
# Each accepted pair is a `{{id}}_tags` / `{{id}}_no_tags` couple, the `_tags` side
# holding the discriminator attribute (speech bubbles / text).
#
# Roles (EasyControl: cache_dir = denoising target, cond_cache_dir = condition):
#   target = the clean `_no_tags` member (+ its caption) — what the model generates
#   cond   = the paired `_tags` latent (the text-bearing reference fed via set_cond),
#            symlinked into `cond/` under the `_no_tags` stem by the preprocess step.
# So the adapter learns: given a text-bearing panel as the reference, regenerate
# the clean version. `path_pattern` keeps only the `_no_tags` members as targets;
# the `_tags` members enter solely as the condition via cond_cache_dir.

[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = '{img_dir}'
  cache_dir = '{cache_dir}'        # denoising target: the clean _no_tags members
  cond_cache_dir = '{cond_dir}'    # condition: the paired _tags latent (keyed by the _no_tags stem)
  path_pattern = '*_no_tags.*'     # targets = clean members only (the _tags twin rides in as the cond)
  recursive = true                 # tree is nested <artist>/<pair_id>_{{tags,no_tags}}; caches mirror the nesting
  flip_aug = false                 # latents can't be flipped post-hoc; the cond cache has no flipped variant
  num_repeats = 1
"""


def write_dataset_config(export_dir: Path, config_path: Path) -> None:
    """Rewrite the blueprint tail of ``config_path``, preserving the head tables.

    First creation scaffolds a ``[staging]`` + ``[preprocess]`` template; re-runs
    keep everything above the sentinel (the user's edited run knobs) verbatim.
    """
    blueprint = _blueprint_text(export_dir)
    if config_path.is_file():
        existing = config_path.read_text(encoding="utf-8")
        head = existing.split(_BLUEPRINT_SENTINEL, 1)[0].rstrip()
        content = f"{head}\n\n{blueprint}" if head else blueprint
    else:
        content = f"{_DEFAULT_STAGING_HEADER}\n{blueprint}"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")
