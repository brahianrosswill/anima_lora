#!/usr/bin/env python3
"""Dataset image grouping — connected-components clustering on PE-Spatial cosine.

A *curation* primitive (distinct from ``library/preprocess``, which makes data
training-ready): group visually near-identical / same-concept images so the GUI
dataset browser can filter by group, dedup is easy to spot, and subsets are easy
to balance. Build a unit-norm PE-Spatial CLS descriptor per image
(:mod:`library.vision.pe_features`), connect any two images whose cosine is at
or above a threshold, and emit the connected components as groups. A tight
threshold (~0.95) surfaces near-duplicates; a looser one (~0.85) broad concept
clusters.

Scope is **per-artist**: components are computed independently within each
top-level folder under the source dir (the artist bucket), so two different
artists never merge into one group. The result is a JSON manifest the GUI reads
(plain JSON — no torch — so the GUI stays torch-free).

``connected_components`` is pure (numpy only). ``build_groups`` lazily imports
the torch-backed embedding primitive, so importing this module for the pure
clustering / manifest schema stays torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MANIFEST_VERSION = 1
DEFAULT_THRESHOLD = 0.92


def connected_components(
    emb: np.ndarray, threshold: float, block: int = 1024
) -> list[list[int]]:
    """Union-find over cosine edges → connected components as row-index lists.

    ``emb`` rows are assumed L2-normed, so ``emb @ emb.T`` is cosine similarity.
    Two rows share a component when their cosine is ``>= threshold``. The
    similarity matrix is computed in row blocks (``block`` rows at a time) so peak
    memory is ``O(block * n)`` rather than ``O(n^2)``. Components are returned
    sorted (each member list ascending) and ordered by descending size.
    """
    n = int(emb.shape[0])
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        sims = emb[i0:i1] @ emb.T  # [i1-i0, n]
        for r in range(i1 - i0):
            i = i0 + r
            # Only look forward (j > i) — edges are symmetric, so this halves the
            # work and avoids self-loops.
            js = np.nonzero(sims[r, i + 1 :] >= threshold)[0]
            for j in js:
                union(i, i + 1 + int(j))

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    groups = [sorted(v) for v in comps.values()]
    groups.sort(key=lambda g: (-len(g), g[0]))
    return groups


def _mean_pairwise_cosine(sub: np.ndarray) -> float:
    """Mean off-diagonal cosine of a small (already-normed) embedding block."""
    if sub.shape[0] < 2:
        return 1.0
    sim = sub @ sub.T
    iu = np.triu_indices(sub.shape[0], k=1)
    return float(sim[iu].mean())


def _artist_of(rel: Path) -> str:
    """Top-level folder under the source dir = the artist bucket ("" if flat)."""
    return rel.parts[0] if len(rel.parts) > 1 else ""


def build_groups(
    source_dir: Path,
    out_path: Path,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_size: int = 2,
    encoder: str = "pe_spatial",
    device: str | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
) -> dict:
    """Embed every image under ``source_dir``, cluster per-artist, write a manifest.

    Reuses the shared PE-Spatial feature cache (``library.vision.pe_features``),
    so a re-run after the near-twin miner — or a re-run with a different
    ``threshold`` — is just the (cached) clustering pass. Returns the manifest
    dict (also written to ``out_path`` as JSON).
    """
    # Lazy torch-backed imports so the pure helpers above import without torch.
    import torch

    from library.vision import load_pe_encoder
    from library.vision.pe_features import Member, embed_members, iter_images

    source_dir = Path(source_dir)
    paths = iter_images(source_dir)

    manifest: dict = {
        "version": MANIFEST_VERSION,
        "source_dir": str(source_dir),
        "encoder": encoder,
        "threshold": threshold,
        "min_size": min_size,
        "n_images": len(paths),
        "n_groups": 0,
        "n_grouped": 0,
        "n_singletons": len(paths),
        "groups": [],
    }
    if not paths:
        _write_manifest(out_path, manifest)
        return manifest

    # Bucket by artist (top-level dir) for per-artist scoping.
    by_artist: dict[str, list[Path]] = {}
    for p in paths:
        rel = p.relative_to(source_dir)
        by_artist.setdefault(_artist_of(rel), []).append(p)

    members = [
        Member(
            artist=_artist_of(p.relative_to(source_dir)),
            stem=p.stem,
            image_path=p,
            txt_path=p.with_suffix(".txt"),
        )
        for p in paths
    ]
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bundle = load_pe_encoder(dev, name=encoder)
    feats = embed_members(bundle, members, batch_size, num_workers)

    groups: list[dict] = []
    gid = 0
    n_grouped = 0
    for artist in sorted(by_artist):
        bucket = [p for p in by_artist[artist] if p.stem in feats]
        if len(bucket) < 2:
            continue
        emb = np.stack([feats[p.stem].cls for p in bucket]).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        for comp in connected_components(emb, threshold):
            if len(comp) < min_size:
                continue
            members_rel = [bucket[i].relative_to(source_dir).as_posix() for i in comp]
            groups.append(
                {
                    "id": gid,
                    "artist": artist,
                    "size": len(comp),
                    "mean_cosine": round(_mean_pairwise_cosine(emb[comp]), 4),
                    "members": members_rel,
                }
            )
            gid += 1
            n_grouped += len(comp)

    manifest["n_groups"] = len(groups)
    manifest["n_grouped"] = n_grouped
    manifest["n_singletons"] = len(paths) - n_grouped
    manifest["groups"] = groups
    _write_manifest(out_path, manifest)
    return manifest


def _write_manifest(out_path: Path, manifest: dict) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
