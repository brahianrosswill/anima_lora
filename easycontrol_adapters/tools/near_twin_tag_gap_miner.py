#!/usr/bin/env python3
"""near_twin_tag_gap_miner — mine in-artist variant pairs by attribute gap.

An **exploration / curation tool** (not a training step) that surfaces
near-duplicate *variant* pairs within a single artist where the two members
differ by a **specified attribute** — e.g. one has a speech bubble and the
other doesn't. It feeds EasyControl builders: eval sets, seed data for unpaired
editing, and a difference-region mask localizing *where* the two members differ.

Pipeline (see ``docs/proposal/near_twin_tag_gap_miner.md`` for the full design):

1. **Gather members** per artist from ``--image-dirs`` (default the raw crawl
   pool ``~/gelcrawl/{retrieved,selected}``), scoped ``union`` so a twin can
   straddle the curated cut. Each member's native pixel ``(W, H)`` is read from
   the image header here (no decode).
1b. **Same-size gate**: a true variant pair (a redraw that adds one attribute)
   shares the **exact** canvas, so only members that share their ``(W, H)`` with
   ≥1 sibling *in the same artist* survive — the rest can never pair and are
   dropped before embedding. The pair loop then only ever compares equal-size
   members, which also makes the dense grid match pixel-aligned by construction
   (the original cross-crop case the PE machinery was hedging against is gone).
2. **Embed** each *surviving* image with **PE-Spatial-B16-512** (``library.vision``)
   at a fixed 512x512 native bucket → CLS descriptor + 32x32 patch grid (pooled to
   16x16, L2-normed). Cached per-image under ``~/.cache/near_twin/``.
3. **Stage A — global prefilter**: within-artist all-pairs cosine on the CLS
   descriptor; keep ``>= --sim-min``.
4. **Stage B — dense grid match**: pool each survivor's grid to ``G x G``, run a
   mutual-NN + ratio test, count inliers ``>= --cell-match-min``; a pair is a
   near-twin when the inlier fraction ``>= --match-frac-min``. Unmatched cells
   are the **difference region**. Optional ``--geom-check`` RANSAC-rejects pose
   twins and estimates the crop offset.
5. **Discriminator** (``--tag`` / ``--tag-any`` / ``--region`` / ``--signal``):
   keep pairs where the attribute is present in **exactly one** member.
6. **Rank by edit-cleanliness**: fewest *other* differences first.
7. **Output**: HTML contact sheet + TSV (curation-only) and a materialized
   ``_tags`` / ``_no_tags`` pair tree (the only training-shaped output), plus a
   ready-to-use EasyControl dataset config under ``configs/easycontrol/``.

Run from the repo root::

    python easycontrol_adapters/tools/near_twin_tag_gap_miner.py \
        --tag-any "speech bubble,thought bubble,blank speech bubble" \
        --artists ama_mitsuki --out output/near_twins/pairs.html

    # tagless visual attribute (recommended for bubbles on an untagged tree):
    python easycontrol_adapters/tools/near_twin_tag_gap_miner.py --region \
        --artists ama_mitsuki --out output/near_twins/pairs.html

Features are cached, so the intended loop is: run → open the HTML → adjust
``--sim-min`` / ``--match-frac-min`` / ``--cell-match-min`` / ``--max-extra-diff``
→ re-run (seconds).
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import os
import re
import shutil
import sys
import tomllib
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    from dotenv import load_dotenv  # picks up CAPTION_CORPUS_DIR from anima_lora/.env
except ImportError:  # dotenv is a soft dependency — env vars still work without it

    def load_dotenv(*_a, **_k):  # type: ignore
        return False


# Run from the repo root; `library` is installed editable (`uv sync`).
from library.vision import load_pe_encoder
from library.vision.encoder import encode_pe_from_imageminus1to1  # noqa: F401  (kept for API parity)

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(
    os.environ.get("NEAR_TWIN_CACHE", Path.home() / ".cache" / "near_twin")
)
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".jxl", ".avif")
PE_NATIVE = 512  # PE-Spatial-B16-512 square bucket → 32x32 patch grid
GRID_NATIVE = 32
GRID_CACHE = 16  # cached pooled grid edge; any --grid <= 16 pools down from here

# ---------------------------------------------------------------------------- captions / tags


def normalize_tag(tag: str) -> str:
    """Space-insensitive canonical form: lowercase, underscores→spaces, collapsed.

    ``speech_bubble`` and ``speech bubble`` map to the same key so either
    danbooru convention works (see the tag-form note in the proposal).
    """
    return " ".join(tag.strip().lower().replace("_", " ").split())


def read_tags(txt_path: Path) -> set[str]:
    """Read a ``.txt`` caption sidecar → set of normalized tags ("" → empty)."""
    if not txt_path.is_file():
        return set()
    raw = txt_path.read_text(encoding="utf-8", errors="ignore")
    return {normalize_tag(t) for t in raw.split(",") if t.strip()}


def caption_text(txt_path: Path) -> str:
    return (
        txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        if txt_path.is_file()
        else ""
    )


# ---------------------------------------------------------------------------- member discovery


@dataclass
class Member:
    artist: str
    stem: str
    image_path: Path
    txt_path: Path
    wh: tuple[int, int] = (0, 0)  # native pixel (W, H); (0, 0) = unreadable header


def _image_size(path: Path) -> tuple[int, int]:
    """Native ``(W, H)`` from the image header (no pixel decode); (0,0) on error."""
    try:
        with Image.open(path) as im:
            return im.size  # PIL returns (width, height)
    except Exception:  # noqa: BLE001 — corrupt/unreadable image
        return (0, 0)


def keep_size_cohabiting(members: list[Member]) -> list[Member]:
    """Drop members with no exact same-size sibling — they can never form a pair.

    The same-size gate's pre-embedding half: a unique canvas size within an
    artist has nothing to pair against, so embedding it would be wasted work.
    """
    sizes = Counter(m.wh for m in members)
    return [m for m in members if m.wh != (0, 0) and sizes[m.wh] >= 2]


def gather_members(
    image_dirs: list[Path], artists_filter: set[str] | None
) -> dict[str, list[Member]]:
    """Walk ``<dir>/<artist>/<stem>.<ext>`` trees → ``artist -> [Member]``.

    Scope is ``union`` across all ``image_dirs`` (a twin can straddle the
    curated cut). A ``(artist, stem)`` seen in more than one dir is kept once;
    the first dir listed wins (so put your preferred source — e.g. the curated
    ``selected`` PNGs — first if it matters for the export symlink target).
    """
    seen: dict[tuple[str, str], Member] = {}
    for d in image_dirs:
        if not d.is_dir():
            print(f"  [warn] image dir not found: {d}", file=sys.stderr)
            continue
        for artist_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            artist = artist_dir.name
            if artists_filter and artist not in artists_filter:
                continue
            for img in sorted(artist_dir.iterdir()):
                if img.suffix.lower() not in IMAGE_EXTS:
                    continue
                key = (artist, img.stem)
                if key in seen:
                    continue
                seen[key] = Member(
                    artist, img.stem, img, img.with_suffix(".txt"), _image_size(img)
                )
    by_artist: dict[str, list[Member]] = {}
    for (artist, _), m in seen.items():
        by_artist.setdefault(artist, []).append(m)
    for artist in by_artist:
        by_artist[artist].sort(key=lambda m: m.stem)
    return by_artist


# ---------------------------------------------------------------------------- embedding + cache


def _dir_hash(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _cache_path(member: Member) -> Path:
    return CACHE_ROOT / _dir_hash(member.image_path.parent) / f"{member.stem}.npz"


def _load_512(image_path: Path) -> torch.Tensor:
    """PIL → [3, 512, 512] in [-1, 1] (PE's Normalize(0.5, 0.5))."""
    with Image.open(image_path) as im:
        im = im.convert("RGB").resize((PE_NATIVE, PE_NATIVE), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    t = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    return t * 2.0 - 1.0


_BAD_TENSOR = torch.zeros(3, PE_NATIVE, PE_NATIVE)  # placeholder for a failed decode


class _ImageDataset(torch.utils.data.Dataset):
    """Decode+resize on DataLoader workers so CPU preprocessing overlaps the GPU
    forward. A corrupt image yields ``ok=False`` (skipped downstream) instead of
    crashing the whole pass."""

    def __init__(self, members: list[Member]):
        self.members = members

    def __len__(self) -> int:
        return len(self.members)

    def __getitem__(self, i: int):
        try:
            return i, _load_512(self.members[i].image_path), True
        except Exception:  # noqa: BLE001 — corrupt/unreadable image
            return i, _BAD_TENSOR, False


def _collate(batch):
    idxs = [b[0] for b in batch]
    tens = torch.stack([b[1] for b in batch])
    oks = [b[2] for b in batch]
    return idxs, tens, oks


@torch.no_grad()
def _forward_pe(bundle, batch: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Run PE-Spatial on a [B, 3, 512, 512] device batch → (cls, grid16) numpy."""
    out = bundle.encoder(batch)
    lhs = out.last_hidden_state.float()  # [B, 1+1024, 768]
    cls = F.normalize(lhs[:, 0], dim=-1)  # global descriptor
    grid = lhs[:, 1:].reshape(lhs.shape[0], GRID_NATIVE, GRID_NATIVE, -1)
    g = grid.permute(0, 3, 1, 2)  # [B, 768, 32, 32]
    g16 = F.adaptive_avg_pool2d(g, GRID_CACHE).permute(0, 2, 3, 1)  # [B, 16, 16, 768]
    return cls.cpu().numpy(), g16.cpu().numpy().astype(np.float16)


def _save_feature(cache_path: Path, f: "Feature") -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, cls=f.cls.astype(np.float32), grid16=f.grid16)


@dataclass
class Feature:
    cls: np.ndarray  # [768] L2-normed float32
    grid16: np.ndarray  # [16, 16, 768] float16


def embed_members(
    bundle, members: list[Member], batch_size: int, num_workers: int = 4
) -> dict[str, Feature]:
    """Load cached PE-Spatial features; embed + cache any misses once.

    Misses are streamed through a ``DataLoader``: worker processes decode +
    resize the next batches while the GPU runs the current forward, the batch is
    copied to the device with pinned-memory async H2D, and the ``.npz`` cache
    writes are handed to a thread pool — so CPU I/O, the host→device copy, and
    the GPU forward overlap instead of running serially.
    """
    feats: dict[str, Feature] = {}
    todo: list[Member] = []
    for m in members:
        cp = _cache_path(m)
        if cp.is_file():
            with np.load(cp) as z:
                feats[m.stem] = Feature(
                    cls=z["cls"].astype(np.float32), grid16=z["grid16"]
                )
        else:
            todo.append(m)
    if not todo:
        return feats

    pin = bundle.device.type == "cuda"
    loader = torch.utils.data.DataLoader(
        _ImageDataset(todo),
        batch_size=batch_size,
        num_workers=min(num_workers, len(todo)),
        pin_memory=pin,
        collate_fn=_collate,
        persistent_workers=False,
    )
    done = 0
    with ThreadPoolExecutor(max_workers=2) as saver:
        for idxs, tens, oks in loader:
            batch = tens.to(bundle.device, bundle.dtype, non_blocking=pin)
            cls_b, grid_b = _forward_pe(bundle, batch)
            for k, i in enumerate(idxs):
                done += 1
                if not oks[k]:
                    print(
                        f"  [warn] skipped unreadable {todo[i].image_path}",
                        file=sys.stderr,
                    )
                    continue
                f = Feature(cls=cls_b[k], grid16=grid_b[k])
                feats[todo[i].stem] = f
                saver.submit(_save_feature, _cache_path(todo[i]), f)
            print(f"  embedded {done}/{len(todo)}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return feats


# ---------------------------------------------------------------------------- Stage B match


@dataclass
class MatchResult:
    n_inliers: int
    match_frac: float
    diff_cells: set[int]  # union of unmatched a/b cells (grid index r*G + c)
    diff_a: set[int]
    diff_b: set[int]
    offset: tuple[float, float]  # estimated (drow, dcol) crop offset (geom-check)
    G: int


def _pool_cells(grid16: np.ndarray, G: int) -> np.ndarray:
    """[16, 16, 768] → [G*G, 768], L2-normed per cell."""
    t = torch.from_numpy(grid16.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    p = F.adaptive_avg_pool2d(t, G)[0].permute(1, 2, 0).reshape(G * G, -1).numpy()
    return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)


def match_grids(
    fa: Feature, fb: Feature, G: int, cell_min: float, ratio: float, geom_check: bool
) -> MatchResult:
    """Mutual-NN + ratio-test dense cell match between two pooled grids.

    Raw "has a >0.9 neighbor" inflates badly on anime art's flat color fields,
    so we require **mutual** nearest neighbours that also pass a distinctiveness
    (ratio) test. Unmatched cells localize the difference region.
    """
    ca, cb = _pool_cells(fa.grid16, G), _pool_cells(fb.grid16, G)
    N = G * G
    sim = ca @ cb.T  # [N, N] cosine (cells are unit-norm)
    col_best = sim.argmax(axis=0)
    matched: list[tuple[int, int]] = []
    for i in range(N):
        row = sim[i]
        order = np.argsort(-row)
        nn1 = int(order[0])
        s1 = float(row[nn1])
        s2 = float(row[order[1]]) if N > 1 else -1.0
        if s1 < cell_min:
            continue
        if col_best[nn1] != i:  # mutual NN
            continue
        if (1.0 - s1) > ratio * (1.0 - s2):  # ratio test on cosine-distance
            continue
        matched.append((i, nn1))

    offset = (0.0, 0.0)
    if geom_check and matched:
        matched, offset = _geom_filter(matched, G)

    matched_a = {i for i, _ in matched}
    matched_b = {j for _, j in matched}
    diff_a = set(range(N)) - matched_a
    diff_b = set(range(N)) - matched_b
    return MatchResult(
        n_inliers=len(matched),
        match_frac=len(matched) / N,
        diff_cells=diff_a | diff_b,
        diff_a=diff_a,
        diff_b=diff_b,
        offset=offset,
        G=G,
    )


def _geom_filter(
    matched: list[tuple[int, int]], G: int
) -> tuple[list[tuple[int, int]], tuple[float, float]]:
    """RANSAC-lite translation consistency: keep matches near the median offset.

    Rejects "same character, different pose" (whose cell offsets scatter) and
    estimates the crop offset from the surviving translation. A full
    LoFTR/SIFT homography is the escape hatch if the coarse grid is too blunt.
    """
    a = np.array([[i // G, i % G] for i, _ in matched], dtype=np.float32)
    b = np.array([[j // G, j % G] for _, j in matched], dtype=np.float32)
    deltas = b - a
    med = np.median(deltas, axis=0)
    keep = (
        np.abs(deltas - med).max(axis=1) <= 1.0
    )  # within 1 cell of the consensus shift
    kept = [m for m, k in zip(matched, keep) if k]
    return kept, (float(med[0]), float(med[1]))


def _largest_blob(cells: set[int], G: int) -> tuple[int, set[int]]:
    """Largest 4-connected component of ``cells`` on the G×G grid → (size, blob)."""
    remaining = set(cells)
    best: set[int] = set()
    while remaining:
        seed = next(iter(remaining))
        comp: set[int] = set()
        q = deque([seed])
        remaining.discard(seed)
        while q:
            c = q.popleft()
            comp.add(c)
            r, col = divmod(c, G)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, col + dc
                if 0 <= nr < G and 0 <= nc < G:
                    n = nr * G + nc
                    if n in remaining:
                        remaining.discard(n)
                        q.append(n)
        if len(comp) > len(best):
            best = comp
    return len(best), best


def diff_bbox_norm(cells: set[int], G: int) -> tuple[float, float, float, float]:
    """Bounding box over ``cells`` (grid idx) → normalized (x0, y0, x1, y1) in [0,1]."""
    if not cells:
        return (0.0, 0.0, 0.0, 0.0)
    rs = [c // G for c in cells]
    cs = [c % G for c in cells]
    return (min(cs) / G, min(rs) / G, (max(cs) + 1) / G, (max(rs) + 1) / G)


# ---------------------------------------------------------------------------- discriminator


@dataclass
class PairVerdict:
    accept: bool
    gap_holder: str  # "a" | "b" | "?"
    n_extra_diff: int
    extra_diff_tags: list[str]
    reason: str = ""


def discriminate_tag(
    tags_a: set[str],
    tags_b: set[str],
    target: set[str],
    max_extra: int,
    rest_jaccard_min: float,
) -> PairVerdict:
    """Keep pairs where the target tag is present in **exactly one** member."""
    in_a = bool(target & tags_a)
    in_b = bool(target & tags_b)
    if in_a == in_b:  # both or neither → not a single-attribute gap on this tag
        return PairVerdict(False, "?", 0, [], "tag present in both/neither")
    gap_holder = "a" if in_a else "b"
    rest_a, rest_b = tags_a - target, tags_b - target
    extra = sorted(rest_a ^ rest_b)
    if max_extra >= 0 and len(extra) > max_extra:
        return PairVerdict(
            False, gap_holder, len(extra), extra, "too many other tag differences"
        )
    if rest_jaccard_min > 0:
        union = rest_a | rest_b
        jac = len(rest_a & rest_b) / len(union) if union else 1.0
        if jac < rest_jaccard_min:
            return PairVerdict(
                False, gap_holder, len(extra), extra, f"rest-jaccard {jac:.2f} < min"
            )
    return PairVerdict(True, gap_holder, len(extra), extra)


def discriminate_region(
    match: MatchResult, region_min_frac: float, region_max_frac: float, scatter_max: int
) -> PairVerdict:
    """Tagless: a single compact diff region of the right rough size IS the gap.

    ``gap_holder`` is a best-effort guess (the member with more unmatched cells
    in the diff region — the side carrying the added content); cross-check by
    eyeballing the HTML.
    """
    N = match.G * match.G
    size, blob = _largest_blob(match.diff_cells, match.G)
    frac = size / N
    scatter = len(match.diff_cells) - size  # diff cells outside the main blob
    if not (region_min_frac <= frac <= region_max_frac):
        return PairVerdict(
            False, "?", scatter, [], f"region frac {frac:.2f} out of range"
        )
    if scatter > scatter_max:
        return PairVerdict(False, "?", scatter, [], f"diff scatter {scatter} > max")
    gap_holder = "a" if len(match.diff_a & blob) >= len(match.diff_b & blob) else "b"
    return PairVerdict(True, gap_holder, scatter, [])


# ---------------------------------------------------------------------------- ranked pair record


@dataclass
class PairRecord:
    artist: str
    a: Member
    b: Member
    cosine: float
    match: MatchResult
    verdict: PairVerdict
    extra_fields: dict = field(default_factory=dict)

    @property
    def ids(self) -> tuple[str, str]:
        return tuple(sorted((self.a.stem, self.b.stem)))

    @property
    def pair_id(self) -> str:
        x, y = self.ids
        return f"{x}-{y}"

    def holder_member(self) -> Member:
        return self.a if self.verdict.gap_holder == "a" else self.b

    def clean_member(self) -> Member:
        return self.b if self.verdict.gap_holder == "a" else self.a


# ---------------------------------------------------------------------------- core run


def run_artist(
    artist: str,
    members: list[Member],
    feats: dict[str, Feature],
    args: argparse.Namespace,
    target_tags: set[str],
) -> list[PairRecord]:
    out: list[PairRecord] = []
    stems = [m.stem for m in members if m.stem in feats]
    cls = np.stack([feats[s].cls for s in stems]) if stems else np.empty((0, 768))
    idx = {m.stem: m for m in members}
    n = len(stems)
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = stems[i], stems[j]
            if idx[si].wh != idx[sj].wh:  # same-size gate: exact (W, H) only
                continue
            if args.id_window and si.isdigit() and sj.isdigit():
                if abs(int(si) - int(sj)) > args.id_window:
                    continue
            cos = float(cls[i] @ cls[j])
            if cos < args.sim_min:  # Stage A prefilter
                continue
            match = match_grids(
                feats[si],
                feats[sj],
                args.grid,
                args.cell_match_min,
                args.ratio,
                args.geom_check,
            )
            if match.match_frac < args.match_frac_min:  # Stage B near-twin gate
                continue
            ma, mb = idx[si], idx[sj]
            if args.mode == "tag":
                verdict = discriminate_tag(
                    read_tags(ma.txt_path),
                    read_tags(mb.txt_path),
                    target_tags,
                    args.max_extra_diff,
                    args.rest_jaccard_min,
                )
            elif args.mode == "region":
                verdict = discriminate_region(
                    match,
                    args.region_min_frac,
                    args.region_max_frac,
                    args.region_scatter_max,
                )
            else:  # signal
                verdict = discriminate_signal(ma, mb, args)
            if not verdict.accept:
                continue
            out.append(PairRecord(artist, ma, mb, cos, match, verdict))

    # Rank by edit-cleanliness: fewest other differences first, then tightest twin.
    out.sort(key=lambda p: (p.verdict.n_extra_diff, -p.cosine))
    if args.per_artist_topk:
        out = out[: args.per_artist_topk]
    return out


# ---------------------------------------------------------------------------- signal mode (MIT text)


_SIGNAL_STATE: dict = {}


def _mit_text_fraction(member: Member, device: str) -> float:
    """Per-image MIT text-area fraction — the detector behind post_image_dataset/masks/."""
    if "mit_model" not in _SIGNAL_STATE:
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "preprocess"))
        from generate_masks_mit import _detect_mask, _load_model  # type: ignore

        _SIGNAL_STATE["mit_model"] = _load_model(None, device=device)
        _SIGNAL_STATE["mit_detect"] = _detect_mask
    cache = CACHE_ROOT / _dir_hash(member.image_path.parent) / f"{member.stem}.mit_text"
    if cache.is_file():
        return float(cache.read_text())
    with Image.open(member.image_path) as im:
        arr = np.asarray(im.convert("RGB"))
    mask = _SIGNAL_STATE["mit_detect"](
        _SIGNAL_STATE["mit_model"], arr, device=device, text_threshold=0.8
    )
    frac = float(np.count_nonzero(mask) / mask.size)
    cache.write_text(f"{frac:.6f}")
    return frac


def discriminate_signal(a: Member, b: Member, args: argparse.Namespace) -> PairVerdict:
    if args.signal != "mit_text":
        raise ValueError(f"unknown signal {args.signal!r}")
    fa = _mit_text_fraction(a, args.device)
    fb = _mit_text_fraction(b, args.device)
    hi, lo = (fa, fb) if fa >= fb else (fb, fa)
    if (
        hi - lo < args.signal_delta or lo > args.signal_delta
    ):  # gap present + low side ≈ 0
        return PairVerdict(False, "?", 0, [], f"signal gap {hi - lo:.3f} insufficient")
    gap_holder = "a" if fa >= fb else "b"
    return PairVerdict(True, gap_holder, 0, [])


# ---------------------------------------------------------------------------- outputs


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
# blueprint) on every run, and leaves everything above it — the user's [miner]
# run knobs + comments — untouched. So one file is both the run config (input,
# read via --config) and the generated subset blueprint (output).
_BLUEPRINT_SENTINEL = (
    "# === generated dataset blueprint (rewritten by the miner; do not edit below) ==="
)

_DEFAULT_MINER_HEADER = """\
# near-twin tag-gap miner config. Edit the [miner] table, then just run:
#   python easycontrol_adapters/tools/near_twin_tag_gap_miner.py
# CLI flags override these. Paths support {CAPTION_CORPUS_DIR}/$VARS/~ expansion.

[miner]
# Discriminator — set exactly one of: tag_any / tag / region (bool) / signal.
tag_any = ["speech bubble", "thought bubble", "blank speech bubble"]
# Source trees (<dir>/<artist>/<id>.<ext>).
image_dirs = ["{CAPTION_CORPUS_DIR}/retrieved"]
# Stage/threshold knobs (any --flag dest works here):
sim_min = 0.85
match_frac_min = 0.66
cell_match_min = 0.9
max_extra_diff = 6
"""


def _blueprint_text(export_dir: Path) -> str:
    img_dir = _rel_to_repo(export_dir)
    cache_dir = _rel_to_repo(
        REPO_ROOT / "post_image_dataset" / "easycontrol" / "near_twins_cache"
    )
    return f"""{_BLUEPRINT_SENTINEL}
# Near-twin pair tree as an EasyControl-style subset. Source images stay
# user-facing; VAE/TE/PE caches land under post_image_dataset/ (the IP-Adapter /
# EasyControl cache_dir pattern). Seed/eval data, NOT a turnkey control-adapter
# dataset — each accepted pair is a `{{id}}_tags` / `{{id}}_no_tags` couple, the
# `_tags` side holding the discriminator attribute (see the proposal doc).

[general]
caption_extension = '.txt'

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = '{img_dir}'
  cache_dir = '{cache_dir}'
  num_repeats = 1
"""


def write_dataset_config(export_dir: Path, config_path: Path) -> None:
    """Rewrite the blueprint tail of ``config_path``, preserving the ``[miner]`` head.

    First creation scaffolds a ``[miner]`` template; re-runs keep everything
    above the sentinel (the user's edited run knobs) verbatim.
    """
    blueprint = _blueprint_text(export_dir)
    if config_path.is_file():
        existing = config_path.read_text(encoding="utf-8")
        head = existing.split(_BLUEPRINT_SENTINEL, 1)[0].rstrip()
        content = f"{head}\n\n{blueprint}" if head else blueprint
    else:
        content = f"{_DEFAULT_MINER_HEADER}\n{blueprint}"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------- config ([miner])


def expand_path(s: str) -> str:
    """Expand ``{VAR}`` / ``${VAR}`` / ``$VAR`` / ``~`` in a path string.

    ``{CAPTION_CORPUS_DIR}`` and ``${CAPTION_CORPUS_DIR}`` both resolve from the
    environment (loaded from ``anima_lora/.env``), so the toml can reference the
    corpus root without hardcoding an absolute path.
    """
    s = re.sub(r"\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), s)
    return os.path.expanduser(os.path.expandvars(s))


# toml [miner] keys that name a comma-joined list flag (accept a TOML array too).
_LIST_FLAGS = {"image_dirs": True, "tag_any": False}  # value→expand-as-path?


def _explicit_dests(argv: list[str]) -> set[str]:
    """dest names the user passed explicitly on the CLI (so they override toml)."""
    return {
        tok[2:].split("=", 1)[0].replace("-", "_")
        for tok in argv
        if tok.startswith("--")
    }


def apply_miner_config(args: argparse.Namespace, argv: list[str]) -> None:
    """Layer a ``[miner]`` table from ``--config`` under the CLI.

    Precedence: explicit CLI flag > ``[miner]`` toml > argparse default. Keys
    mirror the flag dest names (e.g. ``tag_any``, ``sim_min``, ``match_frac_min``,
    ``image_dirs``). List-valued keys (``image_dirs``, ``tag_any``) accept a TOML
    array and are stored as the comma-joined string the flags already parse.
    """
    if not args.config:
        return
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        return
    table = tomllib.loads(cfg_path.read_text(encoding="utf-8")).get("miner")
    if not table:
        return
    explicit = _explicit_dests(argv)
    for key, val in table.items():
        dest = key.replace("-", "_")
        if dest in explicit:
            continue  # CLI wins
        if not hasattr(args, dest):
            print(
                f"  [warn] unknown [miner] key {key!r} in {cfg_path}", file=sys.stderr
            )
            continue
        if dest in _LIST_FLAGS:
            items = val if isinstance(val, (list, tuple)) else [val]
            if _LIST_FLAGS[dest]:  # path list → expand each
                items = [expand_path(str(x)) for x in items]
            setattr(args, dest, ",".join(str(x) for x in items))
        else:
            setattr(args, dest, val)


# ---------------------------------------------------------------------------- CLI


def _default_image_dirs() -> str:
    """Default source trees: ``$CAPTION_CORPUS_DIR/{retrieved,selected}`` when the
    corpus root is set (the Anima-Tagger convention), else ``~/gelcrawl/…``."""
    root = os.environ.get("CAPTION_CORPUS_DIR")
    base = Path(root) if root else Path.home() / "gelcrawl"
    return f"{base / 'retrieved'},{base / 'selected'}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mine in-artist near-twin variant pairs that differ by one attribute.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        default="configs/easycontrol/near_twins.toml",
        help="toml with a [miner] table of run knobs (CLI flags override it; '' disables)",
    )
    disc = p.add_mutually_exclusive_group()
    disc.add_argument(
        "--tag", help="discriminator: this tag present in exactly one member"
    )
    disc.add_argument(
        "--tag-any", help="discriminator: any of these comma-separated synonyms"
    )
    disc.add_argument(
        "--region",
        action="store_true",
        help="discriminator: Stage-B diff region (tagless)",
    )
    disc.add_argument(
        "--signal", choices=["mit_text"], help="discriminator: per-image scalar gap"
    )

    p.add_argument(
        "--image-dirs",
        default=_default_image_dirs(),
        help="comma-separated source trees (<dir>/<artist>/<id>.<ext>); "
        "supports {CAPTION_CORPUS_DIR}/$VARS/~ expansion",
    )
    p.add_argument("--artists", help="comma-separated artist allowlist (default: all)")
    p.add_argument(
        "--per-artist-topk",
        type=int,
        default=0,
        help="keep only the top-K pairs per artist (0=all)",
    )

    p.add_argument(
        "--sim-min",
        type=float,
        default=0.85,
        help="Stage-A CLS-cosine prefilter threshold",
    )
    p.add_argument(
        "--grid", type=int, default=7, help="Stage-B pooled grid edge (G×G cells)"
    )
    p.add_argument(
        "--cell-match-min",
        type=float,
        default=0.9,
        help="per-cell cosine for an inlier match",
    )
    p.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="ratio-test distinctiveness (lower = stricter)",
    )
    p.add_argument(
        "--match-frac-min",
        type=float,
        default=0.66,
        help="inlier fraction to call a near-twin",
    )
    p.add_argument(
        "--geom-check",
        action="store_true",
        help="RANSAC translation consistency (reject pose twins)",
    )

    p.add_argument(
        "--rest-jaccard-min",
        type=float,
        default=0.0,
        help="'same scene' rest-tag overlap floor (0=off)",
    )
    p.add_argument(
        "--max-extra-diff",
        type=int,
        default=6,
        help="cap on non-target differing tags (-1=off)",
    )
    p.add_argument(
        "--signal-delta",
        type=float,
        default=0.04,
        help="signal-mode gap / low-side threshold",
    )
    p.add_argument(
        "--region-min-frac",
        type=float,
        default=0.02,
        help="region mode: min diff-blob cell fraction",
    )
    p.add_argument(
        "--region-max-frac",
        type=float,
        default=0.4,
        help="region mode: max diff-blob cell fraction",
    )
    p.add_argument(
        "--region-scatter-max",
        type=int,
        default=4,
        help="region mode: max diff cells outside main blob",
    )
    p.add_argument(
        "--id-window", type=int, default=0, help="only pair posts within N ids (0=off)"
    )

    p.add_argument(
        "--out",
        default="output/near_twins/pairs.html",
        help="HTML contact sheet (TSV written alongside)",
    )
    p.add_argument(
        "--export-dir",
        default="post_image_dataset/easycontrol/near_twins",
        help="materialized _tags/_no_tags pair tree (empty string disables)",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="copy images into the export tree (default: symlink)",
    )
    p.add_argument(
        "--emit-mask",
        action="store_true",
        help="also write the Stage-B diff-region mask per pair",
    )
    p.add_argument(
        "--config-out",
        default="configs/easycontrol/near_twins.toml",
        help="dataset blueprint written for the export tree (empty string disables)",
    )
    p.add_argument(
        "--batch-size", type=int, default=16, help="PE-Spatial embed batch size"
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers for image decode/resize",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # CAPTION_CORPUS_DIR etc. from anima_lora/.env, before defaults resolve
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    apply_miner_config(args, argv)  # [miner] toml under the CLI
    args.mode = "region" if args.region else "signal" if args.signal else "tag"

    target_tags: set[str] = set()
    if args.mode == "tag":
        raw = args.tag_any or args.tag
        if not raw:
            print(
                "error: tag mode needs --tag or --tag-any (or use --region / --signal)",
                file=sys.stderr,
            )
            return 2
        target_tags = {normalize_tag(t) for t in raw.split(",") if t.strip()}

    image_dirs = [Path(expand_path(d)) for d in args.image_dirs.split(",") if d.strip()]
    artists_filter = (
        {a.strip() for a in args.artists.split(",")} if args.artists else None
    )

    print(f"Gathering members from {len(image_dirs)} dir(s)…", file=sys.stderr)
    by_artist = gather_members(image_dirs, artists_filter)
    total = sum(len(v) for v in by_artist.values())
    print(f"  {len(by_artist)} artist(s), {total} unique images", file=sys.stderr)
    if not total:
        print("No images found — check --image-dirs / --artists.", file=sys.stderr)
        return 1

    device = torch.device(args.device)
    print(f"Loading PE-Spatial-B16-512 on {device}…", file=sys.stderr)
    bundle = load_pe_encoder(device, name="pe_spatial")

    print(
        "Same-size gate ON: pairing only exact-(W×H) members within each artist.",
        file=sys.stderr,
    )
    all_pairs: list[PairRecord] = []
    n_pairable = 0
    for artist, members in by_artist.items():
        members = keep_size_cohabiting(
            members
        )  # drop sizes with no sibling before embedding
        if len(members) < 2:
            continue
        n_pairable += len(members)
        feats = embed_members(bundle, members, args.batch_size, args.num_workers)
        pairs = run_artist(artist, members, feats, args, target_tags)
        if pairs:
            print(f"  {artist}: {len(pairs)} pair(s)", file=sys.stderr)
        all_pairs.extend(pairs)
    print(
        f"  {n_pairable}/{total} image(s) had a same-size sibling (embedded)",
        file=sys.stderr,
    )

    all_pairs.sort(key=lambda p: (p.verdict.n_extra_diff, -p.cosine))
    print(f"\n{len(all_pairs)} accepted pair(s) total", file=sys.stderr)

    out_html = Path(args.out)
    write_html(all_pairs, out_html, thumb=384)
    out_tsv = out_html.with_suffix(".tsv")
    write_tsv(all_pairs, out_tsv)
    print(f"  HTML  → {out_html}", file=sys.stderr)
    print(f"  TSV   → {out_tsv}", file=sys.stderr)

    if args.export_dir and all_pairs:
        export_dir = Path(args.export_dir)
        n = export_pairs(
            all_pairs, export_dir, copy=args.copy, emit_mask=args.emit_mask
        )
        print(
            f"  pairs → {export_dir}/  ({n} pair tree(s), {'copied' if args.copy else 'symlinked'})",
            file=sys.stderr,
        )
        if args.config_out:
            write_dataset_config(export_dir, Path(args.config_out))
            print(f"  cfg   → {args.config_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
