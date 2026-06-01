"""FreeText Stage-1 — attention-guided text-region localization (paper §3.1).

This is the *pure* pipeline: numpy / scipy / scikit-learn only, no model or
torch dependency, so it can be unit-tested on synthetic maps. The driver
(``stage1_localize.py``) captures the endogenous I2T attention with the same
eager-recompute hook the Phase-0 probe uses, hands us per-(timestep, layer)
group-maps, and we turn them into the binary writing mask ``R`` that Stage-2
(SGMI) injects into.

The three sub-stages, with their paper equations:

  3.1.1  Attention extraction      anchor map  M^(t,l)  (Eq 1-2)
  3.1.2  Timestep-layer selection  soft-IoU top-K aggregate  M  (Eq 3-4)
  3.1.3  Topology-aware selection   neighborhood-denoise -> Otsu -> DBSCAN
                                    -> region score q_i -> best region
                                    -> resize to latent -> R  (Eq 5-6)

Faithfulness notes / Anima adaptations (see bench/freetext/README.md):

* **Anchor set T̃_s = Entity ∪ Sink** is the paper's Table-4 winner. On Anima the
  Qwen3-base tokenizer emits *no* BOS/EOS (n_special=0), so the only attention
  sink is the zeroed padding field (447 positions, strategy.py:137). The probe
  hands us the within-group *mean* over the entity tokens and over the padding
  tokens separately, so "Entity+Sink" is a per-map combination of the two
  group-means (``anchor_map(mode="entity_sink")``) rather than a raw union-mean
  over 2+447 tokens — which would be swamped by the sink field.

* **Reference Y for the soft-IoU (Eq 3) is under-specified in the paper**: in
  zero-layout deployment there is no GT box to score against. We default to a
  *self-consensus* reference (the robust mean of all per-(t,l) maps), which
  rewards the maps that agree with the stable majority and down-weights noisy
  outliers — the same "stable across timesteps and layers" cue the paper leans
  on. ``reference_map`` is a thin reducer so the driver can swap in a sink-only
  or band-restricted reference, or a real GT box, via its ``--ref_mode`` knob.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter, zoom
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# 3.1.1  Attention extraction -> anchor map (Eq 2)
# ---------------------------------------------------------------------------
def normalize01(m: np.ndarray) -> np.ndarray:
    """Linear normalize to [0, 1]; all-equal -> zeros."""
    m = np.asarray(m, dtype=np.float64)
    lo = float(m.min())
    hi = float(m.max())
    if hi <= lo:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def anchor_map(
    entity: np.ndarray,
    sink: np.ndarray,
    mode: str = "entity_sink",
    combine: str = "normsum",
) -> np.ndarray:
    """Eq 2 — per-(t,l) localization map over the anchor token set T̃_s.

    ``entity`` / ``sink`` are the within-group mean I2T attention maps (each a
    patch grid ``(hp, wp)``). The result is normalized to [0, 1].

    mode:    ``entity`` | ``sink`` | ``entity_sink`` (Table-4 winner, default)
    combine: ``normsum``  — normalize each group-map to [0,1] then add (each
                            anchor type contributes equally regardless of token
                            count or raw sink magnitude);
             ``rawsum``   — add the raw group-means (paper-literal union-mean up
                            to a constant), sink magnitude can dominate.
    """
    entity = np.asarray(entity, dtype=np.float64)
    sink = np.asarray(sink, dtype=np.float64)
    if mode == "entity":
        m = entity
    elif mode == "sink":
        m = sink
    elif mode == "entity_sink":
        if combine == "rawsum":
            m = entity + sink
        else:
            m = normalize01(entity) + normalize01(sink)
    else:
        raise ValueError(f"unknown anchor mode {mode!r}")
    return normalize01(m)


# ---------------------------------------------------------------------------
# 3.1.2  Timestep-layer selection (Eq 3-4)
# ---------------------------------------------------------------------------
def soft_iou(M: np.ndarray, Y: np.ndarray, eps: float = 1e-9) -> float:
    """Eq 3 — soft IoU between two [0,1] maps (continuous Jaccard)."""
    M = np.asarray(M, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    inter = float((M * Y).sum())
    union = float(M.sum() + Y.sum()) - inter
    return inter / (union + eps)


def concentration(M: np.ndarray, top_frac: float = 0.05) -> float:
    """Fraction of total mass in the top ``top_frac`` patches (Phase-0 metric).

    1.0 = delta, ~``top_frac`` = uniform. This operationalizes the paper's
    "informative" (§3.1.2): mid-timestep shallow-mid maps concentrate on the
    target region, deep/early maps are diffuse.
    """
    v = np.asarray(M, dtype=np.float64).ravel()
    s = v.sum()
    if s <= 0:
        return 0.0
    k = max(1, int(len(v) * top_frac))
    return float(np.sort(v)[-k:].sum() / s)


def reference_map(stack: np.ndarray, reduce: str = "mean") -> np.ndarray:
    """Self-consensus reference Y from a stack of per-(t,l) maps ``(N, hp, wp)``.

    ``mean`` = consensus (default); ``median`` = outlier-robust consensus. The
    driver controls *which* maps go into the stack (all of them for the
    consensus reference, sink-only or band-only for the alternatives).
    """
    stack = np.asarray(stack, dtype=np.float64)
    if reduce == "median":
        Y = np.median(stack, axis=0)
    elif reduce == "mean":
        Y = stack.mean(axis=0)
    else:
        raise ValueError(f"unknown reduce {reduce!r}")
    return normalize01(Y)


def concentration_reference(
    stack: np.ndarray, conc_q: float = 0.75, top_frac: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrapped reference: consensus of the most-*concentrated* maps.

    A plain consensus over *all* per-(t,l) maps is dominated by the diffuse
    deep/early-step majority, so soft-IoU(M, consensus) ends up selecting those
    diffuse maps — the opposite of "informative". Instead we score every map by
    ``concentration``, keep the top ``(1 - conc_q)`` fraction (e.g. conc_q=0.75
    -> top 25% peakiest), and take *their* consensus as a sharp, localized
    reference. soft-IoU against this rewards maps that agree with the
    consistently-peaked region (the writing area), per §3.1.2.

    Returns ``(Y, concentrations)``.
    """
    stack = np.asarray(stack, dtype=np.float64)
    cs = np.array([concentration(m, top_frac) for m in stack])
    thr = np.quantile(cs, conc_q)
    keep = cs >= thr
    if not keep.any():
        keep = cs >= cs.max()
    Y = normalize01(stack[keep].mean(axis=0))
    return Y, cs


def select_and_aggregate(
    maps: np.ndarray, Y: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eq 3-4 — score every map by soft-IoU vs Y, keep top-k, aggregate (mean).

    Returns ``(selected_idx, ious, M)`` where ``M`` is the normalized aggregate.
    """
    maps = np.asarray(maps, dtype=np.float64)
    ious = np.array([soft_iou(m, Y) for m in maps])
    k = max(1, min(k, len(maps)))
    sel = np.argsort(-ious)[:k]
    M = maps[sel].mean(axis=0)
    return sel, ious, normalize01(M)


# ---------------------------------------------------------------------------
# 3.1.3  Topology-aware region selection (Eq 5-6)
# ---------------------------------------------------------------------------
def otsu_threshold(vec: np.ndarray, bins: int = 64) -> float:
    """Otsu threshold on a [0,1] map (maximize inter-class variance)."""
    v = np.asarray(vec, dtype=np.float64).ravel()
    hist, edges = np.histogram(v, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    tot = hist.sum()
    if tot == 0:
        return 0.5
    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
    m1 = (np.cumsum((hist * centers)[::-1])[::-1]) / np.maximum(w1, 1e-9)
    sigma_b = (w0 / tot) * (w1 / tot) * (m0 - m1) ** 2
    return float(centers[int(np.nanargmax(sigma_b))])


def neighborhood_aggregate(M: np.ndarray, size: int = 3) -> np.ndarray:
    """Local mean filter — suppress isolated outliers, promote connected mass."""
    if size <= 1:
        return np.asarray(M, dtype=np.float64)
    return uniform_filter(np.asarray(M, dtype=np.float64), size=size, mode="nearest")


def dbscan_regions(
    B: np.ndarray, eps: float = 1.5, min_samples: int = 4
) -> tuple[list[np.ndarray], np.ndarray]:
    """DBSCAN over foreground patch coordinates -> connected region masks.

    Returns ``(regions, label_map)``. ``eps`` is in patch units (1.5 links the
    8-neighborhood + one-patch gaps); DBSCAN noise (label -1) is dropped, which
    is the paper's "discard sparse noise".
    """
    B = np.asarray(B, dtype=bool)
    ys, xs = np.nonzero(B)
    label_map = np.full(B.shape, -1, dtype=int)
    if len(xs) == 0:
        return [], label_map
    coords = np.stack([ys, xs], axis=1).astype(np.float64)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    label_map[ys, xs] = labels
    regions: list[np.ndarray] = []
    for lab in sorted(set(labels) - {-1}):
        regions.append(label_map == lab)
    return regions, label_map


def region_scores(
    M: np.ndarray, regions: list[np.ndarray], tau_q: float = 0.8
) -> tuple[list[float], float]:
    """Eq 5 — q_i = fraction of region pixels above τ (a high quantile of M
    within the union of candidate regions)."""
    if not regions:
        return [], 0.0
    union = np.zeros(M.shape, dtype=bool)
    for r in regions:
        union |= r
    tau = float(np.quantile(M[union], tau_q)) if union.any() else 0.0
    scores = [float((M[r] > tau).sum()) / max(int(r.sum()), 1) for r in regions]
    return scores, tau


def select_region(
    M: np.ndarray,
    regions: list[np.ndarray],
    tau_q: float = 0.8,
    mode: str = "q",
) -> tuple[int, dict]:
    """Pick the best candidate region. Returns ``(best_idx, stats)``.

    mode: ``q``     — Eq-5 confidence q_i (paper-literal: fraction above τ);
          ``qmass`` — q_i * region size (guards against a 1-patch q=1 winner);
          ``mass``  — mean(M)·size = total attention mass in the region;
          ``peak``  — the region *containing the global argmax* of ``M`` (the
                      most-confident attention point), mass tie-break.

    ``mass`` ranks by total region mass — fine when the threshold cleanly
    isolates the writing band, but it reduces to "largest warm blob" and
    over-grabs when the map is *soft / low-contrast* (e.g. text the base can't
    render crisply): a low threshold merges sign+body into one giant component,
    which then wins on size. ``peak`` is robust to that: it anchors on the single
    most-confident patch (which the n=3 validation shows stays on the sign even
    for un-rendered Korean) and returns whatever connected region holds it,
    regardless of a larger neighbour. Falls back to ``mass`` only if the peak
    landed in DBSCAN noise. See README / stage1_progress.md.
    Returns ``best_idx = -1`` when there are no candidate regions.
    """
    if not regions:
        return -1, {"scores": [], "sizes": [], "means": [], "tau": 0.0}
    scores, tau = region_scores(M, regions, tau_q)
    sizes = [int(r.sum()) for r in regions]
    means = [float(M[r].mean()) for r in regions]
    stats = {"scores": scores, "sizes": sizes, "means": means, "tau": tau}
    if mode == "peak":
        py, px = np.unravel_index(int(np.argmax(M)), M.shape)
        stats["peak_yx"] = [int(py), int(px)]
        holding = [i for i, r in enumerate(regions) if bool(r[py, px])]
        if holding:  # regions are disjoint, so at most one
            stats["peak_in_region"] = True
            best = max(holding, key=lambda i: means[i] * sizes[i])
            return best, stats
        stats["peak_in_region"] = False  # peak fell in DBSCAN noise -> mass
        key = [m * s for m, s in zip(means, sizes)]
        return int(np.argmax(key)), stats
    if mode == "centroid":
        # Mass-weighted centroid of M -> region containing it (nearest region if
        # it lands in DBSCAN noise). Robust where ``peak`` is not: a sparse
        # edge/corner ViT sink can be the single hottest patch on a soft
        # (un-rendered) writing region, but it barely moves the centroid, which
        # tracks where the *bulk* of attention mass sits — the sign. See
        # stage1_progress.md (Validation).
        ys_g, xs_g = np.mgrid[0 : M.shape[0], 0 : M.shape[1]]
        w = M / (float(M.sum()) + 1e-12)
        cy, cx = float((ys_g * w).sum()), float((xs_g * w).sum())
        stats["centroid_yx"] = [round(cy, 1), round(cx, 1)]
        ci = min(M.shape[0] - 1, max(0, int(round(cy))))
        cj = min(M.shape[1] - 1, max(0, int(round(cx))))
        holding = [i for i, r in enumerate(regions) if bool(r[ci, cj])]
        if holding:
            stats["centroid_in_region"] = True
            return max(holding, key=lambda i: means[i] * sizes[i]), stats
        stats["centroid_in_region"] = False
        rc = [(float(np.nonzero(r)[0].mean()), float(np.nonzero(r)[1].mean()))
              for r in regions]
        best = int(min(range(len(regions)),
                       key=lambda i: (rc[i][0] - cy) ** 2 + (rc[i][1] - cx) ** 2))
        return best, stats
    if mode == "q":
        key = scores
    elif mode == "qmass":
        key = [q * s for q, s in zip(scores, sizes)]
    elif mode == "mass":
        key = [m * s for m, s in zip(means, sizes)]
    else:
        raise ValueError(f"unknown region select mode {mode!r}")
    best = int(np.argmax(key))
    return best, stats


def _bbox_fill(m: np.ndarray) -> np.ndarray:
    """Fill the axis-aligned bounding box of a boolean mask (rectangular band)."""
    m = np.asarray(m, dtype=bool)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return m
    out = np.zeros_like(m)
    out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    return out


def grow_region(
    region: np.ndarray,
    *,
    dilate: int = 0,
    bbox: bool = False,
    min_frac: float = 0.0,
    max_dilate: int = 8,
) -> np.ndarray:
    """Grow a (correctly-placed but tight) peak-seeded region to a Stage-2
    injection extent. All knobs default to no-op (returns the seed unchanged).

    The peak-seeded extractor nails *placement* but the region can be tiny
    (un-rendered text → soft map → ~0.2% of latent), which is too small for SGMI
    to splice a glyph string into. This grows the seed without moving it:

      dilate:   isotropic binary-dilation by this many patches.
      bbox:     replace the mask with its bounding-box rectangle (text occupies a
                rectangular band; re-applied after any min_frac dilation).
      min_frac: coverage floor (fraction of the patch grid); keep dilating one
                patch at a time (up to ``max_dilate`` extra iters) until met —
                rescues the tiny soft-attention cases.

    Stage-2 will additionally shape this toward the glyph-raster bbox/aspect; that
    needs the glyph and lives there.
    """
    m = np.asarray(region, dtype=bool).copy()
    if not m.any():
        return m
    if dilate > 0:
        m = binary_dilation(m, iterations=dilate)
    if bbox:
        m = _bbox_fill(m)
    extra = 0
    while min_frac > 0.0 and m.mean() < min_frac and extra < max_dilate:
        m = binary_dilation(m, iterations=1)
        if bbox:
            m = _bbox_fill(m)
        extra += 1
    return m


def to_latent_mask(region: np.ndarray, h_lat: int, w_lat: int) -> np.ndarray:
    """Eq 6 — nearest-resize a patch-grid region mask to latent resolution."""
    region = np.asarray(region, dtype=np.float64)
    hp, wp = region.shape
    R = zoom(region, (h_lat / hp, w_lat / wp), order=0)
    return (R >= 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Optional GT-box validation (only when a reference box is supplied).
# ---------------------------------------------------------------------------
def box_to_grid_mask(box_xyxy, hp: int, wp: int, img_w: int, img_h: int) -> np.ndarray:
    """Rasterize a pixel-space [x0,y0,x1,y1] box onto the (hp,wp) patch grid."""
    x0, y0, x1, y1 = box_xyxy
    gx0 = int(np.floor(x0 / img_w * wp))
    gx1 = int(np.ceil(x1 / img_w * wp))
    gy0 = int(np.floor(y0 / img_h * hp))
    gy1 = int(np.ceil(y1 / img_h * hp))
    m = np.zeros((hp, wp), dtype=bool)
    m[max(0, gy0) : min(hp, gy1), max(0, gx0) : min(wp, gx1)] = True
    return m


def hard_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Top-level orchestration.
# ---------------------------------------------------------------------------
@dataclass
class Stage1Result:
    aggregate: np.ndarray  # M after top-K aggregate (hp, wp) [0,1]   (Eq 4)
    denoised: np.ndarray  # M after neighborhood aggregation
    binary: np.ndarray  # Otsu binary B (hp, wp) bool
    otsu_thr: float
    label_map: np.ndarray  # DBSCAN labels (hp, wp), -1 = noise
    regions: list[np.ndarray]
    selected_idx: int
    region_stats: dict
    region_mask: np.ndarray  # best region (seed) at patch grid (hp, wp) bool
    grown_mask: np.ndarray  # region after grow_region (== region_mask if no grow)
    latent_mask: np.ndarray  # R at latent resolution (h_lat, w_lat) uint8   (Eq 6)
    selected_tl: list  # (step_idx, block) pairs kept by the §3.1.2 selection
    scores: np.ndarray  # selection score per candidate map (soft-IoU or concentration)
    keys: list  # the (step_idx, block) key for each candidate map
    reference: np.ndarray  # the soft-IoU reference Y (hp, wp) [0,1]
    concentrations: np.ndarray | None = None  # per-map concentration (always computed)
    params: dict = field(default_factory=dict)


def localize(
    maps_by_tl: dict,
    *,
    hp: int,
    wp: int,
    h_lat: int,
    w_lat: int,
    anchor_mode: str = "entity",
    anchor_combine: str = "normsum",
    select_mode: str = "concentration",
    ref_mode: str = "concentration",
    ref_reduce: str = "mean",
    ref_conc_q: float = 0.75,
    top_k: int = 24,
    nbhd: int = 3,
    thresh_mode: str = "otsu",
    otsu_bins: int = 64,
    thr_q: float = 0.85,
    thr_rel: float = 0.55,
    dbscan_eps: float = 1.5,
    dbscan_min_samples: int = 4,
    tau_q: float = 0.8,
    region_select: str = "mass",
    grow_dilate: int = 0,
    grow_bbox: bool = False,
    grow_min_frac: float = 0.0,
    grow_max_dilate: int = 8,
) -> Stage1Result:
    """Run the full Stage-1 pipeline.

    ``maps_by_tl`` maps ``(step_idx, block) -> {"entity": (L,), "sink": (L,)}``
    flat per-patch vectors (the probe's reduced group-means; ``sink`` is the
    padding field). We reshape to ``(hp, wp)``, build the anchor map per pair,
    select + aggregate, then run the topology pipeline to a latent mask ``R``.
    """
    keys = sorted(maps_by_tl.keys())
    maps = np.stack(
        [
            anchor_map(
                np.asarray(maps_by_tl[k]["entity"], dtype=np.float64).reshape(hp, wp),
                np.asarray(maps_by_tl[k]["sink"], dtype=np.float64).reshape(hp, wp),
                mode=anchor_mode,
                combine=anchor_combine,
            )
            for k in keys
        ]
    )

    # Always score concentration (the §3.1.2 "informative" proxy) — used for the
    # concentration reference, the concentration selection, and diagnostics.
    concs = np.array([concentration(m) for m in maps])

    # Reference Y (for soft-IoU selection + always rendered as a diagnostic).
    if ref_mode == "concentration":
        Y, _ = concentration_reference(maps, conc_q=ref_conc_q)
    elif ref_mode == "consensus":
        Y = reference_map(maps, reduce=ref_reduce)
    elif ref_mode == "sink":
        sink_stack = np.stack(
            [
                normalize01(
                    np.asarray(maps_by_tl[k]["sink"], dtype=np.float64).reshape(hp, wp)
                )
                for k in keys
            ]
        )
        Y = reference_map(sink_stack, reduce=ref_reduce)
    else:
        raise ValueError(f"unknown ref_mode {ref_mode!r} (driver handles 'gt'/'band')")

    # 3.1.2 timestep-layer selection -> aggregate M (Eq 3-4).
    if select_mode == "concentration":
        # Direct: keep the top-K most concentrated (peakiest) maps. On Anima the
        # padding-field "sink" is not a clean spatial anchor (unlike the paper's
        # special-token sinks), so entity-only + concentration ranking localizes
        # far tighter than soft-IoU vs a diffuse self-reference. See README.
        k = max(1, min(top_k, len(maps)))
        sel = np.argsort(-concs)[:k]
        scores = concs
        M = normalize01(maps[sel].mean(axis=0))
    elif select_mode == "softiou":
        sel, scores, M = select_and_aggregate(maps, Y, top_k)  # Eq 3-4
    else:
        raise ValueError(f"unknown select_mode {select_mode!r}")
    selected_tl = [keys[i] for i in sel]

    # 3.1.3 topology-aware region selection (Eq 5-6)
    denoised = neighborhood_aggregate(M, size=nbhd)
    denoised = normalize01(denoised)
    # Binarize. Otsu (default) maximizes inter-class variance but is
    # contrast-sensitive: on a soft/low-contrast map it picks a low threshold and
    # merges the writing band into its surroundings. ``quantile`` (keep the top
    # 1-thr_q patches) and ``peak_rel`` (keep patches >= thr_rel * peak) are
    # contrast-invariant — pair either with region_select="peak" for un-rendered
    # text. See stage1_progress.md (Validation).
    if thresh_mode == "otsu":
        thr = otsu_threshold(denoised, bins=otsu_bins)
    elif thresh_mode == "quantile":
        thr = float(np.quantile(denoised, thr_q))
    elif thresh_mode == "peak_rel":
        thr = thr_rel * float(denoised.max())
    else:
        raise ValueError(f"unknown thresh_mode {thresh_mode!r}")
    B = denoised >= thr
    regions, label_map = dbscan_regions(B, eps=dbscan_eps, min_samples=dbscan_min_samples)
    best, stats = select_region(denoised, regions, tau_q=tau_q, mode=region_select)
    region_mask = regions[best] if best >= 0 else np.zeros((hp, wp), dtype=bool)
    # Grow the (placed-but-tight) seed to a Stage-2 injection extent (no-op by
    # default). R is built from the grown mask.
    grown_mask = grow_region(
        region_mask, dilate=grow_dilate, bbox=grow_bbox,
        min_frac=grow_min_frac, max_dilate=grow_max_dilate,
    )
    R = to_latent_mask(grown_mask, h_lat, w_lat)

    return Stage1Result(
        aggregate=M,
        denoised=denoised,
        binary=B,
        otsu_thr=thr,
        label_map=label_map,
        regions=regions,
        selected_idx=best,
        region_stats=stats,
        region_mask=region_mask,
        grown_mask=grown_mask,
        latent_mask=R,
        selected_tl=selected_tl,
        scores=scores,
        keys=keys,
        reference=Y,
        concentrations=concs,
        params=dict(
            anchor_mode=anchor_mode,
            anchor_combine=anchor_combine,
            select_mode=select_mode,
            ref_mode=ref_mode,
            ref_reduce=ref_reduce,
            ref_conc_q=ref_conc_q,
            top_k=top_k,
            nbhd=nbhd,
            thresh_mode=thresh_mode,
            otsu_bins=otsu_bins,
            thr_q=thr_q,
            thr_rel=thr_rel,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            tau_q=tau_q,
            region_select=region_select,
            grow_dilate=grow_dilate,
            grow_bbox=grow_bbox,
            grow_min_frac=grow_min_frac,
            grow_max_dilate=grow_max_dilate,
        ),
    )
