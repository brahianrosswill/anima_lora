# FreeText Stage-1 — progress (2026-06-01)

Phase-1 of FreeText (arXiv 2601.00535) on Anima: turn the Phase-0 localization
*premise* (which was GO — see `results/20260601-1442-anima-base-cfg4/verdict.md`)
into the actual Stage-1 product, the high-confidence binary **writing mask `R`**
that Stage-2 (SGMI) injects glyph priors into.

**Status: working end-to-end on the canonical "ANIMA" prompt.** The mask lands on
the sign. Three paper steps faithfully implemented; three Anima-specific
deviations were required and are documented below. Single-image validation only
— multi-prompt/seed + a quantitative IoU pass still to do.

## What was built

| File | Role |
|---|---|
| `stage1.py` | **Pure pipeline** (numpy/scipy/sklearn, no torch). Implements §3.1 Eq 2–6. Unit-testable on synthetic maps. |
| `stage1_localize.py` | **Driver.** Reuses `probe_localization.py`'s faithful eager-recompute attention capture, runs `stage1.localize`, renders + writes `result.json`. |
| dependency | Added `scikit-learn` (DBSCAN, paper §3.1.3). `uv add scikit-learn`. |

**Offline iteration loop** (the reason Stage-1 was tunable in minutes, not a GPU
run per try): `--dump_maps` caches the captured per-(t,l) maps to `maps_raw.npz`;
`--maps_npz <path>` replays them through the pipeline with no generation. Seed-42
deterministic, so the cache is bit-identical to a live run.

```bash
# capture once (~40s GPU), then sweep params offline (CPU, instant):
python bench/freetext/stage1_localize.py --label run --dump_maps
python bench/freetext/stage1_localize.py --maps_npz results/<run>/maps_raw.npz --region_select q ...
```

## The pipeline (paper §3.1)

1. **Attention extraction** (Eq 1–2) — per-(timestep, layer) anchor map
   `M^(t,l)`, head-averaged, normalized to [0,1]. (Capture is the probe's eager
   `softmax(QK^T)` recompute; cross-attn has no RoPE so it's faithful.)
2. **Timestep-layer selection** (Eq 3–4) — score each `M^(t,l)`, keep top-K,
   aggregate → `M`.
3. **Topology-aware region selection** (Eq 5–6) — neighborhood-denoise → Otsu →
   DBSCAN → score regions → pick best → resize to latent → binary `R`.

## Three Anima deviations from the paper (the load-bearing findings)

The paper's literal recipe (Entity+Sink anchor, soft-IoU selection, Eq-5 q-score)
produced a mask over the **whole figure**, not the sign (R=27.7% of latent, lift
1.25×). Diagnosing why surfaced three Anima-specific corrections:

1. **Selection: direct concentration ranking, not soft-IoU vs a self-reference.**
   The paper's Eq-3 soft-IoU needs a reference `Y` that doesn't exist in
   zero-layout deployment (their ablation figures use GT boxes). A
   *consensus-of-all-maps* reference is dominated by the diffuse deep/early-step
   majority, so soft-IoU(M, consensus) **selects those diffuse maps** — the
   opposite of "informative" (it locked onto a deep-block L20–21 stripe). The
   concentration metric (top-5% mass fraction, the Phase-0 metric)
   operationalizes §3.1.2's "mid steps most informative" directly and correctly
   fingers the informative band: **blocks 11 & 14, mid-steps 9–20** = exactly
   Phase-0's L6–L17 mid-t. Default `--select_mode concentration`; the paper's
   `softiou` path is kept (with a concentration-bootstrapped reference) but is
   not the Anima default. `Y` is still rendered as a diagnostic.

2. **Anchor: Entity-only, not Entity+Sink.** The paper's Table-4 winner is
   Entity+Sink, where "Sink" = a few stable special tokens. **Anima's Qwen3-base
   tokenizer emits no BOS/EOS (n_special=0), so the only sink is the 447-position
   zeroed padding field** — and that field is *not* a clean spatial anchor: its
   I2T map points at the body/foreground, dragging localization to the
   lower-center via deep blocks. Entity+Sink → centroid (51,30) = body;
   Entity-only → centroid on the sign. This **refines** the Phase-0 note
   ("Entity+Sink must use padding"): on Anima the padding-sink is harmful for
   localization, so drop it. Default `--anchor_mode entity`.

3. **Region score: total mass, not Eq-5 peakiness.** Eq-5 `q_i` = fraction of a
   region above a high quantile → favors **sharp small peaks**, which on Anima
   are the face and ViT-border attention sinks, not the broad text band. The
   sign region genuinely *is* the highest-mean region (sign-box M 0.169 >
   face-box 0.123) — the premise holds; only the *picker* was wrong. Ranking
   regions by **total attention mass** (mean·size) picks the sign at every
   border margin; `q`/`qmass` pick the face. Default `--region_select mass`.

## Final default config & result

`anchor=entity · select=concentration · top_k=24 · region_select=mass ·
nbhd=3 · Otsu · DBSCAN(eps=1.5, min_samples=4)`

Canonical prompt (girl holding a sign reading "ANIMA", 1024², 28 steps, euler,
CFG=4, seed 42), `results/20260601-1520-stage1-final/`:

- Writing mask `R` centered on the sign — DBSCAN region centroid grid (27.9, 29.8)
  vs sign at ~rows 22–29 (chest level). **HIT.**
- `R` = 447/4096 patches = **10.9% of latent**; attention lift in/out 1.56×.
- Selected (t,l): blocks **[11, 14, 15, 17]**, steps 8–21 (mid) + step 0.
- Artifacts: `stage1_pipeline.png` (6-panel: decoded → aggregate → denoised →
  Otsu → DBSCAN → R), `selection_heatmap.png` (score / concentration matrices +
  reference Y), `stage1.npz`, `result.json`.

### Run trail
- `20260601-1508-anima-base-stage1` — paper-literal (consensus + Entity+Sink + qmass): coarse, whole figure (R 27.7%, lift 1.25).
- `20260601-1514-stage1-conc` — concentration reference, `--dump_maps` (the cache everything below replays). Better (lift 1.83) but still coarse.
- `20260601-1517` / `20260601-1520-stage1-final` — entity + concentration + mass: on the sign.

## Not done / caveats

- **n=1 image.** Validated on the canonical prompt only. Needs multi-prompt
  (poster/caption/multi-line), multi-seed, and CFG=1 vs 4. The three deviations
  are hypotheses backed by one image's anatomy — confirm they generalize.
- **No quantitative IoU yet.** `--gt_box x0 y0 x1 y1` is wired (reports IoU vs a
  pixel box) but unused — eyeballed HIT only. Next: auto-detect the white sign
  card or hand-label a few, report IoU like the paper's Table 4/5.
- **Mask over-covers slightly** (bleeds onto torso). Acceptable for SGMI (over >
  miss) but could tighten — a `border_margin` option was prototyped (kills the
  ViT-edge sinks) but margin=0 + mass already HITs, so it wasn't wired in.
- **Stage-2 SGMI** (glyph raster → VAE encode → noise-align → Log-Gabor band-pass
  → masked-replace into `R` over t∈[0.8T,0.6T]) is the next milestone.
- **Unit test** for the pure pipeline (synthetic blob → R recovers it; the smoke
  used during dev) not yet committed to `tests/`.

## Pointers
- Memory: `project_freetext_phase0_localization_go` (Phase-0 GO + arch fit).
- Premise verdict: `results/20260601-1442-anima-base-cfg4/verdict.md`.
- Paper §3.1 (Stage-1), Fig 2/3, Table 4 (token-set ablation), Table 5 (IoU).
