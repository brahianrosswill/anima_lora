# FreeText Stage-1 — progress (2026-06-01)

Phase-1 of FreeText (arXiv 2601.00535) on Anima: turn the Phase-0 localization
*premise* (which was GO — see `results/20260601-1442-anima-base-cfg4/verdict.md`)
into the actual Stage-1 product, the high-confidence binary **writing mask `R`**
that Stage-2 (SGMI) injects glyph priors into.

**Status: validated across text-content variation (n=3).** Three paper steps
faithfully implemented; three Anima-specific deviations documented below. The
deviations **beat paper-literal** on renderable text (single word + multi-word)
and are *not* overfit to the one image. On text the base can't render (Korean)
the **attention still localizes correctly on the sign** (the WHERE signal
survives) — but the current **region-picker over-grabs** on the softer,
lower-contrast attention and emits a 41% whole-figure blob. The failure is the
*extractor*, not the signal. See *Validation* below.

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

## Validation — text-content variation (2026-06-01, n=3)

Canonical girl-holding-sign layout held fixed; only the quoted text varied.
Maps captured with `--dump_maps`, then Anima-default vs paper-literal A/B'd
offline (free) via `--maps_npz`. GT = eyeball the `R`-over-decoded overlay.

| quoted text | base renders? | R coverage | lift | `R` lands on |
|---|---|---|---|---|
| `ANIMA` (baseline) | yes | 10.9% | 1.56 | sign ✓ |
| `I LOVE ANIMA` (multi-word) | yes (2 lines) | 5.4% | 3.33 | sign ✓ (tighter) |
| `안녕하세요` (Korean) | **no** (squiggles, no card) | 41.2% | 2.49 | whole figure ✗ |

**Finding 1 — deviations validated, not overfit (on renderable text).** On
`I LOVE ANIMA` the Anima default (entity / concentration / mass) gives R=5.4%
lift 3.33; paper-literal (entity_sink / softiou / consensus / q) gives R=20.3%
lift 1.29 and drags in deep diffuse blocks [20,21,22,25]. The deviations *beat*
paper-literal across text content, exactly as their §3.1.2 rationale predicts.
Multi-word / sentence entity localization works. The n=1 worry is relieved.

**Finding 2 — the WHERE signal survives even on un-renderable text; the
*extractor* is the failure.** Korean doesn't render (model emits text-colored
squiggles on the chest, no sign card), yet the entity (Korean-token) attention
**still localizes on the sign/chest** — the concentration reference `Y` is warm
there (see `results/20260601-1539-korean-annyeong/selection_heatmap.png`). What
fails is region extraction on the *softer, lower-contrast* map you get without a
crisply rendered glyph: Otsu picks a low threshold → DBSCAN merges sign+body →
`mass` grabs the 41% blob. The A/B confirms no current picker rescues it:
`mass`/`qmass` → 41%, `q` (Eq-5) → 0.6% spurious face/edge speck, paper-literal
→ 73.8% near-whole-image. **This is a calibration bug in §3.1.3, not a dead
premise** — earlier "structural limit" framing retracted.

**Reframe for un-renderable scripts.** A model that *attends-to-but-can't-draw*
is the canonical FreeText case: Stage-1 supplies WHERE, Stage-2 SGMI supplies
WHAT from a glyph raster **we** render+VAE-encode (not the model). So Korean on
Anima is plausible, gated on: (a) a contrast-robust region extractor
(**centroid-seeded** — see Extractor fix below; the Korean maps *support* a tight
`R` because the attention *mass* is on the sign), and (b) SGMI actually driving
glyphs the base has never drawn (the
real open question — Stage-2, unbuilt). `region_select=mass` is confirmed the
fragile lever (picks largest-blob, not peakiest — predicted from the n=1
region-stats). A low-confidence **abstain** (coverage > ~25% latent, or lift
below threshold → emit no mask so Stage-2 no-ops to base) guards against a
41%-blob mask wrecking the image; thresholds need the broader sweep to calibrate
(don't hardcode from n=3).

### Extractor fix — contrast-invariant threshold + centroid-seeded region (BUILT 2026-06-01)

Two threshold knobs + two new region selectors in `stage1.py`:
- `thresh_mode ∈ {otsu, quantile, peak_rel}` — otsu is contrast-*sensitive*;
  `quantile` (keep top `1-thr_q`) and `peak_rel` (keep `≥ thr_rel·peak`) are
  contrast-*invariant*, so a soft map no longer melts into one giant Otsu blob.
- `region_select="peak"` — region holding the global attention argmax.
- `region_select="centroid"` — region at the **mass-weighted centroid** of the
  aggregate (nearest region if the centroid lands in DBSCAN noise).

**`peak` is a RED HERRING — do not use it.** The global argmax frequently sits on
a sparse **edge/corner ViT sink**, not the writing region. Seed centroids at
`quantile q85` (target sign ≈ row 25, col 32):

| dump | renders? | `peak` seed (yx/size) | `centroid` seed (yx/size) |
|---|---|---|---|
| `ANIMA` | yes | (1.5,12.5)/28 ✗ top-left | **(31.6,30.5)/12 ✓** |
| `I LOVE ANIMA` | yes | (29.4,31.0)/243 ✓ | (29.4,31.0)/243 ✓ |
| `안녕하세요` | no | (2.8,60.1)/32 ✗ corner | **(27.6,32.9)/110 ✓** |
| `사랑해` | no | (2.7,60.2)/28 ✗ corner | **(26.2,32.1)/84 ✓** |
| `커피` | no | (30.8,31.1)/259 ✓ | (30.8,31.1)/259 ✓ |
| `환영합니다` | no | (23.9,12.7)/81 ✗ left | **(27.2,31.6)/136 ✓** |

`peak` is off-sign on **4 of 6** (only the strong-signal cases ANIMA-multiword /
커피 happen to have the sign as the global peak). `centroid` lands on the sign for
**all 6** (rendered + Korean) — a sparse hot sink barely moves the centroid, which
tracks where the bulk of attention *mass* is, and the mass is on the sign. This
also corrects an earlier "n=4 peak PASS" note in this doc — it was a thumbnail
misread; the seeds were at the corners. **Use `centroid`.** (`mass`=largest-blob
still over-grabs the 41% blob; `q`=spurious speck — both retired as the picker.)

### Grow step — seed → injection extent (item 1, BUILT 2026-06-01)

`stage1.py::grow_region` + driver flags `--grow_dilate N`, `--grow_bbox`,
`--grow_min_frac F`, `--grow_max_dilate`. All no-op by default — the no-grow path
is bit-identical (ANIMA dump unchanged at 10.9% / lift 1.56). The centroid seed is
*placed but tight*; grow expands it **without moving it**: `dilate` (isotropic),
`bbox` (rectangularize to a text band), `min_frac` (coverage floor — keep dilating
till met). `R` and the lift metric are now computed on the grown mask; panel 5
overlays the cyan seed inside the spring grown region.

`centroid` + `q85` + grow(dilate=1, floor=0.03) on the failures + a strong case:

| dump | centroid seed | grown R | on sign? |
|---|---|---|---|
| `안녕하세요` | 0.8% | 280p (~6.8%) | ✓ chest |
| `환영합니다` | ~2% | 238p (~5.8%) | ✓ chest |
| `ANIMA` | 12p | grown | ✓ chest |

- **dilate-only + floor preserves lift**; **bbox** is more text-apt but inflates
  spread regions and dilutes lift. Recommend dilate-only + floor as the grow
  default, bbox optional. (Numbers measured earlier on the peak seed: 커피 8.3%@3.38
  dilate-only vs 13.9%@2.45 +bbox — the shape conclusion is selector-independent.)
- With `centroid`, placement holds on the sign for all tested (annyeong/hwanyeong
  now centered on the chest, vs off-sign under `peak`).
- Soft cases sit at low post-grow lift (~1.2–1.4) — genuine attention weakness
  (model barely attended). That's the **abstain** signal: grow gives a usable
  extent where confidence is high; abstain should skip where it isn't.
- Final extent/aspect still wants the glyph raster — grow is the seam Stage-2
  shapes against (glyph bbox/aspect lives there).

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
