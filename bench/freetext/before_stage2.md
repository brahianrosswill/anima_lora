# FreeText — Stage-1 → Stage-2 readiness gate

Go/no-go before building **Stage-2 (SGMI glyph injection)**. SGMI masked-replaces
band-pass glyph spectra into the writing mask `R` at the sampler boundary — so a
wrong/loose `R` paints glyph energy onto skin/background and the output ends up
**worse than base**. Stage-2 amplifies Stage-1 error; do not build it on an
untrustworthy mask. This doc defines "trustworthy" and records whether we're there.

Prereq reading: `stage1_progress.md` (Validation + Extractor-fix + Grow sections).
Recall the load-bearing correction: **`region_select=peak` is a red herring** (its
argmax sits on edge/corner ViT sinks — off-sign on 4/6 dumps); **use `centroid`**
(mass-weighted centroid → region), which lands on the sign for rendered + Korean.

## The gate

**Gate set** — 4 held-out Korean prompts (un-renderable text = the hard case;
canonical held-sign layout, seed 42, 1024², CFG 4):

> `안녕하세요` · `커피` · `환영합니다` · `쏘리현 아니마 로라 프로젝트`

**A single fixed config** (no per-prompt tuning) must, for **all 4**, produce an
`R` that satisfies:

1. **PLACEMENT** *(load-bearing)* — grown-region centroid ∈ the sign region.
   Held-sign layout fixes the sign to center-chest; nominal sign box on the 64×64
   patch grid is **rows [18, 34] × cols [18, 46]**. (For a rigorous close-out,
   hand-label a per-prompt sign box and use `--gt_box` → IoU; the eyeball GT
   already misfired once on `peak`, so do not trust thumbnails alone.)
2. **EXTENT** — post-grow coverage ∈ **[~2%, ~20%]** of latent. Rules out both the
   41% whole-figure blob (old `mass` failure) and a <1% single-patch speck.
3. **CONFIDENCE** *(soft, not a hard fail)* — aggregate lift ≥ ~1.5. Below the
   threshold → Stage-2 should **abstain** (no injection, no-op to base), not error.

**Pass** = (1) ∧ (2) for all 4 under one config. (3) gates per-prompt injection.

## Current status — **PASS (placement 4/4, extent 4/4)**

Config: `--thresh_mode quantile --thr_q 0.85 --region_select centroid
--grow_dilate 1 --grow_min_frac 0.03`

| prompt | grown centroid (yx) | ∈ box? | coverage | lift | run |
|---|---|---|---|---|---|
| `안녕하세요` | (27.9, 32.6) | ✓ | 4.4% | 1.69 | `20260601-1607-cen-annyeong` |
| `커피` | (31.5, 31.2) | ✓ | 8.3% | 3.38 | `20260601-1615-cen-keopi` |
| `환영합니다` | (27.4, 31.6) | ✓ | 5.3% | 1.71 | `20260601-1607-cen-hwanyeong` |
| `쏘리현 아니마 로라 프로젝트` | (30.0, 30.7) | ✓ | 14.3% | 2.02 | `20260601-1610-korean-sorryhyun` |

All 4 centroids land at rows 27–32 / cols 30–33 (center-chest), well inside the
nominal sign box. Coverage in band. The long phrase pushed the base into a
multi-line text *attempt* → the largest, most concentrated region (best extent).

**Caveats before calling it fully closed:**
- **IoU not yet computed.** Placement is centroid-in-nominal-box, not IoU vs a
  hand-labeled box. Close this (it replaces the eyeball that misfired on `peak`).
- **Extent bleed on the long phrase.** `쏘리현…` grows down to row 44 (lower torso),
  cols 14–49 — over-coverage (acceptable for SGMI: over > miss, but tightenable).
- **Soft cases are low-confidence.** `안녕하세요`/`환영합니다` sit at lift ~1.7 — these
  are the abstain candidates; the threshold (3) is not yet calibrated.

## Two acceptance paths

**(A) Validate-as-is (current default).** The recommended config already passes
the placement + extent gate 4/4. To *formally* close before Stage-2:
  1. Hand-label a sign box per gate prompt → run with `--gt_box` → report IoU.
     Promote the gate from "centroid ∈ nominal box" to "IoU ≥ τ_iou" (τ ~0.3–0.5).
  2. Calibrate the abstain threshold (3) — pick a lift floor that keeps the 4 but
     would skip a genuine miss.
  3. (optional) Tighten the long-phrase extent bleed via grow knobs.

**(B) Hyperparameter search** — if (A)'s IoU is marginal, or to optimize extent.
Grid below; **objective = maximize mean IoU(`R`, sign_box) over the 4 prompts,
subject to coverage ∈ [2%, 20%] for every prompt** (a config that nails 3 and
blows up the 4th fails). Offline + free: every cell is a `--maps_npz` replay on
the cached dumps (no GPU). Report the optimum + the per-prompt table.

### Search space
| knob | values | notes |
|---|---|---|
| `thresh_mode` | `quantile` *(rec)*, `peak_rel` | `otsu` is contrast-sensitive — excluded |
| `thr_q` (quantile) | 0.80, 0.85, 0.90, 0.92 | higher = tighter seed |
| `thr_rel` (peak_rel) | 0.50, 0.55, 0.65 | — |
| `region_select` | `centroid` *(rec)* | `peak` retired (sink-prone); `mass`→blob; `q`→speck |
| `grow_dilate` | 0, 1, 2 | isotropic margin |
| `grow_bbox` | off *(rec)*, on | bbox rectangularizes but inflates + dilutes lift |
| `grow_min_frac` | 0, 0.02, 0.03, 0.05 | coverage floor for the soft cases |

Fixed at validated Stage-1 defaults (do **not** search): `anchor=entity`,
`select=concentration`, `top_k=24`, `nbhd=3`, `dbscan_eps=1.5, min_samples=4`.

## What Stage-2 consumes from Stage-1 (the contract)

- `latent_mask` `R` (uint8, `h_lat × w_lat`) — the binary writing region.
- the **grown extent** — where glyph spectra get spliced (seed = placement,
  grown = injection footprint; Stage-2 shapes it further to the glyph-raster
  bbox/aspect — grow is that seam).
- a **confidence scalar** (aggregate lift) → the abstain gate, so Stage-2 no-ops
  when Stage-1 is unsure (the soft Korean cases).

## Risks carried into Stage-2 (out of this gate's scope)

- **Layout coverage.** All 4 gate prompts share ONE composition (held sign,
  center-chest). A poster / t-shirt / multi-line-on-a-wall / two-signs layout is
  **not** gated here — centroid assumes a single writing region; bimodal layouts
  will pull the centroid into the gap. Add such prompts before claiming generality.
- **The real unknown is Stage-2 itself:** can SGMI drive glyphs the base has
  **never drawn** (Korean)? Stage-1 solving WHERE does not guarantee WHAT takes —
  the denoiser may fight unfamiliar glyph spectra. That is the actual experiment
  Stage-2 runs; this gate only guarantees the mask is placed where the glyphs go.
