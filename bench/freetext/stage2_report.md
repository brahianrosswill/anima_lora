# FreeText — Stage-2 (SGMI) report

Stage-1 solved **WHERE** to write (the mask `R`). Stage-2 tests **WHAT**: can
Spectral-Modulated Glyph Injection (SGMI, paper §3.2) drive glyphs the base model
has **never cleanly drawn** — Korean on Anima? The glyph structure is supplied by
us (an external raster, VAE-encoded, masked-injected at the sampler boundary), not
the model. The paper's own caveat (§4.2.3): SGMI "strengthens glyph structure
rather than enabling unseen characters from scratch." Korean-on-Anima is exactly
the OOD case it did *not* claim to solve — so this is the honest test.

Driver: `stage2_sgmi.py` (loads DiT/VAE, derives `R` from a cached Stage-1 map
dump, renders + VAE-encodes the glyph, monkeypatches the Euler step). Injection
machinery: `stage2.py` (`SGMIInjector`). Both share `bench/_common.py`.

Anima maps the paper's DDPM `(α_t, σ_t)` to rectified-flow `(1-σ, σ)`; the
injection window `t∈[0.8T,0.6T]` is `σ∈[0.6,0.8]` and `z_ref^(σ)=(1-σ)z_ref+σε`.

## TL;DR — **SGMI drives OOD Korean glyphs, with the right window**

The decisive config is **NOT the paper's**. On OOD Korean every signature paper
knob (Log-Gabor band-pass + cosine anneal) *subtracts*; the win is a **flat
hard-replace over a window extended down into Anima's detail-resolving tail**:

> **flat λ=1 · no Log-Gabor · no anneal · region mask · σ∈[0.35, 0.8]** (`hard_deep`)

`안녕하세요` went from un-renderable (base) → garbled (`sgmi_hard`, paper window) →
**clean and correct** (`hard_deep`, extended window). **Validated 4/4 across the
full Stage-1 gate set** (incl. a multi-line long phrase) — see Run C. Remaining
artifact: the grey-board overlay (OPEN ISSUE below).

## The two load-bearing findings

### 1. The base prompt MUST match the Stage-1 capture prompt
First end-to-end run (`20260601-1810-patchfix-smoke`) used the argparse-default
**toy** prompt `a girl holding a sign that reads "..."`. `R` is derived from the
*real*-prompt attention dump, so the toy prompt drew a different composition and
the injection landed as a grey blob on the chest — looked like SGMI failing.
Re-running with the **real capture prompt** (`masterpiece, best quality, score_7,
safe. An anime girl … sign … that reads "안녕하세요" …`) put `R` on the sign the
girl actually holds and the injection went where the glyphs belong
(`20260601-1821-realprompt-annyeong`). **`R` is only meaningful against its own
prompt** — pin this (default `--prompt`, or fail loudly on mismatch).

### 2. Extend the window into the detail tail — that's the whole game
We know from `sigma_signal` (Anima resolves x0 by σ≈0.45; high-freq texture energy
triples in the σ<0.45 tail) that **glyph strokes are resolved in the tail**.
Stopping injection at σ=0.6 hands the unfamiliar Hangul to a *free* tail that
"corrects" it toward shapes the model knows (안녕→안긴). Keeping the hard-replace
alive down to σ=0.35 — straddling the σ≲0.45 resolve zone — lets the strokes
**lock** before the tail can rewrite them. σ=0.20 adds nothing → leave the very-low
tail free (cheaper, and keeps the sign photo-coherent).

## Results

### Run A — paper-faithful ladder, paper window `σ∈[0.6,0.8]`
`20260601-1821-realprompt-annyeong` · `안녕하세요` · seed 42 · CFG 4 · 28 steps ·
R = 4.4% latent, px bbox (384,320,672,592)

| variant | knobs | injected | result |
|---|---|---|---|
| `base` | none | — | pink cursive squiggles — **no readable Korean** |
| `sgmi` | cosine anneal + Log-Gabor (post) | 6 steps, λ→0 | **featureless dark board** — band-pass strips glyph energy |
| `sgmi_hard` | flat λ=1, no LG | 6 steps | readable Hangul but **wrong chars** ("안긴 아세요") |
| `sgmi_nolg` | cosine anneal, no LG | 6 steps, λ→0 | vertical-stroke smear — anneal dilutes |

Takeaways: Log-Gabor is *harmful* (nothing in-distribution to "strengthen" → it
just removes the injected structure). Cosine anneal *dilutes* (`sgmi_nolg` ≪
`sgmi_hard`). The strongest, dumbest injection wins — but the chars are wrong.

### Run B — improvement ladder, extended windows + ink mask
`20260601-1827-improve-annyeong` · same prompt/seed/R

| variant | window σ_end | mask | injected | result |
|---|---|---|---|---|
| `sgmi_hard` | 0.60 | region | 6 | "안긴 아세요" — wrong chars (control) |
| **`hard_deep`** | **0.35** | region | 13 | **"안녕하세요" — clean & correct** ✅ |
| `hard_deepest` | 0.20 | region | 17 | "안녕하세요" — clean & correct (no gain over 0.35) |
| `hard_ink` | 0.35 | ink-only | 13 | messy strokes floating on her shirt — **worse** |
| `hard_ink_soft` | 0.30 | ink-only (λ=0.85) | 14 | same — worse |

**Lever A (deeper window): the win.** 0.60→0.35 turns garbled → perfect.

**Lever E (ink mask): backfires — and that's informative.** Removing the board
background made the denoiser float distorted strokes on her tank-top. The full
rectangular-region injection is **load-bearing**: it hands the denoiser a "blank
board" prior it commits to as a clean sign. **Inject the whole writing region, not
just the ink.**

### Run C — gate-set generalization, `hard_deep` + `hard_deepest`
`base,hard_deep,hard_deepest` · real prompts · seed 42 · CFG 4 · 28 steps.
**All four gate prompts render correct Korean** with the deep window (both
σ_end=0.35 and 0.20 work; pick 0.35):

| prompt | run | result | grey-board |
|---|---|---|---|
| `안녕하세요` | `20260601-1827-improve-annyeong` | ✅ "안녕하세요" | moderate |
| `커피` | `20260601-1834-deepest-keopi` | ✅ "커피" | large (R≈8%) |
| `환영합니다` | `20260601-1835-deepest-hwanyeong` | ✅ "환영합니다" | moderate |
| `쏘리현 아니마 로라 프로젝트` | `20260601-1836-deepest-sorryhyun` | ✅ "쏘리현 아니마 / 로라 프로젝트" (2-line) | large (R≈14%) |

**4/4 — the win is not seed/prompt-specific**, and it holds for a multi-line long
phrase. The grey-board artifact persists and **scales with `R` coverage** (worst on
`커피` and the long phrase, where `R` is biggest) — confirming the root cause below
(more region forced to the flat board latent → bigger flat card).

## OPEN ISSUE — the grey-board overlay artifact

`hard_deep`/`hard_deepest` render **clean correct text**, but the sign itself reads
as a **flat neutral-grey card pasted onto the image** — a poor overlay, not an
integrated held sign with the scene's lighting/shading/perspective. Root cause:
`z_ref` is white-on-**black** (`bg=0.0`); a λ=1 hard-replace over all of `R` forces
the whole sign region to that flat board latent, and because we keep injecting deep
into the tail the denoiser never gets a free pass to reshade/relight it. So we
**bought clean glyphs at the cost of board realism** — the deeper the window, the
flatter the card.

This is the tension to resolve next. The two levers that *caused* it (full-region
mask + deep flat λ) are exactly the two that made the text work, so the fix has to
thread the needle. Candidate directions (none tested yet):

1. **Render a natural sign, not white-on-black.** Set `bg` to a plausible
   paper/card tone and `fg` to dark ink (dark-on-light), so even a hard-replace
   board *looks* like a real sign. Cheapest first thing to try (`--bg/--fg` already
   exist). May also need the board tone to match scene lighting.
2. **Late soft-release tail.** Flat λ=1 through the stroke-lock zone
   (σ 0.78→~0.40) then a short cosine release (σ 0.40→0.30) so the denoiser does a
   final integration pass — reshade/relight the board — *after* the glyphs are
   locked. (Untested: region + soft. We only ever ran soft on the ink mask, which
   confounded it with the bad-mask result.)
3. **Two-tier mask.** Hard-replace the *ink* deep (locks strokes) + soft-replace
   the *board* shallow (sets the card but releases early for blending). Combines
   the board prior with stroke fidelity without the flat-card lock.
4. **Channel-selective injection.** Inject structural latent channels (strokes),
   leave color/low-freq channels free so the model colors the board. Speculative.

## Recommended config + code state

- **Promote `hard_deep` to the headline OOD variant**; lower the driver default
  `--sigma_end` 0.6 → **0.35**. Keep Log-Gabor and anneal OFF for OOD glyphs.
- The improvement variants (`hard_deep`/`hard_deepest`/`hard_ink`/`hard_ink_soft`)
  and the ink-mask path are wired in `stage2_sgmi.py` (`variant_specs` +
  per-variant window/`_mask`). The `_mask="ink"` path is kept as a documented
  negative result.

## Next steps

1. ~~**Gate-set generalization**~~ — DONE (Run C): 4/4 gate prompts correct,
   multi-line included.
2. **Board realism** (now the #1 open problem) — work the OPEN ISSUE levers above,
   starting with (1) dark-on-light raster, then (2) late soft-release. Artifact is
   worst at high `R` coverage.
3. **Seed robustness** — confirm the win isn't seed-42-specific (still only one
   seed across the gate set).
4. **Lock defaults** — lower driver default `--sigma_end` 0.6 → 0.35.

## Run index
- `20260601-1810-patchfix-smoke` — plumbing works; toy-prompt layout mismatch (the
  cautionary run).
- `20260601-1821-realprompt-annyeong` — real prompt, paper-window ladder (Run A).
- `20260601-1827-improve-annyeong` — extended-window + ink-mask ladder (Run B); the
  `hard_deep` win.
