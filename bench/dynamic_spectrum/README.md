# Dynamic Spectrum — error-controlled, trajectory-adaptive step scheduling

**Status:** phase-1 probe (decide-before-build). Not a shipped method.
**Owner question:** *Should a sampler's forecast/skip schedule be **fixed** (what Spectrum does today) or **adaptive to the trajectory**? And if adaptive — does a single forward carry enough signal to drive it?*

---

## Motivation

### Where this came from

A theory paper — *High-accuracy and dimension-free sampling with diffusions* (Gatmiry, Chen, Salim; arXiv:2601.10708) — proves that the probability-flow ODE drift, **as a function of time along the trajectory, is low-degree-polynomial** (bounded high-order time derivatives, Lemma 3.1–3.3). That smoothness is what lets their collocation/Picard solver take big steps and reach `polylog(1/ε)` accuracy.

We don't want their solver (it targets `ε→1e-6` on low-radius targets like Gaussian mixtures — irrelevant to image generation). But the *structural fact* is exactly the premise **Spectrum** already bets on: if the drift is smooth in time, you can forecast it and skip block compute on some steps.

The chain that led here:

1. The paper says the drift is smooth-in-time ⇒ forecastable. (This is the theory license for Spectrum.)
2. "Distill the steps into a small net" ⇒ the GENIE idea (Dockhorn et al. 2022): learn the score's **time-derivative** to take fatter, higher-order steps. The paper tells you *how many coefficients* you need (low degree) and *why the head is learnable* (the derivative is a posterior-moment functional of the current state).
3. The most attractive form isn't a fat analytic step (needs the derivative's **direction** — fragile, see landmine below). It's an **error-controlled step**: estimate the *magnitude* of local curvature and step coarsely where it's flat, finely where it's curving. A step-size controller consumes a **scalar**, which is the robust part.
4. That is precisely **"dynamic Spectrum"**: Spectrum skips on a *fixed cadence*; an error estimate makes the skip *adaptive to local flatness*.

### Why fixed schedules are structurally suspect on Anima

Two repo-grounded reasons a hand-set cadence can't be optimal everywhere:

- **The trajectory isn't uniform in time.** `x0_pred` is visually resolved by σ≈0.45, but `e_low` *triples* in the σ<0.45 tail (memory: `sigma_signal_resolves_by_045`). A fixed cadence wastes forwards in the flat middle or under-resolves the tail.
- **Optimal cadence is (CFG × aspect)-dependent** — the same lesson DCW beat into us (the right correction flips sign with CFG and aspect ratio). A schedule tuned for square/CFG=1 is wrong for 16:9/CFG=4. An error-controlled step **auto-tunes per trajectory** instead of shipping a lookup table.

---

## The exact target (why this is cheap)

On this flow-matching schedule the deterministic Euler step is (`library/inference/sampling.py::step`):

```
x_{i+1} = x_i - (σ_i - σ_{i+1}) · v_i
```

Collapse two steps `i, i+1` into one big Euler step from `x_i` (i.e. *fatten* the step, reusing only `v_i`). The discrepancy against the reference two-step result is **exact**:

```
e_i = ‖ x̂_{i+2} − x_{i+2} ‖ = (σ_{i+1} − σ_{i+2}) · ‖ v_{i+1} − v_i ‖
```

So the local fattening error is *exactly the σ-weighted velocity change*. Consequences:

- The whole target is reconstructable from **one reference Euler run** — zero extra forwards.
- The one-shot head's job is precisely: predict `‖v_{i+1} − v_i‖` from **step-i features alone** (before `v_{i+1}` exists).
- The naive **history** baseline `‖v_i − v_{i-1}‖` is last-step curvature predicting this-step curvature — the "you needed a second eval" reference the head must approach.

---

## What the probe measures

Three questions, in order — the answer to each gates the next:

| | Question | Metric | If it fails |
|---|---|---|---|
| **Q1** | Is `e_i` non-uniform along the trajectory? | `p90/p10` of `e_i`, mass below σ=0.45 | flat ⇒ adaptivity pointless, stop |
| **Q2** | Does **σ alone** rank `e_i`? | Spearman(σ, `e_i`) | high ⇒ just reshape the fixed schedule by σ, **no head needed** (cheaper win) |
| **Q3** | Can a **one-shot head** on step-i features rank `e_i`, beating σ and approaching history? | Spearman(head, `e_i`) on held-out trajectories | head ≈ σ ⇒ no learned trigger; head ≫ σ ⇒ dynamic-Spectrum head has legs |

**Success is Spearman, not MSE.** This is deliberate. The mod-guidance text-derivative head (memory: `mod_guidance_text_derivative_orthogonal`) hit low MSE but `cos≈0` — it learned magnitude, not direction. A step-size controller eats a scalar error estimate, so **rank correlation is the honest metric** and the magnitude-not-direction signal is exactly the part that survives.

### Method

- CFG=1, euler, single forward per step (so the target math is exact and one forward/step).
- Tap the **same `final_layer` input Spectrum uses** (module forward hook → `v_i`; `final_layer` pre-hook → patch-pooled feature `feat_i`).
- Tiny MLP `feat_i → log e_i`, **held out by trajectory** (no leakage across steps of the same run), standardized on train.
- Baselines: σ, −σ, `‖v_i‖`, `‖feat_i‖`, history `‖v_i − v_{i-1}‖`.
- Report pooled **and** within-trajectory-averaged Spearman (the latter controls for cross-trajectory scale).

CFG and `er_sde` composability are deliberately **out of scope** for this first read — they change the velocity field and the target's exactness; revisit only if phase 1 passes.

---

## Running it

```bash
# quick correctness smoke (~1 min)
uv run python -m bench.dynamic_spectrum.run_bench \
    --num_trajectories 2 --ref_steps 24 --image_size 768 768 --label smoke

# real probe — real dataset captions (default) + block-compile (~10 min, backgroundable)
uv run python -m bench.dynamic_spectrum.run_bench \
    --num_trajectories 24 --ref_steps 100 --image_size 1024 1024 --compile --label probe-realcap
```

Key flags: `--num_trajectories`, `--ref_steps` (reference solve depth), `--image_size H W` (canonical tier is 1024), `--holdout_frac` (trajectory split for the head), `--head_epochs`. Model paths default to `default_checkpoints()`.

**Prompts** default to **real captions sampled from the dataset** (`--dataset_dir image_dataset`, the tag-style caption master — `os.walk(followlinks=True)` since it's a symlink to nested artist dirs). `--prompts_file` (one/line) overrides; `--dataset_dir ''` falls back to the built-in generic list. The conditioning distribution matters — the trajectory geometry depends on the prompt — so the dataset captions are the representative default.

**`--compile`** block-compiles the DiT (`--compile_blocks`, the blessed path) for ~20% faster forwards on a 5070 Ti at 1024². Bit-exact: it compiles each block's `_forward`; `final_layer` and the top forward stay eager, so the capture hooks are unaffected and the numbers don't move. The inductor warmup is absorbed by the one-time model load (same resolution as the trajectories → graph reused, no per-trajectory recompile).

### Output (`results/<ts>-<label>/`)

- `result.json` — the standard envelope; `metrics.spearman_pooled_test`, `metrics.spearman_within_traj_test`, `metrics.error_profile`, and a one-line `metrics.verdict`.
- `per_step.csv` — per (trajectory, step): σ, `‖v_i‖`, `‖feat_i‖`, history dv, target `e_i`, test flag, head prediction.
- `profile.png` — left: `log e_i` vs σ with binned-median overlay and the σ=0.45 line (Q1/Q2); right: `|Spearman|` per predictor (Q3).

### How to read it (decision tree)

- `p90/p10 < 2` → **FLAT**: stop, fixed schedule is already near-optimal.
- `Spearman(head) − Spearman(σ) > 0.10` → **HEAD WINS**: proceed to phase 2.
- `Spearman(σ) ≥ 0.6` and head ≈ σ → **SIGMA SUFFICES**: reshape the fixed schedule by σ; no head. (A genuine, shippable result — and it would dovetail with DCW's σ/CFG/aspect indexing.)
- else → **INCONCLUSIVE**: weak signal.

---

## Results

Two runs, 24 trajectories × 100 steps @ 1024², 2352 rows, 7 held-out trajectories each. The **prompt distribution turned out to decide the conclusion** — which is exactly why the default was switched to real dataset captions:

- `20260607-2002-probe-1024` — **generic English prompts** (hand-written).
- `20260607-21xx-probe-realcap` — **real dataset captions** (tag-style, `image_dataset/`). **This is the authoritative run** (it matches real conditioning).

**Q1 holds in both — error is strongly non-uniform.** `p90/p10 ≈ 6.8–7.5×`; **~61–66% of total local-error mass sits in σ<0.45** (only 19% of the steps). Error is highest in the low-σ resolve tail. A fixed *uniform* schedule is clearly mistuned in both — adaptivity (even just a static σ-reshape) has headroom. Reproduces `sigma_signal_resolves_by_045`.

**Q2/Q3 — the decision flips with the prompt distribution.** Within-trajectory Spearman (the metric an online per-trajectory controller actually sees; pooled mixes cross-prompt scale):

| within-traj \|ρ\| vs true `e_i` | generic prompts | **real captions (authoritative)** |
|---|---|---|
| **σ** | 0.71 | **0.876** |
| learned head (final-layer feats) | **0.86** | 0.768 |
| free `‖v_i‖` | 0.77 | **0.089** ← collapsed |
| `‖feat_i‖` | 0.53 | 0.226 |

On **real captions the story inverts**:
- **σ becomes a near-sufficient statistic** for local fattening error (0.876) — the real trajectories are cleaner / more monotone in σ than my generic ones.
- **The free `‖v_i‖` heuristic collapses to noise (0.77 → 0.09).** Its apparent value on generic prompts was an artifact; do **not** gate on velocity norm.
- **The learned head drops *below* σ** (0.768 < 0.876) — pooled final-layer features can't out-rank σ itself.

### Verdict (on real data): SIGMA SUFFICES — no head, no `‖v‖` gate

1. **Error is non-uniform (Q1 yes), so a fixed *uniform* cadence is suboptimal** — but
2. **σ alone ranks it at 0.876 within-trajectory, and nothing beats σ** — not the free velocity norm, not a trained head.
3. Therefore the win is **a static σ-reshaped schedule** (denser steps where σ<0.45), *not* anything dynamic or learned. That's the cheapest possible outcome and it dovetails with DCW's σ/CFG/aspect indexing.

**Methodological takeaway:** the generic-prompt run would have sent us to build a `‖v‖`-gate / trained head — both of which evaporate on real conditioning. Always run this kind of probe on the real caption distribution.

---

## Phase 2 (revised by the real-data result) — σ-reshaped schedule, not a gate

The dynamic/learned trigger is **not** worth building: σ already captures the rankable signal. The actionable follow-up is much cheaper:

1. **Static σ-reshaped step schedule** — bias `get_timesteps_sigmas` to place more steps in σ<0.45 (where 60%+ of the error mass is). Pure schedule change, no model, no per-step decision. Measure NFE-at-fixed-CMMD vs the current linear-in-σ (shifted) schedule.
2. **If** a sweep across **CFG × aspect** shows the *optimal* σ-reshape moves with CFG/aspect (it might — cf. DCW), ship a small CFG/aspect-indexed schedule table, still static.

What's explicitly **dropped**: `‖v‖`-gated Spectrum (heuristic collapsed on real data) and the trained one-shot head (below σ on real data). Spectrum's own *fixed* forecast cadence could likewise just be σ-reshaped, but that's a Spectrum change, not a new method.

## Landmine (read before building a head)

The mod-guidance head trained a small MLP with **pointwise MSE on a derivative** and got `cos≈0` — magnitude right, direction wrong, and σ-FiLM couldn't rescue it (memory: `mod_guidance_sigma_film`). This probe sidesteps it by predicting a **scalar** (error magnitude → step size), validated by **Spearman**, not by MSE. Any future fat-*analytic*-step head re-enters the directional regime and must use a directional/geometric loss + cosine probe, or it will relive that dead-end.

## References

- Gatmiry, Chen, Salim. *High-accuracy and dimension-free sampling with diffusions.* arXiv:2601.10708 (2026). — the smoothness-in-time result.
- Dockhorn, Vahdat, Kreis. *GENIE: Higher-Order Denoising Diffusion Solvers.* NeurIPS 2022. — distilling the score time-derivative for bigger steps.
- `networks/spectrum.py`, `docs/inference/spectrum.md` — the fixed-cadence forecaster this would make adaptive.
