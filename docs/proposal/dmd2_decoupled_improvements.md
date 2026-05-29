# Decoupled DMD2 — improvement proposal (diagnostics + validated levers)

> Improvement proposal for the shipped Decoupled DMD2 (Turbo) distillation. For the
> method itself see `docs/experimental/dmd2-decoupled.md` (usage / ops) and
> `docs/structure/dmd2-decoupled.md` (math / walkthrough). This doc is the live
> decision log: which levers survived contact with Anima's constraints, and the
> metric-driven rules for picking the next one.

Constraints that survived contact with Anima: the output must bake to a **plain
standard LoRA** run at `--infer_steps 4 --cfg 1.0`; we train at batch=1 on a single
16 GB GPU at native-token buckets (so no full-BPTT); the student-loss **sign fix landed
2026-05-27** (every pre-fix log reads like "base 4-step blur," not a blow-up).

**Current best checkpoint:** `anima_turbo_G_noprep` family, specifically **`g_agg_250`**
(`alpha_warmup_steps=250`, `iterations=250`, grabbed at ramp-completion) — see §1.
Everything below is organized around *why* that checkpoint wins and how to beat it.

---

## 0. The mechanism (read this first — it reorganizes everything)

Reading the actual loss assembly (`scripts/distill_turbo/distill.py:388–438`) is what
unlocked the rest of this doc. The student gradient has **two branches with opposite
character**:

```
warmup_frac = min(1, (step+1)/alpha_warmup_steps)
alpha_eff   = teacher_cfg·warmup_frac + 1·(1−warmup_frac)        # ramps 1 → teacher_cfg

grad_signal = grad_dm                                            # DM branch
            + tau_ca·(alpha_eff − 1)·delta_cfg                   # CA branch (do_ca steps)
```

- **DM branch** (`grad_dm`): matches the student to the *real conditional* distribution,
  measured against the fake/critic. It has a **real target → it converges.** `dm_rel_gap`
  is this branch's gap.
- **CA branch** (`(alpha_eff−1)·delta_cfg`, `delta_cfg = v_real_cond − v_real_uncond`):
  bakes the classifier-free-guidance direction into the student. `delta_cfg` is a
  **constant teacher-supplied bias with NO fixed point.** Every step with α>1 adds another
  dose of CFG; it never settles, it just keeps pushing the student toward higher effective
  CFG — i.e. off the real-data manifold, toward oversaturation.

**Over-distillation is therefore structural, not a tuning bug.** Training is a *race*:
the DM branch converging (good, bounded) vs the CA branch over-accumulating CFG (bad,
unbounded). The total CFG baked ≈ **`student_lr × Σ(alpha_eff − 1)`** = the area under
the α-ramp. There is an optimal dose: too little → under-baked (washed-out, weak
contrast); too much → oversaturated / letters destroyed / off-manifold.

Consequences that drive every lever below:

1. **Best checkpoint = where DM has converged but CFG isn't over-baked yet.** That is at
   (or just after) α-ramp completion, *before* any hold accumulates more dose.
2. **`student_lr` is a red herring for quality — it only sets the timescale.** Dose =
   `lr × Σ(α−1)`. Lower LR just moves the optimum to a later step; higher LR reaches
   over-bake sooner. You cannot fix a dose problem by tuning LR; you change the dose or
   change the *ratio* of the two branches' rates.
3. **A persistent CA bias means "train longer to fully adapt" is a category error** — the
   CA branch has no fixed point to adapt *to*. More steps at α>1 = more over-bake.
4. **`dm_rel_gap` rising is the over-bake *symptom*** — the CA term drags the student off
   the real-data manifold, so the DM branch's gap to real data *grows*. Minimize it.

To genuinely beat `g_agg_250` you must improve the **ratio of DM-convergence-rate to
CFG-accumulation-rate** — make the good branch finish before the bad branch over-bakes.

### 0b. The paper says exactly this — and reframes the fix (Liu et al., arXiv:2511.22677)

The source paper ("Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as
the Shield") independently derives the §0 mechanism and names our failure mode. Its
central Eq. 6 **is** our code: `∇L = −[ (s_cond^real − s_cond^fake)[DM] + (α−1)(s_cond^real
− s_uncond^real)[CA] ]·∂G/∂θ`. "CA is the engine, DM is the shield." Key passages:

- **§3.1.1 (CA-alone ablation):** *"training with CA alone is unsustainable… the generated
  images progressively suffer from over-saturation and high-frequency noise, eventually
  leading to training collapse."* — verbatim our 500/750 over-bake (dark, letters destroyed,
  worse over steps; their Fig. 2).
- **§3.2:** *"training with the CA engine leads to a monotonic increase in the variance of
  generated images."* — our `x_pred_std` inflation; confirms **low = in-domain, inflation =
  oversaturation** (not diversity).
- **The fix they prescribe:** *"the Distribution Matching term eliminates these issues,
  enabling stable training over extended periods and yielding higher-quality final results."*

**This reframes our whole finding.** Our "best checkpoint is early, then it over-bakes" is a
symptom of an **under-powered DM shield** — the CA-alone failure leaking through — not the
intended operating point. In a properly-shielded run (their Fig. 2 CA+DM column) training is
*stable for thousands of steps and later checkpoints are better*. We already know *why* our
shield is weak: **fake under-tracking** (`dm_rel_gap`↑ / `dm_cos`↓), our long-standing metric.

Consequences (these reorder §3):
- The **paper-faithful fix is to strengthen DM, keeping α at the target CFG** — not to grab
  early or lower `teacher_cfg`. Those weaken the *engine* (the few-step conversion power).
- §3.2: oversaturation is a **low-frequency** artifact that DM corrects **only if τ_DM spans
  full [0,1]** — which we have (`tau_dm_strategy="uniform"`), so our DM *can* fix it once
  strong enough.
- Our schedule (`τ_CA>t`, `τ_DM∈[0,1]`) **is** the paper's best "Decoupled-Hybrid" (Table 1
  row 4) — leave it alone.
- The **α-warmup ramp is NOT in the paper** (they use fixed α). It is almost certainly *our
  workaround* for the weak shield (ramp slowly so CA doesn't overwhelm the lagging fake
  early). With a strong DM shield the "grab-at-ramp-completion" sweet spot may dissolve and
  late checkpoints become best — as in their Fig. 2.

---

## 1. The dose findings (2026-05-29 — supersedes the old "fake under-tracking" framing)

Per-step checkpoint saving is now wired into `distill.py` (step-tagged intermediates +
canonical bare name at the final step), so the whole trajectory survives for eyeballing.
The `warmup=250` run (`output/logs/turbo/20260529-184159`, images
`comfy/output/turbo/g_agg_*`) is the cleanest probe:

| checkpoint | window | α_eff | `dm_rel_gap` | `x_pred_std` | eyeball |
|---|---|---:|---:|---:|---|
| **250** (ramp done) | 0–250 | →4 | **0.066** | **0.666** | **best** — light/soft, in-domain, clean letters |
| 500 | 250–500 | 4 (hold) | 0.100 | 0.710 | worst — dark/over-saturated (over-bake onset) |
| 750 | 500–750 | 4 (hold) | 0.104 | 0.696 | peak damage |
| 1k | 750–1000 | 4 (hold) | 0.099 | 0.703 | partial recovery, still < 250 |

The gap is lowest at ramp-completion and **spikes the instant α pins at 4**, then only
partially unwinds. This is the dose curve made visible.

Cross-run, all grabbed at *their* ramp-completion (so all "no hold"):

| run | warmup | grab | CFG dose ∝ ½·3·W | result |
|---|---:|---:|---:|---|
| g_agg_250 | 250 | 250 | **375** | best (soft, near-ideal) |
| gnp / F | 1000 | 1000 | 1500 (4×) | more saturated → lost to g_agg_250 |
| g_agg_500/750 | 250 | +hold | 1125 / 1500 | over-baked |

Two corroborations from the historical log: the `anima_turbo_B_8k` run (§ old Post-8k
read) found "more same-config steps are not the lever" — exactly the over-bake prediction
at 8k. And **`x_pred_std` is reframed: LOW = good** (in-domain/soft); the over-bake
*inflates* it (oversaturation), it is **not** a diversity signal. (Earlier readings of it
as "higher = more diverse = good" were backwards.)

### Falsified directions (do not run)
- **Longer warmup / 4k "full adaptation"** (`alpha_warmup ≈ iterations = 4000`): dose ∝
  ½·3·4000 ≈ 6000 = **16× g_agg_250** → massively over-baked unless LR is cut ~16×; even
  dose-matched it's the "adapt to a non-existent fixed point" error. Burns compute to
  rediscover that the best checkpoint lands early.
- **Higher `student_lr` (e.g. 5e-5):** more dose/step → over-bake *sooner*. At a fixed
  grab step it's strictly worse. LR is the clock, not the lever.
- **Curation by low `hf_ratio`** (`use_prep_list` prep): `hf_ratio ⊥ noise (r=0.055)`, so
  penalizing it strips high-frequency *content*, not noise → smoother target → blurrier
  output. See `project_item5_turbo_curation_phase0`; drop the `hf_ratio` term (noise-only)
  or set its α=0 before re-enabling prep.

---

## 2. Metrics

`dm_rel_gap` is now the **primary grab signal**, not just a fake-lag trigger.

| metric | definition | read |
|---|---|---|
| `dm_rel_gap` | `rms(τ·Δ_dm) / rms(τ·v_real_dm)` | **minimize / grab at its minimum.** Rising = CFG over-bake dragging the student off-manifold (or fake-lag — disambiguate with `dm_cos`). |
| `dm_cos` | `cosine(v_fake_dm, v_real_dm)` | →1 healthy; dropping = fake pointing wrong (true fake-lag, not over-bake). |
| `dm_mag_ratio` | `rms(v_fake_dm)/rms(v_real_dm)` | ≈1 healthy; off = fake mis-scaled. |
| `dm_to_ca` | `rms(τ_dm·Δ_dm)/rms(τ_ca·(α−1)·Δ_cfg)` | Decoupled guard: CA is the *engine*, DM the *shield*. Logged on `do_ca` steps. |
| `x_pred_std` | std of x_pred | **~0.66 good (in-domain); inflation = oversaturation/over-bake.** Not a diversity signal. |
| `v_student_rms` | — | steady (~1.2) healthy; rising fast = runaway. |
| `alpha_eff` | schedule | ramps 1→`teacher_cfg`; **the dose driver.** |
| `fake_loss` | — | may rise; **not a trigger by itself.** |

Disambiguation rule: `dm_rel_gap`↑ **with** `dm_cos`↓ → fake-lag (bump fake). `dm_rel_gap`↑
**with** `dm_cos` ~flat **and** `x_pred_std`↑ → CFG over-bake (you've passed the grab).

---

## 3. Validated levers (ordered by the paper's engine/shield framing, §0b)

> Reordering rationale (§0b): the paper says the over-bake is the CA-alone failure leaking
> through a **weak DM shield**. So the primary levers *strengthen the shield* (A, B); the
> schedule/CFG/grab-timing knobs (C, D) are **workarounds** for a weak shield — keep them as
> safety nets, but they're not the real fix.

### A. Strengthen the DM shield: better/faster critic ✅ **#1, paper-faithful**
`fake_steps_per_student_step` 2→4 (+ `fake_lr` 3e-5→5e-5, `fake_warmup_steps` a touch
higher). Per §3.1.1, DM is what *"eliminates over-saturation… over extended periods."* Our
over-bake = the shield can't keep up (fake under-tracking: `dm_rel_gap`↑ / `dm_cos`↓). A
fresher, more-accurate critic = a cleaner DM gradient = a stronger shield that suppresses
the CA oversaturation, ideally letting later checkpoints win (their Fig. 2). Cheap at short
iteration counts; low risk. **Try this first, at fixed α=4 and the full schedule.**

### B. Mean-variance regularization (paper Eq. 7) ✅ **WIRED — cheap second shield**
A KL reg on each generated image's `(μ_i, σ²_i)` toward real-data stats directly clamps the
**variance inflation** that *is* the oversaturation (§3.2):
`L_mv = (1/B)Σ ½[ (σ_i² + (μ_i−μ_t)²)/σ_t² − 1 − log(σ_i²/σ_t²) ]`. In the paper this
is a *weaker standalone* regularizer than DM, but it targets our exact symptom for almost no
compute → stacks on top of DM as an auxiliary shield (small weight).

**Implemented 2026-05-29** (`distill.py::mean_var_kl`, config `[mean_var]`, TB
`train/mean_var_kl`): a real differentiable loss on `x_pred` added to the student loss as
`mean_var_weight·L_mv`. The paper's SDXL stats μ_t=0.075/σ_t²=0.81 **don't transfer** to
Anima's 16-ch VAE, so the target defaults to an **EMA over the real latents**
(`sigma2_t ≤ 0` → auto; set `sigma2_t > 0` to pin). Stats are per-image over (C,H,W),
full-frame even under masked loss (it's a global distribution clamp). Run S2 with
`mean_var_weight ∈ {0.01, 0.03, 0.05}`. Benched: not yet.

### C. (Workaround) short ramp, no hold, grab at the `dm_rel_gap` minimum ⚠️ compensates for weak shield
`iterations ≈ alpha_warmup_steps`, per-step saves on, grab the `dm_rel_gap` minimum
(≈ ramp-completion). This is `g_agg_250`, the current best **given the present weak shield**.
A faster ramp gives a cleaner grab (less cumulative high-α exposure) — g_agg_250 (warmup 250)
beat gnp/F (warmup 1000). **Caveat (§0b):** the α-warmup ramp is *not* in the paper (they use
fixed α); it's our workaround. If A/B fix the shield, the early-grab sweet spot should dissolve
and you train the full schedule. Keep this only as a fallback; bracket `alpha_warmup_steps ∈
{150,250}` if pursuing it.

### D. (Workaround) lower the CFG ceiling: `teacher_cfg` 4 → 3 / 3.5 ⚠️ weakens the engine
Evidence the ideal *baked* CFG is below 4 (g_agg_250 baked less and is softest/best).
`teacher_cfg` only scales the bake (`(α−1)·delta_cfg`); lowering it caps the saturation
ceiling and slows CA accumulation. **But the paper keeps α at the target CFG** — lowering it
trades away the CA engine's few-step conversion power. Use only if strengthening DM (A+B)
isn't enough. Test `teacher_cfg ∈ {3, 3.5}` last, not first.

### E. Masked loss — protects the background from CFG over-bake ✅
`use_masked_loss=true` zeroes the *student* push (including the CA/CFG bias) in
background latents, so the background never gets CFG-over-baked and stays on-manifold —
this is the mechanistic reason masked runs (F) had cleaner backgrounds and more in-domain
characters than full-frame runs (G). Keep it **on**. (The fake/critic regression stays
full-frame; normalization stays /numel, so a masked run sees a lower effective student
gradient — fold that into the dose when comparing.)

### F. DMD per-sample normalization — in x0 space ✅ SHIPPED (`dm_x0_norm=true`)
(b) won the A/B decisively on samples — (a) τ-damping collapsed seeds (mode-seeking) and
blurred text; (b) per-sample magnitude-normalization restored seed diversity and fixed
text. `denom ≈ τ·mean|v_real|` so τ roughly cancels (only `clamp_min(norm_floor)` bites
for τ<~0.056). (a)/(b)/(c) are *not* additive — (c) is just (b) with τ re-multiplied;
**do not ship (c).** `dm_rel_gap`/`dm_cos` are policy-invariant here; the decisive signal
was multi-seed 4-step samples. (Full original A/B writeup preserved in git history.)

### G. Detached on-trajectory anchors — only if 4-step eval plateaus
Single-anchor training vs 4-step inference is a real DMD2 mismatch, but anchors are a
*trajectory-distribution fix*, not a general quality booster. Keep cheap: sample
k∈{0,1,2,3}, roll student to schedule state x_tk under `inference_mode` (~1.5 fwd avg),
train one normal DMD update from (x_tk, t_k) with CA τ_CA>t_k. Start `anchor_prob=0.05`.
**Verify `blocks_to_swap=0` first** (swap offloader desyncs on a 2nd DiT forward; turbo's
`use_custom_down_autograd=true` keeps swap off — expected-safe, confirm before wiring).

### H. Consistency auxiliary / schedule-jitter — defer behind G
Low-weight (0.01–0.05) output-space consistency after CA warmup (redundant with G).
Schedule-jitter only buys mild solver robustness, never true multi-stride (a plain LoRA
must average antagonistic per-t corrections; Shortcut/MeanFlow Δt-conditioning can't
survive the plain-LoRA bake). Both low priority.

### Recommended next run matrix
The paper says fix the *shield* first (A+B), then see if the early-grab workaround (C) is
even still needed. So:

- **Control:** `g_agg_250` schedule exactly.
- **S1 (shield, the real test):** **full schedule, fixed α=4** — `teacher_cfg=4,
  alpha_warmup=250, iterations=2000, fake_steps=4, fake_lr=5e-5, fake_warmup≈100,
  student_lr=2e-5, use_masked_loss=true, save_every=100`. Watch whether a *stronger* shield
  keeps `dm_rel_gap` flat/low through the hold instead of spiking — i.e. whether **late
  checkpoints now beat 250** (the paper's Fig. 2 prediction). This is the decisive
  engine/shield experiment.
- **S2 (shield + Eq. 7):** as S1 plus the mean-variance reg (lever B, now wired) —
  `mean_var_weight ∈ {0.01, 0.03, 0.05}`, target auto-calibrated by EMA over real latents
  (`[mean_var].sigma2_t = -1`). Watch `train/mean_var_kl` fall and `x_pred_std` settle near
  ~0.66. Tests the cheap second shield.
- **Fallbacks only if S1/S2 still over-bake:** the early-grab bracket (C, `alpha_warmup ∈
  {150,250}`, iters=warmup) and lower `teacher_cfg` (D, {3, 3.5}).

---

## 4. `ca_band_weight` removal (item 2 — FALSIFIED, ✅ REMOVED 2026-05-29)

The CA band-deficit reweighting (`item2_plan.md`) is dead and the §0 mechanism confirmed
it can't be revived in its current form. **Removed 2026-05-29** — `scripts/distill_turbo/
ca_band.py` deleted; the wiring, config knobs, and 5 `band_*` TB scalars ripped out of
`distill.py` / `config.py` / `metrics.py` (the packed-flush array re-indexed) / `__init__.py`
/ `turbo.toml`. The findings docs (`docs/findings/turbo_fei_band_deficit_falsified.md`,
`turbo_caband.md`) are kept as the why. The reasoning, preserved:

- **Falsified live** (2026-05-29, `docs/findings/turbo_fei_band_deficit_falsified.md`,
  check-item `turbo_caband.md` PASS): the wired loss measures FEI on renoised-data x_t,
  which fires the *inverse* arm — logged `band_w_high ≡ 1.00001` (intended arm never
  fired), `band_w_low ≈ +1%` (~no-op, mildly anti-sharpening). `g_noprep ≡ F` numerically
  across every bin → disabling it changed nothing.
- **Wrong direction for the real failure.** It up-weights bands where the student
  *under-fills* (adds *more* CFG). §0 shows the failure is the **opposite** — too *much*
  accumulated CFG. A deficit-filling reweight points the wrong way.
- **Wrong altitude.** The problem is the scalar CFG *dose* (`lr × Σ(α−1)`); a per-band
  gain on `delta_cfg` only redistributes *which* band over-bakes first — it can't stop the
  unbounded accumulation.
- **Redundant with the shipped schedule (paper §4.1).** Liu et al. show CA at re-noising
  level τ already enhances the frequency band *at that noise level*, and the correct way to
  target bands is the **`τ_CA > t` schedule** ("concentrate on the remaining, unresolved
  aspects") — which we already ship (`tau_ca_strategy="above_t"`). Explicit band-reweighting
  of `delta_cfg` duplicates what the re-noising schedule does for free.

**Removed, not kept dormant.** It was ~250 lines of dead surface + 5 TB scalars that
confuse the grab-signal reading. (If a band-aware tool is ever wanted here it'd be the
*inverse* — an HF-protecting *damp* on the CA push, a new method, only after the dose is
dialed in. Not this.) The packed-flush array in `metrics.py` was re-indexed when the 5
trailing `band_*` elements were dropped; `flush()` now packs `[…9 always-on…, mv,
dm_to_ca, ca_steps]` and the round-trip is covered by the CPU smoke check.

---

## 5. Explicitly skipped (with reasons)
- **`ca_band_weight` / FEI band-deficit** — falsified + wrong-direction + wrong-altitude;
  removed 2026-05-29 (§4).
- **Longer warmup / 4k "full adaptation" & higher `student_lr`** — over-bake the dose
  (§1 falsified directions).
- **Curation by low `hf_ratio`** — strips detail, not noise (§1).
- **Share `τ_CA = τ_DM`** — breaks the load-bearing `τ_CA > t` schedule; 25% saving not
  worth it.
- **Timestep-conditioned student (T-LoRA / per-step scale)** — T-LoRA mask is
  training-only; inference is full-rank at every t → nothing after the plain-LoRA bake
  (`project_tlora_inference_full_rank`).
- **Shortcut / MeanFlow Δt-conditioning** — needs a step-size input at inference; doesn't
  survive plain-LoRA export unless schedule-locked.
- **Adversarial / LADD feature discriminator** — out of v1 scope; LADD's "frozen
  generative features" exert ~zero transferable pressure on a rank-r adapter
  (`docs/findings/selfflow.md`).
- **Lowering fake cadence / fake rank** — backwards vs the DM:CA rate-ratio lever (§3C).
