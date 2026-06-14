# FLAIR — training-free inverse-problem solver on the Anima flow prior

Status: **PHASE 0 PASSED 2026-06-14.** The port-validation gate (SR ×8,
uncalibrated `λ_R=t`) is implemented in `bench/flair/` (`solver.py` = Algorithm 1,
`operators.py`, `sanity_sr.py`) and **passes**: FLAIR beats the bicubic baseline on
**5/5** images (mean PSNR 23.76 vs 22.99, +0.77; SSIM 0.742 vs 0.704; per-image
gain +0.44…+1.05), sharp + artifact-free, runs on a 16 GB card at 512px with
`--compile` (no OOM). So the σ/t mapping, the `v_θ` sign, the adjoint init, the
VAE-decode HDC, and the dim-2 handling are all correct — the prior **can** drive
inverse problems on Anima. Nothing is wired into the inference engine yet (solver
stays in `bench/` until the pilots prove out). Bench notes: `bench/flair/README.md`;
project memory: `[[project_flair_phase0_port_validated]]`. Green field — no prior
inverse-problem / posterior-sampling line exists in the repo or `_archive/`.
**Open before Phase 0 fully closes:** the formal N=100 run (above is N=5), a
`reg_scale`/`hdc`/`alpha` sweep, and LPIPS/pFID (currently PSNR/SSIM only).

**Phase 1 (λ_R calibration) — CALIBRATED 2026-06-14, A/B owed.**
`bench/flair/calibrate_lambda.py` mirrors Eq. 14 over the *deployed* flow-shifted σ
grid (not linear t), auto-detects Anima's low-noise cutoff from the error knee, and
writes `networks/calibration/flair_lambda_r.npz`. The solver
(`load_lambda_table` + `λ_R(t)` selector) and the SR gate driver (`--calib auto`,
peak-normalized so `reg_scale` stays comparable + loop stops at the cutoff) are
wired for the A/B. **N=100/768px run done** (`bench/flair/results/20260614-1417-n100-768px`):
the FM-error curve is U-shaped, **most reliable at σ=0.486 — confirming the σ≈0.45
hypothesis** (`[[project_sigma_signal_resolves_by_045]]`); error blows up only in the
deep tail (max 0.40 at σ=0.029), so the **zeroing cutoff lands at σ=0.111**, *lower*
than the paper's SD3 0.2 — Anima's prior stays usable deeper into low-σ. **Owed to
fully close Phase 1:** the calibrated-vs-`λ_R=σ` SR×8 A/B verdict (PSNR/SSIM now,
LPIPS/CMMD when wired). If it doesn't beat `λ_R=σ` → ship the linear weight.

Source: *Solving Inverse Problems with FLAIR* (Erbach et al., NeurIPS 2025,
`12635_Solving_Inverse_Problems.pdf` in repo root; code
`https://inverseflair.github.io/`). FLAIR is a **training-free variational
framework** that uses a flow-matching latent model (SD3) as the prior for inverse
imaging problems `y = Ax + ν` — super-resolution, motion deblur, box-inpaint, and
(as a special case of masked inpaint) **text-prompted editing**. Anima is the
same recipe — DiT, flow-matching v-field, nonlinear Qwen VAE latent space — so the
method ports without retraining anything.

## TL;DR — what this buys us and how it sits next to what we ship

The repo has Spectrum / SPD / DCW / SMC-CFG / CNS / DAVE inference stacks but **no
inverse-problem solver at all** — no SR / deblur / restoration path, and the only
"inpaint" is the *trained* EasyControl variant. FLAIR is a **new capability
category**, not a reshuffle. Two of its instances land on known repo pain points:

- **vs DirectEdit** — *complementary, more robust for localized edits.* DirectEdit
  is full-image: invert, swap ψ_src→ψ_tgt. Its documented failure is "edit
  leverage collapses if ψ_src is off-manifold" — it needs a good *source* prompt
  and can drift the whole frame. FLAIR's editing is **inpaint-with-target-prompt**:
  `A` = the mask, **no source inversion**, and hard data consistency keeps the
  unmasked region *bit-exact*. So FLAIR is the tool for "change this region, lock
  everything else"; DirectEdit stays the tool for global, mask-free semantic edits.
  Add alongside, don't replace.
- **vs EasyControl 'edit'** — *orthogonal, with one concrete overlap worth chasing.*
  EasyControl is a *trained* adapter (frozen DiT + per-block cond-LoRA + mined
  dataset) and FLAIR can't replace its non-linear structural conditioning. **But
  colorization is a linear inverse problem** (`A` = luma projection), and the
  EasyControl colorize path has a documented data-bias wash-out:
  `[[project_easycontrol_comfy_washout_is_cfg]]` — targets are 94 % desaturated
  because the miner selects pale manga. FLAIR-colorize is **training-free with no
  mining bias** — color comes from the prior under a hard luma constraint. That is
  exactly the failure case that has been hard to fix by training.

## The method in one screen (Algorithm 1, mapped to Anima)

FLAIR optimizes a variational posterior mean `μ` (a **latent**) by alternating a
cheap forward-eval prior pull with a hard data-consistency projection. Init from
the adjoint, then descend the schedule:

```
μ  ← E(A^T y)                       # adjoint init: encode A-transposed observation
ε̂  ← N(0, I)
for t = 1 → t_stop (descending):
    x_t ← (1−t)·μ + t·ε̂                          # noisy latent
    u_t ← (ε̂ − x_t)/(1−t)                        # reference velocity of the variational dist
    μ   ← μ − λ_R(t)·( v_θ(x_t, t) − u_t )        # (i) flow-matching regularizer pull
    μ   ← argmin_μ ‖y − A(D(μ))‖²                 # (ii) hard data consistency (SGD, pixel space)
    x̂_1 ← x_t + (1−t)·v_θ(x_t, t)                 # one-step manifold predictor
    ε    ← N(0, I)
    ε̂   ← α·x̂_1 + √(1−α²)·ε                       # (iii) deterministic trajectory adjustment
```

Three load-bearing parts, in ablation-importance order (paper Table 3):

1. **Deterministic Trajectory Adjustment (DTA)** — re-noise toward `x̂_1` instead of
   fresh Gaussian (`α`∈[0,1] trades deterministic↔stochastic). **Biggest lever** —
   drop it and you get the classic variational blur / mode-collapse.
2. **Hard Data Consistency (HDC)** — decoupled `argmin ‖y − A(D(μ))‖²` projection,
   SGD through the VAE decoder `D`. Guarantees exact agreement with `y`.
3. **Calibrated Regularizer Weight (CRW)** — `λ_R(t)` = inverse expected model error
   per timestep, measured offline; **zeroed below a low-noise cutoff** (the paper
   uses `t < 0.2` because SD3 is inaccurate at low noise).

Crucially the regularizer is the **Wasserstein-gradient-flow reformulation** (paper
Eq. 10): `∇_μ R = λ_R(t)·v_θ − u_t` — a **forward eval of the DiT, no backprop
through it**. Only the HDC step backprops, and only through the VAE decoder.

## Why the Anima port is non-trivial in exactly four places

The math transfers; these four are where a naive port produces black images or
silent garbage, and three of them have a memory note already.

1. **σ/t convention.** FLAIR's `t` (with `x_t=(1−t)μ+tε`, `t=1` noise → `t=0` data)
   **is** Anima's flow-matching `σ`. Anima's `noise_pred` already **is** the
   velocity `v_θ` (`generation.py:832` computes `denoised = x_t − σ·noise_pred`,
   i.e. `v = (x_t − x_0)/σ` = `ε − x_0`), so `v_θ` is exactly what FLAIR's Eq. 10
   wants — no ε/v reparam needed. The only wrinkle is `flow_shift=3` reshaping the
   grid (`get_timesteps_sigmas(..., shift)`, `library/inference/sampling.py:14`):
   **calibrate λ_R on the deployed σ grid**, not on linear t.
2. **The low-noise cutoff is backbone-specific and we already know Anima's break.**
   The paper's `t<0.2` zeroing is an SD3 fact. Anima **resolves x₀ by σ≈0.45**
   (`[[project_sigma_signal_resolves_by_045]]` — base reconstructs to 20 % norm-MSE
   by σ=0.55, x₀_pred visually done by σ=0.45, `e_low` triples in the σ<0.45 tail).
   So the cutoff and the whole λ_R curve are Anima-specific — Phase 1 **re-measures**
   them; expect the break near σ≈0.45, not 0.2.
3. **5D dim-2 latent.** FLAIR blends `x_t`, `μ`, and the adjoint reference `A^T y`
   together every step. The DiT wants 5D `(B,C,1,H,W)`; cached/VAE latents are 4D.
   Every DiT call needs `unsqueeze(2)` and the regularizer math must ndim-match the
   reference — this is the recurring freetext bite-point flagged in CLAUDE.md
   ("target dim 2 explicitly, never `squeeze()`"). Get this wrong → corrupted layout.
4. **Max-padded text encoder + domain prompt.** Trimming TE output → black images
   (CLAUDE.md invariant). And the paper leans on a generic face prompt ("A high
   quality photo of a face"); Anima's prior is a different domain, so the pilots
   pass either the real caption or a domain-appropriate generic — never an empty/
   trimmed embed.

## Where it hooks (verified against the live tree)

> **Productization split.** This doc is the **method + bench** (Algorithm 1, the
> ladder, the λ_R calibration). Shipping a validated task as a user-facing app is its
> own proposal. The first front door is **editing**, not SR: [`flair_edit.md`](flair_edit.md)
> productizes the Phase-3 inpaint pilot into training-free *localized editing with a
> delta-only prompt* (no ψ_src — the DirectEdit/tagger pain point), incl. a SAM3
> text→mask front-end, the `make exp-test-flair-edit` target, and the
> HDC-grad-in-a-node edge. (The earlier SR-deploy proposal `flair_sr.md` was retired
> 2026-06-14 in favor of the editing track — SR stays validated in `bench/flair/` but
> is not the shipped app.)

A new training-free stack, composing at the **sampler boundary** like CNS/DCW.

- **New module** `library/inference/corrections/flair.py` — a `FLAIRSolver` holding
  the forward operator `A`/`A^T`, the λ_R table, and `α`. It does **not** ride the
  normal Euler loop step-for-step; FLAIR *replaces* the sampling loop with
  Algorithm 1 (it optimizes `μ`, it doesn't denoise a fixed trajectory). So the
  cleanest wiring is a **branch in `generate_body`** (`library/inference/generation.py:505`):
  when `args.flair_task` is set, dispatch to `flair_solve(...)` instead of the
  standard `euler/er_sde` path at `:757`. The DiT velocity call (`anima(x_t, t, embed, …)`,
  cf. `:815`) and `get_timesteps_sigmas` are reused verbatim.
- **Forward operators** `library/inference/corrections/flair_operators.py` — each a
  `(A, A_adjoint)` pair in **pixel** space (FLAIR's `A(μ)` is really `A(D(μ))`,
  decode then degrade): `sr` (bicubic down ×s / up), `deblur` (conv with a kernel /
  its transpose), `inpaint` (binary mask, self-adjoint), `colorize` (RGB→luma 1×3
  projection / its transpose). Adjoint init `μ=E(A^T y)` uses
  `vae.encode_pixels_to_latents` (`library/models/qwen_vae.py:1433`); HDC decodes
  via `vae.decode_to_pixels` (`:1411`) — both auto-handle 4D↔5D.
- **Calibration npz** `networks/calibration/flair_lambda_r.npz` (keys `lambda_r`,
  `sigmas`), loaded by the CNS pattern: `from_path("auto")` → `resolve_under_home`
  (`corrections/cns.py:101`). Auto-download from a GitHub release like cns_gamma.
- **Request/CLI plumbing** — add `flair_task` / `flair_scale` / `flair_alpha` /
  `flair_hdc_steps` / `flair_calib` to `GenerationRequest` (`request.py:50`) +
  `build_parser` (`library/inference/args.py:23`); long-tail can ride `extra_argv`
  first. Task surface: `make test FLAIR=sr SCALE=8 REF_IMAGE=lowres.png` etc.
- **Docs** — user-facing `docs/inference/flair.md`; one-liner row in root CLAUDE.md.

## What to bench — the ladder (each rung gates the next)

Tier-2 method ⇒ bench + invariant test mandatory (CONTRIBUTING). All runs drop the
`bench/_common.py` `result.json` envelope into `bench/flair/results/<ts>-<label>/`
and build the model through `library/runtime/harness.py::build_anima`
(load→apply→compile ordering). Metrics throughout: **LPIPS, pFID, PSNR, SSIM**
vs ground truth (the paper's set) + **CMMD** (our live signal,
`[[project_cmmd_val_signal]]`); FM-MSE is uninformative
(`[[project_fm_val_loss_uninformative]]`).

### Phase 0 — port-validation sanity (the cheap gate, NO calibration) — ✅ PASSED 2026-06-14

`bench/flair/sanity_sr.py`. Replicate the paper's easiest win — **SR ×8** —
with the *uncalibrated* `λ_R(t)=reg_scale·t` setting (the CRW-ablated row of paper
Table 3) so calibration isn't a confound yet. This proves the port, not the method:
validates the σ/t mapping (#1), the `v_θ` sign, the adjoint init, the VAE-decode HDC
projection, and the dim-2 handling (#3).

**Gate.** Reconstruction is sharp and data-consistent (PSNR > the `A^T y` upsample
baseline, no black frames, no layout corruption). If it's garbage/black → the
**port** is broken (dim-2 / TE-pad / σ-sign), fix that before reading any quality
number. Cheap close if Anima's prior simply can't drive SR at all.

**Result (N=5, 512px, 50 steps / 8 HDC steps / reg_scale 0.5, `--compile`):** PASS.
FLAIR beats bicubic on **5/5** images — mean PSNR 23.76 vs 22.99 (+0.77), SSIM 0.742
vs 0.704, per-image gain +0.44…+1.05 — sharp and artifact-free (triptychs recover
hair/face/fabric detail bicubic smooths away). Runs on a 16 GB card with no OOM.
One process lesson baked into the bench: **data consistency must keep pace with the
prior pull** — the under-converged smoke (10 steps / 2 HDC) let the prior drag μ
off-manifold (hazy + hallucinated watermark, FAIL); longer/stronger HDC fixes it.
Still owed for the *formal* close: N=100, a `reg_scale`/`hdc`/`alpha` sweep, and
LPIPS/pFID. See `bench/flair/README.md` + `[[project_flair_phase0_port_validated]]`.

### Phase 1 — λ_R(t) calibration + the Anima cutoff

`bench/flair/calibrate_lambda.py`. Mirror paper Eq. 14: sample N calibration images
from `post_image_dataset`, over the **deployed σ grid** compute the conditional FM
error `mean_i ‖v_θ(x_t^i,t) − u_t(x_t^i|ε)‖²`, set `λ_R(t) = 1/error`. **Find
Anima's low-noise cutoff** (paper's `t<0.2`; pre-registered hypothesis: it lands
near σ≈0.45 per `[[project_sigma_signal_resolves_by_045]]`). Output
`networks/calibration/flair_lambda_r.npz`.

**Readout (pre-registered).** The error-vs-σ curve shape + the σ below which λ_R is
zeroed. **A/B against Phase 0:** calibrated λ_R must beat `λ_R=t` on the SR ×8
LPIPS/CMMD, replicating the paper's CRW ablation direction. If calibration doesn't
move the needle on Anima → ship `λ_R=t`, skip the npz, note it.

**Results — N=100, 768px, 100-step σ grid (`results/20260614-1417-n100-768px/`).**
Conditional FM error `‖v_θ − u_t‖²` over 100 calibration images from
`post_image_dataset/resized`, flow_shift 3.0, RTX 5070 Ti, compiled. Canonical table
written to `networks/calibration/flair_lambda_r.npz` (keys `sigmas`, `lambda_r`,
`error`, `cutoff_sigma`; curve in `lambda_r_curve.png`).

- **Curve shape — clean U.** Error is **minimal in the mid-σ band** (0.0683 @ σ=0.444,
  **0.0679 @ σ=0.486**, 0.0687 @ σ=0.551 — a flat basin σ≈0.4–0.55) and rises at
  *both* ends: a moderate climb to 0.299 at the noise end (σ=1.0, 4.4× the min) and a
  **steeper blow-up in the deep tail** to 0.402 at σ=0.029 (5.9× the min). So the
  calibrated `λ_R(σ)=1/error` is a **hump peaking at σ=0.486** that sits *above* the
  Phase-0 `λ_R=σ` diagonal across the mid-band and *below* it past σ≈0.75 — i.e. CRW
  concentrates the prior pull where the model is trustworthy and backs off in the
  noisy high-σ regime the linear weight over-trusts.
- **Resolve-σ vs hypothesis — CONFIRMED.** The error minimum (`resolve_sigma`) is
  **σ=0.486**, against the pre-registered **σ≈0.45** (`[[project_sigma_signal_resolves_by_045]]`:
  x₀_pred visually done by σ=0.45, base reconstructs to 20% norm-MSE by σ=0.55). The
  measured peak-reliability σ lands inside that window, slightly toward the 0.55 edge —
  Anima's flow prior is most reliable exactly where the σ-signal work said it resolves.
- **Cutoff — σ=0.111, deeper than SD3's 0.2.** Auto knee detection (`knee_mult=2.0`)
  zeros λ_R where error first exceeds 2× the basin minimum on the clean side:
  err(σ=0.111)=0.157 ≈ 2×0.0679. So Anima stays usable **below** the paper's SD3
  `t<0.2` cutoff — the deep-tail blow-up only takes over under σ≈0.11, not 0.2. (The
  hypothesis expected the *break* near 0.45; the **error minimum** is at 0.45–0.49 as
  predicted, while the **zeroing cutoff** sits well below it because the basin is wide
  and only the σ<0.11 tail is unreliable. Both numbers are consistent with the σ-signal
  picture: reliable through the mid-band, degrading only in the cleanest steps.)

**Still owed to close Phase 1:** the calibrated-vs-`λ_R=σ` SR×8 **A/B** (PSNR/SSIM
now, LPIPS/CMMD once wired). The table ships as the default (`--calib auto`,
peak-normalized so `reg_scale` stays comparable); if it does *not* beat `λ_R=σ` on the
A/B → fall back to the linear weight per the pre-registered rule.

### Phase 2 — ablation: confirm DTA is the lever on Anima too

`bench/flair/ablation.py`, mirroring paper Table 3 on SR ×12 (the hardest /
most-collapse-prone task), 100 FFHQ-equivalent + anime samples: toggle **HDC ×
DTA × CRW** (2³ minus redundant) + an **α sweep** (DTA strength) + an
**NFE sweep** + an **HDC-step-count** sweep.

**Pre-registered expectation:** DTA off ⇒ the biggest LPIPS/CMMD regression (blur /
mode-collapse), reproducing the paper. If DTA is *not* the lever on Anima, that's a
real finding about our prior — investigate before trusting the pilots.

### Phase 3 — Pilot A: FLAIR-inpaint-with-prompt vs DirectEdit

`bench/flair/pilot_inpaint.py`. Masked-region regeneration with a target prompt,
head-to-head with **DirectEdit** and the proposed **EasyControl-inpaint**
(`docs/proposal/easycontrol_inpaint.md`). Test set: real images + seeded masks
(reuse the colorize/inpaint mask generator) + a small prompt suite.

- **Hard gate (the whole point vs DirectEdit):** unmasked-region exactness. HDC
  should make `A(output)=y` on the kept pixels ⇒ PSNR there ≈ ∞. **Verify it; if
  the unmasked region drifts, HDC is mis-wired** — this is FLAIR's structural
  advantage over DirectEdit's whole-frame drift.
- **Quality:** masked-region LPIPS/pFID vs a plausible target; prompt adherence
  (CLIP-sim + a qualitative anime-style pass).
- **Readout:** FLAIR wins iff it holds the unmasked region bit-exact *and* matches
  or beats DirectEdit on masked-region quality. Tie on quality but exact context ⇒
  still a win (DirectEdit can't promise that).

### Phase 4 — Pilot B: FLAIR-colorize vs the EasyControl adapter

`bench/flair/pilot_colorize.py`. The headline overlap. `A` = RGB→luma; recover
chroma. Head-to-head with the trained **EasyControl colorize** adapter on the exact
failure it suffers — **vivid inputs** (`[[project_easycontrol_comfy_washout_is_cfg]]`).

- **Primary (the bias test):** output **saturation distribution** on vivid-input
  test images (Hasler–Süsstrunk colorfulness + the sat-histogram the wash-out memo
  used). FLAIR should *not* collapse to <0.35 sat, because it has no mined
  desaturated targets pulling it pale.
- **Consistency gate:** luma-MSE — `A(output)` must match the grayscale input
  exactly (HDC enforces it; verify).
- **Fidelity:** LPIPS to GT color where a GT exists.
- **Readout:** FLAIR-colorize avoids the wash-out (sat distribution not collapsed)
  while holding luma ⇒ a training-free win over a trained, data-biased adapter. If
  it washes out *too*, the bias is in the prior, not the data — informative either
  way.

### Invariant test (mandatory, ships with v1)

`tests/test_flair.py`: (a) each operator's adjoint passes the dot-product test
`⟨Ax,y⟩=⟨x,A^Ty⟩`; (b) `inpaint` with a full-keep mask + `α=1` + 0 prior weight is
a near-identity on a clean latent (the loop doesn't corrupt data it's handed);
(c) dim-2 round-trip: encode→solve→decode preserves `(W,H)`.

## Risks / honest limits

- **HDC backprops through the VAE decoder every step** — the real compute/memory
  cost, and a potential OOM at high res or many HDC SGD steps. `compile_blocks`
  covers the `v_θ` forwards but the decode loop is separate. Mitigations: cap
  `flair_hdc_steps`, or a latent-space approximate `A` for tasks where it's linear
  enough (inpaint mask downsampled to latent grid). Bench the cost in Phase 0.
- **Three-part method, DTA load-bearing.** A half-port (regularizer only, no DTA)
  will underwhelm — the ablation says so. Implement all three or don't bother.
- **Posterior *sampling*, not deterministic.** Diversity is a feature for
  restoration but a liability for editing — run the pilots near `α→1` (deterministic
  end) when repeatability matters.
- **Domain prior.** The paper's face-prompt trick is FFHQ-specific; Anima's prior
  is a different domain, so colorize/inpaint quality is bounded by what the Anima
  prior actually knows. Phase 3/4 qualitative gates catch this.
- **Linear-A only.** FLAIR covers SR/deblur/inpaint/colorize — operators expressible
  as a linear `A`. It **cannot** replace EasyControl's non-linear structural
  conditioning (depth, sketch, sanitize) or DirectEdit's global mask-free edits.
- **Paper caveats inherited.** Single backbone (SD3), face/natural datasets; the
  λ_R cutoff and α are tuned per task there. Treat the paper as the recipe, the
  bench ladder as the validation.

## Explicitly NOT doing

- **Pixel-space FLAIR** (the paper's secondary variant on model [33]) — Anima is
  latent-only; out of scope.
- **Replacing EasyControl structural control or DirectEdit global edits** — FLAIR is
  additive, covering the linear-operator tasks neither does cleanly.
- **Any training.** FLAIR is training-free by construction; if a task needs a
  trained operator, it's the wrong tool — use EasyControl.
- **Wiring before Phase 0 passes.** The port-validation gate is cheap and decides
  whether the Anima prior can drive inverse problems at all.
