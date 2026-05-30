# spectral_guidance — using Anima's diffusion-operator spectrum for preservation-aware steering

Source: Moreira et al., "Spectral Guidance for Flexible and Efficient Control of
Diffusion Models" ([arXiv:2605.28900v1](https://arxiv.org/abs/2605.28900), CMU /
IST, May 2026). Local copy: `2605.28900v1.pdf`.

One line: the paper learns, *without backprop through the denoiser*, the low-rank
time-indexed subspace of directions that survive the forward process, and shows
guidance is only effective during the spectral **phase transition** of that
subspace. This proposal asks whether that **property** (not the paper's
unconditional-CIFAR use case) buys us anything on Anima — primarily in
**DirectEdit**, secondarily in **mod-guidance**.

> ⚠️ This is a *property-transfer* proposal with a real chance of dying at Phase 0.
> The paper is pixel-space DDPM at ≤121M params; Anima is a latent flow-matching
> DiT, and our latent manifold is already collapsed (`project_pe_feature_diagnostics`,
> PR≈6.2). Phase 0 exists to kill it cheaply before anyone trains an `f_φ`.

## What the property actually is

Two *separable* claims. Keep them separate — they fail independently.

- **(P1) Low-rank, time-indexed guidable subspace.** As noise grows, only a few
  features survive. Formally these are the leading left singular functions
  `φ_{t,k}` of the conditional-expectation operator `T_t` (clean → noisy data),
  i.e. the principal modes of the round-trip covariance `T_t T_t*`. Guidance can
  *only* act along `span{φ_{t,k}}`; everything else is noise-destroyed. The paper
  recovers this basis with a denoiser-free SSL objective (whitened VICReg/Barlow,
  Algorithm 1) where the "augmentation" is diffusion noise itself — two noised
  views of one clean sample.

- **(P2) Guidance window = spectral phase transition.** The spectrum of `T_t T_t*`
  collapses midway through the reverse process (Fig 6–7). Guidance is most
  effective precisely in that transition window — early it's redundant (operator
  ≈ identity), late it's hopeless (all information erased).

The paper's *deliverable* is a small `f_φ` (≈16M params) that recovers `φ_{t,k}`
plus cached reference features `Φ_t`; at sampling time any clean-data signal
`h(x₀)` (label / CLIP embed / mask) is projected onto the basis to produce a
classifier-guidance-style step, gradient flowing through `f_φ` only.

**This is fundamentally a *preservation* tool.** `span{φ_{t,k}}` is the
information-*preserving* subspace. The paper explicitly contrasts itself with the
*editing-direction* literature (NoiseCLR; Park et al. 2023; Chen et al. 2024 —
Jacobian spectral decomposition). That distinction is what makes DirectEdit the
right home and plain guidance the wrong one (Anima is already text-conditional;
adding label/CLIP guidance to it is redundant).

## Where it could pay off — candidate consumers (ranked)

### A. DirectEdit spectral-subspace anchoring (primary)

DirectEdit's hard half is **preserve identity/structure while the edit takes**,
and that preservation is currently done with two blunt instruments
(`docs/experimental/directedit_editing_v3.md`):

- the **Δz anchor**, which pins the *entire* trajectory to source (global recon,
  fights the edit everywhere), and
- **`t_inj`**, a single scalar trading preservation against edit leverage
  ("higher = stronger source-feature preservation, weaker edit leverage").
- The surgical tool — **mask blending (paper Eq. 12)** — is *stubbed*: `--mask`
  warns and is ignored.

The idea: replace/augment the blunt preservation with a **spectral mask**. At each
edit step, decompose the latent into its projection onto the preserved subspace
`span{φ_{t,k}}` (anchor *that* to source) and its complement (leave *that* free
for ψ_tar to drive the edit):

```
z_new = P_t · z_anchor  +  (I − P_t) · z_pred          # P_t = projector onto span{φ_{t,k}}
```

This is exactly the per-step linear combination the un-wired Eq. 12 mask would be
— except the "mask" is a *learned semantic projector in spectral coordinates*
rather than a hand-drawn spatial region. It decouples *what to keep* (spectral
subspace) from *what to change* (its complement), which `t_inj` conflates into one
scalar. And the `(P2)` window tells you *when* in the schedule the projector
matters — which should line up with `project_sigma_signal_resolves_by_045`
(σ≈0.45 ≈ `t_inj`=12/28 today), giving a principled schedule instead of a
hand-tuned one.

Bonus: the `h(x₀)` route lets you steer toward a target attribute/CLIP-embed
*without an on-manifold ψ_src at all*, sidestepping DirectEdit's documented top
failure mode (edit leverage collapses when the tagger's ψ_src is off-manifold).

### B. Mod-guidance window/subspace gating (secondary)

Mod-guidance steers AdaLN with a fixed unit direction
`delta_unit = proj(pool(p₊)) − proj(pool(p₋))`, scaled per-*block* but **not
per-timestep** (`docs/methods/mod-guidance.md`, Strategy 4). Two known pains:

1. **Directional double-count** — base + steering both read
   `maxpool→pooled_text_proj` (`project_mod_guidance_quality_tag_axis`).
2. The per-block schedule (`step_i8_skip27`) is hand-tuned to dodge DC blowout /
   pink-collapse in early blocks.

Two property-driven moves, both cheap once `f_φ` exists:

- **(P2) per-timestep gating.** Schedule `mod_w` over t by the spectral
  transition rather than applying it flat across all steps. Hypothesis: the
  double-count degradation concentrates *outside* the effective window, so tapering
  there is free quality. Testable directly against the existing
  `bench/mod_guidance/` harness.
- **(P1) project the steering delta onto `span{φ_{t,k}}`.** A steering direction
  with components outside the guidable subspace can only push the latent
  off-manifold (the pink-collapse mode). Projecting `delta_unit` onto the
  surviving subspace is a principled version of "protect blocks 0–7".

This is more speculative than (A) and the payoff is a refinement, not a new
capability. Treat as a follow-on consumer, not the lead.

### C. Spectral phase-transition as a validation/diagnostic (byproduct)

The project has a documented val-signal pain (`project_fm_val_loss_uninformative`;
`project_cmmd_val_signal`). The `T_t T_t*` spectrum is a *model-intrinsic*,
denoiser-only signal: "does this adapter preserve the guidable subspace / shift
the transition window?" This falls out of Phase 0 for free and is worth keeping
even if (A) and (B) die.

## Phase 0 — the free falsification gate (no `f_φ`)

**Do not train anything yet.** Phase 0 answers one question: *do (P1) and (P2)
even hold on Anima's latents?* If the collapsed manifold means the spectrum is
already degenerate or shows no clean transition, the whole proposal is moot and
the `f_φ` recast is wasted.

Use the frozen DiT as its own denoiser-based stand-in for `φ_{t,k}` (the paper's
`f_φ` is just an amortized estimator of the posterior-mean spectrum we can measure
directly):

1. **Spectrum probe (tests P1 + P2).** For each σ bin over the schedule: take N
   clean latents from `post_image_dataset/lora`, FM-noise them, run the frozen DiT
   to the posterior-mean estimate `x̂₀(x_t, σ)`, and compute the covariance
   spectrum of `{x̂₀}` across the batch. Report effective rank vs σ and locate the
   collapse.
   - **(P1) GO** if a small number of modes dominate (effective rank ≪ latent dim)
     in the mid-σ band.
   - **(P2) GO** if effective rank shows a clear transition (not flat, not
     monotone-trivial) and the transition lands near σ≈0.45 — cross-checks
     `project_sigma_signal_resolves_by_045` and the paper's Fig 7.
   - **NO-GO** if the spectrum is flat/degenerate (collapsed manifold dominates)
     or shows no usable window.

2. **DirectEdit anchoring probe (tests whether the subspace is *useful*, not just
   present).** Along a real inversion trajectory, anchor the **top-K covariance
   subspace** of `x̂₀` to source (crude `P_t` stand-in) instead of the full Δz,
   and check whether a fixed edit *takes more strongly while identity is
   preserved* than `t_inj=12`. Sweep K and the σ-window.
   - **GO** if crude-subspace anchoring beats the `t_inj` scalar on the
     preserve↔edit frontier for ≥1 setting.
   - **NO-GO** if it never beats the scalar — then the learned `f_φ` won't save it
     (the basis quality isn't the bottleneck).

Phase 0 is a few hundred GPU-seconds: no training, reuses cached latents + the
inversion primitive that already exists.

## Phase 1+ — only if Phase 0 GO

1. **Recast `f_φ` for Anima.** Latent-space, FM schedule. Two FM-noised views at
   matched σ; whitened SSL loss (Algorithm 1). Recast the DDPM `ᾱ_t` math to
   Anima's σ-parameterization. ~10 GPU-h offline per the paper; budget more for
   latent + larger model. Cache `Φ_t` over a reference set.
2. **Wire `P_t` into DirectEdit.** Add a spectral-projector path to
   `library/inference/directedit.py::edit_forward` in the exact slot the stubbed
   `mask=` Eq. 12 blend targets (a per-step linear combination — no inversion-loop
   surgery). Keep `t_inj`/Δz as the fallback.
3. **(deferred) Mod-guidance consumers** (B) once the basis exists.

## Bench harness — `bench/spectral_guidance/`

Standard envelope (`bench/_common.py::make_run_dir` / `write_result`).

| Phase | Script | Primary metric | GO criterion |
|---|---|---|---|
| 0.1 | `probe_spectrum.py` | effective-rank(σ) curve + transition σ | low-rank mid-band **and** clear transition near σ≈0.45 |
| 0.2 | `probe_anchor.py` | preserve (DreamSim/LPIPS vs src) ↔ edit-leverage frontier vs `t_inj` baseline | crude subspace beats scalar for ≥1 setting |
| 1 | `bench_directedit_spectral.py` | same frontier, learned `f_φ` vs crude-PCA vs `t_inj` | learned basis ≥ crude basis ≥ scalar |

Artifacts: `effective_rank.png` (rank vs σ), `frontier.csv` (preserve vs edit per
K/window), side-by-side edit grids (READ THE GRIDS — DreamSim/LPIPS are blunt on
pose/identity, cf. the PE-pooled-cosine blindness noted in
`project_dpdmd_pivot_phase0`).

## Kill criteria

- Phase 0.1 NO-GO (no low-rank window / no transition) → **shelve**; write up in
  `docs/findings/` like `l2p_pixel_transfer.md`. The collapsed manifold won.
- Phase 0.2 NO-GO (crude subspace never beats `t_inj`) → **shelve the DirectEdit
  arm**; the scalar is already at the frontier. Keep (C) as a diagnostic.
- Phase 1 learned `f_φ` ≤ crude PCA → the SSL amortization isn't buying anything
  on our latents; ship the crude per-trajectory PCA projector instead of `f_φ`
  (cheaper, no offline train) or shelve.

## Relationship to existing work

- **`postfix_residual_for_directedit.md`** — orthogonal. That carries the visual
  *residual* in cross-attn conditioning space (ψ_src fidelity); this controls the
  preserve↔edit split in *latent* space. They compose: postfix fixes "ψ_src can't
  recon", spectral anchoring fixes "`t_inj` is a blunt preserve knob".
- **`project_sigma_signal_resolves_by_045`** — Phase 0.1 is a direct cross-check;
  if the transition doesn't land near σ≈0.45 something is wrong with the probe.
- **`project_pe_feature_diagnostics`** — the reason for the ⚠️ preface; the
  collapsed manifold is the most likely Phase-0 killer.
- **Editing-direction refs** (Chen et al. 2024; Park et al. 2023) — if the goal
  shifts from *preservation* to sharper *edit directions*, those are more
  on-target than this paper. Noted so we don't reach for Spectral Guidance to do a
  job it explicitly isn't for.
