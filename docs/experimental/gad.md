# Geometry-Aware Distillation (GAD)

**Paper:** Huang et al., *Restoring Initial Noise Sensitivity in Text-to-Image
Distillation via Geometric Alignment*, arXiv:2606.01651.
**Where it lives:** **Mod-guidance** (text-direction sensitivity):
`scripts/distill_mod/distill.py` (post-base-loss insert),
`scripts/distill_mod/config.py` (`--gad_*` flags). See the section below — wired
after `bench/mod_guidance/text_jacobian.py` confirmed the deficiency.
**Status:** experimental, off by default. `--gad_weight 0` (default) is
bit-for-bit the MSE-only head.

> **Turbo GAD removed (2026-06-10).** GAD was also wired into the turbo / DP-DMD
> distillation (initial-noise sensitivity), folded into the DMD2 surrogate. An A/B
> showed no measurable benefit, so the turbo instantiation (`[gad]` /
> `--gad_weight` / `--gad_h` in `configs/methods/turbo.toml` +
> `scripts/distill_turbo/`) was removed. DP-DMD's first-step teacher anchor already
> carries diversity; the local JVP shield didn't add to it. The
> `base_loss="dmd"` + `dmd.grad_step` rollout-grad machinery is unrelated to GAD
> and stays — it's documented in `docs/experimental/dpdmd.md`.

## What GAD is

Standard distillation objectives are **pointwise** — they match the student's
output to the teacher's for each input independently. That flattens the
input→output landscape and destroys the distilled model's **sensitivity to the
input** (initial noise, or the conditioning): different inputs collapse to
near-identical outputs.

GAD adds a **first-order** term: match the student's *directional derivative*
w.r.t. the input to the teacher's. For a random perturbation `v`, the
finite-difference Jacobian-vector product

```
Φ(z + h·v) − Φ(z)            (response to a small input shift)
```

is matched between student and teacher. `L_total = L_base + λ·L_GAD`. It is a
plug-in regularizer — the paper instantiates it for output-matching (LADD),
DMD/SiD, and score-based distillation.

## GAD for mod-guidance (output-matching instantiation)

The deficiency was measured first and is real:
**`bench/mod_guidance/text_jacobian.py`** compares the distilled head's local
text response to the teacher's, per σ, on held-out pairs. On the 0602 baseline
head (probed on its training distribution) it found `cos(ΔS, ΔT) ≈ 0` at *every*
σ — within ~1 SE of zero, orthogonal not merely degraded — while the magnitude
ratio `‖ΔS‖/‖ΔT‖` collapses `0.83 → 0.05` from σ=0.1 to σ=0.9 (the head ignores
text exactly where the teacher leans on it most). Full write-up + table:
`docs/findings/mod_guidance_text_derivative_orthogonal.md`. This is the textbook
precondition for a first-order term: outputs match pointwise, the derivative is
unconstrained.

So `distill_mod` carries an **off-by-default** GAD term (the LADD-style
*output-matching* instantiation — there's no DMD/critic surrogate here, so it is
simply "also match the teacher's finite-difference response to a text change"):

```
L = L_mse + λ · L_gad
ΔT = teacher(cross_pert) − teacher(cross_A)       # no_grad, via cross-attn
ΔS = student(uncond, pool_pert) − student(uncond, pool_A)   # via the modulation MLP
L_gad = MSE(ΔS, ΔT)            # or 1 − cos(ΔS, ΔT)
```

`(cross_pert, pool_pert)` is another sample's text (the perturbation direction),
optionally scaled by `--gad_h` (`1.0` = full prompt swap, best SNR since the text
signal is ~2%). Where it lives: `scripts/distill_mod/distill.py` (insert after the
base-loss block) + `scripts/distill_mod/config.py`. Flags (all default to the
reproduction-exact off state):

| flag | default | meaning |
|---|---|---|
| `--gad_weight` | `0.0` | λ; `0` = off → bit-for-bit the MSE-only head |
| `--gad_h` | `1.0` | text-perturbation scale (`1.0` = full A→B swap) |
| `--gad_loss` | `l2` | `l2` (paper-faithful; also penalizes the magnitude gap) \| `cosine` (direction-only A/B lever) |
| `--gad_pair_source` | `auto` | `auto` → batch-roll if B>1 else dataset-random; `batch`; `dataset` |

**Cheap seam.** GAD adds +1 grad student forward (`pool_pert`) and +1 `no_grad`
teacher forward (`cross_pert`); `student(uncond, pool_A)` and `teacher(cross_A)`
are already computed for `L_mse`. A **two-phase backward** keeps peak VRAM at
*one* student graph (the base term is back-propagated first, freeing
`student_pred`'s activations before the perturbed student graph is built), so it
fits without `--grad_ckpt`. The trade: GAD's gradient flows through the perturbed
(B) endpoint only — `student(A)` is a detached constant. The target is then
`student(B) → student(A) + ΔT`, i.e. the head's `student − teacher` **residual
must not depend on the text**; since `L_mse` already drives the A-residual to ~0,
GAD pulls the B-residual the same way (synergistic, not competing). A
strictly-symmetric both-endpoints gradient would hold two student graphs and need
`--grad_ckpt`.

**RESOLVED 2026-06-05 — architectural ceiling confirmed; ship `gad_weight=0`.**
A σ-FiLM head (`pooled_text_proj.safetensors`, 1500 iters, `gad_weight=1.0
gad_loss=l2`, trained on synth) was probed head-to-head against the 0602 baseline,
**both on synth** (`bench/mod_guidance/results/20260605-1620-sigma-film-dcac-synth`
and `…-1627-0602-dcac-synth`). By the decision rule it **fails on both axes**:
`cos` stays ~0 at every σ, and the high-σ `ratio` does *not* rise (σ=0.9: 0602 0.049
→ σ-FiLM 0.056). σ-FiLM is a **no-op** vs the plain head — including on its own
stated target (the magnitude collapse).

The *why* is now measured, not inferred. `text_jacobian.py` was extended with a
**DAVE DC/AC decomposition** of the response deltas (DC = per-channel spatial mean,
AC = residual). The teacher's text response is **~99% AC** (`dT_ac_frac` 0.997→0.967,
identical across heads since `dT` is head-independent), so the hard ceiling for *any*
AdaLN-modulation head — `cos_ceiling = √(DC frac)` — is just **0.05 (low σ) → 0.17
(high σ)**. The full `cos` sits at that ceiling. AdaLN `shift` injects a spatially-
*uniform* per-channel constant (pure DC) and `scale`/`gate` only rescale the AC that
cross-attn already wrote — so the head structurally cannot synthesize the *new*
spatial structure a text change demands. The magnitude collapse was a *symptom* of
this directional ceiling (GAD-l2 has no incentive to grow ‖ΔS‖ in a direction
orthogonal to ΔT), which is why σ-FiLM's per-σ magnitude knob did nothing.

**Takeaway:** mod-guidance via AdaLN is a global-tone/contrast lever, not a content
lever — content lives in the AC, which is cross-attention's job. Reaching it needs an
AC-writing route (mini-cross-attn / pooled-text-gated *spatial* LoRA), i.e. abandoning
the pure-AdaLN premise. Don't keep tuning the head (σ-FiLM, more steps, gad_h, or
retargeting GAD to `dT_DC` — the DC piece is 0.3–5% of the response *and* unaligned).
Evidence: the finding doc
(`docs/findings/mod_guidance_text_derivative_orthogonal.md`) + the per-σ DC/AC
tables in the two `result.json`s above.
