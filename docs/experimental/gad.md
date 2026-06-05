# Geometry-Aware Distillation (GAD)

**Paper:** Huang et al., *Restoring Initial Noise Sensitivity in Text-to-Image
Distillation via Geometric Alignment*, arXiv:2606.01651.
**Where it lives:**
- **Turbo** (initial-noise sensitivity): `scripts/distill_turbo/distill.py`
  (signal-assembly insert), `scripts/distill_turbo/config.py` (`base_loss` +
  `[gad]`), `configs/methods/turbo.toml`.
- **Mod-guidance** (text-direction sensitivity): `scripts/distill_mod/distill.py`
  (post-base-loss insert), `scripts/distill_mod/config.py` (`--gad_*` flags). See
  the dedicated section below — wired after `bench/mod_guidance/text_jacobian.py`
  confirmed the deficiency.
**Status:** experimental, off by default in both. Turbo: `base_loss="dpdmd"` +
`gad.weight=0` reproduces the shipped DP-DMD behavior bit-for-bit. Mod-guidance:
`--gad_weight 0` (default) is bit-for-bit the MSE-only head.

## What GAD is

Standard distillation objectives are **pointwise** — they match the student's
output to the teacher's for each input independently. That flattens the
input→output landscape and destroys the distilled model's **sensitivity to the
initial noise**: different seeds collapse to near-identical images, and any
downstream noise-based control (NoiseQuery retrieval, layout-via-initial-noise,
diversity modulation) goes dead.

GAD adds a **first-order** term: match the student's *directional derivative*
w.r.t. the input to the teacher's. For a random perturbation `v`, the
finite-difference Jacobian-vector product

```
Φ(z + h·v) − Φ(z)            (response to a small input shift)
```

is matched between student and teacher. `L_total = L_base + λ·L_GAD`. It is a
plug-in regularizer — paper instantiates it for output-matching (LADD), DMD/SiD,
and score-based distillation.

## Why it's interesting *here* (relative to DP-DMD)

DP-DMD already preserves diversity, but by a **different-order** mechanism: it
supervises the first rollout step toward the teacher's own diverse K-step anchor
(a zeroth-order, mode-fixing constraint) and detaches it. That couples diversity
to rollout depth — hence `student_steps >= 2`.

GAD's diversity mechanism is **rollout-free** (it matches score-field directional
derivatives at a single renoised timestep), so it composes with a **1-step**
student — something DP-DMD structurally cannot produce. The open question GAD
poses for turbo is whether a *local* Jacobian constraint can *replace* DP-DMD's
*global* anchor:

- **Replacement** (`base_loss="dmd"` + `student_steps=1` + `gad.weight>0`) — the
  cell that would be an outright win: a genuine 1-step student that still has
  noise sensitivity. The risk: local Jacobian matching may preserve "wiggle
  sensitivity" while still permitting *global* mode collapse (distant seeds
  landing on the same mode). Validate on **global** diversity (Vendi over
  multiple seeds + seed self-identifiability), not just local seed-ID.
- **Hybrid** (`base_loss="dpdmd"` + `gad.weight>0`) — keep the global anchor, add
  GAD as a local-geometry shield. The fallback if the replacement underperforms
  on global diversity.

## How it's wired (the cheap seam)

GAD does **not** fork the DMD loss. In the score/DMD setting the gradient is
already routed via the DMD2 surrogate `loss = (grad_signal · x_pred).mean()`,
where `grad_signal` is a **detached latent-space vector**. GAD's term is the
*same kind of object*, so it folds **additively** into `grad_signal`:

```
gad_signal = gad_weight · (1 − τ) · [ (v_real(x_t+h·v) − v_real(x_t))
                                     − (v_fake(x_t+h·v) − v_fake(x_t)) ]
grad_signal = grad_signal + gad_signal.detach()
```

- Same operand order as the DM signal (`v_real − v_fake`), so it inherits the
  repo's verified DMD sign convention. *If an A/B shows it anti-correlating with
  diversity, flip the sign — the ε-space vs velocity parametrization is the one
  place the sign could be off.*
- The per-sample weight is `(1 − τ)`, the exact jacobian of the repo's
  `renoise = (1−τ)·x_pred + τ·ε` — **not** the DM branch's `τ`-damping heuristic.
- `h` is absorbed into `gad_weight`; the paper's fixed `h = 1e-2` was best
  (their Table 9), so the schedule is left fixed.
- Cost: **+2 `no_grad` forwards** (perturbed teacher + fake), **zero extra
  backward graph**. Both perturbed forwards funnel through the existing
  `_forward(view, …)` helper, so nothing below that seam (the `TurboDMDNetwork`
  views, per-step heads, routing) changes.

**Caveat — hard routing.** A hard-routed fake/critic can route-flip across the
perturbation, making `Δv_fake` discontinuous and the finite difference garbage.
The teacher view is the frozen base DiT (always smooth), so only the critic side
is exposed. Plain-LoRA fake (the turbo default) is fine; keep `gad_h` small if you
ever hard-route the critic.

## `dmd.grad_step` — which rollout step carries gradient

`base_loss="dmd"` has no first-step anchor to detach, so the naive plain
multi-step DMD backprops through the **full N-step rollout** — the student
backward holds `N` forward graphs. At `student_steps=2` that's ≈2× the activation
memory of DP-DMD@2 (which frees step-0 via `detach_after_first`), and it OOMs a
16 GB card. `dmd.grad_step` selects which step(s) actually carry gradient; the
rest are **backward-simulated** under `no_grad` (the generator trains on inputs
from its *own* sampling trajectory — DMD2's train/inference input match, Yin et
al. 2024 — not forward-noised real latents).

| `grad_step` | Memory | What it supervises |
|---|---|---|
| `"all"` | `N` forward graphs (OOM-prone) | Full-rollout BPTT — every step grads into the endpoint `x_pred`. |
| `"last"` | **1 forward graph** | Only the final, cleanest-σ step. Memory-flat, but the noisy steps where the big denoise jumps live are never *directly* supervised — they improve only indirectly via the shared LoRA. |
| `"random"` | **1 forward graph** | **Canonical DMD2 multistep.** Each iteration samples `g~U{0..N-1}`, backward-simulates to `g` under `no_grad`, then grads **only** step `g`'s one-step x0-prediction `x_g − σ_g·v_g`. Spreads supervision over *every* grid point while staying memory-flat. |

For `student_steps >= 2` on a 16 GB card, prefer `"random"` (faithful) or
`"last"` (cheapest); `"all"` needs `--grad_ckpt` (~1.3–2× compute, activations
recomputed) or `student_steps=1` (the 1-deep replacement arm).

The DP-DMD step-0 diversity step is **exempt** — it keeps its own grad+detach
regardless, so under `dpdmd` `grad_step` only governs the DMD-refined steps
`1..N-1` and `"random"` is downgraded to `"last"` (randomizing would fight the
structural anchor; config emits a warning). For `dpdmd@2` it's a no-op (the
single DMD step is already the last).

**`per_step_expert`:** `"random"` and `"all"` train every up-head over time
(`"random"` trains the sampled step's head each iteration); `"last"` trains only
the final head and leaves heads `0..N-2` untouched (config emits a warning).

```toml
[dmd]
grad_step = "random"   # canonical DMD2 multistep; "last" / "all" also available
```

## Running the A/B/C matrix

```bash
make exp-turbo                                                              # A: DP-DMD@2 (baseline)
make exp-turbo ARGS="--base_loss dmd --student_steps 1 --gad_weight 1.0"    # B: replacement — 1-step + GAD
make exp-turbo ARGS="--base_loss dmd --student_steps 2 --gad_weight 1.0 --dmd_grad_step random"  # C: GAD at matched depth (memory-flat)
make exp-turbo ARGS="--gad_weight 0.5"                                      # D: hybrid (dpdmd@2 + GAD)
```

`gad_weight` is **untuned** — `1.0` is a starting guess; it competes directly with
the DM gradient in the same surrogate, so sweep it and watch `grad_signal_rms`.

**Decision rule:** score on **global** diversity (Vendi over ≥8 seeds/prompt +
seed self-identifiability), not just local seed-ID. If **B** holds Vendi vs **A**,
the replacement wins and you've bought a step. If **B** recovers seed-ID but lags
**A** on global diversity, fall back to the hybrid **D**.

## Does GAD transfer to other methods (soft-tokens, IP-Adapter, …)?

**No — not as-is.** GAD's prerequisites are (1) a teacher→student *distillation*
pair and (2) the student *compresses* the teacher's sampling trajectory such that
pointwise matching flattens its input→output geometry. That failure mode is
specific to few-step distillation.

- **Soft Tokens / IP-Adapter / EasyControl** — frozen-DiT *additive* methods (soft
  text tokens, decoupled image cross-attn, cond LoRA). No teacher pair, no
  trajectory compression, so no sensitivity collapse for GAD to repair. The JVP
  term has nothing to align against.
- **Spectrum / SPD** — training-free inference accelerators; there's no training
  loop to attach a regularizer to.
- **Mod-guidance distillation** (`make distill-mod`) is the one other genuine
  teacher→student fit in the repo (distills a `pooled_text_proj` MLP from
  teacher-synthetic data). GAD-style JVP matching is applicable there — and
  unlike turbo, the sensitivity at stake is **text-direction sensitivity in
  modulation space**, not initial-noise sensitivity (the inference steering
  `emb + w·delta` *is* a first-order text perturbation, so GAD shapes the exact
  Jacobian the steering rides on). **This is now wired up** (see below); it's the
  only non-turbo home GAD has.

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

So `distill_mod` now carries an **off-by-default** GAD term (the LADD-style
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

**Cheap seam (mirrors turbo's "+2 forwards, zero extra graph").** GAD adds +1 grad
student forward (`pool_pert`) and +1 `no_grad` teacher forward (`cross_pert`);
`student(uncond, pool_A)` and `teacher(cross_A)` are already computed for `L_mse`.
A **two-phase backward** keeps peak VRAM at *one* student graph (the base term is
back-propagated first, freeing `student_pred`'s activations before the perturbed
student graph is built), so it fits without `--grad_ckpt`. The trade: GAD's
gradient flows through the perturbed (B) endpoint only — `student(A)` is a
detached constant. The target is then `student(B) → student(A) + ΔT`, i.e. the
head's `student − teacher` **residual must not depend on the text**; since `L_mse`
already drives the A-residual to ~0, GAD pulls the B-residual the same way
(synergistic, not competing). A strictly-symmetric both-endpoints gradient would
hold two student graphs and need `--grad_ckpt`.

**RESOLVED 2026-06-05 — architectural ceiling confirmed; ship `gad_weight=0`.**
A σ-FiLM head (`pooled_text_proj.safetensors`, 1500 iters, `gad_weight=1.0
gad_loss=l2`, trained on synth) was probed head-to-head against the 0602 baseline,
**both on synth** (`bench/mod_guidance/results/20260605-1620-sigma-film-dcac-synth`
and `…-1627-0602-dcac-synth`). By the decision rule above it **fails on both axes**:
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
Plan + decision rule: `scripts/distill_mod/plan.md` (Phase 3); finding doc + the
per-σ DC/AC tables in the two `result.json`s above.

So GAD's homes in this codebase are **turbo** (initial-noise sensitivity under
trajectory compression) and **mod-guidance** (text-direction sensitivity under
output-matching); both off by default, and nowhere else.
