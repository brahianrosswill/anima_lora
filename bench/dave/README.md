# DAVE — DC Attenuation for diVersity Enhancement (Anima port)

Training-free, representation-level intervention to recover same-prompt sample
diversity. Reconstructed from the abstract + teaser figure of an unrevealed
**ICML'26** paper (no code/weights public as of 2026-06-05) — treat the exact
mechanism as our best-faithful guess, not a spec.

## The idea

Text-to-image samples from one prompt look overly similar. DAVE's diagnosis:
the **DC component** of intermediate Transformer-block features — the spatial
average `μ^ℓ` of `h^ℓ` over the patch grid, per channel — converges across seeds,
pinning global layout and collapsing later variation. The fix is a per-block
output rewrite that attenuates the DC while leaving the AC residual intact:

```
ĥ^ℓ = α · μ^ℓ + (h^ℓ − μ^ℓ)        α < 1,  μ^ℓ = mean over (T,H,W)
```

The teaser figure pairs this with a **Power Ratio** (`‖μ‖²/‖h‖²`) gate — i.e. the
attenuation is meant to be *block-selective*, hitting only where the DC carries
energy. See the root `image.png` for the figure we worked from.

### How it maps onto Anima

- Anima blocks emit 5D `(B, T, H, W, D)`; the DC is `h.mean(dim=(1,2,3))`. The
  intervention is a **post-`forward` hook** (forward_hook-not-override invariant),
  branchless (`h − (1−α)·μ`, α=1 → exact no-op) so it survives `compile_blocks()`.
- It is a *representation* edit inside the forward, **not** a sampler-boundary
  correction like DCW/CNS/SMC — so it lives as a hook + per-step model buffer
  (the Spectrum/mod-guidance pattern), not in `library/inference/corrections/`'s
  post-step callback shape.

## Phase 0 — premise probe ✅ DONE (2026-06-05)

`probe_dc_convergence.py` — read-only. Generates the **`make test` prompt** across
N seeds (real `generate()`, model injected via `shared_models`, hooks on every
block), and per (step, block) measures cross-seed cosine similarity of the **DC**
vs the **AC residual `h−μ`** (pooled to GxG, DC-disjoint by construction), plus
the DC power ratio. Verdict restricted to **DC-heavy blocks** (early-step power
ratio ≥ 0.30) — block-averaging smears the bimodal per-block power structure.

**Question:** is the DC the shared/locked component while the AC carries the
seed-specific diversity? If DC sim ≈ AC sim even where the DC has energy, there's
nothing to unlock and we stop.

### Result — PASS (`results/20260605-1442/`, 8 seeds × 24 steps, CFG 4.0)

| early steps | DC sim | AC sim | gap |
|---|---|---|---|
| all blocks | **0.989** | 0.482 | +0.51 |
| DC-heavy blocks | **0.9993** | 0.611 | +0.39 |

The DC is near-perfectly cross-seed-shared (it's the conditioning); the AC
residual that holds seed structure sits at 0.48–0.61. DAVE's decomposition is
real on Anima — there is a shared, energetically-significant channel whose
attenuation would let the diverse AC breathe.

> ⚠️ **False start, kept as a lesson:** the first run (`results/20260605-1432/`)
> compared DC against a plain **coarse 2×2 pool**, which *contains* the DC. The
> gap collapsed to +0.022 and the auto-verdict printed `premise_holds: false`.
> It was a measurement artifact — DC and AC must be **disjoint** (`h−μ`) to be
> compared. Don't reintroduce a DC-containing AC proxy.

### What the probe settled about *where* to intervene

From the power-ratio heatmap + the DC−AC gap heatmap (`dc_minus_ac_gap.png`):

- **Block-selective, not global.** Strongest targets (high DC power ∩ high
  DC−AC gap) are the **late blocks ~19–27** and mid blocks ≥11.
- **Block 0 is a false target.** DC-heavy by power, but its AC is *also*
  cross-seed-locked (gap ≈ 0) — attenuating DC there perturbs a shared signal
  without unlocking anything. Exclude it.
- **Temporal story differs from the paper.** On Anima the DC is shared at *all*
  steps (sim ≈0.99 even at σ≈1.0), not "converging early." Its *energy* is small
  early (ratio 0.28) and grows late (0.63, σ<0.45). So the paper's "attenuate
  early" isn't obviously the lever here — the early-vs-late attenuation window is
  an **open knob**, not a settled default.

Artifacts per run: `result.json`, `convergence_curves.png` (all vs DC-heavy),
`power_ratio_heatmap.png`, `dc_minus_ac_gap.png`, `per_block.npz` (the (S,L)
matrices — reuse instead of re-running).

## Build-out (Phases 1–2, both done)

The probe localized the targets and the saved `per_block.npz` let us derive the
**per-block α schedule offline (no new GPU run)**: weight the attenuation by
`power_ratio × max(0, DC−AC gap)` so block 0 and AC-shared regions self-zero —
that *is* the figure's "Power Ratio" gate, fit from our own data. Phase 1 was that
desk exercise; Phase 2 wired the intervention and eyeballed it.

### Phase 1 — derive the per-block α mask ✅ DONE (2026-06-05)
`derive_alpha_mask.py` — offline, no GPU. Turns `per_block.npz` into a per-block
weight `w(ℓ) = power_ratio(ℓ) · max(0, DC−AC gap(ℓ))` (both averaged over the
early window), normalized to `[0, 1]`, shipped as
`networks/calibration/dave_alpha.npz`. The product self-zeros exactly the two
families DAVE must not touch — **block 0** (gap ≈ 0) and the **early-mid AC-shared
blocks 1–8** (~zero DC power). Sanity gates pass: block 0 → 0.026, peak at **block
22 (1.00)**, late blocks 19–27 dominate (0.59–1.00). The figure's "Power Ratio"
gate, fit from our own data.

At inference `--dave` reads `w(ℓ)` and forms the attenuation factor
`(1−α_ℓ) = strength · w(ℓ)`, so `--dave_strength` is a single live knob and the
self-zeroed blocks stay no-ops.

### Phase 2 — `--dave` intervention ✅ DONE (2026-06-05) — eyeballed, not bench'd
**Shipped** (the README design, faithfully): per-block **post-`forward` hook**
`h ← h − (1−α_ℓ)·μ` (`library/inference/corrections/dave.py`), branchless (α=1 →
no-op, survives `compile_blocks()`), + a per-step model buffer `_dave_cur_sigma`
restamped from the timestep so the hooks can σ-gate without seeing the step.
Standard loop only (no Spectrum/SPD compose), applied to both CFG passes. Knobs:
`--dave_strength`, `--dave_sigma_lo/hi`, `--dave_block_lo/hi`. Make lever:
`make test DAVE=1 [DAVE_STRENGTH= DAVE_SIGMA='lo,hi' DAVE_BLOCKS='lo,hi']`.

We chose to **eyeball** the diversity/quality tradeoff (`eyeball.py`, 2 seeds ×
{baseline, s0.05/s0.10 all-blocks, s0.10/s0.20 mid-only}) rather than build the
PE-Core diversity bench — there's no text tower wired for an honest "alignment
held constant" number, and the failure mode turned out to be visually obvious.

**Findings:**
- **The default must be small.** At `strength=0.6` the output collapses to a
  color-blown **grid** — the mask weights the *latest* blocks hardest, but there
  the DC carries 60–97% of the energy (it *is* the image content, not a removable
  shared-layout signal), so attenuating it across 9 blocks before `final_layer`
  goes fully off-manifold. Default is now **0.1**.
- **The knee is `s≈0.05–0.10` all-blocks.** `s0.05` is a subtle tone/background
  nudge; `s0.10` gives genuine **per-seed recomposition** (framing pulls back,
  layout changes) while staying coherent and keeping the "ANIMA" sign legible.
- **The cost is tonal flatness, not collapse.** Removing the global per-channel
  DC removes tonal/color mean, so quality degrades *gracefully* as **posterization**
  (flatter range, higher saturation) that scales with strength — not the grid.
- **Diversity lives in the late blocks, not the mid blocks.** Mid-only (9–18) is a
  weak lever: `s0.10` ≈ baseline, `s0.20` warms tone and **breaks the sign text
  before it recomposes**. This matches the probe (late blocks had the highest DC
  power ∩ DC−AC gap); the "mid is the safe lever" guess was wrong.

**Verdict — plausibly ports, with a caveat.** There is a usable regime (≈0.05–0.10
all-blocks) where composition diversifies at near-fixed quality, so DAVE is not a
no-go on Anima. But it is **not free**: above ~0.1 the global-DC removal trades
tonal richness for diversity. Open follow-up: **RMS-renormalize after the DC
subtraction** to conserve tonal energy (recolor-not-remove, à la CNS) — would test
whether the posterization is separable from the diversity gain.

## Spin-off — DAVE as a training diversity signal (turbo / DP-DMD)

The intervention is the less interesting half; the **diagnostic** (DC = seed-shared
conditioning, AC = seed-specific diversity) ports to training as a *measurement*.
DP-DMD's whole pitch is *diversity-preserved* distillation, but its in-loop `div`
loss only measures how close the student's first step lands to the teacher's
K-step anchor — it says nothing about whether the student's own same-prompt
samples have collapsed across seeds (the canonical DMD failure mode). The DAVE
cross-seed **AC sim** is exactly that mode-collapse detector.

Wired as a `validate_every_n_steps`-style pass in the turbo loop (mirrors
`distill_mod`'s validation): fix one **held-out** conditioning, roll the student's
N-step Euler grid across N seeds under `no_grad` (the live `_forward` /
`set_student_step` / `student_sigmas` primitives), hook every block, and log:

- `val/div_ac_sim` — cross-seed cosine of the AC residual; **lower = more diverse**
  (rising over training = collapse). The headline signal.
- `val/div_dc_sim` — reference; should stay high (~conditioning lock).
- `val/div_gap` (`dc−ac`) and `val/div_xpred_ac_sim` (same split on the final latent).
- `val/fm_mse` — flow-matching reconstruction MSE on the held-out sample (the
  **fidelity** half of the fidelity↔diversity tradeoff view). ⚠️ FM val loss has
  *not* tracked sample quality on Anima (CMMD replaced it as the quality signal),
  so read it as a divergence/sanity number paired against AC-sim, not a score.

The metric is layout-robust (DC = mean over all non-(batch,channel) axes), so it
reads correctly under both eager 5D blocks and the compiled native-flatten.

```bash
make exp-turbo  # then add to the turbo config or CLI:
#   --validate_every_n_steps 750 --val_diversity_seeds 8 [--val_prompt_idx N]
```

Code: `scripts/distill_turbo/diversity.py` (probe), `config.py` (`io.validate_every_n_steps`
/ `val_diversity_seeds` / `val_prompt_idx`), `distill.py` (held-out capture + in-loop call).
Status: wired + unit-smoke-tested (layout-invariance, cosine math, config round-trip);
**not yet exercised in a live distill run**.

## Files

| File | Role |
|---|---|
| `probe_dc_convergence.py` | Phase-0 premise probe (read-only block hooks). |
| `derive_alpha_mask.py` | Phase-1 offline mask: `per_block.npz` → `networks/calibration/dave_alpha.npz`. |
| `eyeball.py` | Phase-2 sweep: baseline vs DAVE configs × seeds → `output/tests/dave/`. |
| `results/<ts>/` | Standard `result.json` envelope + PNGs + `per_block.npz`. |

Intervention code lives outside `bench/`: `library/inference/corrections/dave.py`
(hooks), the `_dave_*` buffers in `library/anima/models.py`, `--dave*` flags in
`library/inference/args.py`, the `DAVE=1` lever in `scripts/tasks/inference.py`.
