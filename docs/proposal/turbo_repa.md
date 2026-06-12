# Turbo × REPA — relational alignment for the DP-DMD student

Status: **proposal — not started**. Depends on REPA v2 Phase 0 (closed
2026-06-12, relational/Gram arm won — `docs/experimental/repa.md`) and the
turbo feature-tap API (landed — `forward_mini_train_dit(...,
return_block_features=, return_features_early=)`, bit-exact no-op when off).
Sibling proposals on the same loop: `turbo_caption_ranking.md` (Phase 1
wired, soft-rank at the step-0 diversity site) and `turbo_gan.md` (Idea 1
implemented, off by default).

## Why now

Three lines converge:

1. **The student's representation measurably degrades.** The caption-ranking
   probe (Phase 0 of `turbo_caption_ranking.md`) showed the DP-DMD student
   loses caption discriminability at the step-0 state: shuffled rank@1
   **0.750 vs base 0.958** at σ=1.0. That is *text-side* evidence that
   distillation pulls the student's mid-block representation off the base
   manifold. Soft-rank attacks the text axis directly; nothing currently
   anchors the **visual** axis.
2. **REPA is validated on this model.** Phase 0 + the quarter-run readouts
   established the mechanism end to end on the LoRA family: relational Gram
   vs PE-Spatial at block 8, weight 0.05 ≈ 4% of total loss, align curve
   plateaus by ~18% of a run, decisively better grids. The same machinery
   (grid match, spatial_norm, the Gram form's no-head property) transplants.
3. **The loop already has everything within reach.** The distill batch
   carries real cached latents + matched captions (used by the diversity
   anchor and mean-var calibration); PE-Spatial sidecars exist for the same
   cache; and the feature-tap API can capture student block-8 features with
   an early exit (~9/28 of a forward) — no hooks, no second graph.

Positioning vs the GAN head (`turbo_gan.md` Idea 1): the discriminator puts
*adversarial, distribution-level* pressure on teacher features of
**generated** samples. REPA is *direct, per-sample* alignment on **real**
data. Different data, different failure modes, compose freely. Together with
soft-rank the three span text / visual / distribution axes of the same
degradation.

## Mechanism

Every `every_n`-th student step, one extra **partial** student forward on
noisy real data:

```
x_τ   = (1 − τ)·latents_real + τ·ε            # the loop's renoise primitive
feats = _forward("student", x_τ, τ, c_matched,
                 return_block_features={repa.layer},   # default 8
                 return_features_early=True)            # exit after block 8
L_repa = GramMSE(pool(feats), spatial_norm(PE_sidecar))  # relational, fp32
loss_student += repa.weight * L_repa
```

- **Relational (Gram) form only** — no projection head on the student, no new
  params, dimensions never need to match (`library/training/repa.py`
  `extra_forwards` is the reference implementation of the pooling + Gram;
  factor the math into a reusable helper rather than copying).
- **Backward site**: the term joins the student loss at the DMD-phase
  accumulation (where the GAN generator loss lands, `distill.py:~1069`) —
  NOT inside the step-0 `split_bwd` diversity graph. The load-bearing
  detach-after-first structure stays untouched.
- **σ choice**: sample τ from the same renoise distribution the DMD path
  uses, so the alignment pressure lives on the student's actual operating
  inputs. Per-step-expert students (`step_expert_K > 1`) select head
  k = nearest student-grid step for the sampled τ.
- **Cost**: `every_n = 4` ⇒ ~0.08 extra partial forwards per step amortized
  (~9/28 depth, grad-bearing but early-exited) — small next to the existing
  2·k_anchor teacher + N student + 4 fake forwards per step.

**Bespoke-loop caveat (load-bearing)**: none of train.py's REPA
infrastructure reaches `scripts/distill_turbo/` — the dataset PE gate
(`dataset.load_repa_pe`), the adapter dispatch, and the step metrics all have
to be mirrored explicitly in the distill loop (the established
bespoke-loop-mirroring pattern). Concretely: set `load_repa_pe` on the
distill `CachedDataset`, extend the batch tuple, and log
`repa/align_loss` + `repa/active` to the loop's own metric stream.

## Phase 0 — drift probe (no training, the gate for everything else)

`bench/turbo_repa/probe_alignment_drift.py`: measure the unweighted
relational align loss (block 8 vs PE-Spatial, spatial_norm on) for **base**
vs the **existing trained turbo student** checkpoint, over real renoised
latents at σ ∈ {1.0, 0.97, 0.75, 0.5, 0.25} (matching the caption-probe σ
grid), N ≥ 200 images, fixed seed. Envelope per `bench/_common.py`.

Pre-registered readout:

- **Student align loss clearly worse than base at σ ≥ 0.75** (where the
  ranking gap lives) → the premise holds, proceed to Phase 1.
- **Student ≈ base everywhere** → CLOSE the line without a training run: the
  representation drift isn't PE-visible, the caption degradation is a
  text-conditioning problem, and soft-rank alone owns the fix.

This mirrors how the heatmap diagnostic closed REPA lever 3 in one
measurement — buy the answer before buying the training runs.

## Phase 1 — wiring + A/B (gated on Phase 0)

New `[repa]` section in `configs/methods/turbo.toml`, following the `[gan]`
pattern (**`weight = 0.0` default ⇒ byte-identical training**):

| key | default | note |
|---|---|---|
| `weight` | `0.0` | off → byte-identical; first live value 0.05 (LoRA-validated scale) |
| `layer` | `8` | block tap (matches LoRA REPA) |
| `encoder` | `"pe_spatial"` | sidecar `{stem}_anima_pe_spatial.safetensors` next to the TE cache |
| `every_n` | `4` | amortization (soft-rank convention) |
| `spatial_norm` | `true` | iREPA target standardization (eyeball-validated on LoRA) |

A/B at fixed seed/data/iterations, `--infer_steps 4 --cfg 1.0`,
`weight = 0` vs `0.05` (one order-of-magnitude sweep only if directionally
right). Pre-registered gates, in order:

1. **Caption-ranking probe** (`bench/dpdmd/caption_ranking_probe.py`) —
   shuffled rank@1 at σ=0.97/1.0 must not regress; recovery toward base is
   the win condition (same metric contract as soft-rank Phase 1, so the two
   levers are directly comparable and their composition measurable).
2. **CMMD** non-regressing.
3. **Grids** — no diversity collapse (the DP in DP-DMD is the point of the
   method; an alignment term that collapses diversity fails regardless of
   probe gains), anatomy intact.

If both this and soft-rank individually pass, a third composed run (both on)
decides whether they stack.

## Secondary arm (fallback only)

If real-data PE loading proves awkward in the distill loop, a cheaper
variant exists: relational alignment of student block-8 Grams to **teacher**
block-8 Grams at the GAN tap (`x_renoised`, generated data — the teacher
feature capture at `distill.py:~1021` already exists). No PE cache, no
dataset change. It is, however, a proximal-to-base pressure on generated
states (teacher features at few-step states can be off-manifold, and it may
directly fight the distillation update), so it is the fallback, not the
primary. Don't run both arms blind — pick by Phase-0's σ profile.

## Non-goals

- **Absolute / MLP-head arm** — closed by REPA Phase 0 + iREPA.
- **REPA on the fake critic** — the critic must track the student's current
  distribution, not real data; anchoring it breaks the DMD score estimate.
- **Input-token masking** — same constant-token-bucketing constraint as the
  LoRA family.
- **Token-subset Gram** — lever 3 closed ~uniform on the LoRA family; if it
  is ever revisited here, the `repa_grad_heatmap` machinery is reusable.

## References

- `docs/experimental/repa.md` — validated mechanism + operating numbers;
  §"Annealing plan" carries the live lever status (anneal open, candidate
  0.25; spatial_norm on; lever 3 closed by heatmap). Historical plan archived
  at `_archive/proposals/repa_phase1_operating_point.md`.
- `docs/proposal/turbo_caption_ranking.md` — degradation evidence + the
  shared Phase-1 gate contract.
- `docs/proposal/turbo_gan.md` — feature-tap API + the adversarial sibling.
- `docs/experimental/dpdmd.md`, `docs/structure/dpdmd.md` — loop structure.
- `library/training/repa.py` — Gram/pooling reference implementation.
