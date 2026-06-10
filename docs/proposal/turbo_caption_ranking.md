# Turbo prompt-following — caption-ranking probe, then (maybe) a soft-rank auxiliary

Status: **proposal / not started**. Probe-first: Phase 0 is measurement-only and can
kill the whole line for free.

Premise sources: `docs/findings/agsm_reward_premise_holds.md` (the validated ranking
reward), `docs/findings/turbo_fei_band_deficit_falsified.md` (the "measure at the
distribution the loss sees" lesson this design obeys), `docs/experimental/dpdmd.md` +
`docs/experimental/soft_tokens.md` (the two implementations being crossed).

## The question

**Does DP-DMD distillation preserve the teacher's caption discriminability — and if
not, can the soft-tokens soft-rank machinery restore it?**

Nothing currently measures turbo prompt-following. The val signal is CMMD
([[project_cmmd_val_signal]], distribution match — not text alignment) plus the
diversity probe; the known evaluation gap is "pose/prompt effects only show in
grids". Meanwhile distillation compresses a 28-step CFG=4 teacher into a 2-step
CFG-baked student, and prompt-following degradation is a classic few-step-distill
failure mode. We have a validated, model-intrinsic way to measure it and never
pointed it at a turbo student.

## Why the reward premise transfers unusually well here

`agsm_reward_premise_holds.md` established on the frozen base:

1. **Relative FM-ranking is a valid compass** even though absolute FM-MSE is
   uninformative — everything that pollutes the absolute number is held constant
   across candidate captions at a shared `(x_t, ε, t)` and cancels in the ranking.
2. **Discriminability grows monotonically with σ** and is perfect (hard-pool
   rank@1 = 1.000) for σ ≥ 0.45; the margin at σ=0.90 is ~40× the σ=0.15 margin.

The 2-step student operates **only** at σ ∈ {1.0, 0.75} (`student_sigmas`,
`scripts/distill_turbo/distill.py:685`, flow_shift=3). Its entire operating band sits
in the regime where the ranking reward is *most* trustworthy. That alignment is the
whole reason this proposal exists — the same idea aimed at a low-σ refiner would be
measuring noise.

Frozen reference numbers to compare against (run `20260529-1157-phase0-agsm`,
24 anchors, k=2, 2 seeds):

| arm | shuffled rank@1 | hard rank@1 |
|---|---|---|
| base, LoRA-off | 0.993 | 0.958 |

## Phase 0 — probe (no training, ~1 GPU-hour)

Extend `bench/soft_tokens_contrastive/reward_premise_probe.py` (or fork to
`bench/turbo/caption_ranking_probe.py` if the diff gets ugly) with three pieces:

### 0a. `--lora` arm (renoised-real ranking, student vs base)

The probe already builds `x_t = (1−σ)x0 + σε` from cached real latents and ranks
candidate captions by `−‖v(x_t, c_j) − v_target‖²`. Two changes:

- **Load the student.** `build_anima(args, adapter=<student.safetensors>)` — the
  turbo output is a plain LoRA, so the harness's standard adapter path
  (`library/runtime/harness.py:57`, `create_network_from_weights` → `apply_to` →
  `load_weights` → compile-after-apply) handles it. The probe currently passes
  `adapter=None` and hand-loads soft-tokens banks (`reward_premise_probe.py:280-300`);
  the LoRA arm is *simpler* than what's already there.
- **Extend the σ grid up.** `DEFAULT_SIGMAS` stops at 0.90; add 0.97 and 1.0
  (at σ=1.0, `x_t = ε` exactly — the model must rank purely from text, which is
  precisely the student's step-0 situation; `v_target = ε − x0` still carries the
  caption-specific answer). Read the student arm mainly at σ ∈ {0.75, 0.90, 0.97, 1.0}.

Output: student rank@1 / margins vs the base arm, same anchors, same seeds, same
negatives (shuffled + hard pools via the caption index).

### 0b. Anchor-ranking arm (on-trajectory — the `turbo_fei` lesson)

The falsified FEI lever died because Phase 0 measured a different distribution than
the loss saw. So measure ranking **at the exact state + target the diversity loss
trains**: from a fresh `ε`, roll the teacher CFG anchor `k_anchor` steps conditioned
on the *matched* caption (the `distill.py:810-820` construction, reproduced in the
probe) → `v_target = (ε − z_tk)/(1 − t_k)`. Candidates = matched + k negatives;
reward_j = `−‖v_student(ε, t=1, c_j) − v_target‖²`; rank@1 + margins as usual.

This is the trained quantity (`div_loss`) read out as a discrimination test. Cost
per anchor: `2·k_anchor` teacher forwards (CFG pair) + `(k+1)` student forwards.

### 0c. Caption-contrast transfer ratio (secondary scalar)

At shared states (`x_t` from 0a at σ=0.75/0.90, and the student's own `z1` from a
1-step rollout), compute the text-conditioning channel's gain directly:

```
contrast_S = v_S(x_t, c_pos) − v_S(x_t, c_neg)
contrast_T = v_T(x_t, c_pos) − v_T(x_t, c_neg)      # teacher CFG'd, matching what was distilled
ratio  = ‖contrast_S‖ / ‖contrast_T‖                 # kept gain
cosine = cos(contrast_S, contrast_T)                 # kept direction
```

This catches a failure 0a can't: a student that still *ranks* correctly but with a
collapsed contrast magnitude (text channel attenuated, prompts "work" but weakly).

### Gate (pre-registered, provisional numbers)

- **Degradation = any of:** student shuffled rank@1 < 0.93 at σ ∈ {0.75…1.0}
  (base: 0.993); hard rank@1 more than 0.05 below the base arm at matched σ;
  contrast ratio < 0.7 with cosine < 0.8.
- **No degradation → STOP and write the finding.** "DP-DMD preserves caption
  discriminability" is a publishable negative — it retires the prompt-following
  worry and Phase 1 never happens.
- Run against the current best student (and ideally one *over*-distilled checkpoint
  — [[project_turbo_alpha4_overdistill]] suggests over-baking is where text response
  would die first).

## Phase 1 — soft-rank auxiliary in `distill.py` (only if Phase 0 shows degradation)

### Why soft-rank (and not InfoNCE or AGSM)

The soft-tokens program already burned down this decision tree: AGSM was **removed
2026-05-30** (its `w_matched` pinned at chance all run); InfoNCE's negative branch is
unbounded (the SoftREPA degrade-while-loss-drops failure). Soft-rank won the live A/B
and the offline gradient probe (`cos(∂L/∂V, ∂margin/∂V) ≈ 0.86` vs AGSM's 0.25,
[[project_softrank_agsm_gradient_probe]]). It is bounded by construction
(`L → 0` as the matched caption's rank → 1, self-annealing at the win). Reuse
`bench/soft_tokens_contrastive/_softrank.py` / the `softtorch` dependency (already
present).

### Wiring (the real seams)

Site it at **step 0, the diversity site** — `distill.py:832-846`. Everything needed
is already in scope there:

- `v_target` (the matched-caption teacher anchor) is already computed — **free**.
- `v_first = v_student(ε, t=1, c_pos)` is already computed live — **free**.
- New per firing step: `k` extra student forwards `v_j = v_student(ε, t=1, c_neg_j)`
  under `no_grad`, same `(ε, t)`, only `crossattn_emb` swapped — the
  `extra_forwards` contract soft-tokens validated.

```python
r = stack([-mse(v_first, v_target), *(-mse(v_j, v_target) for j in negs)])
L_rank = soft_rank(r, method, softness)[0] - 1          # 0 when matched wins
loss_div_total = cfg.div_weight * div_loss_t + cfg.softrank_weight * L_rank
```

It rides the existing step-0 backward (`split_bwd` branch at `distill.py:842-847`
backwards the diversity term before the detach — the rank term joins that backward,
so the DMD graph separation is untouched). v0 detaches the negatives: gradient flows
through the live `v_first` only ("make the matched caption explain the anchor
better"), the bounded-objective property holds, and there is no second retained
graph. The soft-tokens `after_backward` grad-cache replay (push on the negatives
too) is a documented fidelity upgrade, not v0.

**Negative sourcing.** `CachedDataset` already yields per-sample `crossattn_emb`;
v0 negatives = TE tensors of other samples in the pool (`shuffled` — the validated
kill-gate mode). `hard` (same-artist / different-character via
`IdentityPairSampler` + `caption_index.json`) is a later knob; note its strict pool
covers only ~29% of stems.

### Config (`configs/methods/turbo.toml`, new `[softrank]` section)

| key | default | note |
|---|---|---|
| `weight` | `0.0` | **off → byte-identical training** (no negatives loaded, no extra forwards) |
| `k` | `2` | `softtorch` standardizes the candidate axis — needs k ≥ 2 |
| `every_n` | `4` | fire on every Nth student step; effective strength ≈ weight/N (manual, not auto-scaled — soft-tokens precedent) |
| `softness` / `method` | `0.1` / `neuralsort` | straight from the soft-tokens defaults |

### Cost & caveats

- `k=2, every_n=4` ≈ +0.5 no-grad student forwards per step amortized — small next
  to the existing `2·k_anchor` teacher + N student + 4 fake forwards.
- **Block swap:** extra DiT forwards are the offloader's audited-risk area
  ([[project_blockswap_extra_forwards_gradcache]]). The turbo default is
  `blocks_to_swap=0` and the loop already calls
  `prepare_block_swap_before_forward` per forward, but do not enable this together
  with swap without auditing.
- **σ=1.0 only.** The auxiliary trains ranking at the step-0 state. If Phase 0
  shows degradation concentrated at the σ=0.75 refinement state instead, the site
  moves (rank at `z1` against a teacher velocity referee) — different design,
  re-scope rather than stretch this one.

### Decision gate (Phase 1)

A/B at fixed seed/data/iterations, `--infer_steps 2 --cfg 1.0`:
`softrank.weight = 0` vs `0.05` (sweep one order of magnitude if the first point is
inert). Ship only on: Phase-0 probe metrics recover toward base **and** CMMD
non-regressing **and** grids show no diversity collapse (reuse `diversity.py`).
The probe is the regression metric for the thing being treated — re-run it on the
trained checkpoint, don't trust the training-time rank scalar alone.

## Sequencing

```
Phase 0 probe (0a + 0b + 0c, ~1 GPU-hour)
   ├─ no degradation ──► STOP, write docs/findings/ entry (negative result)
   └─ degradation ──► Phase 1 soft-rank auxiliary (off-by-default flag)
                          └─ Gate: probe recovery + CMMD + grids ──► ship / kill
```

## Contributing tier

Phase 0 is a bench probe (no trainer change). Phase 1 is numerics-changing →
**Tier 1.5**: bench script (the Phase-0 probe doubles as it) + invariant test
(`softrank.weight=0` byte-identical to current `distill.py`).

## References

- `docs/findings/agsm_reward_premise_holds.md` — the reward premise + σ-trend.
- `bench/soft_tokens_contrastive/reward_premise_probe.py` — the probe to extend;
  `_softrank.py` — the rank loss; `negative_audit.py` — hard-pool coverage.
- `scripts/distill_turbo/distill.py:685` (grids), `:805-846` (ε / anchor / step-0
  diversity site), `:842-847` (split backward the rank term joins).
- `docs/experimental/soft_tokens.md` §"Soft-rank objective" — objective history
  (AGSM removal, boundedness, knob semantics).
- `docs/findings/turbo_fei_band_deficit_falsified.md` — why Phase 0b measures
  on-trajectory; the probe/loss distribution-match rule.
- [[project_fm_val_loss_uninformative]] — why the reward must stay *relative*.
