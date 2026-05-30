# Soft-Tokens soft-rank — native-gradient listwise ranking objective

**Status:** Proposal. Tier-A gradient-quality probe **GO** (`bench/soft_tokens_contrastive/gradient_quality_probe.py`, run `results/20260530-1344-n12s3`); Tier-B live A/B not yet run. Nothing wired into the training path.

**Reference:** SoftSort (Prillo & Eisenschlos, ICML 2020) / softtorch (a-paulus, Apache-2.0, arXiv:2603.08824) for the differentiable rank relaxation. Live method deep-dive: `docs/experimental/soft_tokens.md`. Sibling of the AGSM and InfoNCE objectives (proposals archived under `_archive/proposals/soft_tokens_agsm.md` / `_archive/proposals/soft_tokens_contrastive.md`) — same extra-forward plumbing, different loss head.

## TL;DR

The shipped AGSM objective computes a Plackett–Luce ranking of the candidate
captions and **`.detach()`-es it** into a frozen shifted-MSE target — the
gradient never flows *through* the ordering, only through an MSE pull toward a
constant. This proposal replaces that detached-target regression with a
**differentiable listwise ranking loss** (`soft_rank` of the matched caption among
the candidates), so the gradient flows through the soft ordering directly. A
frozen-DiT probe shows the native gradient is **3.5× better aligned** with the
actual ranking objective (cos 0.86 vs 0.25) at the **same** boundedness budget.

## The bracket (why this is the missing corner)

The three objectives sit at the corners of a direction × boundedness trade:

| objective | gradient through ranking? | bounded? | shipped |
|---|---|---|---|
| **InfoNCE** | yes (softmax-CE = soft top-1) | **no** — unbounded negative push | yes |
| **AGSM** | **no** — ranking detached into MSE target | yes (self-annealing Δ) | yes (default) |
| **soft-rank** | **yes** | **yes** (sigmoid saturation) | *this proposal* |

AGSM was adopted precisely to bound InfoNCE's blow-up, but it bought boundedness
by throwing the ranking gradient away. soft-rank is the corner that keeps both.

## The objective

Per step, for anchor latent `x0` at σ, matched caption `c⁺` and `k` negatives:

```
v̂_j = v_θ(x_t, c_j)            # one DiT forward per candidate (j = +, 1..k)
r_j  = −‖v̂_j − v_target‖²      # per-candidate FM-error reward (higher = better)
```

The loss is the **differentiable rank of the matched candidate** among all
candidates (0 = best), pushed toward 0:

```
L = soft_rank(r)[matched]  =  Σ_{j≠+} sigmoid( (r_j − r⁺) / τ )
```

`soft_rank` is the pairwise-sigmoid relaxation of the rank operator (the ~20-line
core of softtorch `st.rank`, vendored as `bench/soft_tokens_contrastive/_softrank.py`).
Gradient flows through every `r_j` — i.e. through the ordering. Two properties
that matter:

- **Bounded by construction.** `∂sigmoid/∂· ≤ ¼` and saturates: as the matched
  caption loses badly (`r⁺ ≪ r_j`) the sigmoid pins at 1 and the gradient → 0.
  No EMA shadow, no γ damping — the relaxation is its own regularizer.
- **Self-annealing.** As the matched caption wins (`r⁺ ≫ r_j`), `L → 0` and the
  gradient → 0; the objective relaxes to plain FM at the fixed point, the same
  bounded behaviour AGSM engineers with its Δ → 0 baseline.

No detach anywhere on the ranking; `v_target` is the only constant.

## Tier-A evidence (gradient-quality probe, frozen base DiT)

`bench/soft_tokens_contrastive/gradient_quality_probe.py` makes the candidate
velocities `V = [v⁺, v⁻₁…v⁻_k]` a leaf and compares `∂L/∂V` for each objective
against `ideal = ∂(−margin)/∂V`, the gradient of the probe's own goal metric
(`margin = r⁺ − max_j r⁻_j`). On the same `V`, so `∂V/∂ψ` cancels and the
comparison is splice-independent. Run `20260530-1344-n12s3` (7 surviving anchors
× 3 seeds × 6 σ; 5/12 anchors skipped for thin hard-negative pool):

**Alignment `cos(g_obj, g_ideal)` — soft-rank points where the margin improves:**

| σ band ≥0.45 | agsm | infonce | soft-rank |
|---|---|---|---|
| | **0.248** | 0.859 | **0.858** |

**Boundedness — ‖g_matched‖ as the matched error is scaled up:**

| scale | agsm | infonce | soft-rank |
|---|---|---|---|
| 1.0× | 0.0011 | 0.0015 | 0.0011 |
| 4.0× | 0.0013 | **0.0087** | **0.0013** |
| tail slope | 0.0000 | 0.0024 | 0.0000 |

InfoNCE blows up (the AGSM-motivating failure, reproduced live); soft-rank peaks
at 2× then falls back — bounded on both ends. Slope ratio 0.015.

**Honest framing of the GO.** soft-rank does **not** beat InfoNCE on alignment
(0.858 vs 0.859 — identical). The entire value of soft-rank over just using
InfoNCE is the boundedness. The win over AGSM is the alignment (0.86 vs 0.25) at
no boundedness cost. Both deltas are real; neither is "best at everything."

**Two nulls, reported straight (do not claim these as wins):**

- **Near-miss credit:** all three objectives sit at chance (0.498 ≈ 1/k) for
  gradient mass on the binding competitor. The hypothesized advantage did not
  materialize at k=2.
- **Self-anneal correlation** (metric 4): weak positive, σ-confounded, a weak
  instrument. The boundedness sweep is the clean self-anneal evidence, not this.

## Why it's the right shape for Anima

The same reasoning that justified AGSM applies — the objective only needs the
*relative ordering* of captions for the same noised latent to be informative,
which `reward_premise_probe` confirmed (rank@1 0.71 on the σ≥0.45 band,
[[project_agsm_reward_premise_holds]]) even though absolute FM-MSE is
uninformative ([[project_fm_val_loss_uninformative]]). soft-rank consumes exactly
that ordering signal, and consumes it through the gradient rather than around it.

Bonus: soft-rank needs **no EMA value passes** (its boundedness comes from the
sigmoid, not from a lagged bank), so it costs `k` extra forwards/firing-step —
the same as InfoNCE and *cheaper* than AGSM's ~(2k+1).

## Migration plan (phased — add, A/B, then flip default)

This is a migration, not a rip-and-replace: AGSM stays loadable for existing
checkpoints throughout.

1. **Phase 1 — wire `contrastive_objective="softrank"`** as a third branch in
   `networks/methods/soft_tokens.py`, reusing all existing plumbing (negatives,
   warmup, `contrastive_every_n`, the grad-cache split, `after_backward`). Only
   the `w → target → loss` block changes:
   - Skip the EMA value passes (not needed); run the `k` negative forwards live.
   - **Anchor gradient (rides primary backward):** `soft_rank` loss on
     `stack(v⁺_live, v⁻.detach())` — the partial ∂L/∂v⁺ holding negatives fixed.
   - **Negative gradient (grad-cached + replayed in `after_backward`):**
     `g_neg = ∂/∂v⁻ soft_rank(stack(v⁺.detach(), v⁻_leaf))`, cached and replayed
     exactly as AGSM caches `g_neg` today. The split is exact because the two
     partials are w.r.t. disjoint variables.
   - This reuses `_build_gradcache` / `after_backward` verbatim — the only new
     code is the `soft_rank` loss function (move `_softrank.py` from `bench/` into
     the network module or `library/training/`).
2. **Phase 2 — Tier-B live A/B.** Three runs from identical seed/init, everything
   else fixed: plain-FM / `agsm` / `softrank`. Evaluate with the signals we trust:
   - `reward_premise_probe --adapter <ckpt>` → does **rank@1 / margin** move where
     AGSM's `w_matched` stalls?
   - **CMMD** val ([[project_cmmd_val_signal]]).
   - **Read the sample grids** — FM-val and pooled-cosine are pose-blind.
3. **Phase 3 — flip default** to `softrank` in `configs/methods/soft_tokens.toml`
   only if Phase 2 wins on CMMD *and* rank@1. Keep `agsm` selectable; mark it the
   fallback in the config comment.

## Risks & kill conditions

- **Gradient quality ≠ signal quality.** Tier-A proves the gradient is better, not
  that the trained bank is. If Phase-2 CMMD/rank@1 don't move, the bottleneck is
  the reward premise / char-tag coverage
  ([[project_soft_tokens_hard_negative_untagged]]), not the gradient — stop and
  keep AGSM.
- **New τ knob.** soft-rank's sigmoid temperature controls near-miss spreading and
  is *not* the same quantity as the PL τ. Start by reusing `contrastive_tau`; add
  a separate `softrank_temp` only if a sweep shows it matters. Per
  [[project_network_kwarg_toml_allowlist]], any new net kwarg must be registered in
  `networks/__init__.py` `*_KWARG_FLAGS` or it is silently dropped.
- **Block-swap interaction.** The grad-cache exists because the offloader desyncs
  on a 2nd DiT forward+backward ([[project_blockswap_extra_forwards_gradcache]]).
  soft-rank reuses the same cached-replay path, so it inherits the fix — but the
  Tier-B A/B must run at the production `blocks_to_swap` to confirm.
- **Smaller forward count is a behaviour change**, not just a speedup: dropping the
  EMA passes means soft-rank reads live (not lagged) rewards. That is intended
  (native gradient), but it means a `softrank` run is not a drop-in numerical match
  to an `agsm` run — the A/B is the only valid comparison.

## Files

- `bench/soft_tokens_contrastive/gradient_quality_probe.py` — Tier-A probe (built).
- `bench/soft_tokens_contrastive/_softrank.py` — vendored `soft_rank`/`soft_sort`
  (built; promote into the network module on Phase 1).
- `networks/methods/soft_tokens.py` — `extra_forwards` / `after_backward`; add the
  `softrank` branch alongside `agsm`/`infonce`.
- `configs/methods/soft_tokens.toml` — `contrastive_objective` option + Phase-3
  default flip.

## Bench

```bash
# Tier-A (built): gradient quality, no training
uv run python -m bench.soft_tokens_contrastive.gradient_quality_probe \
    --num_samples 24 --num_seeds 3 --contrastive_k 2
# Tier-A on the kill-gate pool (more anchors survive)
uv run python -m bench.soft_tokens_contrastive.gradient_quality_probe \
    --negative_mode shuffled --num_samples 32
# Tier-B (Phase 2, after wiring): live A/B + reward-premise on the trained bank
uv run python -m bench.soft_tokens_contrastive.reward_premise_probe \
    --adapter output/ckpt/anima_soft_tokens_softrank.safetensors --label softrank
```
