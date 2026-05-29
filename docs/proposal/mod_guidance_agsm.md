# Mod Guidance — AGSM: a contrastive term on the modulation path

Status: **proposal** (2026-05-29, rev. 2026-05-29 — Phase-0 premise probe `--mod_agsm_probe`
built and wired into `scripts/distill_mod/distill.py`; gates whether Phase 1 is worth building). Builds on
`docs/methods/mod-guidance.md` and **directly reuses the soft-tokens contrastive
machinery** — negative sourcing, AGSM target-shift, EMA shadow — re-targeted from
the soft-token bank onto `pooled_text_proj`.

Reference: Lee, Hong, Kwon, Ye, *Alignment-Guided Score Matching for Text-to-Image
Alignment in Diffusion Models* (ICML 2026; https://jaayeon.github.io/AGSM/), in
`_archive/Alignment-Guided Score Matching…`. Mod guidance: Starodubcev et al., ICLR
2026, arXiv:2602.09268. Sibling proposal that built the machinery:
`docs/proposal/soft_tokens_agsm.md`.

## TL;DR

Mod guidance's **inference** mechanism is already a preference direction in
modulation space (`delta = proj(pool(p₊)) − proj(pool(p₋))`), but its **training** is
a single-pair distillation MSE — for each image, one caption, regress student
(text-through-modulation) onto teacher (text-through-cross-attention). The projection
is only ever taught to *faithfully carry* text; nothing teaches it to **discriminate**
captions in modulation space, which is the geometry the steering delta rides on.

**The proposal:** add the soft-tokens **contrastive/AGSM term** to the distillation,
re-targeted onto `pooled_text_proj`. Same image latent, the **matched caption** plus
**k real mismatched captions** (soft-tokens' existing cached-TE negatives), each
injected through the projection; the bounded AGSM target-shift trains the projection
so the matched caption's modulation is more on-manifold (lower FM-error) than the
mismatched ones. This makes the modulation path **text-discriminative**, which is
mod-guidance's whole reason to exist ("make AdaLN text-aware") — and the bet is that a
sharper, less-entangled modulation geometry gives cleaner steering directions, so the
per-block `w` schedule that exists to fight pink/DC collapse becomes optional rather
than load-bearing.

**No synthetic negatives.** An earlier draft fabricated `worst quality, score_1`
strings as negatives; that invents a distribution the model never sees and is dropped.
Negatives are real mismatched captions from the same plumbing soft tokens use
(`setup_contrastive_negatives` / `IdentityPairSampler`), so they are in-distribution
and B=1-safe.

## The mapping (mod guidance gives the AGSM structure for free)

AGSM's structure is *"same noised latent, several conditionings, rank by FM-error,
shift the target to prefer the matched one."* The distill loop already supplies it on
the exact pathway we want to shape:

| AGSM ingredient | Where it comes from |
|---|---|
| same `x_t`, several conditionings | one `noisy_input`, several pooled vectors through the **same** MLP via `forward_mini_train_dit(..., pooled_text_override=…)` (`distill.py:653`) |
| matched conditioning (D⁺) | the image's own caption pooled vector (`pooled_text`), already per-batch |
| negatives (D⁻) | **real mismatched captions** — soft-tokens' `setup_contrastive_negatives` / `_load_te_for_stem` cached-TE swap (`library/datasets/base.py`); pool each negative's `crossattn_emb` the same way the base path pools the matched one |
| reward `r(x_t,c)` | `−‖v_student(pool_c) − v_teacher‖²` — which injection best reproduces the gold teacher velocity |
| v_target | `teacher_pred` (already computed, `distill.py:619/628`) — the gold |
| PL weights / Δ / γ⁺/γ⁻ | `_agsm_pl_weights` / `agsm_targets` / `agsm_losses` (`soft_tokens.py`) — pure tensor ops, network-agnostic |
| EMA shadow | EMA of the ~8M-param MLP weights — same `decay·ema + (1−decay)·cur` |

**No dual ψ⁺/ψ⁻ bank** (soft-tokens §3a) is needed or possible: there is one MLP, and
matched/mismatched are different *inputs* to it, not different parameter banks. The
single open question that soft-tokens AGSM is still falsifying does not exist here.

### Loss

Keep the distillation MSE as the carry term; add the contrastive term on the same
batch (only the injected pooled vector differs, so the extra forwards are the same
`(k+1)×` soft tokens already pays):

```
L_carry  = ‖ v_student(pool(caption))    − v_teacher ‖²                  # unchanged distillation
L⁺       = ‖ v_student(pool(caption))    − (v_teacher + γ⁺·Δ⁺)   ‖²       # matched
L⁻ⱼ      = ‖ v_student(pool(neg_capⱼ))   − (v_teacher − γ⁻·Δ⁻ⱼ) ‖²       # k real mismatched
L        = L_carry + λ_pref · ( L⁺ + mean_j L⁻ⱼ )
```

with `Δ_j = v̂_ema_j − Σ_k w_k v̂_ema_k`, `w_j = softmax_j(−‖v̂_ema_j − v_teacher‖²/τ)`
exactly as in `agsm_targets`. ε→v is free (Anima is velocity FM, `v=ε−x₀`, `x₀`
constant — an ε-target shift is a v-target shift). Start `Ã(t)=1`. `λ_pref` and a
`--mod_agsm` switch default **off**, so the shipped MSE-only distillation stays the
default until the Phase-2 A/B clears.

## Phase-0 premise probe — folded into the warmup (`--mod_agsm_probe`)

An earlier draft argued the Phase-0 reward-premise probe was unnecessary here because
the soft-tokens version **already passed** it (rank@1 0.993 shuffled / 0.958 hard,
margin grows with σ; `[[project_agsm_reward_premise_holds]]`). **That transfer
argument was wrong**, and the probe is now wired (`scripts/distill_mod/distill.py
--mod_agsm_probe`, observation-only). Two things the soft-tokens probe measured differ
from what this method needs:

1. **Different reward target.** Soft tokens' reward anchors on the **ground-truth data
   velocity** `v = ε − x₀` (`soft_tokens.py:1022`): "which caption best explains the
   real image." Here the reward anchors on **`teacher_pred`** (the matched caption
   through cross-attention): "which pooled injection best reproduces the teacher." The
   held Phase-0 result is about the former; it does not cover the latter.
2. **Different channel.** Soft tokens routes conditioning through a per-layer crossattn
   splice that the **frozen** DiT already responds to strongly — and that frozen
   response is what the probe measured. Here the entire discriminative response is
   mediated by a **trainable, near-zero-init pooled → AdaLN MLP**. The signal is *not*
   a frozen-DiT property; it only exists once the carry MSE has trained the projection,
   and only to the extent the (narrow, entangled — `[[project_mod_guidance_quality_tag_axis]]`)
   pooled channel allows.

So *narrowness is the premise question*, not a separate Phase-2-only unknown: if the
trained pooled channel can't separate captions, the PL weight `w_matched` stays at
chance, `Δ → 0`, and the full term is a no-op — exactly the soft-tokens AGSM Phase-3b
failure signature (*"Δ rotates but w stays at chance"*, `[[project_soft_tokens_agsm_pl_correction]]`).

**The probe is cheap and folds into the run.** After the warmup ratio (default 0.1 —
by then the carry MSE has trained the projection), on each log step it reuses the
matched student forward and adds `k` no-grad forwards on mismatched pooled vectors (a
ring buffer of recently-seen pooled vectors; B=1-safe), then reads `w_matched` off the
**exact** `_agsm_pl_weights` math the real term would use. The loss is unchanged while
it runs. Decision rule:

- **`w_matched` hugs `1/(k+1)`** (0.33 at the default `k=2`) → channel too narrow, the
  full term will no-op → **revert, having paid ~nothing**.
- **`w_matched` climbs above chance** (and `reward_margin > 0`) → premise holds *live on
  the real pathway* (the thing the soft-tokens Phase-0 could not tell us) → wire the
  full gated term.

Mod guidance's distillation val loss stays a useful companion signal here:
`project_fm_val_loss_uninformative` is about generic data-FM-MSE, whereas the
distillation loss `MSE(student, teacher)` is a *reconstruction-of-the-gold-teacher*
metric, and on this method lower distillation val loss **has** tracked better samples
(operator-confirmed; also why the synth-pool was added to floor the real-vs-teacher
gap, mod-guidance.md §distill-prep Phase 2). But it is the companion, not the gate —
the gate is `w_matched`.

## Phasing — gates, cheapest-first

- **Phase 0 — premise probe, folded into a normal distill run (`--mod_agsm_probe`).**
  No loss change, ~`k` no-grad forwards on log steps after the warmup ratio. Watch
  `agsm_probe/w_matched` vs `1/(k+1)`. **GATE:** stays at chance → the pooled channel
  can't discriminate, the full term will no-op → stop here, don't build Phase 1.
  Above chance → proceed. (Built; see `scripts/distill_mod/distill.py`.)
- **Phase 1 — wire the contrastive term into distillation, single MLP.** Add the k
  negative-pool injections + EMA shadow + `agsm_targets`/`agsm_losses` call to
  `distill.py`'s loss (the `pooled_text_override` student forward already exists; the
  negative sourcing is the soft-tokens path). Knobs mirror soft tokens: `agsm_gamma`
  (γ⁺), `agsm_gamma_neg` (γ⁻, 0.1), `agsm_ema_decay` (0.99), PL τ, `k ∈ {1,2}`, plus a
  new `lambda_pref` and `--mod_agsm` (default off). Train against the synth pool
  (`--synth_data_dir`) so the real-vs-teacher gap doesn't confound the term. Watch the
  distillation val loss (informative here) doesn't regress.
- **Phase 2 — the A/B that decides ship-vs-revert.** AGSM-distilled projection vs the
  shipped MSE-only projection, **on the steering quality axis** (CMMD,
  `[[project_cmmd_val_signal]]`, + qualitative drift sweeps):
  - **Primary win condition — safer steering at uniform `w`.** Re-run the pink/DC
    collapse that forced the per-block schedule (the "channel" LoRA at `w=3`,
    mod-guidance.md §"Why schedule instead of uniform?"). If the contrastive-trained
    projection stays clean at `--mod_start_layer 0` (uniform) where the MSE-only one
    collapsed, the discrimination sharpening paid off and the schedule becomes optional.
  - **Secondary — CMMD quality-steering gain** at matched `w`, and the
    steering-direction-consistency metric (mod-guidance.md §"Quality direction
    consistency", 0.814 mean cosine) must not degrade.
  - **KILL → revert:** no uniform-`w` safety gain and no CMMD gain → the faithful-carry
    distillation already captured the modulation geometry well enough. Keep `--mod_agsm`
    off.

## Why this might help where it counts

The per-block `w` schedule exists because the learned quality direction is *entangled
with the early-block tonal-DC direction* — uniform `w` blows up the DC component (pink
collapse). That entanglement is the symptom of a modulation geometry trained only to
carry, never to discriminate. `[[project_mod_guidance_quality_tag_axis]]` already
showed the axis is real but **directionally double-counted** and the score ladder is a
*rotation*, not a clean axis — i.e. structured but entangled. A contrastive term that
pushes matched-caption modulation to higher likelihood than mismatched is a direct
attempt to disentangle that geometry. The σ-growth of the caption margin (Phase-0
finding) also lines up with mod guidance's own observation that the modulation path is
most sensitive at high noise (mod-guidance.md §"Modulation sensitivity") — AGSM's
natural band is where mod guidance has the most leverage.

## Costs to keep honest

- **Content-discrimination ≠ quality-steering (the real open risk).** Real mismatched
  captions sharpen *content* discrimination in the modulation path; the inference use
  is a *quality* direction. The bet is that a cleaner, more text-discriminative geometry
  yields cleaner steering — but that transfer is exactly what Phase 2 tests, not a given.
- **Double-count interaction.** `pooled_text_proj` feeds *both* the base modulation and
  the steering delta (`[[project_mod_guidance_quality_tag_axis]]`). Sharpening could
  amplify the base double-count; watch the steering-direction-consistency metric in
  Phase 2 — if it degrades, the term is over-rotating the base path.
- **Extra forwards.** `(k+1)×` student forwards/step + EMA value passes (same as
  soft-tokens AGSM). Teacher forward is cached (`teacher_cache_K`), unaffected. `k ∈ {1,2}`.
- **Redundancy risk.** Mod guidance ships and works; this is a *refinement bet*, not a
  rescue — it must earn its keep on uniform-`w` safety or CMMD, or revert.
- **EMA memory.** One shadow of the ~8M MLP. Negligible.

## What this does NOT do

- No external reward model, scorer, or teacher checkpoint — the reward is
  `−‖v_student − v_teacher‖²` off the frozen DiT (reward-free, in-house).
- No fabricated quality negatives — negatives are real mismatched captions from the
  soft-tokens plumbing.
- No dual ψ⁺/ψ⁻ bank — one MLP; matched/mismatched are inputs to it.
- No *separate* Phase-0 premise probe — but the premise **is** checked, folded into the
  warmup as `--mod_agsm_probe` (observation-only). The soft-tokens Phase-0 result does
  **not** transfer (different reward target: `teacher_pred` vs data-velocity; different
  channel: trainable pooled MLP vs frozen crossattn), so the probe re-measures it live.
- No inference-path change — the trained projection drops into the existing
  `--pooled_text_proj` / `--mod_w` / per-block surface unchanged (the *hypothesis* is
  the schedule becomes optional, not that the surface changes).

## Reference points

- Mod-guidance method + inference surface: `docs/methods/mod-guidance.md`
- Distillation loss site (where the term composes): `scripts/distill_mod/distill.py`
  (teacher forward `:628`, student forward w/ `pooled_text_override` `:653/658`, MSE
  loss `:662`)
- Phase-0 probe (built): `scripts/distill_mod/distill.py` `--mod_agsm_probe`
  / `--mod_agsm_probe_{warmup,k,tau}`; logs `agsm_probe/{w_matched,reward_margin,w_chance}`
- AGSM helpers to reuse (network-agnostic): `networks/methods/soft_tokens.py`
  (`_agsm_pl_weights`, `agsm_targets`, `agsm_losses`, `update_bank_ema`)
- Negative sourcing to reuse: `library/datasets/base.py::setup_contrastive_negatives`
  / `_load_te_for_stem`, `library/datasets/identity_pairs.py::IdentityPairSampler`
- Sibling proposal (built the machinery + ran Phase 0): `docs/proposal/soft_tokens_agsm.md`
- Context: `[[project_fm_val_loss_uninformative]]`, `[[project_cmmd_val_signal]]`,
  `[[project_mod_guidance_quality_tag_axis]]`, `[[project_agsm_reward_premise_holds]]`,
  `[[project_sigma_signal_resolves_by_045]]`
- Papers: AGSM (ICML 2026, https://jaayeon.github.io/AGSM/); Mod guidance (ICLR 2026,
  arXiv:2602.09268)
