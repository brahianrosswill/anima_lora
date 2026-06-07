# σ-reshape — does packing steps into the low-σ tail buy NFE efficiency?

**Status:** phase-2 of the dynamic-Spectrum line (decide-before-build). Not a shipped method.
**Owner question:** *`bench/dynamic_spectrum` proved a fixed **uniform** schedule is mistuned and that **σ alone** ranks the error — so the cheapest fix is a static σ-reshaped schedule, no head. Does that reshape actually improve image quality at fixed NFE?*

---

## Where this came from

`bench/dynamic_spectrum` (phase 1) measured the local **Euler step-fattening error** `e_i = (σ_{i+1} − σ_{i+2})·‖v_{i+1} − v_i‖` along real-caption trajectories and found:

- **Q1 — error is non-uniform:** p90/p10 ≈ 6.8×, ~**60% of the error mass sits in σ<0.45** (only 19% of the steps). A fixed *uniform* cadence is mistuned.
- **Q2/Q3 — σ is a near-sufficient statistic:** within-trajectory Spearman(σ, e_i) ≈ **0.876**; a trained head (0.768) and the free `‖v‖` heuristic (0.089, collapsed) both **lose to σ**. So **no learned head, no `‖v‖`-gate** — just reshape the fixed schedule by σ.

That points at the cheapest possible follow-up: **a static σ-reshaped step schedule** that places more knots in σ<0.45. This bench asks whether that reshape is worth shipping — i.e. whether it improves *output quality* at fixed NFE, which phase 1 did **not** measure (it scored ODE truncation error, not image quality, and `flow_shift` keeps steps at the noisy end for a reason — early structure formation).

---

## The knob

`library/inference/sampling.py::get_timesteps_sigmas(..., tail_power=p)`:

```
u = linspace(1, 0, n+1)            # uniform grid
if p != 1.0:  u = u ** p           # p>1 pulls knots toward σ=0 (the tail)
σ = shift·u / (1 + (shift−1)·u)    # flow-shift map (unchanged)
```

`p=1.0` is the canonical schedule **bit-for-bit** (verified). `p>1` densifies the low-σ resolve tail; `p<1` does the opposite. Endpoints σ∈{0,1} are fixed, so only the spacing moves. Fraction of steps with σ<0.45 at 12 NFE: **p=1.0 → 0.17, 1.5 → 0.33, 2.0 → 0.42, 2.5 → 0.50.**

Exposed as `--sigma_tail_power` (inference CLI) and `GenerationRequest.sigma_tail_power`. Default `1.0` everywhere → zero behaviour change unless opted in.

---

## What the bench measures

Euler, CFG=1. **Reference = a deep converged solve** (`--ref_steps`, p=1.0) per prompt/seed. The schedule's only job is to land the ODE at that same endpoint with fewer steps, so **distance-to-converged isolates discretization error** (the schedule's lever) from model quality. For each (`tail_power`, `NFE`) on the **same prompt/seed**, three distances to the converged ref (all ↓ = better; converged scores ~0 by construction):

| metric | what | why |
|---|---|---|
| `latent_endpoint` | mean ‖x_lowNFE − x_ref‖ / ‖x_ref‖ (paired) | the most direct discretization-error signal — free, no decode. **Not FM loss** (compares trajectory *endpoints*; FM loss is structurally blind to the sampling schedule). |
| `pe_cosine` | mean 1 − cos(PE(x_lowNFE), PE(x_ref)) (paired) | perceptual, same PE-Core space CMMD uses; the **decision metric** (paired → sensitive at small N). |
| `cmmd` | CMMD²(PE set low-NFE, PE set converged) | the repo's blessed quality metric, but **noisy at small N** — read alongside the paired two, not alone. |

Plus a **`grid.png` eyeball montage** (rows = prompts, cols = [converged, p=1.0, p=1.5, …] at the smallest NFE) so the number is sanity-checked by eye.

**Out of scope for this first read:** CFG>1 and er_sde. CFG×aspect is the documented follow-up *if* a single p wins here (cf. DCW, whose optimal correction moves with CFG/aspect — the σ-reshape optimum may too).

---

## Running it

```bash
# correctness smoke (~2 min)
uv run python -m bench.sigma_reshape.run_bench \
    --num_prompts 4 --ref_steps 40 --nfe 8 16 --tail_powers 1.0 2.0 --label smoke

# real read — real captions + block-compile (backgroundable)
uv run python -m bench.sigma_reshape.run_bench \
    --num_prompts 24 --ref_steps 100 --nfe 8 12 16 \
    --tail_powers 1.0 1.5 2.0 --compile --label reshape-realcap
```

Key flags: `--num_prompts`, `--ref_steps` (converged depth), `--nfe` (list of low budgets), `--tail_powers` (list; must include `1.0` for the baseline-beaten verdict), `--guidance_scale` (CFG, default 1.0), `--image_size H W`, `--compile`.

### Output (`results/<ts>-<label>/`)

- `result.json` — standard envelope; `metrics.table` (per config), `metrics.per_nfe_winner` (best p + improvement vs p=1.0 at each NFE), one-line `metrics.verdict`.
- `results.csv` — `tail_power, nfe, latent_endpoint, pe_cosine, cmmd`.
- `metrics.png` — the three metrics vs NFE, one line per `tail_power`.
- `grid.png` — eyeball montage at the smallest NFE.

### How to read it

- **`RESHAPE HELPS`** (a p>1 beats p=1.0 on `pe_cosine` at ≥½ the NFE budgets) → proceed to the CFG×aspect sweep, then ship a static (optionally CFG/aspect-indexed) schedule.
- **`NO RESHAPE WIN`** (p=1.0 best/tied everywhere) → the tail-densify doesn't improve the converged-target match; flow_shift's noisy-end emphasis already pays for itself. Stop — the dynamic_spectrum line is fully closed.
- **`MIXED`** → inspect the table + grid before building.

---

## Results — NO RESHAPE WIN (Euler, CFG=1)

`20260607-2106-reshape-realcap` — 24 real captions × 1024² × {100-step ref}, Euler, CFG=1.

**`pe_cosine` (decision metric, ↓ better):**

| NFE | p=1.0 (canonical) | p=1.5 | p=2.0 |
|---|---|---|---|
| 8  | **0.118** | 0.124 | 0.128 |
| 12 | **0.068** | 0.084 | 0.109 |
| 16 | **0.064** | 0.064 | 0.071 |

All three metrics agree and the trend is monotone: **densifying the σ<0.45 tail (p>1) strictly worsens the converged-target match** at every NFE (`latent_endpoint` and `cmmd` tell the identical story; the only blip is a marginal `latent_endpoint` tie at NFE=16 that the perceptual/distributional metrics don't echo). The canonical `flow_shift=3` schedule (p=1.0) is already the best step placement here.

### Why this doesn't contradict `dynamic_spectrum`

Phase 1 found ~60% of the **local Euler-fattening-error mass** in σ<0.45 and concluded a fixed *uniform* schedule is mistuned. This phase shows that **moving step budget into that tail hurts** — not a contradiction, a distinction between two different errors:

- Phase 1's `e_i` is a **within-trajectory, step-to-step** curvature signal — large in the tail because `v` is changing fast there.
- But the σ<0.45 tail is also where **`x0_pred` is already visually resolved** (memory `sigma_signal_resolves_by_045`). So that large local curvature is being spent refining an *already-converged* image, while the steps it steals come from the **high-σ structure-forming region the global endpoint actually depends on**. Local fattening error ≠ global endpoint sensitivity to budget reallocation.

In short: `flow_shift`'s noisy-end emphasis is load-bearing, exactly the caveat flagged before building this. **The dynamic-Spectrum line closes** — phase 1 (σ ranks local error, no head/`‖v‖`-gate) and phase 2 (but don't reshape toward the tail) together say the shipped schedule is already well-placed.

**The one untested escape hatch:** CFG>1 and/or non-square aspect. Per the DCW lesson the optimum can move with CFG/aspect, so this CFG=1/square verdict isn't proof the reshape loses *everywhere* — but the burden is now on that hypothesis. Re-run with `--guidance_scale 4` and a non-square `--image_size` if pursuing it.

> **er_sde + CFG=4 were checked as escape hatches — both close the line.** On stochastic `er_sde` the direction *reverses* at **CFG=1** (a `p≈2` schedule beats p=1.0 on CMMD at NFE=16) — but the effect ≈ the n=12 noise floor, and crucially it **does not survive at production CFG=4** (reshape hurts at low NFE, ties at NFE=16). So the reshape never robustly wins where it matters; canonical p=1.0 stays. **See `er_sde_results.md`** for the full log, the noise-floor analysis, and the per-CFG tables. (Decision metric switches to CMMD for er_sde — no fixed endpoint → paired metrics carry SDE-variance noise.)

---

## References

- `bench/dynamic_spectrum/README.md` — phase 1 (σ ranks the fattening error; SIGMA SUFFICES).
- memory `dynamic_spectrum_vnorm_gate`, `sigma_signal_resolves_by_045`.
- `library/inference/sampling.py::get_timesteps_sigmas` — the reshape knob.
