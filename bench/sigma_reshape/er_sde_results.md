# σ-reshape under er_sde — results log

Companion to `README.md` (which carries the **Euler** result: NO RESHAPE WIN). This
file logs the **er_sde** runs, where the picture *flips* — but the effect currently
sits at the n=12 noise floor, so it's **suggestive, not established**. Read this
alongside the README's "Why this doesn't contradict dynamic_spectrum" section.

## TL;DR

- **Euler, CFG=1 (deterministic):** densifying the σ<0.45 tail (`tail_power=p>1`) **strictly hurts** the converged-target match at every NFE. Canonical p=1.0 wins. (README, n=24.)
- **er_sde, CFG=1 (stochastic):** **direction reverses** — at NFE=16 the ordering is monotone **p=2.0 < p=1.5 < p=1.0** (lower CMMD = closer to converged), reproduced across two independent runs. Effect ≈ the n=12 noise floor (~0.12 CMMD), so *direction* trustworthy, magnitude not.
- **er_sde, CFG=4 (PRODUCTION):** ⚠ **the CFG=1 win does NOT survive.** At low NFE=8 the reshape **actively hurts** (p2.0=0.636 vs p1.0=0.223, far beyond noise); NFE=12 is non-monotone (noise); NFE=16 is **tied** within the noise floor (0.107/0.104/0.100). The DCW lesson holds — **the optimum moved with CFG.**
- **Net verdict: the σ-reshape is NOT a robust production win.** The only regime it helped (er_sde CFG=1) is not production; at production CFG=4 it's tied-to-harmful. **Line effectively closed** — see "Status" below.
- **Qualitative (eyeball, CFG=1 NFE=16):** the reshape benefit looked **content-dependent** — complex/busy scenes favored p=2.0, simple scenes were indistinguishable. But this was a CFG=1 observation; at CFG=4 NFE=16 the columns are near-identical by eye (matching the tied CMMD), so the content-dependence did not carry to production either.

## Runs

Both: 12 real captions, 1024², er_sde, CFG=1, flow_shift=3, converged ref = 100-step er_sde (same seed), block-compiled, RTX 5070 Ti.

### Run 1 — `20260607-2121-ersde-half` (NFE 8/12/16)

CMMD (↓ better, vs converged):

| NFE | p=1.0 | p=1.5 | p=2.0 |
|---|---|---|---|
| 8  | 0.685 | **0.611** | 0.725 |
| 12 | 0.303 | **0.287** | 0.325 |
| 16 | 0.435 | 0.181 | **0.161** |

Verdict string: `RESHAPE HELPS (er_sde): p>1 beats p=1.0 on cmmd at 3/3 NFE`.
Paired metrics (`latent_endpoint`, `pe_cosine`) are **confounded by SDE variance** here (stochastic sampler has no single endpoint) and are non-monotone in NFE — that's expected; they are *not* the decision metric for er_sde. CMMD (distribution-to-distribution) is.

### Run 2 — `20260607-2133-ersde-grid16` (NFE 16 only, same config/seeds)

| NFE | p=1.0 | p=1.5 | p=2.0 |
|---|---|---|---|
| 16 | 0.313 | 0.266 | **0.188** |

### Run 3 — `20260607-2226-ersde-cfg4` (CFG=4, production) ⚠

CMMD (↓ better), n=12:

| NFE | p=1.0 | p=1.5 | p=2.0 | read |
|---|---|---|---|---|
| 8  | **0.223** | 0.320 | 0.636 | reshape **hurts** — p2.0 loss (0.4) far beyond noise |
| 12 | 0.240 | 0.308 | **0.140** | non-monotone (p1.5 worst) = noise |
| 16 | 0.107 | 0.104 | **0.100** | **tied** within the ~0.1 noise floor; grid confirms by eye |

The clean monotone `p2.0 < p1.5 < p1.0` ordering from CFG=1/NFE=16 **vanishes** at CFG=4. At production guidance the reshape gives no reliable benefit and clear harm at low NFE. (Aside: CFG=4 CMMD magnitudes are overall lower than CFG=1 — strong guidance locks trajectories closer to the converged reference — but that's orthogonal to the reshape ranking.)

## The noise floor (key methodological finding)

Run 1 and Run 2 are **identical config/seed** at NFE=16, yet p=1.0 CMMD came out **0.435 vs 0.313** — a **~0.12 swing from pure run-to-run nondeterminism** (bf16 + flash-attn + torch.compile are not bit-deterministic). At n=12 that noise floor is the *same size* as the reshape effect (p1.0→p2.0 ≈ 0.12–0.27). So:

- **No single n=12 run can establish the magnitude.**
- **What survived the noise:** the *ordering* p=2.0 < p=1.5 < p=1.0 at NFE=16 held in **both** independent runs. Direction reproducing across two draws is weak-but-real positive evidence.

## Why er_sde might flip Euler's verdict (hypothesis)

1. er_sde is **higher-order** (stages 2/3 use finite-difference curvature of the denoised history) — it already handles high-σ curvature efficiently, so reshaping a few steps *out* of the structure-forming high-σ region costs less than it does for plain Euler.
2. er_sde **injects stochastic noise per step**, scaled by the σ-gap (`noise_coeff = sqrt(er_λ_t² − er_λ_s²·r²)`). Denser tail steps → smaller σ-gaps in σ<0.45 → finer, smaller stochastic corrections exactly where the image is resolving. Plausibly that's what helps the busy/high-frequency scenes converge.

The **content-dependence** observation fits (2): the σ<0.45 tail is where fine/high-frequency detail resolves, so complex scenes (more high-freq content) have more to gain from denser tail steps under the stochastic refinement, while flat scenes don't. *Untested* — a complexity-stratified CMMD (split prompts by edge density / detail) would confirm or kill it.

## Status & next steps

**Status: LINE EFFECTIVELY CLOSED — the σ-reshape is not a robust production win.** The CFG×aspect sweep (the documented escape hatch) was run for CFG and it *killed* the case rather than rescuing it:

- Euler CFG=1 → reshape hurts.
- er_sde CFG=1 → reshape helps (p≈2 at NFE=16), but only ≈ the noise floor, and **only at non-production CFG=1**.
- er_sde CFG=4 (production) → reshape gives no reliable benefit and **hurts at low NFE**.

So across the three regimes the reshape never robustly wins where it matters. The canonical `flow_shift=3`, p=1.0 schedule stays. **Do not ship a tail-densified schedule.**

What would *re-open* it (low priority — burden is high after three negative/null regimes):
1. The CFG dependence itself is a (negative) instance of DCW's CFG/aspect-indexed optimum. If someone ever ships a CFG-indexed schedule table for another reason, re-check whether a per-CFG `tail_power` falls out — but it's not worth a dedicated effort now.
2. The content-dependence (complex scenes) was a CFG=1-only eyeball that didn't carry to CFG=4; a complexity-stratified CMMD would only matter if a production-CFG regime first showed *any* aggregate signal, which it doesn't.

**The `tail_power` knob stays in the codebase** (default 1.0, inert, bit-identical) — it's the cheap way to re-test if the above ever becomes relevant, and reverting it buys nothing.

## Files

- `results/20260607-2121-ersde-half/` — Run 1 (table, metrics.png, grid@8).
- `results/20260607-2133-ersde-grid16/` — Run 2 (grid@16 — the complex-scene p=2.0 win is visible here).
