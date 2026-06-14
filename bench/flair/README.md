# bench/flair — FLAIR inverse-problem solver (Phase 0: port validation)

Proposal: [`docs/proposal/flair_inverse.md`](../../docs/proposal/flair_inverse.md).
Paper: *Solving Inverse Problems with FLAIR* (Erbach et al., NeurIPS 2025).

A self-contained port of FLAIR's Algorithm 1 onto the Anima flow prior, kept in
`bench/` (not yet wired into the inference engine) so the port can be validated
**before** promotion to `library/inference/corrections/flair.py`.

- `solver.py` — Algorithm 1 (regularizer pull + hard data consistency + DTA).
  Holds `load_lambda_table()` (Phase-1 calibrated CRW) + the `λ_R(t)` selector.
- `operators.py` — forward operators `A` (Phase 0: `sr` bicubic ×s only).
- `sanity_sr.py` — SR gate driver. Default arm is the uncalibrated
  `λ_R(t)=reg_scale·t`; pass `--calib auto` for the Phase-1 calibrated arm.
- `calibrate_lambda.py` — **Phase 1** λ_R(σ) calibration (paper Eq. 14).

## Phase 0 status — PORT VALIDATED (2026-06-14)

Runs end-to-end on a **16 GB** card at 512px with the DiT + VAE both resident and
HDC backprop through the VAE decoder — **no OOM, no black frame, no layout
corruption**, with or without `--compile` (one block graph — the FLAIR latent
shape is fixed across the solve). The gate (FLAIR PSNR/SSIM > bicubic-upsample
baseline, sharp output) **passes on all 5 images** of a representative config:

```bash
uv run python bench/flair/sanity_sr.py --n 5 --res 512 \
    --steps 50 --hdc_steps 8 --hdc_lr 0.1 --reg_scale 0.5 --compile
# → FLAIR PSNR 23.76 / SSIM 0.742  vs  bicubic 22.99 / 0.704   (mean Δ +0.77, GATE PASS)
#   per-image PSNR gain +0.44 … +1.05 — beats bicubic on every image.
```

Triptychs (GT | bicubic | FLAIR) confirm the FLAIR panel recovers high-frequency
detail (hair, facial features, fabric) that bicubic blurs out — lower SSIM on
detailed images is recovered texture, not artifacts.

The smoke default (`--steps 10 --hdc_steps 2 --reg_scale 1.0`) **fails** — too few
steps + under-converged HDC let the prior drag μ off-manifold (hazy, hallucinated
watermark). That's the expected under-constrained signature, not a port bug:
stronger/longer HDC fixes it. **Lesson: data consistency must keep pace with the
prior pull.**

## Still open before the Phase 0 gate is "closed"

- **Full N=100 run** (the proposal's gate size) — the numbers above are N=1.
- **Hyperparameter sweep** — `reg_scale` / `hdc_steps` / `hdc_lr` / `alpha` were
  hand-set to one working point, not swept.
- **LPIPS / pFID** — not wired (`lpips` not installed; pFID is Phase 2). PSNR/SSIM
  + the eyeball triptychs are the current gate. PSNR favors the blurry baseline by
  the paper's own argument, so it's a *floor* check, not where FLAIR's value lives.

## Phase 1 — λ_R(σ) calibration (CALIBRATED 2026-06-14, A/B owed)

`calibrate_lambda.py` mirrors paper Eq. 14: encode N real latents from
`post_image_dataset`, sweep the **deployed** flow-shifted σ grid
(`get_timesteps_sigmas(n_timesteps, flow_shift)`, not linear t), and at each σ
measure the conditional-FM error `mean_i ‖v_θ(x_σ,σ) − (ε−x0)‖²` → `λ_R(σ)=1/error`.
The Anima low-noise cutoff is **re-derived** from the error knee (pre-registered
hypothesis: σ≈0.45, not the paper's SD3 0.2 — see `sigma_signal_resolves_by_045`).

```bash
uv run python bench/flair/calibrate_lambda.py --n 100 --res 768 --batch 4 --compile
# → networks/calibration/flair_lambda_r.npz  (keys: sigmas, lambda_r, error, cutoff_sigma)
#   lambda_r_curve.png shows error-vs-σ + the calibrated λ_R(σ) vs the λ_R=σ baseline.
```

**Result (N=100, 768px, 100 σ-steps, compiled — `results/20260614-1417-n100-768px`):**
the error-vs-σ curve is U-shaped; the prior is **most reliable (λ_R peak) at
σ=0.486 — HIT on the σ≈0.45 hypothesis** (error min 0.068). Error blows up in the
deep tail (max 0.40 at σ=0.029, ~6× the min) but only there: the auto-detected
**zeroing cutoff is σ=0.111** — *lower* than the paper's SD3 0.2, i.e. Anima's prior
stays usable deeper into low-σ than SD3's. The calibrated λ_R thus peaks at the
resolve σ, tapers toward both ends, and is zeroed (+loop stops) below σ≈0.11.

A/B against Phase 0 (the proposal's readout — calibrated λ_R must beat `λ_R=σ`):

```bash
uv run python bench/flair/sanity_sr.py --n 100 --reg_scale 0.5 --hdc_steps 8 --hdc_lr 0.1 --compile               # λ_R=σ arm
uv run python bench/flair/sanity_sr.py --n 100 --reg_scale 0.5 --hdc_steps 8 --hdc_lr 0.1 --compile --calib auto  # calibrated arm
```

`load_lambda_table` peak-normalizes the curve over the active region, so `reg_scale`
stays the same comparable knob across both arms; the calibrated arm sets
**`t_stop = cutoff`** (0.111 here — refines *deeper* than the linear arm's 0.2,
since the calibration says the prior is reliable that far down), and `--t_stop` is
ignored in the calibrated arm.

**Preliminary A/B (N=5, SR×8, PSNR/SSIM only):** a **wash** — calibrated 23.78/0.740
vs linear `λ_R=σ` 23.76/0.742 (±0.02 PSNR, inside N=5 noise). The cutoff lever
(loop→0.111) matters: with it disengaged the calibrated arm was slightly worse
(23.73), engaging it recovered the deficit. **NOT a verdict** — PSNR/SSIM are floor
checks that favor blur; the real signal is **LPIPS/CMMD at N=100**, still unwired/
unrun. If calibration still doesn't beat `λ_R=σ` there → ship the linear weight,
delete the npz, note it.

## Later phases (proposal)

2. **HDC×DTA×CRW ablation** — confirm DTA is the lever on Anima too.
3. **Pilot A** FLAIR-inpaint vs DirectEdit · **Pilot B** FLAIR-colorize vs EasyControl.
