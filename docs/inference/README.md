# Inference stacks

Training-free runtime methods — sampler acceleration and sampler-boundary corrections that ride on top of any checkpoint. None of these need training (mod-guidance is the exception — its `pooled_text_proj` head is distilled, but it *applies* at inference). They compose at the sampler boundary; read the relevant doc before touching one.

The DiT operates on 5D latents `(B, C, T=1, H, W)`; sampler-boundary plug-ins here receive 5D — match `ndim` against any 4D reference latent they blend against (see root `CLAUDE.md` §"The DiT operates on 5D latents").

## Acceleration

| Doc | What it is | Flag | Load-bearing gotcha |
|-----|-----------|------|---------------------|
| [spectrum.md](spectrum.md) | Chebyshev feature forecasting — cached steps skip all blocks; `final_layer` pre-hook captures outputs. | `--spectrum` | Structure walkthrough in `../structure/spectrum.md`. |
| [spd.md](spd.md) | Spectral Progressive Diffusion — early steps at low res, spectral noise-expansion handoff to full res. Runner in `networks/spd.py`. | `--spd` | v0 = Euler-only, no DCW/SMC/Spectrum compose; single-late `0.5→1.0 @ σ0.7` default. `bench/spd/plan.md` Phase 3, `../proposal/spd_finetune_lora.md` (Case B). |
| [channel_scaling.md](channel_scaling.md) | SmoothQuant-style per-channel LoRA input pre-scaling. | `--per_channel_scaling` | Calibration at `models/calibration/channel_stats.safetensors`; content-agnostic (`../findings/channel_stats_content_independence.md`). |

## Sampler-boundary corrections

| Doc | What it is | Flag | Load-bearing gotcha |
|-----|-----------|------|---------------------|
| [dcw.md](dcw.md) | SNR-t bias correction at the sampler boundary; composes with everything. Scalar or v4 learnable. | `--dcw` / `--dcw_v4 auto` | **Bias direction is (CFG × aspect)-dependent** — shipped scalar `−0.015` is CFG=1-only and wrong-sign on CFG=4 non-square. |
| [smc_cfg.md](smc_cfg.md) | α-adaptive sliding-mode CFG correction in velocity space (λ=5, α=0.2). | `--smc_cfg` | Paper's fixed k was ~14× off; ships `sign()` only (tanh ε removed). |
| [cns.md](cns.md) | SDE noise recolorer — per-step injected noise is `sqrt(1−γ)`-shaped toward unresolved freq bands, RMS-renormalized (zero-sum). | `--sampler er_sde --cns auto` | **er_sde-only** (no-op on euler/lcm); faithful to paper Alg. 1. |
| [mod-guidance.md](mod-guidance.md) | Text-conditioned AdaLN via a learned `pooled_text_proj` MLP, distilled with `make distill-mod`. | `make test MOD=1` | Global-tone lever, not a content lever (σ-FiLM probe was a geometric ceiling). |

## Other

| Doc | What it is | Gotcha |
|-----|-----------|--------|
| [invert.md](invert.md) | Embedding inversion — optimize a text embedding to match a target image through the frozen DiT (full and K-slot reference). | Postfix slot-collapse: `anima_postfix.safetensors` is effectively K=1. |

User-facing flag reference: [`../guidelines/inference.md`](../guidelines/inference.md).
