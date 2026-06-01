# bench/freetext

Phase-0 probes for **FreeText** (arXiv 2601.00535) — training-free, base-model-
agnostic text-rendering enhancement for DiTs. Two stages: *where to write*
(read endogenous I2T cross-attention → writing mask) and *what to write*
(Spectral-Modulated Glyph Injection of a noise-aligned, band-pass glyph prior).

## probe_localization.py — Stage-1 premise check

Does base-Anima's image→text cross-attention localize the writing region? Anima
runs cross-attn through fused flash/flex kernels that never materialize the
attention matrix, so the probe installs an **eager recompute** on each block's
`cross_attn` (cross-attn has no RoPE, so `softmax(QK^T)` from the post-RMSNorm
q/k reproduces the kernel's scores), head-averages, and reduces on the fly to
per-patch attention mass over token groups (entity / special / padding-sink).

```bash
python bench/freetext/probe_localization.py --label anima-base-cfg4
python bench/freetext/probe_localization.py \
    --prompt 'a poster that reads "HELLO"' --target HELLO --guidance_scale 4.0
```

Artifacts per run (`results/<ts>-<label>/`): `decoded.png`,
`entity_timestep_layer.png` (timestep×layer grid), `token_group_comparison.png`
(Entity/Special/Sink — FreeText Table-4 analog), `aggregate_mask.png` (crude
Stage-1 preview), `reduced.npz` (per-(fwd,block) reduced vectors), `result.json`,
optional `verdict.md`.

**2026-06-01 verdict: GO.** Entity attention concentration 2–3.6× uniform, on the
sign region, strongest in shallow-mid blocks (L6–L17) at mid timesteps. Anima's
only sink is the zeroed padding field (Qwen3-base adds no special tokens). Raw
maps are coarse — FreeText's timestep-layer selection + topology refinement is
what sharpens them. See `results/20260601-1442-anima-base-cfg4/verdict.md`.

## Not yet built
- Stage-1 refinement: soft-IoU timestep-layer selection, neighborhood-denoise,
  Otsu, DBSCAN component pick, topology score → binary latent mask `R`.
- Stage-2 SGMI: glyph raster → VAE encode → forward-diffuse to σ_t → Log-Gabor
  band-pass (FFT) → masked-replace into `R` over t∈[0.8T,0.6T], cosine-annealed.
