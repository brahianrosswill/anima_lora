# ComfyUI-Anima-PiD

NVIDIA **PiD** (Pixel Diffusion Decoder) as a drop-in replacement for **VAE Decode**
on Anima / Qwen-Image latents. It takes a `LATENT` and emits a **4× super-resolved
`IMAGE`** in a single 4-step pass — decode *and* upscale fused into one node.

The gemma text encoder is **skipped entirely** (zero caption embeddings): the
distilled 4-step path uses no classifier-free guidance and the net's caption
embedder is RMSNorm-fronted, so null text is numerically safe. → no ~5 GB
text-encoder download, and no prompt input.

## Flow

PiD replaces VAE Decode. Drop `AnimaPiDDecode` where `VAEDecode` was:

```
 checkpoint → KSampler → LATENT ─┐
                                 ├─► Anima PiD Decode (4x SR) ─► IMAGE → Save Image
 Anima PiD Loader (PiD .pth) ────┘
```

There is **no second KSampler** and your Anima model does **not** connect to PiD —
PiD runs its own internal 4-step pixel diffusion. Output size = `latent_grid × 8 × 4`
(e.g. a 64×64 latent → 2048×2048; a 128×128 latent → 4096×4096).

## Install

1. Copy/clone this folder into `ComfyUI/custom_nodes/`.
2. Download the PiD Qwen checkpoint and place it under `ComfyUI/models/pid/`:
   ```bash
   hf download nvidia/PiD --local-dir /tmp/pid \
     --include "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/*"
   mkdir -p ComfyUI/models/pid
   cp /tmp/pid/checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth \
      ComfyUI/models/pid/pid_qwenimage_2kto4k_4step.pth
   ```
   (The Qwen VAE and gemma are **not** needed — PiD emits pixels directly and we
   null the caption.)

## Nodes

- **Anima PiD Loader** — `ckpt_name` (from `models/pid/`), `dtype` → `ANIMA_PID`.
- **Anima PiD Decode (4x SR)** — `ANIMA_PID` + `LATENT` → `IMAGE`.
  - `steps` (default 4) — distilled student steps.
  - `sigma` (default 0.0) — assumed latent degradation; 0 = clean, higher = more
    synthesized detail.
  - `tile_latent` (default 64) — `0` decodes the whole image at once (4K may OOM
    on ≤16 GB); `>0` tiles the latent (each tile → `tile×32` px) with feather
    blending. **64 → 2048 px tiles** (~7 GB peak in bf16).
  - `tile_overlap` (default 16) — latent overlap between tiles (px = `overlap×32`).
  - `compile` (default off) — `torch.compile` the PiD net. ~1.8× faster warm
    (e.g. 3.8s → 2.1s for a 2048px tile), after a one-time ~37s compilation **per
    output resolution**. With tiling on, every tile is the same size so it
    compiles once and all tiles reuse the graph — keep `tile_latent` fixed across
    runs to keep hitting the cache.

## Latent convention

ComfyUI stores raw Qwen/Wan VAE latents; PiD wants the per-channel **normalized**
latent. The node applies `(latent − mean)/std` (the standard Qwen
`latents_mean/std`, `scale_factor=1.0`) internally — the same convention
`anima_lora`'s `encode_pixels_to_latents` produces. If a future Anima latent
format uses a non-1.0 `scale_factor`, update `pid_core.QWEN_LATENTS_*`.

## Licensing

- **This wrapper code**: MIT (`LICENSE`).
- **Vendored PiD network** (`pid_net/`): Apache-2.0, from
  [nv-tlabs/PiD](https://github.com/nv-tlabs/PiD), cross-imports rewritten to be
  self-contained (no hydra/imaginaire). Refresh by re-copying
  `pid/_src/networks/{pid_net,pixeldit_official,lq_projection_2d}.py` and
  re-applying the local-import rewrites in `pid_net/`.
- **PiD weights**: NVIDIA **NSCLv1 — non-commercial only**. Not redistributed
  here; you download them yourself. Do not ship them in a commercial product.

## Provenance

Net constructor config + the 4-step SDE schedule (`t_list=[0.999, 0.866, 0.634,
0.342, 0.0]`, velocity prediction, timescale 1000) were captured from the live
`qwenimage` 2kto4k checkpoint and baked into `pid_core.py` so no hydra config
resolution is needed at runtime.
