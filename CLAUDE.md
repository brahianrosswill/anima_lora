# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

## Project Overview

Anima — LoRA/T-LoRA training and inference pipeline for the Anima diffusion model (DiT-based, flow-matching). Supports several adapter families (LoRA / OrthoLoRA / T-LoRA / HydraLoRA / FeRA / ChimeraHydra / EasyControl) selectable via method config + hardware preset. The LoRA family is routed via a three-axis surface — `use_moe_style` / `route_per_layer` / `router_source` — see `configs/methods/lora.toml`.

## Setup

```bash
uv sync                    # Install dependencies (Python 3.13)
hf auth login              # Authenticate for model downloads
make download-models       # Download DiT, text encoder, VAE, SAM3, MIT, PE-Core, PE-Spatial
# Training images go in image_dataset/ with .txt caption sidecars
make preprocess            # Resize → post_image_dataset/resized/, cache → post_image_dataset/lora/
```

## Commands

Both `make` (Unix) and `python tasks.py` (cross-platform/Windows) work — the `Makefile` is a thin dispatcher forwarding every target to `python tasks.py <target> $(ARGS)`. **`tasks.py` is the source of truth**; command bodies live in `scripts/tasks/{training,inference,preprocess,masking,gui,downloads,utilities,tagger,dcw}.py` and `scripts/experimental_tasks/` (for `exp-*`). Don't grep the Makefile for a recipe — look there.

All training runs `train.py --method <name> --preset <name>`. By default it's invoked **directly** (single-GPU fast path — skips the ~5s accelerate launcher bootstrap; `train.py` builds its own single-process `Accelerator()` and reads `mixed_precision` from the config chain). Set `ANIMA_ACCELERATE_LAUNCH=1` to wrap it in `accelerate launch` for multi-GPU / distributed runs (see `build_launch_cmd` in `scripts/tasks/_common.py`). Override any config value from CLI (`--network_dim 32 --max_train_epochs 64`) or the preset via `PRESET=low_vram make lora`. `exp-*` targets are experimental — may break or be removed.

`make help` lists every target; the canonical bodies are in `tasks.py`. Non-obvious knobs and gotchas worth knowing up front:

- **Training**: `make lora PRESET=low_vram|fast_16gb|half` (half → `sample_ratio=0.5`); `make lora-gui GUI_PRESETS=tlora` runs the clean per-variant `configs/gui-methods/` tree (`ls` it for the live list). `exp-soft-tokens | exp-chimera | exp-turbo` are the experimental methods.
- **Inference compose flags**: `SPECTRUM=1` / `MOD=1` / `NOLORA=1` compose into **every** `test-*` target (`make test`, `test-hydra`, `test-merge`, `test-dcw{,-v4}`, `test-smc-cfg`, `test-easycontrol REF_IMAGE=…`, `exp-test-*`).
- **Daemon** (local FIFO job queue, auto-starts on first submit): `make daemon | daemon-attach [JOB=<id>] | daemon-kill | daemon-terminate`. Append `--queue` to any train/distill target to enqueue instead of running inline (`make lora --queue`, `make exp-turbo --queue`). GUI Train button, ComfyUI trainer node, and preprocessing all submit here. **Agent surface**: discovery is pidfile-based (`output/daemon/daemon.json` / `~/.anima/daemon.json` → `{port, root}`; never hardcode 8765 — the port falls back to ephemeral on collision). `make daemon-status` prints one JSON object (health + resolved `base_url` + compact job summaries; `--full` for raw records; passive, exit 1 when down); the daemon self-describes at `GET /` (README) and `GET /tools` (JSON-Schema manifest); `scripts/daemon/mcp.py` is a stdio MCP bridge over the same surface (register the script path as the MCP command — it discovers the daemon itself). Full contract: `scripts/daemon/README.md`.
- **Gotchas**: `make merge ADAPTER_DIR=… [MULTIPLIER=0.8]` bakes LoRA into the DiT (LoRA/Ortho/T-LoRA only) and refuses Hydra-moe / postfix unless `--allow-partial`. `turbo` output is a normal LoRA — infer with `--infer_steps` matched to the DP-DMD `student_steps` rollout (currently 4) and `--cfg 1.0`. `make print-config METHOD=… PRESET=…` dumps the merged chain; `make test-unit` runs pytest; `ruff check . --fix && ruff format .` (touched files only — see [[feedback_ruff_scope_collateral]]).

## Key entry points

| File | Purpose |
|------|---------|
| `anima_lora/__init__.py` | **Programmatic front door** — lazy (PEP 562) re-export of the curated embedder entry points (`generate`, `get_generation_settings`, `GenerationRequest`, `load_method_preset`, `load_dit_model`, `load_vae`, …) + `ROOT` (repo root). `import anima_lora` instead of reverse-engineering `main()`s. |
| `examples/` | Runnable API scripts (`01`–`04` high-level flows, `05`–`06` raw primitives). `examples/README.md` is the embedder guide. |
| `train.py` | `AnimaTrainer` — main training loop via HF Accelerate |
| `inference.py` | Standalone image generation (`--help` for all flags) |
| `networks/spectrum.py` | Spectrum inference acceleration |
| `gui/` | PySide6 GUI package |
| `tasks.py` | Cross-platform task runner — source of truth for every `make` target |
| `scripts/tasks/` + `scripts/experimental_tasks/` | Where command bodies actually live (`_common.py` = shared helpers) |

Docs: shipped method deep-dives in `docs/methods/`, experimental in `docs/experimental/`, active proposals in `docs/proposal/`, retired material under `_archive/`.

## Programmatic API (embedders)

`uv sync` installs the repo editable, so `anima_lora` is importable anywhere. It's a thin façade — canonical homes are unchanged (`library.inference` / `library.config.io` / `library.anima.weights` / `library.models.qwen_vae` / `library.runtime.device`). Inference is **request-driven**: build a typed `GenerationRequest`, call `.to_args()` (which routes through `inference.parse_args` so every `getattr()`-read knob is populated; long-tail method flags ride `extra_argv`). Adapter family lives **in the checkpoint metadata**, not the call — the DiT loader merges-or-keeps-live accordingly. Prompt encoding installs two process-global strategy singletons lazily (`ensure_text_strategies`). Repo-relative model/config paths resolve against the **repo home** (`library.env.anima_home()` / `resolve_under_home()`), not the CWD — so `import anima_lora` works from any directory; set `ANIMA_HOME` for a relocated checkout, or override individual model paths with `ANIMA_DIT` / `ANIMA_VAE` / `ANIMA_TEXT_ENCODER`. The anchor is wired at the config-loader chokepoint (`library/config/io.py`) and the model-loader leaves (`load_anima_model` / `load_vae` / `load_qwen3_text_encoder`); new code opening a repo-relative path should call `resolve_under_home()` rather than assuming CWD.

## Config flow

Config-driven via a three-layer merge chain: `base.toml → presets.toml[<preset>] → methods/<method>.toml → CLI args`. **Method settings win over preset settings on overlap**, so a method can force its own hardware requirements (e.g. a frozen-DiT method forcing `blocks_to_swap=0`).

- `configs/base.toml` — shared infra (model paths, optimizer, compile) AND the default LoRA dataset blueprint (`[general]` + `[[datasets]]` + `[[datasets.subsets]]`, consumed by `BlueprintGenerator`, skipped by the flat method+preset merge — see `_DATASET_CONFIG_SECTIONS`). Use `--dataset_config` for a different blueprint, or drop a `[general]`/`[[datasets]]` block into the method TOML to shallow-override top-level scalars (`_apply_dataset_overrides` in `library/config/io.py`; subset-level overrides not supported this way).
- `configs/preprocess.toml` — preprocess knobs split out of base.toml (`source_image_dir`, `drop_lowres_images`, `min_pixels`, **`target_res`**). Read by the preprocess pipeline via `load_path_overrides`, layered **`preprocess.toml → base.toml → preset → method`** (preprocess.toml read first, so a legacy copy of any of these keys still in base.toml keeps winning — backward compatible). It lives here (not base.toml) because **base.toml is overwritten on `make update`** — preprocess.toml is user-owned and preserved. `train.py` never reads the others, **but `target_res` is dual-use**: `load_method_preset` seeds it from preprocess.toml (lowest priority, preset/method/CLI still override) so the training side matches preprocess. The rest of the **shared** path/tier contract (`resized_image_dir`, `lora_cache_dir`, model paths) stays in base.toml because the dataset blueprint interpolates `{resized_image_dir}`/`{lora_cache_dir}`.
- `configs/presets.toml` — hardware profiles as sections: `[default]`, `[fast_16gb]`, `[low_vram]` (also Windows 8GB), `[half]`. Holds `blocks_to_swap`, gradient/offload checkpointing, etc.
- `configs/methods/` — one flat file per family read by `train.py` (`lora`, `chimera`, `easycontrol`, `soft_tokens`), each holding rank + routing knobs + opinionated LR/epochs/output_name. `turbo.toml` is the **odd one out**: a bespoke sectioned schema read only by `scripts/distill_turbo/` — don't `print-config METHOD=turbo`. Variants inside `lora.toml` are comment-toggle blocks; default stacks LoRA + OrthoLoRA + T-LoRA + shared_A FEI-routed Hydra. **Pre-three-axis checkpoints (`ss_use_hydra`/`ss_use_fei_router` metadata) no longer load** — legacy fallback removed.
- `configs/gui-methods/` — clean per-**variant** parallel tree, no toggle blocks (what you see is what runs). Selected via `--methods_subdir gui-methods` (wrapped by `make lora-gui`). `ls` for the live list.

Subsets accept `cache_dir` — redirects all VAE/TE/PE caches to that dir with stem-mirrored names (EasyControl uses this to keep source dirs user-facing while caches live under `post_image_dataset/`). `library.config.io.load_method_preset(method, preset, methods_subdir=...)` is the reusable merge helper (not re-exported via `train_util`). All config paths are relative to `anima_lora/`. Outputs split by kind: checkpoints (+ `.snapshot.toml` + `_moe` siblings) in `output/ckpt/`, inference images in `output/tests/`.

## Architecture

- **Modular `library/`** (`train_util.py` is a re-exporting facade): domain subpackages `anima/` (DiT model, weights, strategy), `datasets/` (`cache.py` = `CachedDataset`), `training/` (optimizer/scheduler/checkpoint + loss/sampler/metric registries), `inference/` (engine + `request.py` typed `GenerationRequest`; plug-ins split `corrections/` — DCW / SMC-CFG / mod-guidance — vs `editing/` — DirectEdit + postfix inversion), `preprocess/` (caching orchestration), `models/`, `captioning/`, `vision/`, `config/`, `io/` (cache-path resolution), `runtime/` (device/offloading + `cli.py` argparse + `harness.py` `build_anima`). Full per-subpackage map in `docs/structure/`.
- **Tooling layering contract**: **primitives** (`library/*` — load a model, encode a batch, resolve a cache path) → **façade** (`anima_lora/` — embedder entry points) → **orchestration** (`library/preprocess/`, `library/runtime/harness.py` — drive primitives over a whole dataset/run) → **entry points** (`scripts/preprocess/*.py`, `bench/**/run_bench.py`, `scripts/**`, `tasks.py` — thin argparse wrappers). `scripts/preprocess/*.py` are now thin CLI shells over `library/preprocess/`. `bench/`, `scripts/` are **not** installed packages (only `anima_lora`/`library`/`networks` are) — they keep a `sys.path` bootstrap to import siblings.
- **Strategy pattern** for tokenization/encoding (`library/anima/strategy.py`, `library/strategy_base.py`).
- **Pluggable adapters** under `networks/` — selected via `network_module` + (for LoRA family) the three-axis routing cfg. LoRA modules in `networks/lora_modules/` coordinated by `networks/lora_anima/`; EasyControl in `networks/methods/`; attention dispatcher `networks/attention_dispatch.py`; Spectrum `networks/spectrum.py`; SPD `networks/spd.py`. **See `networks/CLAUDE.md`** for the per-module map, three-axis surface, and dispatch invariants.

## Critical invariants

### Text encoder padding
The pretrained model expects max-padded text encoder outputs — zero-padded positions act as attention sinks in cross-attention softmax. Trimming to actual text length produces **black images**. Both training and inference must pad to `max_length` and must NOT mask out padding via `crossattn_seqlens`. Regenerate disk-cached `.npz` after any tokenizer/padding change.

### Constant-token bucketing — native shapes are the only mode
`CONSTANT_TOKEN_BUCKETS` (`library/datasets/buckets.py`) is **two token-count families — 4032 and 4200** — each entry *exactly* filling its count (zero intra-bucket padding by construction), tuples in `(W, H)` order. Each forward runs at its real token count. `compile_blocks()` is the single switch: when `torch_compile` is on it sets `_native_flatten` so the forward flattens each bucket's patch grid to a fake-5D `(B, 1, seq_len, 1, D)` shape, keying the block graph on **token count alone (2 graphs)** instead of per-resolution (24). No padding → no flash static-pad leak; bit-exact to the eager 5D path (eager forwards skip the flatten). The legacy pad-to-static path (`set_static_token_count(pad=True)` / `compile_core` / `--compile_mode full` / `static_token_count` / `static_pad`) was **removed 2026-05-24** — it leaked padding into flash self-attn and couldn't run this table (4200 > 4096).

**Multi-scale tiers (opt-in)**: `CONSTANT_TOKEN_BUCKETS` is the canonical **1024** tier (and stays frozen — DCW keys off it). Preprocess `--target_res 512 768 896 1024 1280 1536` (any subset; `CONSTANT_TOKEN_BUCKETS_BY_EDGE` in `buckets.py`) adds per-tier tables (768→2160 / 1280→6300 / 1536→8640 tok = one graph each; 512→{1024 square, 1008} and 896→{3024, 3000} = two graphs each). Each image is assigned to the tier that **resizes it the least** (`choose_edge` — nearest bucket by cover-scale, scale-symmetric so a 0.95MP image stays at 1024 rather than downscaling to 768), reproducing v1.0's diverse 512–1536 spread. `--target_res` is a **preprocess-only** knob (it decides what each image is resized to). **Training is self-describing and does NOT need `--target_res`**: the bucket table is the full native-shape catalog (`all_constant_token_buckets` — every tier), so every cached latent exact-matches its true `(W,H)` and nothing AR-snaps; the `compile_blocks(n_token_families=…)` dynamo budget is derived from the buckets the path_pattern-filtered images **actually populate** (`train.py::_derive_token_budget`) **plus the sample-prompt resolutions when sampling is enabled** (sample generation runs through the same compiled blocks — a prompt outside the training range, e.g. `--w 1024 --h 1536` over 1024-tier data, used to crash mid-run with a dynamic-seq ConstraintViolation; prompts added to the file *mid-run* are instead skipped with a warning at sample time), not from the arg. So the on-disk caches are the source of truth for which tiers are present — you can no longer silently drop a tier by forgetting to pass it at train time, and a filtered run sizes the compile cache to only the families it really uses. All tiers stay within the rope cap (≤256 patches/axis).

### Lazy model loading
DiT loads AFTER text-encoder/VAE caching and unloading, to avoid OOM: text encoder → cache → free → VAE → cache → free → load DiT → attach adapter → train.

### compile-after-apply (`build_anima`)
`torch.compile` traces the adapter's monkey-patched forward, so `compile_blocks()` MUST run **after** `network.apply_to` + `load_weights`. `library/runtime/harness.py::build_anima` is the shared harness encoding this ordering (promoted from `bench/_anima.py`); use it from `bench`/`scripts`/`preprocess` rather than open-coding load→apply→compile.

### The DiT operates on 5D latents `(B, C, T=1, H, W)` — the singleton is **dim 2**
The DiT forward (and `PatchEmbed`, which `assert x.dim() == 5`) takes a **5D** latent with a singleton temporal/frame axis at **dim 2** (`T=1` for images — Anima reuses a video-shaped layout). Everything *around* the DiT is 4D `(B, C, H, W)`: VAE `encode_pixels_to_latents` returns 4D, cached `.npz` latents are 4D, the training inner loop works in 4D, FFT/spectral helpers (Spectrum, CNS γ, Log-Gabor) want 3D/4D `(C,H,W)`/`(B,C,H,W)`, and the vision tower (PE-Core `encode_pe_from_imageminus1to1`) wants 4D `(B,3,H,W)`. So the boundary dance is **always `unsqueeze(2)` going into the DiT and `squeeze(2)` coming out** — target **dim 2 explicitly**, never `squeeze()`/`squeeze(0)` (which silently hits batch when B=1 and corrupts the layout). Two recurring bite points: **`vae.decode_to_pixels` returns 5D `(B,3,1,H,W)` when fed a 5D latent** (squeeze dim 2 before handing RGB to a vision tower / `F.interpolate`), and **sampler-boundary plug-ins (DCW/SMC/CNS/SGMI/etc.) receive 5D** while any reference latent they blend against is often 4D (match ndim first — see the archived FreeText `_match_latent_ndim`). Mishandling dim 2 was a repeated source of subtle freetext bugs.

## Methods

Adapter families (training methods) below — one-line orientation plus the load-bearing gotcha; read the linked deep-dive before working on one.

**Training-free inference stacks** (Spectrum, SPD, DCW, SMC-CFG, CNS, mod-guidance, embedding inversion, DAVE) are documented separately under [`docs/inference/`](docs/inference/README.md) — read the relevant doc when you touch one rather than carrying their details here. Most ride on the sampler boundary and compose with any checkpoint (DAVE is the exception — a block-forward hook for same-prompt diversity). Channel scaling (per-channel LoRA gradient rebalance, on by default) is a training-time feature — see [`docs/optimizations/channel_scaling.md`](docs/optimizations/channel_scaling.md); note it's exactly inert on frozen-basis ortho variants.

| Method | What it is | Gotcha / pointer |
|---|---|---|
| **DirectEdit + Anima Tagger** | Inversion + edit-conditioning swap; Tagger (`library/captioning/`) maps image → Anima-format tags for ψ_src. | Edit leverage collapses if ψ_src is off-manifold — verify with `exp-test-directedit-dry`. `docs/experimental/directedit_editing_v3.md`, `anima_tagger.md` |
| **EasyControl** | Extended self-attn image conditioning; frozen DiT, per-block cond LoRA + scalar `b_cond` gate. Source `easycontrol-dataset/`. | `docs/experimental/easycontrol.md` |
| **Soft Tokens** | SoftREPA per-layer × per-t soft text tokens (~1M params); frozen DiT, per-block `Block.forward` splice into `crossattn_emb`. | InfoNCE objective intentionally skipped. `configs/methods/soft_tokens.toml` |
| **ChimeraHydra** | Dual-pool additive MoE: content pool (network ContentRouter on pooled `crossattn_emb`) + freq pool (network FreqRouter on FEI+σ), two A's per Linear off disjoint SVD subspaces. Both pools always centered-gate; the per-Linear `lx_c` content router + non-centered path were removed. | T-LoRA mask hits content branch only. `docs/experimental/chimera-hydra.md`, `networks/lora_modules/chimera.py` |
| **Turbo** | DP-DMD (diversity-preserved DMD) distillation; output is a normal LoRA. | Bespoke schema read by `scripts/distill_turbo/` — don't `print-config`. Bespoke two-optimizer loop (student + fake/critic) kept out of `train.py`; converges only the leaves — honors `--queue` (daemon command-job) + writes a canonical `output/ckpt/<name>.snapshot.toml`. `docs/experimental/dpdmd.md` (ops), `docs/structure/dpdmd.md` (structure); CA-era history in `docs/proposal/dmd2_decoupled_improvements.md`. |
| **Postfix-tail inversion** | Per-image inversion *probe* (training method archived 2026-05-20). | Observation tool, not a deployable adapter. `library/inference/postfix_inversion.py` |

## Preprocessing & scripts

Data-prep scripts in `scripts/preprocess/` are thin argparse wrappers (resize → VAE latents → text embeddings → PE features → masks); **the caching logic lives in `library/preprocess/`** — edit orchestration there, flags in the script. `make preprocess-{resize,vae,te,pe,pooled}` / `make mask`. Resize is **idempotent + size-aware** (skips images already at the correct bucket; `--overwrite` forces all). After a `target_res` tier change, run `make preprocess-reconcile` (dry-run; `ARGS="--delete"` to act) to drop orphaned latent npz / stale resized PNG / PE sidecar / mask for every image whose bucket moved — TE caches are text-only and never touched. Other utility scripts: `distill_mod/`, `merge_to_dit.py`, `dcw/`, `anima_tagger/cli.py`, `edit.py`, `export_logs_json.py`.

Caches live under `post_image_dataset/lora/`: `{stem}_{WxH}_anima.npz` (VAE), `{stem}_anima_te.safetensors` (text), `{stem}_anima_pe.safetensors` (PE). TE caching reads `.txt` from `image_dataset/` (the caption master); training reads only cached embeddings.

## Custom nodes

Spectrum KSampler + mod-guidance nodes live in a separate repo (https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler; ships DCW scalar default `+0.01` + `auto` mode). The PiD decode node ships from its own repo too (https://github.com/sorryhyun/ComfyUI-Anima-PiD — full handoff 2026-06-04; symlinked into `../comfy/custom_nodes/comfyui-anima-pid`), as does the EasyControl KSampler node (`~/ComfyUI-EasyControl-KSamplerCompat`). In-tree under `custom_nodes/`: `comfyui-hydralora/` (Adapter / FeRA / Soft Tokens loaders — see its `CLAUDE.md` for the `forward_hook`-not-override invariant), `comfyui-anima-directedit/`, `comfyui-anima-tagger/`, `comfyui-anima-trainer/` (daemon-backed one-shot trainer), `comfyui-anima-blockcompile/`.

Several nodes carry a `_vendor/` subset of the live tree. **Regenerate vendor trees with `make vendor-sync` (`scripts/sync_vendor.py`), never `cp` by hand** — re-run before every node publish. See [[feedback_vendor_sync]]. Note `../comfy/custom_nodes/` is symlinked into this repo — edit the source here, not the symlink.

## External tools

ComfyUI, SAM3, and manga-image-translator live in the parent directory (`../comfy/`, `../sam3/`, etc.).

## Contributing

PRs follow a tier system — see `CONTRIBUTING.md`. Key constraint for code work: numerics/efficiency changes (Tier 1.5) and new methods (Tier 2) **require a bench script + invariant test**. Bench scripts share `bench/_common.py` and drop a `result.json` envelope into `bench/<method>/results/<YYYYMMDD-HHMM>[-label]/`.
