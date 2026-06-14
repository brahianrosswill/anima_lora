# FLAIR-SR — training-free super-resolution as a shipped Anima application

Status: **PROPOSAL (2026-06-14).** Productionizes the one FLAIR task whose port is
already **validated end-to-end** — super-resolution — from the `bench/flair/`
solver into a user-facing inference path: a `make exp-test-flair-sr` target and a
ComfyUI custom node. This is the **application** layer; the **method/bench** layer
is the parent proposal [`flair_inverse.md`](flair_inverse.md) (Phase 0 PASSED, Phase 1
λ_R calibrated 2026-06-14). Nothing here changes the math — it promotes the solved,
benched code to the engine and wires two front doors.

Scope: **SR only.** Inpaint/colorize/deblur stay in `bench/` behind their pilots
(parent proposal Phase 3/4); this proposal does *not* gate on them. SR is the right
first app because (a) its port passed the cheap gate (PSNR > bicubic on 5/5, sharp,
no OOM on 16 GB), (b) it has the cleanest UX (low-res image in, high-res out, one
scale knob), and (c) it fills a real hole: **the repo has no SR / restoration path
at all** today (Spectrum/SPD/DCW/CNS/DAVE are all generation-time accelerators or
correctors, none invert a degradation).

## What ships

Three deliverables, in dependency order:

1. **Promote the solver** `bench/flair/{solver,operators}.py` → `library/inference/corrections/flair.py` (+ `flair_operators.py`). The bench keeps a thin shim importing from `library` so the Phase-1/2 benches keep running. This is a near-verbatim move — the solver is already written against `library.inference.sampling.get_timesteps_sigmas`, `library.runtime.harness.build_anima`, the Qwen VAE, and `library.env.resolve_under_home` (the calibration loader). Promotion is the standard bench→library lift the layering contract already blesses.
2. **CLI / Request plumbing + `make exp-test-flair-sr`** — a branch in the inference engine that, when `--flair_task sr` is set, dispatches to the promoted `flair_solve` instead of the Euler/er-SDE loop.
3. **ComfyUI node** `custom_nodes/comfyui-anima-flair-sr/` — MODEL + VAE + low-res IMAGE → IMAGE, the full Algorithm-1 solve behind one node, `_vendor/`-synced like the in-tree directedit/tagger nodes.

## Why SR needs more than "call the bench from a flag" — three real app-layer problems

The bench fixed every variable (512px target, square crop, synthetic bicubic `y`).
A shipped SR app removes those fixings, and exactly three of them bite:

1. **Arbitrary output resolution vs the token-bucket / rope cap.** The DiT runs on
   the constant-token bucket families and caps at ≤256 patches/axis (CLAUDE.md
   invariant). A 4× SR of a 768px input is 3072px → far past the cap; the `v_θ`
   forwards simply can't run at that latent size. So the app must **tile the
   solve** (run Algorithm 1 on overlapping crops at a supported tier, blend in
   latent space) or **cap the target tier and report it**. The bench never hit
   this — it's the load-bearing new engineering. Proposed default: solve at the
   nearest supported tier ≤ the requested output, then a cheap final bicubic lift
   to the exact requested size if the user asked for more, with a `--flair_tile`
   path for true high-res (Phase B below). HDC keeps each tile data-consistent, so
   seams are bounded by the operator, not hallucinated.
2. **HDC backprops through the VAE decoder — inside a ComfyUI node.** The solver's
   one backward pass is `argmin ‖y − A(D(μ))‖²` through `D` (the VAE decoder).
   ComfyUI runs its VAE under `no_grad` / tiled-decode by default, so the node
   **cannot** reuse Comfy's decode path for HDC — it must drive the decoder with
   grad enabled on `μ` (the `_hdc_project` Adam loop from `solver.py`, decoder in
   eval but autograd on). This is the node's single sharp edge; everything else is
   tensor-marshalling. Mitigation already proven in the bench: cap `hdc_steps`,
   keep the decoder resident, fp32 `μ` leaf.
3. **Real degradations ≠ clean bicubic.** The bench synthesizes `y` with a known
   bicubic kernel; real low-res images carry compression / unknown blur. FLAIR is
   robust here (HDC only enforces `A(D(μ))=y` for the `A` you give it), but the app
   should expose `--flair_sr_kernel {bicubic,bilinear,area}` and default to bicubic
   (what the calibration + Phase-0 gate used), with a documented "your `A` should
   match your real downsampler" note.

## Where it hooks (verified against the live tree)

### Engine branch + Request/CLI

- **Promoted module** `library/inference/corrections/flair.py` — `FLAIRSolver` /
  `flair_solve` (lifted from `bench/flair/solver.py`) + `flair_operators.py`
  (`SROperator` lifted from `bench/flair/operators.py`). Loads the calibrated
  λ_R table via the existing `load_lambda_table("auto")` (reads
  `networks/calibration/flair_lambda_r.npz`).
- **GenerationRequest** (`library/inference/request.py:50`) — add, mirroring the
  `easycontrol_image` / `easycontrol_weight` optional-field pattern already there:
  `flair_task: Optional[str]` (`"sr"`), `flair_sr_image: Optional[str]` (the low-res
  input), `flair_sr_scale: int = 8`, `flair_reg_scale: float = 0.5`,
  `flair_alpha: float = 0.5`, `flair_hdc_steps: int = 8`, `flair_hdc_lr: float = 0.1`,
  `flair_calib: str = "auto"` (or `"off"` for the λ_R=σ arm), `flair_sr_kernel: str`.
  Long-tail knobs can ride `extra_argv` first per the embedder contract.
- **argparse** (`library/inference/args.py` `build_parser`) — the matching
  `--flair_task` / `--flair_sr_image` / `--flair_sr_scale` / … flags; post-parse
  validation in `build_default_args` (sr_scale > 1, sr_image exists when
  flair_task=sr). Defaults chosen as the **Phase-0 working point** (reg_scale 0.5,
  hdc_steps 8, hdc_lr 0.1) so the shipped defaults are the benched ones.
- **Dispatch branch** — when `args.flair_task == "sr"`, the inference engine routes
  to `flair_solve(...)` instead of the standard sampler. FLAIR *replaces* the
  sampling loop (it optimizes `μ`, it does not denoise a fixed trajectory), so this
  is a top-of-`generate_body` branch, not a sampler-boundary plug-in like CNS/DAVE.
  It reuses `build_anima` (compile-after-apply) and `get_timesteps_sigmas` verbatim.

### `make exp-test-flair-sr` target

Experimental inference target, mirroring `exp-test-directedit`
(`scripts/experimental_tasks/inference.py:215`) for the REF_IMAGE plumbing and
`test-easycontrol` (`scripts/tasks/inference.py:265`) for the image-arg injection:

```bash
make exp-test-flair-sr REF_IMAGE=lowres.png SCALE=8                  # SR ×8
make exp-test-flair-sr REF_IMAGE=lowres.png SCALE=4 PROMPT="…" CALIB=off
```

- `REF_IMAGE` env → positional fallback → resolved via the `_resolve_ref_image`
  helper (`scripts/experimental_tasks/inference.py:70`), exactly like directedit.
- `SCALE` → `--flair_sr_scale`; `CALIB=off` → `--flair_calib off` (λ_R=σ arm);
  `PROMPT` → the domain caption (or the image's `.txt` sidecar). The body builds
  argv off `INFERENCE_BASE` (`scripts/tasks/_common.py:714`) + `--flair_task sr
  --flair_sr_image <resolved>`, then `run()`s it.
- Registered in `tasks.py` `COMMANDS` alongside the other `exp-test-*` entries
  (line ~327), imported from `scripts.experimental_tasks.inference`.
- **Compose flags**: FLAIR-SR is its own sampling path, so it does **not** compose
  with `SPECTRUM` / `SPD` / `MOD` / `DAVE` (those decorate the standard loop). The
  target ignores them and says so — documented, not silently dropped.

### ComfyUI node

`custom_nodes/comfyui-anima-flair-sr/`, templated on `comfyui-anima-directedit/`
(`nodes.py:167` `AnimaDirectEdit` is the closest analog — training-free, IMAGE in,
solves on the DiT, returns a result):

- **Layout**: `__init__.py` (re-exports `NODE_CLASS_MAPPINGS`), `nodes.py`
  (`AnimaFlairSR`), `pyproject.toml`, `_vendor/` (synced by `make vendor-sync` —
  never `cp`; see `[[feedback_vendor_sync]]`). Vendor subset: `library/inference/`
  (sampling, corrections/flair, flair_operators), `library/datasets/buckets.py`,
  `library/anima/models.py`, the Qwen VAE, and the calibration npz.
- **INPUT_TYPES**: `MODEL` + `VAE` sockets (Comfy-loaded), `IMAGE` (low-res input,
  `[B,H,W,C]`), `scale` (INT 2–8), `prompt` / `negative` (multiline STRING),
  `reg_scale` / `alpha` / `hdc_steps` / `hdc_lr` (FLOAT/INT), `use_calibration`
  (BOOLEAN, default on). **RETURN_TYPES** `("IMAGE",)` (decode inside, since HDC
  already needs the decoder resident) — or `("LATENT",)` for a downstream
  `VAEDecode`, matching directedit's `{"samples": …}` convention. Default IMAGE.
- **The node body**: comfy IMAGE → `[-1,1]` pixel `y` → build `SROperator(scale)` →
  `flair_solve(model, vae, y=…, operator=…, embed=…)` → decode → comfy IMAGE. The
  prompt is encoded through Comfy's CLIP exactly as directedit does
  (`_encode_prompt_comfy`), honoring the max-pad TE invariant.
- **HDC grad caveat** (node-specific, the §-problem-2 edge): the node runs its own
  grad-enabled decode loop on `μ`, not Comfy's `VAEDecode` — call out loudly in the
  node doc + code comment so a future refactor doesn't "optimize" it into the
  no-grad tiled decoder and silently kill data consistency.
- **Lifecycle**: ship in-tree first (vendor-synced), promote to a standalone repo on
  publish like the EasyControl-KSampler / DAVE / PiD nodes once it stabilizes.

## What to bench (Tier-2: bench + invariant test mandatory)

The **method** numbers are the parent proposal's job (SR×8 LPIPS/pFID/CMMD, the
λ_R A/B). This proposal owns the **app-layer** acceptance:

- **`bench/flair/sr_app.py`** — parity: the promoted `library` path must be
  **bit-for-bit identical** to the `bench/flair/solver.py` output on the Phase-0
  config (same seed → same μ). Guards the promotion.
- **Tiling correctness** (Phase B): tiled solve vs whole-image solve at a res both
  can run — seam PSNR/SSIM within the overlap, and the global metric within ε of
  the untiled run. No tiling ships until this passes.
- **Node smoke**: a headless ComfyUI graph (IMAGE→AnimaFlairSR→SaveImage) produces
  a sharp, data-consistent upscale; HDC keeps `A(output)≈y` (verify the grad path
  actually engaged — the node-edge regression).

### Invariant test (`tests/test_flair_sr.py`)

(a) `SROperator` adjoint dot-product test `⟨Ax,y⟩=⟨x,A^Ty⟩`; (b) promoted
`flair_solve` == bench `flair_solve` on a fixed seed (parity lock); (c) dim-2
round-trip (encode→solve→decode preserves `(W,H)`, never `squeeze()` the batch);
(d) Request→argv→engine reaches the FLAIR branch (a `flair_task=sr` request does
not fall through to the Euler loop).

## Phasing

- **Phase A — single-tier SR (the MVP).** Promote solver, wire Request/CLI +
  `exp-test-flair-sr`, ship the node at a capped output tier (no tiling), parity +
  invariant tests. This is shippable on its own and covers the common
  "upscale-to-a-supported-resolution" case.
- **Phase B — tiling for true high-res.** Latent-space tiled solve with HDC-bounded
  seams + the tiling bench. Gated on Phase A.
- **Phase C — kernel options + UX polish.** `--flair_sr_kernel`, the "match your
  real downsampler" doc, node tooltips, `docs/inference/flair_sr.md`.

## Risks / honest limits

- **Compute/memory**: HDC backprops through the decoder every σ-step — the real
  cost, and the OOM risk at high res. Capping `hdc_steps` + the tier cap keep
  Phase A on 16 GB (Phase 0 proved 512px fits); tiling (Phase B) is what unlocks
  big outputs without a bigger card.
- **Posterior sampling, not deterministic**: SR diversity is a feature, but for
  repeatable upscales run near `α→1` (the deterministic end). Expose `alpha`; note
  it in the node.
- **Domain prior bound**: output quality is bounded by what the Anima prior knows —
  the parent proposal's domain-prompt caveat applies. Anime/illustration inputs are
  in-domain; photographic SR is out-of-domain and will look like Anima, not like a
  photo. Document it.
- **Not a sampler plug-in**: FLAIR-SR can't compose with Spectrum/SPD/DAVE/MOD (it
  replaces the loop). That's a hard boundary, not a TODO.

## Explicitly NOT doing

- **The other FLAIR tasks** (inpaint/colorize/deblur) — they stay in `bench/`
  behind the parent proposal's pilots; this is SR-only.
- **A trained SR adapter** — FLAIR is training-free; if a task wants learned
  structural conditioning it's EasyControl's job, not this.
- **Replacing Spectrum/SPD** — those accelerate *generation*; FLAIR-SR *inverts a
  degradation*. Different category, additive.
- **Shipping the node before parity passes** — the promotion parity lock
  (`bench/flair/sr_app.py`) gates everything downstream.
