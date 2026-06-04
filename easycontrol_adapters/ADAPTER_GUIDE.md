# Building your own EasyControl adapter

A step-by-step guide to adding a **new EasyControl control task** to Anima for
your own use — not as a git contribution, just a local adapter living under
`easycontrol_adapters/<your_task>/`. We walk the canonical example, **colorize**
(`easycontrol_adapters/colorization/`), in detail and call out exactly what you
change for your own task.

If you only read one thing: an EasyControl adapter is **not new model code**. The
network (`networks/methods/easycontrol.py`), the two-stream forward, the
`b_cond` logit-bias, the inference KV cache — all shipped and shared. A control
task differs from every other control task in *one* dimension only:

> **how the condition image is built.**

Everything below is plumbing around that single idea.

---

## 0. Mental model — what an EasyControl adapter actually is

EasyControl conditions generation on a reference image. The reference is
VAE-encoded into *cond tokens* that flow through every DiT block alongside the
target stream; target self-attention attends to an extended key set
`[target_k; cond_k]`. (Full architecture: `docs/experimental/easycontrol.md`.)

The **default** EasyControl uses `cond == target` (reference *is* the image being
reconstructed). A **control-task adapter** breaks that: it pairs each color
target with a *different* condition image, so the model learns
`condition → target` instead of identity.

| | default EasyControl | a control-task adapter (e.g. colorize) |
|---|---|---|
| target | image X | color image X |
| condition | image X (same latent) | a *transform* of X (B&W manga of X) |
| what it learns | reconstruct | manga → color |
| text channel | full caption | (optional) reduced caption |

The colorize insight, which you can reuse: **real B&W manga has no color ground
truth**, so you can't make `(B&W, color)` pairs by collecting them. You *invert*
the direction — take color images you already have as the target, and
**synthesize** the B&W condition from each one (XDoG lineart + algorithmic
screentone). Synthesizing the condition to match the inference distribution is
the whole game. If you're building, say, `depth → image` or `pose → image`, your
"mangafy" step is a depth estimator or a pose extractor run over the existing
training images.

So your job is: **write a function `color_image → condition_image`, cache its
output as a parallel latent set, point a dataset blueprint at it, and add a
config + a one-line selector entry.**

---

## 1. The four surfaces you touch

Adding an adapter named `<task>` means creating/editing exactly these:

| # | Surface | colorize instance | what it does |
|---|---------|-------------------|--------------|
| 1 | `easycontrol_adapters/<task>/` project | `colorization/` (`mangafy*.py`, `color_caption.py`, `prep.py`) | builds + caches the condition (and optionally a reduced text cache) |
| 2 | `configs/datasets/<task>.toml` | `configs/datasets/colorize.toml` | dataset blueprint pairing target latents with the **cond_cache_dir** |
| 3 | `configs/methods/<task>.toml` (+ `configs/gui-methods/<task>.toml`) | `configs/methods/colorize.toml` | method config — points at the dataset, sets LR/epochs/`network_args` |
| 4 | `scripts/experimental_tasks/{training,inference}.py` selector | `_EASYADAPTERS = {"colorize"}` + branches | wires `EASYADAPTER=<task>` into the `make exp-easycontrol*` targets |

We take them in order. Nothing here needs `networks/` edits.

---

## 2. Surface 1 — the adapter project (`easycontrol_adapters/<task>/`)

This is where the real work is. It has two jobs: **synthesize the condition** and
**cache it** (plus, optionally, build a task-specific text cache).

### 2a. The condition synthesizer

A pure function: color RGB `uint8 (H,W,3)` + a per-stem seed → condition RGB
`uint8 (H,W,3)` of the **same size**. (Same size matters — see §3 on token-count
matching.) It must be deterministic given the seed so re-runs and parallel
workers are bit-identical.

In colorize this is `mangafy.py::mangafy_array` (and its CUDA twin
`mangafy_gpu.py::mangafy_array_gpu`):

```python
# easycontrol_adapters/colorization/prep.py
Screener = Callable[[np.ndarray, int], np.ndarray]  # (img_rgb, seed) → cond_rgb
```

Key design points worth copying:

- **Per-stem deterministic seed.** colorize uses `zlib.crc32(stem)` —
  *not* Python's `hash()`, which is salted per-process and would make workers
  disagree. Seed jitter (per-page tone angle/period in colorize) gives variety
  without nondeterminism.
- **Deferred heavy imports.** colorize has three engines (`cv2` / `gpu` / `sd`)
  and imports the 3.5 GB SD stack only if a page actually routes to it. If your
  synthesizer needs a model (depth net, line extractor), import it lazily so a
  no-download fallback stays cheap.
- **A no-model fallback is gold.** colorize's `cv2`/`gpu` engines need zero
  downloads. If your task has one, you can prep + train on a fresh checkout with
  no `make exp-easycontrol-download` step.
- **Atomic writes.** colorize's `_save_png_atomic` writes to a temp file +
  `os.replace`, so an interrupted run never leaves a truncated PNG that the
  `out.exists()` skip-check would then trust forever. Copy this; it's a real bug
  you'll hit otherwise.

If your condition is a *real* artifact (you already have depth maps / sketches on
disk), you can skip synthesis entirely and just point the encode stage at them.
Synthesis is only needed when you must derive the condition from the targets.

### 2b. (Optional) a reduced text cache

colorize doesn't just change the condition — it also **reduces the caption to
color tags only** (`color_caption.py::filter_to_colors`). The reasoning is
task-specific and instructive: the condition (lineart + screentone) already
encodes *everything spatial*, so the only thing left for text to carry is the one
variable B&W can't — **hue**. Filtering the caption to color tags makes every
surviving token a fact the model can't get from structure, giving a strong
`prompt → color` binding instead of weak steering with color tags buried in a
full caption.

Ask the same question for your task: **what does the condition already determine,
and what's left ambiguous for text to carry?** A `pose → image` condition fixes
pose but not appearance/clothing/setting — so you'd probably keep the *full*
caption. A `depth → image` condition fixes layout but not identity or palette.
The colorize "filter the caption to the residual ambiguity" move is a *pattern*,
not a requirement; many adapters keep captions as-is and skip the text cache
entirely (just omit `text_cache_dir` from the dataset blueprint — §4).

If you do reduce captions, note colorize's two independent knobs (don't conflate
them — the README spends real ink on this):

- **`caption_dropout_rate`** — the auto-color *floor*. ~5% of steps drop the
  caption entirely (→ uncond), training the empty-prompt default. Keep it **low**
  (`0.05`); a high rate over-trains the unconditional path into weak steering.
- **`use_shuffled_caption_variants`** — the full-vs-partial *balance*. The text
  cache is multi-variant (v0 = full color set, v1+ = shuffled with each tag
  dropped at p≈0.5), and the loader draws 20% v0 / 80% v1+, so partial prompts
  ("pink hair" alone) work.

### 2c. `prep.py` — the cache builder

Three **idempotent** stages (skip already-done work; safe to re-run):

1. **Synthesize** — walk every color image under `--src`
   (`post_image_dataset/resized`), run the synthesizer, write the condition PNG
   to a `--staging` dir mirroring the source subpath.
2. **Encode** — VAE-encode the staged condition images into `--cond_cache_dir`
   via `library.preprocess.cache_latents`, at each image's **native size** so the
   cond latent shape matches its target latent exactly. Same `{stem}_{WxH}_anima.npz`
   format as the target cache.
3. **(Optional) Text** — re-encode captions through your filter into a
   `--text_cache_dir`, via `library.preprocess.cache_text_embeddings` with a
   `caption_transform=` (and `caption_shuffle_variants` / `caption_tag_dropout_rate`).

Reuse the library primitives — `library.preprocess.{cache_latents,
cache_text_embeddings, tqdm_progress}` and `library.preprocess._dataset.walk_images`.
Don't hand-roll the encode loop; `prep.py` is a thin orchestration shell over
them, exactly like `scripts/preprocess/*.py`.

Two correctness gotchas colorize handles, that you inherit for free by copying its
structure:

- **Stem key-matching.** The text stage reads `.txt` from the caption master
  (`image_dataset/`, nested identically to `resized/`) so the resulting TE cache
  paths key-match the loader's `image_dir=post_image_dataset/resized` lookup. If
  your caches don't key-match the target stems, the loader silently won't pair
  them.
- **Uncond sidecar.** colorize's text stage re-stages the shared `T5("")` uncond
  sidecar idempotently, in case the colorize run is the first to touch it. If you
  build a text cache and use caption dropout, do the same
  (`library.inference.uncond.stage_uncond_sidecar_with_models`).

---

## 3. Critical invariant — cond token count must match the target's

The DiT operates on Anima's **native-shape bucketing** (two token-count families,
4032 / 4200; see CLAUDE.md and `docs/experimental/easycontrol.md` §"Cond token
count"). There is **no static-pad knob** — the cond stream runs at the cond
latent's native token count.

This is why §2a insists the synthesizer output is the **same size** as the input,
and §2c encodes at **native size**: the condition latent then lands on the same
bucket family as its target latent automatically, and `_extended_target_attention`
just works. If you downsample the condition (legit — smaller cond = less memory /
faster), do it *upstream* at the image level so the encoded latent still lands on
a real bucket; don't try to cap the token count in the network.

---

## 4. Surface 2 — the dataset blueprint (`configs/datasets/<task>.toml`)

This is what makes `cond ≠ target`. It's a normal dataset blueprint (`[general]`
+ `[[datasets]]` + `[[datasets.subsets]]`) with **one extra subset knob**:
`cond_cache_dir` (and optionally `text_cache_dir`).

colorize (`configs/datasets/colorize.toml`), annotated:

```toml
[general]
caption_extension = '.txt'
keep_tokens = 3

[[datasets]]
batch_size = 1
validation_split = 0.005
validation_seed = 42

  [[datasets.subsets]]
  image_dir = 'post_image_dataset/resized'        # the COLOR targets
  cache_dir = 'post_image_dataset/lora'           # target latents+TE — REUSED, not re-encoded
  cond_cache_dir = 'post_image_dataset/colorize_cond'   # ← the synthetic condition latents (prep.py)
  text_cache_dir = 'post_image_dataset/colorize_text'   # ← color-only TE cache (prep.py); TE-only redirect
  recursive = true
  flip_aug = false        # latents can't be flipped post-hoc, and the cond cache has no flipped variant
  num_repeats = 1
```

What each redirect does:

- **`cond_cache_dir`** — the only knob that distinguishes a control-task adapter.
  The loader stem-matches each target to a condition latent here. This is what
  EasyControl's two-stream forward consumes as the reference.
- **`text_cache_dir`** — a **TE-only** redirect (latents still come from
  `cache_dir`). Omit it entirely if you keep full captions — then the loader
  reads the shared TE cache and you skip `prep.py`'s text stage.
- **`flip_aug = false`** — required. A flipped target would need a flipped
  condition latent, which you didn't cache. Leave flip off.

Note colorize reuses the **shared** `post_image_dataset/lora` cache for the target
latents and TE — `make preprocess` already built those, nothing is re-encoded.
Your adapter only adds the *condition* cache (and maybe a reduced text cache).

---

## 5. Surface 3 — the method config (`configs/methods/<task>.toml`)

A near-clone of `configs/methods/easycontrol.toml`. The only structural change is
`dataset_config` pointing at your blueprint; the rest is hyperparameters.

colorize (`configs/methods/colorize.toml`), the load-bearing lines:

```toml
dataset_config = "configs/datasets/colorize.toml"   # ← your blueprint from §4

network_module = "networks.methods.easycontrol"     # SHARED — same network as default

network_dim = 32
network_alpha = 32
network_args = [
    "b_cond_init=-6.0",     # logit-bias init (see below)
    "cond_scale=1.0",
    "apply_ffn_lora=1",     # 0 → drop FFN LoRA, ~halves trainable params
]

output_name = "anima_colorize_full"   # ← checkpoint name; the inference selector greps this

use_easycontrol = true
easycontrol_drop_p = 0.0              # image-CFG dropout; 0.1 default, colorize wants 0
masked_loss = false

caption_dropout_rate = 0.05           # auto-color floor (§2b)
use_shuffled_caption_variants = true  # full-vs-partial balance (§2b)
easycontrol_cond_noise_max = 0.02     # small — high noise erases the lineart structure

learning_rate = 2e-5
max_train_epochs = 3
blocks_to_swap = 0                    # recommended for EasyControl
gradient_checkpointing = true
unsloth_offload_checkpointing = true
```

Knobs to think about for your task:

- **`b_cond_init`** — the step-0 baseline-equivalence init. `-10` makes the cond
  contribute ~`e⁻¹⁰` of the softmax mass at step 0 (= baseline DiT, then learns
  up); the bench in `docs/experimental/easycontrol.md` §"Step-0 baseline
  equivalence" derives this. colorize loosens it to `-6` so the cond bites
  sooner — a stronger-condition task can afford that. It's learnable either way.
- **`easycontrol_cond_noise_max`** — per-step training noise on the cond latent
  (σ ~ U(0, max), applied as `cond + σ·ε`). `0` = cond is a perfect blueprint;
  a positive value degrades it to a "lossy hint", forcing text to carry residual
  detail. colorize uses `0.02` (tiny — the lineart *is* the signal). The default
  easycontrol.toml uses `0.3`.
- **`easycontrol_drop_p`** — per-batch full-cond dropout for image-CFG. colorize
  sets `0` (the condition is always wanted); default is `0.1`.
- **`output_name`** — must be a unique stem; the inference selector resolves the
  latest checkpoint by this name (§6).

Optionally also add `configs/gui-methods/<task>.toml` — a self-contained variant
(no toggle blocks) with a `[variant]` block (`family = "easycontrol"`, `label`,
`description`, `order`) so it shows up in the GUI's EasyControl tab dropdown. See
`configs/gui-methods/colorize.toml`. Skip this if you only run from the CLI.

---

## 6. Surface 4 — wire `EASYADAPTER=<task>` into the task runner

The `make exp-easycontrol*` targets dispatch on the `EASYADAPTER` env var. Three
small edits make `EASYADAPTER=<task>` route to your config + prep + checkpoint.

**`scripts/experimental_tasks/training.py`:**

1. Register the name in the allowlist:
   ```python
   _EASYADAPTERS = {"colorize", "<task>"}   # was {"colorize"}
   ```
   (`_easyadapter()` validates against this set and errors on a typo.)

2. Route preprocess to your `prep.py` in `cmd_easycontrol_preprocess`:
   ```python
   adapter = _easyadapter()
   if adapter == "colorize":
       run([PY, "easycontrol_adapters/colorization/prep.py", *extra]); return
   if adapter == "<task>":
       run([PY, "easycontrol_adapters/<task>/prep.py", *extra]); return
   ```

3. Training itself needs **no** edit — `cmd_easycontrol` already does
   `train(_easyadapter() or "easycontrol", extra)`, so `EASYADAPTER=<task>` runs
   `configs/methods/<task>.toml` automatically once the name is in the allowlist.

4. (Only if your synthesizer needs downloads) add a branch in
   `cmd_easycontrol_download` pointing at your weight-fetch task.

**`scripts/experimental_tasks/inference.py`** (`cmd_test_easycontrol`): the
selector currently hard-codes colorize. Generalize the three colorize-specific
values for your task — checkpoint name, output subdir, ref fallback dir, and the
empty-prompt default:

```python
adapter = (os.environ.get("EASYADAPTER") or "").strip()
is_colorize = adapter == "colorize"
weight_name = "anima_colorize" if is_colorize else "anima_easycontrol"
out_sub     = "colorize"       if is_colorize else "easycontrol"
ref_fallback_dir = (ROOT/"post_image_dataset"/"resized") if is_colorize else (ROOT/"easycontrol-dataset")
```

Add your `adapter == "<task>"` cases alongside (weight name must match your
config's `output_name`, modulo the `latest_output` stem match). If your task,
like colorize, wants an empty-prompt default and a real conditioning image from a
specific fallback dir, mirror the `is_colorize` branches further down.

---

## 7. Run it

```bash
# 1. Build the condition cache (synthesize + VAE-encode). Idempotent.
make exp-easycontrol-preprocess EASYADAPTER=<task>
#    QA a handful first:
python easycontrol_adapters/<task>/prep.py --limit 8
#    Inspect the staged condition PNGs under post_image_dataset/<task>_staging/

# 2. Train (frozen DiT, adapter-only).
make exp-easycontrol EASYADAPTER=<task>

# 3. Inference — feed a real in-distribution condition image as the reference.
REF_IMAGE=path/to/condition.png make exp-test-easycontrol EASYADAPTER=<task>
#    Steer with text:  ... ARGS='--prompt "..."'
```

Prereq: `make preprocess` once, so the shared target latents + TE exist in
`post_image_dataset/lora` (your adapter reuses them).

### Inference settings worth knowing (from colorize's experience)

- **Feed a real in-distribution condition.** The reference is VAE-encoded *as-is*
  at inference — no synthesis. colorize feeds a real screentoned B&W page; a flat
  grayscale photo is out-of-distribution and degrades. Whatever your synthesizer
  *imitated* is what inference expects to receive.
- **`--easycontrol_image_match_size`** — picks the token bucket matching the
  reference aspect ratio so tall pages don't squash. colorize forces it on.
- **`--easycontrol_scale`** (`EC_SCALE=`, structure adherence) — `1.0` trained
  default; raise (1.1–1.2) if the condition bleeds, lower (0.7–0.8) for looser
  output.
- **`--guidance_scale`** — interacts with your text policy. colorize: empty
  prompt → low cfg (1.0–1.5, nothing to push toward); text prompt → higher
  (3.0–4.5, that's what makes the prompt bite).

---

## 8. Checklist

- [ ] `easycontrol_adapters/<task>/` with a deterministic, atomic-writing
      synthesizer (`(img, seed) → cond` same-size) and an idempotent `prep.py`
      (synthesize → encode → optional text).
- [ ] Condition encoded at **native size** so token count matches the target
      bucket family (§3).
- [ ] `configs/datasets/<task>.toml` with `cond_cache_dir` (+ optional
      `text_cache_dir`), `flip_aug = false`, reusing the shared target cache.
- [ ] `configs/methods/<task>.toml` → points at the blueprint, unique
      `output_name`, `network_module = "networks.methods.easycontrol"`,
      `use_easycontrol = true`. (Optional GUI variant.)
- [ ] `EASYADAPTER=<task>` registered in `_EASYADAPTERS` + a preprocess branch
      (training.py) + generalized checkpoint/output/fallback in inference.py.
- [ ] QA'd a `--limit 8` staging batch by eye before the full run.

---

## 9. Where to read more

- **`easycontrol_adapters/colorization/README.md`** — the colorize design notes
  in full (caption policy, screentone bands, Phase B roadmap). The reference
  implementation of everything above.
- **`docs/experimental/easycontrol.md`** — the network architecture: two-stream
  forward, `b_cond` step-0 equivalence bench, inference KV cache, memory
  envelope, limitations. Read this before touching `network_args`.
- **`networks/methods/easycontrol.py`** — `EasyControlNetwork` + the patched
  `Block.forward` closure. You should *not* need to edit this for a new adapter;
  if you think you do, reconsider whether the difference really lives in the
  condition.
- **`networks/CLAUDE.md`** — the per-module map and dispatch invariants.
