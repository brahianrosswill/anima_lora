# API surface suggestions

Friction I hit while writing `examples/`, ranked by impact-per-effort. These are
observations from an embedder's seat — none are bugs, and most of the codebase
is internally consistent. The theme: **there is no programmatic front door**, so
every embedder reverse-engineers `inference.py` / `train.py` `main()`.

> **Status (2026-05-24):** the safe, additive "Tier A" items — **#3, #4, #5, #6,
> #7** — are now implemented and the examples use them. The remaining items
> (**#1, #2, #8, #9, #10**) are API-design changes left open for a follow-up.
> Resolved items are marked ✅ inline below.

---

## 1. No installable public package  ·  high impact / low effort

`pyproject.toml` has `packages = []` and `library/__init__.py` is empty, so
nothing is importable without the repo root on `sys.path` + CWD == repo root
(every example needs the `sys.path.insert` bootstrap). An embedder can't
`pip install anima-lora` then `import anima`.

**Suggestion:** ship a thin top-level package that re-exports the handful of
real entry points:

```python
# anima/__init__.py
from library.inference import generate, get_generation_settings, save_output
from library.config.io import load_method_preset
from library.anima.weights import load_anima_model
from library.models.qwen_vae import load_vae
```

Even without fixing the CWD-relative path assumptions, this collapses
"reverse-engineer `main()`" into "read four exports."

---

## 2. Generation is `argparse.Namespace`-driven, no request object  ·  high / medium

`generate(args, gen_settings)` reads ~40 fields off an `argparse.Namespace` via
`getattr`. The only safe way to build one programmatically is to call the CLI
parser (what `01`/`02`/`03`/`05` all do) — and `--save_path` / `--prompt` are
`required`, so non-generation tasks pass dummy values (`--save_path /tmp/unused.png`).

**Suggestion:** a `@dataclass GenerationRequest` (prompt, negative, size, steps,
cfg, seed, lora_weights, …) with a `.to_args()` adapter for the CLI. The
codebase already uses frozen dataclasses well (`sampler_context.py:27`); this is
the same move for the inference inputs. The CLI parser becomes one consumer
instead of the only constructor.

---

## 3. `read_config_from_file` re-reads `sys.argv` for overrides  ·  high / low  ·  ✅ DONE

**Resolved:** `read_config_from_file(args, parser, argv=None)` now threads `argv`
into both inner `parse_args` calls (defaults to `sys.argv`, so the CLI path is
byte-identical). `04_train_lora.py` passes its assembled `argv` explicitly.


`library/config/io.py` applies CLI overrides via
`parser.parse_args(namespace=Namespace(**merged))` — with **no argv**, so it
reads the global `sys.argv`. This means you cannot drive the merge from a passed
argv list: `04_train_lora.py` has to forward `sys.argv[1:]` and document the
footgun, and `--method`/`--preset` injected via a list silently vanish from the
final namespace (they're consumed before the re-parse).

**Suggestion:** accept an explicit override source, e.g.
`read_config_from_file(args, parser, argv=None)` passing `argv` through to the
inner `parse_args`. Defaulting to `sys.argv` keeps the CLI path unchanged.

---

## 4. `load_method_preset` import path drift  ·  low / trivial  ·  ✅ DONE

**Resolved:** `library.train_util` now re-exports both `load_method_preset` and
`read_config_from_file` from `library.config.io`, so the convention import works
(the canonical home is still `library.config.io`).

---

## 5. Two DiT loaders with different calling conventions  ·  medium / low  ·  ✅ DOCSTRINGS

**Partially resolved:** both loaders now carry docstrings cross-linking each
other — `load_anima_model` is labelled the explicit-argument primitive,
`load_dit_model` the Namespace adapter that adds LoRA-attach + compile. The
deeper refactor (splitting `attach_adapters` out of the loader) is left open.


`load_anima_model(device, dit_path, attn_mode, loading_device, dit_weight_dtype, …)`
(`library/anima/weights.py:113`) takes explicit args; `load_dit_model(args, …)`
(`library/inference/models.py:76`) wraps it but reads everything off a Namespace
and also handles LoRA attach. An embedder has to know which one to reach for.

**Suggestion:** keep `load_anima_model` as the explicit primitive and make
`load_dit_model` an obvious thin Namespace-adapter (docstring cross-link), or
collapse the LoRA-attach into a separate `attach_adapters(dit, weights, …)` so
the loader does one thing.

---

## 6. VAE load leaves dtype/eval to the caller  ·  low / trivial  ·  ✅ DONE

**Resolved:** `load_vae(..., dtype=None, eval=False)` — pass
`dtype=torch.bfloat16, eval=True` for a ready model. Both default off to keep
the historical raw-load behaviour. All examples now use the kwargs.

---

## 7. No in-memory "latent → image" exit  ·  medium / low  ·  ✅ DONE

**Resolved:** `library.inference.output.decode_to_pil(vae, latent, device) ->
Image` (+ the shared `pixels_to_pil` it and `save_images` both call). Exported
from `library.inference`; `06_vae_and_dataset.py` now uses it instead of
open-coding the denormalization.


`save_output` couples VAE decode + denormalization + metadata + file write. An
embedder who wants a `PIL.Image` (to composite, to return over HTTP, to score)
has to either write a temp PNG or reassemble the decode by hand.
`decode_to_pixels` exists on the VAE but the full latent→PIL path (incl. the
`[-1,1]→[0,1]` + channel handling) lives inside `save_output`.

**Suggestion:** factor a `decode_to_pil(latent, vae) -> Image` that `save_output`
also calls. (`06_vae_and_dataset.py` open-codes exactly this.)

---

## 8. Text encoding needs the DiT, and via global strategy singletons  ·  medium / medium

Producing a DiT-ready cross-attn embedding requires (a) setting two global
strategy singletons (`TokenizeStrategy.set_strategy` / `TextEncodingStrategy`)
and (b) the DiT itself, because the encoder-hidden-state projection lives on
`Anima._preprocess_text_embeds`. Forget the strategies and you get a cryptic
`get_strategy()` failure; the coupling to the DiT is surprising given it's
nominally "text" encoding (`05_load_models.py` documents both).

**Suggestion:** a `TextConditioner` (or a `prepare_text_inputs` that lazily
initializes the strategies from paths if unset) would make prompt-encoding a
one-liner that fails loudly with a clear message when misconfigured.

---

## 9. Naming/location: `CachedDataset` under `library.datasets.distill`  ·  low / low

It's the general reader for the preprocessed train cache (latent + TE + pooled),
not distill-specific — `06`'s cache iteration is a mainstream use. The `distill`
module path undersells it.

**Suggestion:** move/alias to `library.datasets.cache` (or re-export from
`library.datasets`) so general consumers don't import from `distill`.

---

## 10. Vestigial `GenerationSettings`  ·  low / trivial

`GenerationSettings` (`generation.py:137`) carries `device` + an
explicitly-unused `dit_weight_dtype`. It's threaded through `generate()` but adds
a step without earning it.

**Suggestion:** fold `device` into the request object (#2) and drop the class, or
document why it's a seam (future per-call dtype override).

---

### Smaller notes
- `generate()` mutates its input (`args.seed = seed`, `generation.py:861`) — a
  surprise for callers reusing one args object across seeds. Return the resolved
  seed instead.
- `inference.parse_args` is now `parse_args(argv=None)` (added for these
  examples) — `train.setup_parser()` already returns a parser, so both paths can
  take an explicit argv; #3 is the remaining gap.
