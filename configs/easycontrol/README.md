# configs/easycontrol/

Auto-generated EasyControl dataset blueprints for **curated / mined** pair
trees — the output side of the EasyControl builder workflow, kept apart from the
shipped training config (`configs/methods/easycontrol.toml`) and the canonical
dataset blueprint (`configs/datasets/easycontrol.toml`).

| File | Producer | Points at |
|---|---|---|
| `near_twins.toml` | `python -m easycontrol_adapters.tools.near_twin` | the materialized `_tags`/`_no_tags` near-twin pair tree under `post_image_dataset/easycontrol/near_twins/staging/` (VAE/TE caches land beside it under `near_twins/cache/`) |

`near_twins.toml` carries three hand-edited tables above the generated
blueprint: `[staging]` (mining run knobs, read by the miner — legacy name
`[miner]` still accepted), `[preprocess]` (VAE/TE caching knobs, read by
`make easycontrol-preprocess EASYADAPTER=near_twin`), and the optional
`[training]` table (folded into `--key value` CLI overrides by
`make easycontrol EASYADAPTER=near_twin`). All three survive re-runs of the miner
verbatim; only the blueprint tail (below the sentinel) is rewritten.

Train on the mined tree with:

```bash
make easycontrol-staging    EASYADAPTER=near_twin   # mine pair tree
make easycontrol-preprocess EASYADAPTER=near_twin   # VAE/TE caches
make easycontrol            EASYADAPTER=near_twin   # train (easycontrol method
                                                    #   + near_twins blueprint
                                                    #   + [training] overrides)
```

The training run uses the shipped `configs/methods/easycontrol.toml` method.
train.py's dataset-config validator only accepts `[general]`/`[[datasets]]`, so
the train step first extracts the blueprint into a generated sidecar
(`post_image_dataset/easycontrol/near_twins/dataset_config.toml`, regenerated
each run) and points `--dataset_config` there; keys in `[training]` override the
method defaults (CLI wins the merge chain).

**It's a text-removal control task.** The generated blueprint pairs each twin via
`cond_cache_dir` (EasyControl roles: `cache_dir` = denoising target,
`cond_cache_dir` = `set_cond` reference):

- **target** = the clean `_no_tags` member (+ its caption) — what the model
  generates. `path_pattern = '*_no_tags.*'` keeps only these as targets.
- **cond** = the paired `_tags` latent (the text-bearing reference), symlinked
  into `cond/` under the `_no_tags` stem by the preprocess step (step 4 of
  `make easycontrol-preprocess EASYADAPTER=near_twin` — `_near_twin_build_cond`).
  Same-bucket twins only; a member that bucketed to a different resolution is
  skipped with a warning.

So the adapter learns: given a text-bearing panel as the reference + a clean
caption, regenerate the clean version. At inference, feed a `_tags`-style page as
the EasyControl condition. The `cond/` tree is pure symlinks over `cache/` and is
rebuilt from scratch on every preprocess run.

These start as **seed / eval** pair sets: each accepted pair is a `{id}_tags` /
`{id}_no_tags` couple (the `_tags` side holds the discriminator attribute). See
`docs/proposal/near_twin_tag_gap_miner.md`.

Source images stay user-facing; VAE/TE/PE caches land under
`post_image_dataset/` via the subset `cache_dir` (the IP-Adapter / EasyControl
pattern). Re-running the miner regenerates the matching `.toml` here.
