# configs/easycontrol/

Auto-generated EasyControl dataset blueprints for **curated / mined** pair
trees — the output side of the EasyControl builder workflow, kept apart from the
shipped training config (`configs/methods/easycontrol.toml`) and the canonical
dataset blueprint (`configs/datasets/easycontrol.toml`).

| File | Producer | Points at |
|---|---|---|
| `near_twins.toml` | `easycontrol_adapters/tools/near_twin_tag_gap_miner.py` | the materialized `_tags`/`_no_tags` near-twin pair tree under `post_image_dataset/easycontrol/near_twins/` |

These are **seed / eval** datasets, not turnkey control-adapter blueprints: each
accepted pair is a `{id}_tags` / `{id}_no_tags` couple (the `_tags` side holds
the discriminator attribute). Wire one into a control-adapter run by pairing the
clean member as the condition (`cond_cache_dir`) — see
`docs/proposal/near_twin_tag_gap_miner.md`.

Source images stay user-facing; VAE/TE/PE caches land under
`post_image_dataset/` via the subset `cache_dir` (the IP-Adapter / EasyControl
pattern). Re-running the miner regenerates the matching `.toml` here.
