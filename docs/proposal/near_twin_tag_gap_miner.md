# near_twin_tag_gap_miner — mining in-artist variant pairs by attribute gap

An **exploration / curation tool** (not a training step) that surfaces
near-duplicate *variant* pairs within a single artist where the two members
differ by a **specified attribute** — e.g. one has a speech bubble and the other
doesn't. Output is an eyeball-first HTML contact sheet plus a TSV (so the criteria
can be tuned by looking at what comes back) and a materialized `_tags`/`_no_tags`
pair tree under `post_image_dataset/easycontrol/near_twins/` ready for an
EasyControl subset.

It's a tool **for EasyControl builders**: the bottleneck in building an
EasyControl adapter (colorize / removal / attribute-edit) is finding real
with-/without-attribute data. This miner finds those pairs *and localizes the
differing region for free* (see Stage B), which is exactly the curation lens that
work needs — both for eval sets and as seed data.

It lives in `anima_lora/` (so it can import the shipped `library.vision`
encoder), but it's dataset-agnostic: point `--image-dirs` at any image tree —
the raw crawl pool (`~/gelcrawl/{retrieved,selected}`) by default, or an
`image_dataset/` you're curating for a control adapter.

## Why this exists / what it feeds

The earlier idea of training an EasyControl *removal* adapter directly on mined
real pairs is the wrong fit: inspecting the canonical example
(`ama_mitsuki/5908478` ↔ `5908479`) showed the two members differ in **bubbles +
expression + crop + watermark all at once** and are **not pixel-aligned**. That's
poor supervision for a single-attribute edit.

So this tool's output is **not** direct EasyControl training data. It feeds:
- **Eval sets** — real with/without-attribute pairs to score a *synthetically*
  trained adapter (e.g. the synthetic speech-bubble-removal variant) on real data.
- **Seed/eval data for unpaired editing** — the `byg_unpaired_editing.md`
  proposal is built exactly for loose, non-aligned, single-attribute pairs.
- **A difference-region mask** — Stage B's unmatched grid cells localize *where*
  the two members differ. That region doubles as a conditioning/eval mask for an
  EasyControl builder ("score the adapter only inside the changed area").
- **Dataset spelunking** — "show me every pair in the corpus that differs only by
  `X`" is a generally useful curation lens.

The ranking is designed around the entanglement problem: it surfaces the
*cleanest* pairs (differ by the target attribute and as little else as possible)
first.

## Backbone: PE-Spatial only (no extra gate)

The embedding backbone is **PE-Spatial-B16-512**, the spatial-fine-tuned
Perception Encoder already shipped with the repo — `make download-models` (and
`make download-pe-spatial`) pulls it from the **public** `facebook/PE-Spatial-B16-512`
HF repo, so it adds **no new gated download** to a checkout that already runs the
pipeline. Loaded via the existing registry:

```python
from library.vision import load_pe_encoder
bundle = load_pe_encoder(device, name="pe_spatial")   # ships in library/vision/encoders.py
out = bundle.encoder(pixel_values)                    # → last_hidden_state [B, T, 768]
#   out.last_hidden_state[:, 0]   = CLS token   → global descriptor (Stage A prefilter)
#   out.last_hidden_state[:, 1:]  = 32×32 patch grid (768-d) → dense match (Stage B)
```

This is why DINOv3 was dropped from the original design: DINOv3 needs a **separate
HF gate** the user has to accept, whereas PE-Spatial is already part of the model
download. And crucially, PE-Spatial gives **both** halves we need from one
forward — a pooled global vector (CLS) *and* the dense grid — so a single encoder
covers the cheap prefilter and the precise rerank. (For pure global
near-duplicate retrieval a self-supervised backbone is arguably crisper, but the
gate cost isn't worth it, and the dense grid is the part that actually carries
this tool.)

## Reused infra

| Asset | Path | Use |
|---|---|---|
| PE-Spatial-B16-512 | `models/pe/PE-Spatial-B16-512.pt` | image embedding backbone (32×32 grid + CLS) |
| Encoder loader | `library.vision.load_pe_encoder(device, name="pe_spatial")` | build + auto-download the tower; no gate |
| Feature cache | `~/.cache/near_twin/<dir-hash>/<stem>.npy` (CLS 768-d + pooled grid) | per-image features, computed once per image |
| Captions | `<dir>/<stem>.txt` (comma-separated danbooru tags) | tag-gap discriminator + "rest-of-tags" similarity |

There is **no DINO `.clf` cache reuse** anymore — the tool maintains its own
PE-Spatial feature cache keyed by source dir, so first run embeds the pool once
and re-runs are seconds.

## Algorithm

Per artist (artists are small — ≤ a few hundred images — so all-pairs is fine; a
kNN bound is the optional escape hatch). The search is a **prefilter → rerank**
pipeline so the expensive dense match only runs on plausible pairs.

1. **Gather members.** Scope = `union` by default (a twin can straddle the
   curated cut — `5908479 ∈ selected`, `5908478 ∈ retrieved`), or restrict to a
   single `--image-dirs` entry.
2. **Embeddings.** Load the cached PE-Spatial features; for any missing stem, run
   `load_pe_encoder(..., name="pe_spatial")` once and cache the CLS vector + the
   pooled grid. L2-normalize.
3. **Stage A — global prefilter (cheap).** All within-artist pairs; cosine on the
   **CLS** descriptor. Keep pairs with `cosine ≥ --sim-min` (default ~0.85). This
   is one dot product per pair and discards the obvious non-pairs for free.
4. **Stage B — dense grid match (the near-twin confirm + diff localizer).** Only
   on Stage-A survivors:
   - Pool each image's 32×32 patch grid down to a coarse `G×G` (default ~7×7 = 49
     cells — hence "matched N/49") via avg-pool, L2-norm per cell. Coarsening is
     cheaper *and* robust to local pixel noise while still localizing.
   - **Mutual nearest-neighbor + ratio test** between the two cell sets; count
     inliers with cell-cosine `≥ --cell-match-min` (default 0.9). Mutual-NN +
     ratio is required — raw "has a >0.9 neighbor" inflates badly on anime art's
     flat color fields (many-to-one matches that mean nothing).
   - Optional **geometric consistency** (RANSAC/homography on matched cell
     coords) to reject "same character, different pose" and estimate the crop
     offset. This folds in what v1 had deferred to a v2 structural-confirm step.
   - **Near-twin** ⇔ inlier fraction `≥ --match-frac-min` (default ~0.66, i.e.
     ~32/49). The **unmatched cells are the difference region** → free attribute
     localizer (where the bubble is), feeding the discriminator and the
     diff-region mask above.
5. **Discriminator (pluggable).** Keep pairs where the chosen attribute is present
   in **exactly one** member:
   - **tag mode** (`--tag "speech bubble"`, or `--tag-any "a,b,c"` for synonyms):
     the tag is in exactly one tagset. **Tags are matched space-insensitively**
     (`speech bubble` ≡ `speech_bubble`), so either danbooru convention works —
     see the tag-form note below.
   - **region mode** (`--region`): use Stage B's unmatched-cell region directly —
     a pair whose difference is one compact region with the right rough size *is*
     a single-attribute pair, no caption tag needed. This is the recommended path
     for **bubbles** (and any visual attribute the captions don't name).
   - **signal mode** (`--signal mit_text`): a per-image scalar differs by
     `≥ --signal-delta` with the low side ≈ 0. The shipped signal is MIT
     text-area fraction (the detector behind `post_image_dataset/masks/`).
   - the interface is `image → set[str]` (tag) / `image → float` (signal) /
     `pair → region` (region), so new discriminators are a few lines.
6. **Rank by edit-cleanliness.** Primary sort = number of *other* differing tags
   (symmetric tag-difference minus the target) ascending, or for tagless
   attributes the compactness / count of extra unmatched cells → pairs that
   differ **only** by the target float to the top; secondary sort = CLS cosine
   descending. `--max-extra-diff N` hard-caps the other-differences.
7. **Output.** Two eyeball artifacts plus a materialized pair export.
   - **HTML contact sheet** (`--out pairs.html`): side-by-side thumbnails with the
     **difference region overlaid**, the tag diff highlighted (added/removed),
     cosine, match fraction, extra-diff count, and which member holds the
     attribute. This is the artifact to eyeball.
   - **TSV** (`--out pairs.tsv`): `artist, id_a, id_b, cosine, match_frac,
     gap_holder, n_extra_diff, extra_diff_tags, diff_bbox` for downstream use.
   - **Materialized pair tree** (`--export-dir`, default
     `post_image_dataset/easycontrol/near_twins/`): each accepted pair is written
     under `{export-dir}/{artist}/` as a `_tags` / `_no_tags` couple — the member
     that **holds** the discriminator attribute (`gap_holder`) is the `_tags`
     side, the clean member is `_no_tags`:

     ```
     post_image_dataset/easycontrol/near_twins/
       ama_mitsuki/
         5908478-5908479_tags.png       # has speech bubble  (gap_holder)
         5908478-5908479_tags.txt        #   its caption sidecar
         5908478-5908479_no_tags.png    # textless counterpart
         5908478-5908479_no_tags.txt
         5908478-5908479_mask.png        # Stage-B diff region (optional, --emit-mask)
     ```

     `{pair}` = the two post ids joined `{id_a}-{id_b}` (sorted, stable). Images
     are **symlinked** by default (copy with `--copy`) from the source tree, and
     the `.txt` caption sidecars are written so an EasyControl `[[datasets.subsets]]`
     can point its `cache_dir` straight at this tree (source dirs stay
     user-facing, caches land under `post_image_dataset/` — the IP-Adapter /
     EasyControl pattern). The export is the **only** training-shaped output; the
     HTML/TSV remain curation-only.

## Tag-form note (was: the speech-bubble caveat)

An earlier draft claimed the curated captions carry **zero** `speech_bubble`
tags. That was a false alarm caused by grepping the **underscore** form: these
captions use danbooru's **display form with spaces**, so the tag is `speech bubble`
— present in **1,212 of 15,905** caption files (plus `thought bubble` ×81). Tag
mode therefore **works for bubbles today**; just query the spaced form, or rely
on the space-insensitive matching in step 5.

The practical consequence:
- **Bubbles via tag mode:** `--tag-any "speech bubble,thought bubble,blank speech bubble"`.
- **Bubbles without captions:** `--region` (Stage B's diff region) — works even on
  an untagged image tree.
- Re-deriving full tags from the gelbooru API is **no longer needed** for bubbles;
  it remains an option only if you want meta tags the curation genuinely dropped.

Tag mode is also the right tool for caption-present attributes — expression
(`closed eyes` ↔ `open mouth` / `tongue out`), clothing, pose, etc. (note the
spaced form on this corpus).

## Knobs (the iteration surface)

```
--tag <t> | --tag-any <a,b,c> | --region | --signal mit_text   # discriminator
--image-dirs <d1,d2,..>   # source trees (default: ~/gelcrawl/{retrieved,selected})
--sim-min 0.85            # Stage-A CLS-cosine prefilter threshold
--grid 7                  # Stage-B pooled grid edge (G×G cells)
--cell-match-min 0.9      # per-cell cosine for an inlier match
--match-frac-min 0.66     # inlier fraction to call a pair a near-twin
--geom-check              # RANSAC consistency on matched cells (reject pose twins)
--rest-jaccard-min 0.6    # "same scene" tag-overlap floor (0 to disable)
--max-extra-diff 6        # cap on non-target differing tags (single-attribute-ness)
--signal-delta 0.04       # signal-mode gap threshold
--id-window N             # optional: only pair posts within N IDs (cheap strong prior)
--artists a,b,..          # restrict run; --per-artist-topk K; --out pairs.html
--export-dir <path>       # materialized pair tree (default post_image_dataset/easycontrol/near_twins/)
--copy                    # copy images into the export tree instead of symlinking
--emit-mask               # also write the Stage-B diff-region mask per pair
```

The intended loop: run → open the HTML → adjust `--sim-min` /
`--match-frac-min` / `--cell-match-min` / `--max-extra-diff` → re-run (features
are cached, so re-runs are seconds).

## Scope / non-goals (v1)

- **Detection + ranking + difference-region localization.** No pixel-exact
  alignment, no training.
- The dense-grid match already provides what the old design deferred to "v2
  structural confirm" — inlier-counting + optional RANSAC reject pose twins and
  estimate crop offset. A full LoFTR/SIFT homography is still an escape hatch if
  the coarse grid proves too blunt on hard cases.
- All-pairs within artist is the v1 search; `--id-window` / kNN are the escape
  hatches if any single artist is too large. The Stage-A prefilter keeps the
  expensive grid match off the all-pairs set.
