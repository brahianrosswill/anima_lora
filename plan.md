# DirectEdit smart-edit integration — status + next steps

DirectEdit's edit-instruction handling now does REPLACE, REMOVE, and NOOP in
addition to APPEND, with auto-replace driven by Qwen3 last-token cosine
geometry rather than a hand-curated tag-families YAML. Slot-level surgery
on `crossattn_emb` is wired but off by default.

## TL;DR (shipped)

- **Detection**: Qwen3 last-non-padding-token cosine, threshold + gap-to-#2
  gate. Probe `scripts/probes/edit_nearest_tag.py` scored 7/17 confident
  REPLACE matches with **zero false positives** at the shipped defaults
  (`replace_threshold=0.92`, `replace_gap=0.04`); the remaining 10 are
  graceful APPEND fallbacks. Real-caption sub-probe (cases 11–21) matches
  these numbers under realistic tag density.
- **Execution**: slot-level surgery on `crossattn_emb` (`directedit_splice.py`).
  Probe `scripts/probes/edit_slot_alignment.py` scored 10/10 clean
  contiguous diff spans. Wired behind `--use_slot_surgery` / a node toggle,
  default off — needs Phase 6 A/B before becoming the default.
- **Dispatcher**: REMOVE on explicit `-X` or `no X` (matching tag); NOOP when
  edit phrase is already in ψ_src; REPLACE on confident cosine match; APPEND
  otherwise.
- **Failure mode is graceful**: when detection is uncertain, fall through to
  APPEND — same as the pre-dispatcher world, no regression.

## Decisions (resolved)

| Decision | Why |
|---|---|
| Drop YAML tag-families. | Brittle, never complete; can't handle out-of-vocab concepts. |
| Use last-non-padding-token pool, not mean pool. | Probe: mean-pool 7/17, last-pool 12/17. Decoder-only LMs concentrate phrase semantics in the final token. |
| Threshold + gap gating on detection. | Probe shows ambiguous and no-conflict cases sit at gap < 0.01; real REPLACE cases sit at gap ≳ 0.07. Requiring both top-1 cosine AND a notable gap to #2 separates them honestly. |
| Slot-level surgery on crossattn_emb, not full re-encode. | T5 tokenization produces a clean contiguous diff span for every edit kind in the probe. Surgery preserves untouched slots from ψ_src; only the diff range is transplanted from ψ_tar. |
| Default to APPEND when uncertain. | Honest failure mode — preserves baseline behaviour. |
| Default `use_slot_surgery` OFF. | Cross-attn drift outside the diff span hasn't been A/B'd against full re-encode. Phase 6 territory. |
| `use_dispatcher` defaults ON in the comfy node. | Worst case = APPEND = legacy behaviour. Only cost is one batched Qwen3 forward over the tag list. |
| NOOP short-circuits before the encoder forward. | Cheaper, AND avoids the dispatcher picking a *different* cosine-near tag (e.g. user retypes "smile" while caption has "grin" — REPLACE would corrupt; NOOP keeps caption clean). |

## Dispatcher behaviour table

| Edit input | Caption has it (exact, case-insensitive)? | Intent |
|---|---|---|
| `-X` | yes | REMOVE (strip) |
| `-X` | no | REMOVE (no-op, caption unchanged) |
| `no X` | yes | REMOVE (strip) |
| `no X` | no | falls through; `no X` literal goes to NOOP/DETECT/APPEND |
| `X` (plain) | yes | **NOOP** (caption unchanged, no encoder call) |
| `X` (plain) | no, but cosine-near another tag (top1 ≥ 0.92, gap ≥ 0.04) | REPLACE |
| `X` (plain) | no, far from everything | APPEND |

## Architecture (shipped)

### `library/inference/edit_dispatcher.py`

```python
def derive_target_caption(
    src_caption: str,
    edit_instruction: str,
    *,
    encode_last_pooled: EncodeLastPooledFn,   # (list[str]) -> (N, D) fp32
    replace_threshold: float = 0.92,
    replace_gap: float = 0.04,
) -> EditPlan
```

Encoder shim is caller-supplied so the Anima CLI (strategy trio) and the
ComfyUI node (comfy `CLIP` socket) share this code path. Anima callers wrap
`encode_last_pooled_via_anima_strategy`; the comfy node uses
`_comfy_encode_last_pooled` which does **one batched** Qwen3 forward via
the bare `cond_stage_model.qwen3_06b` (bypassing `encode_from_tokens` to
avoid N+1 `load_models_gpu` spam).

### `library/inference/directedit_splice.py`

```python
def find_t5_diff_span(src_ids, tar_ids, pad_id) -> T5DiffSpan
def splice_crossattn_emb(*, crossattn_emb_src, crossattn_emb_tar,
                         t5_ids_src, t5_ids_tar, pad_id) -> (Tensor, T5DiffSpan)
```

`T5DiffSpan` carries `(start, src_end, tar_end, src_len, tar_len)` for both
the splice machinery and downstream logging.

### Wiring

| File | What changed |
|---|---|
| `scripts/edit.py` | `--edit_instruction`, `--replace_threshold`, `--replace_gap`, `--use_slot_surgery` flags. TE loaded once, shared between dispatcher and prompt encoding. |
| `scripts/experimental_tasks/inference.py` (`cmd_test_directedit`) | Passes `--edit_instruction` so the dispatcher runs in-process — avoids double-loading Qwen3. |
| `custom_nodes/comfyui-anima-directedit/nodes.py` | Sockets: `use_dispatcher` (default ON), `replace_threshold`, `replace_gap`, `use_slot_surgery` (default OFF). `_encode_prompt_comfy` returns T5 IDs alongside the embedding for surgery. |
| `scripts/sync_vendor.py` | Bundles `edit_dispatcher.py` + `directedit_splice.py` into the directedit `_vendor/` tree. |
| `tests/test_edit_dispatcher.py` | 23 unit tests covering REMOVE/REPLACE/APPEND/NOOP, threshold gating, span finder, splice replace/add/remove + shape guards. |

## What's left

### Phase 6 — empirical validation (the big one)

Bench-style script in `bench/directedit/` (or `scripts/probes/`): fixed set
of reference images × edit instructions × variants:

1. baseline: current append-only (legacy behaviour)
2. dispatcher-only, string-level (REPLACE/REMOVE/APPEND/NOOP, no surgery)
3. dispatcher + slot surgery
4. dispatcher + **full re-encode** (no surgery, but uses the dispatcher's
   ψ_tar)

The (3)-vs-(4) comparison is the load-bearing one — it tells us whether
slot surgery is worth the complexity over just re-encoding the
dispatcher-derived ψ_tar. The plan currently assumes surgery preserves
unchanged content better; that's untested. **Don't promote
`use_slot_surgery` to default-on without this.**

Expand the probe regression set from ~22 cases to ~50, including the known
miss modes: pose swaps (sitting/standing), fine hair-length granularity
(medium ↔ long), and hairstyle-vs-length conflicts (twintails vs long+ponytail).

### Phase 7 — docs

Update `docs/experimental/directedit_editing_v3.md` with:
- The new edit-instruction syntax (`-X`, `no X`, plain tag).
- Dispatcher behaviour table (copy from this file).
- A "failure modes" section listing the 3 probe misses by category:
  pose swaps, fine hair-length, no-surface-overlap synonyms.
- Slot-surgery caveats (the list below).

## Known issues / follow-ups (deferred)

These are documented now so the next pass knows where to look:

**Slot-surgery semantic edge cases** (all silent — would produce
suboptimal results, not crashes):

1. **Cross-attn drift outside the diff span.** The LLM Adapter cross-attends
   Qwen3's whole hidden state, so even the "unchanged prefix" slots of
   ψ_src and ψ_tar differ subtly. We keep ψ_src's. Phase 6's (3)-vs-(4)
   tells us if this is visible.
2. **REMOVE leaves residual context in the suffix.** The slots we keep
   from ψ_src "saw" the removed tag through Qwen3's causal context. Faint
   echo of the removed concept may survive.
3. **Two-or-more non-contiguous edits widen the span.** Diff = LCP + LCS,
   so unchanged tags between two edits get engulfed in the diff range and
   come from ψ_tar's encoding instead. Not wrong, just weakens the
   "minimal perturbation" sales pitch.
4. **First-tag edits hit T5 leading-space quirks.** Position-0 tokens
   (e.g. `1girl → 2girls` without a leading comma) may tokenize
   asymmetrically. Probe case 9 happened to be clean; not exhaustive.
5. **Wildly different ψ_src/ψ_tar degenerates to ~full re-encode.** If
   captions share almost nothing, LCP ≈ 0 and LCS ≈ just `</s>`. Surgery
   transplants ~all of ψ_tar with a 1-token ψ_src tail.

**Cheap fixes available** (haven't shipped yet):

- **NOOP + use_slot_surgery fast-path.** When the dispatcher returns NOOP,
  `tar_caption == src_caption` and splice produces something equal to
  `embed_src`. Skip the second TE pass entirely. ~5 LOC.
- **>512-token-caption guard in `splice_crossattn_emb`.** T5 truncation can
  land mid-tag. Assert non-pad length < 512 and abort surgery cleanly if
  violated. ~10 LOC.
- **REMOVE + use_slot_surgery warning.** When `intent == "remove"` and
  surgery is on, log the residual-context caveat so users know the
  tradeoff. ~5 LOC.
- **T5 pad_id validation in the comfy node.** Currently hardcoded to 0
  (correct across the T5 family but unvalidated). Tiny audit: read it from
  the loaded tokenizer if accessible. ~10 LOC.

**Dispatcher detection gaps** (real, would need a different approach to fix):

- **Pose swaps** (sitting/standing, lying/standing). Cosine geometry
  doesn't separate these well. Probe case 6 + 14 both miss.
- **No-surface-overlap synonyms** (twintails vs long blonde hair, hairstyle
  vs hair-length). Cosine sees them as distant tags. Probe case 5 + 17.
- **Fine hair-length granularity** (medium ↔ long, near-tie at gap ≈ 0.003).
  Threshold correctly abstains into APPEND — honest, but not a "fix".
- **Single-letter typos / paraphrases.** Out of scope — dispatcher is for
  tag-level edits only.

## Explicitly out of scope (long-term parking lot)

- **LLM dispatcher.** Falls into the toolbox if cosine detection + Phase 6
  surgery still aren't enough. Adds latency + dep + hallucination risk.
- **Post-LLM-Adapter (crossattn_emb) similarity probe.** Last-token-pool on
  Qwen3 is good enough; loading the DiT just to compute detection scores
  isn't worth it.
- **SEGA-style score-space guidance.** Different mechanic; could compose
  later as an independent feature.
- **Continuous-slider edits** ("30% larger"). Not in the APPEND/REMOVE/
  REPLACE/NOOP universe.
- **General natural-language edit rewriting.** Dispatcher treats the edit
  instruction as a single tag or short tag phrase; full free-form requires
  the deferred LLM path.

## Probe regression set (last run)

`scripts/probes/edit_nearest_tag.py` — 22 cases. Last-pool 12/17 (the 5
no-conflict APPEND cases aren't scored).

| # | edit | expected | result |
|---|---|---|---|
| 0–4, 10 | various REPLACE cases | various | last-pool OK |
| 5, 6 | twintails / sitting→standing | hairstyle / pose | last-pool MISS (graceful APPEND) |
| 7, 8 | holding sword / cat ears | no-conflict | OK (APPEND) |
| 9 | large breasts (w/ "breast tattoo" present) | breast tattoo | MISS (real-caption case 19 with denser context lands correctly) |
| 11–21 | real-caption cases | various | 12 wins, 10 graceful APPEND fallbacks |

Phase 6 should grow this to ~50.
