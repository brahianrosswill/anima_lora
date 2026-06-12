# REPA Phase 1 — operating point on the relational arm

Status: **levers 1–2 implemented (2026-06-12, default-off); lever 2
(spatial_norm) eyeball-validated and ON in lora.toml; lever 3 CLOSED by the
gradient-heatmap diagnostic (~uniform, see below); lever 1 (anneal) A/B not
started — curve-justified candidate 0.25**. Phase 0 closed 2026-06-12 — see
`docs/experimental/repa.md` (live doc) and
`_archive/proposals/repa_v2_patchwise_pe_spatial.md` (the implemented v2
proposal this succeeds). Relational (Arm B) is the validated arm and the
shipped default; this proposal is about *where to run it*, not whether.

## Phase-0 outcome (context)

Three short runs (baseline / absolute / relational, weight 0.05, block 8,
PE-Spatial): REPA decisively better than baseline on eyeballed grids; B
preferred over A roughly 6:4–7:3. Two papers landed the same week and
independently corroborate the relational read:

- **iREPA** (Singh et al., arXiv:2512.10794): across 27 vision encoders,
  *spatial structure* of the target (pairwise patch-token cosine structure —
  exactly what the Gram form aligns) correlates with generation FID at
  |r| > 0.85, while global semantic quality (linear probing) correlates at
  |r| ≈ 0.26. PE-Spatial-B beats PE-Core-G as a REPA target despite 30pt
  worse ImageNet accuracy (our encoder choice, externally validated). Mixing
  global info into patch tokens monotonically *hurts* generation. And the
  standard MLP projection head is shown to be lossy for spatial structure —
  a direct indictment of Arm A's mechanism.
- **MaskAlign** (Pang et al., arXiv:2606.08788): full-token alignment
  concentrates alignment-gradient norm at stable spatial positions (~21×
  ratio) — a feature-fitting shortcut; aligning on random token subsets
  (25% mask) regularizes it away and improves FID.
- **HASTE** (arXiv:2505.16792, cited by both): REPA helps early training
  and *degrades* generation late — they fix it with early stopping.

## Levers (in priority order)

Each is a small, independent knob on the relational loss in
`library/training/repa.py::extra_forwards`. All default-off; new kwargs need
`NETWORK_KWARGS` registration in `networks/__init__.py`.

### 1. Anneal / cutoff — `repa_anneal_steps` ✅ implemented

Carried over from the v2 proposal ("alignment matters early; the late-run
gradient is where style drifts"), now with published support (HASTE). Hard
cutoff at a step fraction; default candidate 50% of the run. Implementation:
the adapter already sees `ctx` per step — gate the returned scalar on
`global_step / max_steps`. Optionally compare hard cutoff vs linear decay to
zero; hard cutoff first (one knob).

Highest prior of the three: it attacks the exact v1 failure axis (late-run
style drift) and costs one comparison run.

*Implemented semantics*: value in (0, 1] = fraction of `max_train_steps`,
value > 1 = absolute optimizer steps, 0 = off (default). The adapter keeps
its own micro-batch counter (validation passes excluded) and converts via
`gradient_accumulation_steps`; past the cutoff the term is skipped before
the PE-feature transfer.

### 2. Spatial normalization of the PE target — `repa_spatial_norm` ✅ implemented

iREPA's second modification, transplanted to the target side of the Gram:

```
pe = (pe − γ · pe.mean(dim=tokens)) / (pe.std(dim=tokens) + ε)   # γ = 1
```

before per-token L2-norm + Gram. Rationale: pretrained patch tokens carry a
large shared global component, which compresses pairwise cosines toward each
other (foreground tokens look similar to background tokens). The Gram form
cancels a *common additive direction* from the loss only imperfectly — the
global component still sits inside every token's normalization. Subtracting
the spatial mean first sharpens the target affinity contrast — iREPA shows
exactly this raises spatial-structure metrics and generation quality.

Cheap (two lines, no params). Risk: changes the target geometry mid-line —
run as its own A/B, not bundled with lever 1.

*Implemented*: relational mode only, applied after the CLS drop and before
per-token L2-norm (ε = 1e-6, γ = 1).

### 3. Token-subset Gram (MaskAlign, loss-side only) — `repa_token_keep`

Sample a random index subset `S` per step (keep ratio 0.75 to mirror their
25% mask), compute `MSE(G_dit[S,S], G_pe[S,S])`. Breaks the stable
high-gradient-position shortcut MaskAlign documents, and shrinks the Gram
~2× at 0.75 keep.

**Hard constraint: never mask the DiT input.** MaskAlign's input-dropping
variant (and its pre-mask mixing block) is incompatible with constant-token
bucketing + `compile_blocks()` (graphs keyed on token count) and would
perturb the frozen backbone — loss-side subsampling only. Lowest prior:
their shortcut formed over 400K pretraining iterations; a short LoRA run may
never develop it. Run the diagnostic first (below) and skip the lever if the
pathology isn't there.

**Diagnostic (cheap, do before lever 3):** log the per-token alignment-
gradient-norm heatmap (top-10% position frequency, MaskAlign Fig. 2a) on a
current-default run. If the distribution is ~uniform in our regime, close
lever 3 without a training run.

**Result (2026-06-12, `repa_grad_heatmap=1` on `anima_repa_normed_quarter`,
2292 samples): ~UNIFORM — lever 3 CLOSED without a training run.** Top-10%
mean recurrence 1.51× uniform (paper pathology ~21×); no position recurs
>50% of steps (max single-position freq 0.34 = 3.4× uniform, confined to a
top-right corner/edge cluster — flat-background/edge effect, not interior
content positions); every position lands in the top-10% at least once;
coarse 4×4 map flat (0.083–0.135). The shortcut needs pretraining-scale
iteration counts to form, exactly as suspected above. Artifact:
`output/ckpt/anima_repa_normed_quarter_repa_grad_heatmap.npz`; rerun via the
`repa_grad_heatmap` knob (lora.toml, default off) if the regime changes
(much longer runs / different weight).

### 4. Weight sweep — `repa_weight ∈ {0.02, 0.05, 0.1}`

Carried over from the v2 proposal. Only after levers 1–2 settle, on the
winning configuration — weight interacts with cutoff (a shorter window
tolerates a higher weight).

## Protocol

Sequential short runs (the Phase-0 recipe: ~tenth-scale, fixed seed/data,
current lora.toml default stack), one lever at a time vs the current
default (relational, w=0.05, block 8, no anneal):

```
1. anneal 50% vs none          ── expect: equal-or-better style, anatomy kept
2. spatial_norm on vs off       ── on the lever-1 winner
3. gradient-heatmap diagnostic  ── only if concentrated: token_keep 0.75 A/B
4. weight sweep on the winner   ── only if 1–2 moved anything
```

Gates unchanged from Phase 0 (pre-registered): **sample grids first** (style
intact, anatomy — hands/limbs/eyes), CMMD-vs-own-PE-Core as the drift
tripwire, never FM val loss, never fraction-of-Δ readouts. Any lever that
visibly moves style fails regardless of its anatomy delta.

## Phase 2 — graduation (only on a settled operating point)

Tier 1.5 obligations, still outstanding from Phase 0:

- `bench/repa_v2/` bench script (`bench/_common.py` envelope; grids + CMMD
  in `result.json`).
- Invariant test: `use_repa = false` ⇒ training loss byte-identical to
  baseline (extend `tests/test_repa.py`).
- One full-length run at the final operating point vs baseline.
- `configs/gui-methods/` variant + default-on decision for lora.toml.

## Non-goals

- **Arm A revival** — closed by Phase 0 + iREPA's MLP-projector finding. The
  head stays implemented for reference; don't spend runs on it. (If anyone
  revisits absolute alignment, iREPA's conv-projector would be the form to
  try, not the MLP — but that's a new proposal.)
- **Input-token masking / pre-mask mixing** (MaskAlign's full recipe) — see
  the hard constraint under lever 3.
- **Re-litigating encoder choice** — PE-Spatial is doubly validated (in-house
  tagger routing + iREPA's spatial-structure ranking).

## References

- `docs/experimental/repa.md` — live operating reference; Phase-0 verdict.
- `_archive/proposals/repa_v2_patchwise_pe_spatial.md` — implemented v2
  proposal (v1 autopsy, hypothesis, Phase-0 gate design).
- `library/training/repa.py`, `tests/test_repa.py`,
  `networks/__init__.py::NETWORK_KWARGS`.
- iREPA: Singh et al., arXiv:2512.10794 (repo-root PDF `2512.10794v1.pdf`).
- MaskAlign: Pang et al., arXiv:2606.08788 (repo-root PDF `2606.08788v1.pdf`).
- HASTE: arXiv:2505.16792 (early-stopped alignment).
