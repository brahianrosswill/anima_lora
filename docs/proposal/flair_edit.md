# FLAIR-edit — training-free localized editing without a source prompt

Status: **PHASE A + B IMPLEMENTED (2026-06-14).** Phase A (explicit-mask MVP) and
Phase B (SAM3 text→mask) are wired into the engine; Phase C (ComfyUI node) is not
yet done. The solver+operators are promoted to
`library/inference/corrections/flair.py` / `flair_operators.py` (`InpaintOperator`
added; `bench/flair/{solver,operators}.py` are now re-export shims), the FLAIR
branch dispatches in `generate()` on `--flair_task edit`, and Request/CLI carry the
`flair_*` knobs. The user-facing target is **`make exp-test-flair`** — reframed from
the proposed `exp-test-flair-edit` into a **reconstruction observer**: its default
(no `MASK`/`MASK_PROMPT`/`PROMPT`) knocks out a *random* region and refills it from
the image's own **canonical caption** so the prior's fill can be eyeballed before
trusting real edits; `MASK=` / `MASK_PROMPT=` / `PROMPT=` turn it into the
productized delta-only edit. Invariant tests in `tests/test_flair_edit.py`. The
app-layer benches (`edit_vs_directedit`, `edit_mask_prompt`) and the unmasked-
exactness hard gate remain owed. The **method/bench** layer is the parent proposal
[`flair_inverse.md`](flair_inverse.md) (Phase 0 PASSED, Phase 1 λ_R calibrated
2026-06-14); its **Phase 3 — Pilot A: FLAIR-inpaint-with-prompt** is the bench gate
this proposal productizes. Nothing here changes the math — it takes the validated
inpaint solver and wires the one property that makes it worth shipping over
DirectEdit: **the user describes only the edit (the delta), never the source image.**

This proposal replaces the retired SR-deploy proposal (`flair_sr.md`, removed
2026-06-14) as FLAIR's first user-facing application. Editing is the higher-value
front door: it lands on a documented repo pain point (DirectEdit's ψ_src burden,
the whole reason the Anima Tagger exists) instead of merely filling a missing
capability.

## The thesis — FLAIR-inpaint eliminates ψ_src structurally

DirectEdit's source-prompt requirement is **structural, not incidental**
(`docs/experimental/directedit_editing_v3.md`):

- Inversion queries `v_θ(z, σ, embed_src)` at every step and records `Δz`. The edit
  is then the **embedding difference** ψ_tar − ψ_src driving the forward pass off
  that anchor. A bad ψ_src ⇒ wrong inversion trajectory ⇒ "edit leverage collapses"
  (the documented failure on line 47 of the v3 doc).
- The edit is a *vector in caption space*; a meaningful delta vector needs a clean
  basepoint. That is the entire job of the Anima Tagger — not reconstruction (Δz
  makes that bit-exact regardless) but making ψ_tar − ψ_src a *correct direction*.

So DirectEdit asks the user (via the tagger) to **describe the whole image
accurately, then describe it again with the change.**

FLAIR-inpaint is `A = mask`, `y = the visible pixels`. Algorithm 1 has **no
inversion and no ψ_src anywhere**. The prompt enters in exactly one place — the
prior pull `v_θ(x_t, t, ψ_tar)` — and Hard Data Consistency (HDC) re-projects the
unmasked region bit-exact every step. Consequences:

- The **unmasked region is locked by the data term**, not by a prompt. The user
  does not describe it at all.
- The **masked region is filled by the prior conditioned on ψ_tar**, with the
  surrounding visible context bleeding in through the trajectory (HDC re-pins
  context each step, so the fill is coherent with what's around it).

That is exactly the property the user wants: **a sparse, delta-only prompt
(`"blue hair"`, `"glasses"`) works because the rest is held by pixels, not words.**
The ψ_src burden — and with it the tagger dependency for editing — disappears.

## The honest catch — it trades a semantic burden for a spatial one

FLAIR does not make editing free; it **moves the cost from "describe the image" to
"localize the edit."** The win is real because spatial localization is usually
cheaper than accurate full-image tagging — and (below) it can be automated from the
edit instruction itself. But the boundary must be stated up front:

| | DirectEdit | FLAIR-edit (this proposal) |
|---|---|---|
| Source spec | full ψ_src (Anima Tagger) | **none** |
| Edit spec | ψ_tar = ψ_src + delta | **delta-only prompt** |
| Localization | none (mask-free, global) | **requires a mask** |
| Edit class | any global semantic edit | linear-`A` / masked region only |
| Unmasked region | can drift (whole-frame) | **bit-exact (HDC gate)** |

So "describe only the delta" is achievable **iff the edit is spatially
localizable.** Add glasses / change a hairstyle / recolor a garment / remove an
object — yes. Mask-free global re-stylings ("make it a sunset," "change the pose")
have no linear `A` and no mask — that stays **DirectEdit's** domain. FLAIR-edit is
**additive and complementary**, not a DirectEdit replacement (parent proposal line
312).

## The front-end that makes it "just tell me the change" — text→mask via SAM3

A manual mask still asks the user for spatial input. The product-grade version
removes even that by deriving the mask from the edit instruction, using assets the
repo **already ships**:

1. User says `"give her glasses"`.
2. **SAM3 open-vocabulary concept segmentation** localizes the region from a short
   text phrase — `processor.set_text_prompt(prompt="eyes")` /`"face"` (SAM3 lives
   in `../sam3/`, driven by `make mask` / `scripts/tasks/masking.py`, masks land in
   `post_image_dataset/masks/`; it is built for exactly this — "exhaustively segment
   all instances of an open-vocabulary concept specified by a short text phrase").
3. **FLAIR-inpaint** runs with prompt `"glasses"`, HDC locking everything else.

This removes **both** burdens at once — no ψ_src (no tagger) **and** no hand-drawn
mask — which is the closest the stack can get to "the user only describes the
delta." The mask becomes auto-derived rather than user-drawn. Crucially this reuses
the SAM3 path verbatim; the only new glue is mapping the *edit* phrase to a *region*
phrase (often the same words, sometimes a noun extracted from the instruction).

Both paths ship: **explicit mask** (drawn / supplied binary mask) as the reliable
core, **text→mask** as the convenience layer on top. The explicit path is the
correctness anchor; the SAM3 path is best-effort localization that degrades to "draw
it yourself" when the concept isn't found.

## Relationship to the three neighbors (all distinct, all kept)

- **vs DirectEdit** (`docs/experimental/directedit_editing_v3.md`) — complementary.
  FLAIR-edit owns *localized* edits with a delta-only prompt and bit-exact context;
  DirectEdit owns *global mask-free* semantic edits. Keep both.
- **vs `easycontrol_inpaint.md`** (the proposed **trained** EasyControl variant) —
  the head-to-head worth running. EasyControl-inpaint is a frozen-DiT adapter that
  *learns* to fill gray holes from a mined dataset; FLAIR-inpaint is **training-free**
  with **no mining bias** and a **hard** (not learned) data-consistency guarantee on
  the kept region. FLAIR wins on "no training, exact context, any caption"; the
  trained adapter may win on fill *quality* where its data covers the distribution.
  This proposal does **not** block `easycontrol_inpaint.md` — they are the two ends
  (training-free vs trained) of the same task and the pilot benches them against each
  other.
- **vs parent `flair_inverse.md` Phase 3** — that is the *method* pilot
  (`bench/flair/pilot_inpaint.py`, the unmasked-exactness gate + masked-region
  quality). This proposal is the *application*: the engine branch, the `make` target,
  the SAM3 front-end, the ComfyUI node, and the **equal-effort** comparison against
  DirectEdit.

## Method recap — FLAIR-inpaint (Algorithm 1, `A = mask`)

From the parent proposal's Algorithm 1, specialized to inpaint:

```
mask m (1 = keep, 0 = fill), source image x_src
y      ← m ⊙ x_src                              # observation = visible pixels
μ      ← E(A^T y) = E(m ⊙ x_src)                # adjoint init (gray hole encodes flat)
for t = 1 → t_stop (descending):
    x_t ← (1−t)·μ + t·ε̂
    u_t ← (ε̂ − x_t)/(1−t)
    μ   ← μ − λ_R(t)·( v_θ(x_t, t, ψ_tar) − u_t )    # (i) prior pull, ψ_tar = DELTA-ONLY prompt
    μ   ← argmin_μ ‖ m ⊙ (D(μ) − x_src) ‖²           # (ii) HDC: lock kept region bit-exact
    x̂_1 ← x_t + (1−t)·v_θ(x_t, t, ψ_tar)
    ε̂   ← α·x̂_1 + √(1−α²)·ε                          # (iii) DTA (run near α→1 for repeatable edits)
```

The `inpaint` operator (binary mask, self-adjoint) and `pilot_inpaint.py` are
already scoped in the parent proposal (lines 152, 253-268). For editing repeatability
run near the **deterministic end (`α→1`)** — posterior diversity is a feature for
restoration but a liability for "apply this one edit" (parent risk, line 304).

## What ships — where it hooks (verified against the live tree)

In dependency order, mirroring the bench→library→engine→node lift the layering
contract blesses.

1. **Promote the inpaint solver** `bench/flair/{solver,operators}.py` →
   `library/inference/corrections/flair.py` (+ `flair_operators.py`). Shared with any
   other promoted FLAIR task; the bench keeps a thin shim so Phase 1/2 benches keep
   running. The solver is already written against
   `library.inference.sampling.get_timesteps_sigmas`,
   `library.runtime.harness.build_anima`, the Qwen VAE, and
   `library.env.resolve_under_home` (the λ_R loader) — a near-verbatim move. Add the
   `InpaintOperator` (binary mask, self-adjoint) alongside the existing `SROperator`.

2. **Engine branch + Request/CLI plumbing.** FLAIR *replaces* the sampling loop (it
   optimizes `μ`, it does not denoise a fixed trajectory), so this is a top-of-body
   branch in `generate_body` (`library/inference/generation.py:505`), **not** a
   sampler-boundary plug-in like CNS/DAVE. Dispatch when `args.flair_task == "edit"`.
   - **GenerationRequest** (`request.py:50`) — mirror the `easycontrol_image` /
     `easycontrol_weight` optional-field pattern already there: `flair_task:
     Optional[str]` (`"edit"`), `flair_edit_image: Optional[str]` (the source),
     `flair_mask: Optional[str]` (binary mask PNG), `flair_mask_prompt:
     Optional[str]` (the SAM3 concept phrase; when set, derive the mask),
     `flair_reg_scale: float = 0.5`, `flair_alpha: float = 0.9` (near-deterministic
     default for editing — *unlike* SR's 0.5), `flair_hdc_steps: int = 8`,
     `flair_hdc_lr: float = 0.1`, `flair_calib: str = "auto"`. The **prompt** field
     carries the delta. Long-tail knobs ride `extra_argv` first per the embedder
     contract.
   - **argparse** (`library/inference/args.py` `build_parser`) — matching `--flair_*`
     flags; post-parse validation in `build_default_args` (exactly one of
     `flair_mask` / `flair_mask_prompt` set when `flair_task=edit`; edit_image
     exists). Defaults = the benched working point.

3. **`make exp-test-flair-edit` target** — experimental inference target, mirroring
   `exp-test-directedit` (`scripts/experimental_tasks/inference.py`) for the
   `REF_IMAGE`/`_resolve_ref_image` plumbing and `test-easycontrol`
   (`scripts/tasks/inference.py`) for image-arg injection:

   ```bash
   # explicit mask:
   make exp-test-flair-edit REF_IMAGE=girl.png MASK=hair.png PROMPT="blue hair"
   # text→mask (SAM3 derives the region from MASK_PROMPT):
   make exp-test-flair-edit REF_IMAGE=girl.png MASK_PROMPT="eyes" PROMPT="glasses"
   ```

   `REF_IMAGE` → source; `MASK` → `--flair_mask`; `MASK_PROMPT` → `--flair_mask_prompt`
   (routes through the existing SAM3 path); `PROMPT` → the delta. Registered in
   `tasks.py` `COMMANDS` alongside the other `exp-test-*` entries. **Compose flags:**
   FLAIR-edit is its own sampling path, so it does **not** compose with
   `SPECTRUM`/`SPD`/`MOD`/`DAVE` (those decorate the standard loop) — documented, not
   silently dropped.

4. **SAM3 text→mask glue** `library/vision/` (or reuse `scripts/tasks/masking.py`)
   — a small `mask_from_concept(image, phrase) -> binary_mask` wrapper over SAM3's
   `set_text_prompt`, returning the union of matched instances (optionally dilated a
   few px so the fill has margin). This is the one genuinely new capability beyond the
   parent bench; everything else is marshalling. Cache nothing — it's per-edit.

5. **ComfyUI node** `custom_nodes/comfyui-anima-flair-edit/`, templated on
   `comfyui-anima-directedit/` (the closest analog — training-free, IMAGE in, solves
   on the DiT, returns latent). `MODEL` + `VAE` sockets, `IMAGE` (source), `MASK`
   (Comfy mask socket) **or** `mask_prompt` STRING (SAM3), `prompt`/`negative`,
   `reg_scale`/`alpha`/`hdc_steps`. Returns `LATENT` (compose with `VAEDecode`),
   matching directedit. **The one sharp edge** (inherited from the parent + the
   retired SR node design): HDC backprops through the VAE decoder, but ComfyUI runs
   VAE under `no_grad`/tiled-decode — the node must drive its **own grad-enabled
   decode loop on `μ`** (decoder in eval, autograd on `μ`), never Comfy's `VAEDecode`.
   Call it out loudly in the node doc + code comment so a future refactor doesn't
   "optimize" it into the no-grad decoder and silently kill data consistency.
   `_vendor/`-synced via `make vendor-sync` (never `cp` — `[[feedback_vendor_sync]]`).

6. **Docs** — user-facing `docs/inference/flair_edit.md`; one-liner row + the
   `make exp-test-flair-edit` knobs in root `CLAUDE.md` once landed.

## What to bench — the headline is "equal user effort" (Tier-2: bench + test)

The **method** numbers (unmasked exactness, masked-region LPIPS/CMMD, the DTA
ablation) are the parent proposal's Phase 3 job. This proposal owns two app-layer
claims:

1. **`bench/flair/edit_vs_directedit.py` — the delta-prompt advantage at equal
   effort.** The point of the whole proposal. Same edit set, two arms:
   - **FLAIR-edit**: delta-only prompt (`"glasses"`) + mask (drawn or SAM3-derived).
   - **DirectEdit**: full ψ_tar = tagger(ψ_src) + delta, the shipped pipeline.

   Metrics: **(a) edit success** (CLIP-sim to the intended change + a qualitative
   anime-style pass); **(b) context preservation** — FLAIR's structural advantage:
   PSNR on the unmasked region should be ≈ ∞ (HDC), DirectEdit's whole-frame drift
   shows up as finite/lower PSNR away from the edit; **(c) effort** — FLAIR's input is
   "2-word prompt + mask"; DirectEdit's is "tagger run + edited tag string." The
   readout: **FLAIR-edit wins iff it matches DirectEdit's edit success while holding
   context bit-exact at strictly lower user-specification burden.** A tie on quality
   with exact context is still a win (DirectEdit can't promise the context).

2. **`bench/flair/edit_mask_prompt.py` — the SAM3 front-end is good enough.** Does
   text→mask localize well enough that the delta-only UX actually holds end-to-end?
   Compare FLAIR-edit with a hand-drawn mask vs the SAM3-derived mask on the same
   edits: masked-region quality delta, and a localization-IoU sanity (SAM3 mask vs a
   reference mask) per concept. Documents which concept classes the SAM3 path is
   reliable for vs where the user must fall back to drawing — **no silent cap**.

3. **Node smoke**: headless graph (IMAGE + MASK → AnimaFlairEdit → VAEDecode →
   SaveImage) produces a coherent local edit with bit-exact context; **verify the
   grad path actually engaged** (the node-edge regression — if HDC didn't backprop,
   the kept region drifts).

### Invariant test (`tests/test_flair_edit.py`)

(a) `InpaintOperator` adjoint dot-product test `⟨Ax,y⟩=⟨x,A^Ty⟩` and self-adjointness
`A = A^T`; (b) **full-keep mask + α=1 + 0 prior weight ⇒ near-identity** on a clean
latent (the loop doesn't corrupt data it's handed — the parent's invariant (b)); (c)
**unmasked-region exactness**: after a solve, `m ⊙ D(μ) ≈ m ⊙ x_src` to a tight
tolerance (the load-bearing HDC guarantee — if this fails the whole ψ_src-free claim
collapses); (d) dim-2 round-trip (encode→solve→decode preserves `(W,H)`, never
`squeeze()` the batch); (e) Request→argv→engine reaches the FLAIR branch (a
`flair_task=edit` request does not fall through to the Euler loop).

## Phasing

- **Phase A — explicit-mask MVP.** Promote the inpaint solver + operator, wire
  Request/CLI + `exp-test-flair-edit` with a **supplied/drawn mask**, parity +
  invariant tests, the `edit_vs_directedit` bench. Shippable on its own: it already
  delivers "delta-only prompt, bit-exact context" — the core thesis — for users
  willing to provide a mask. Gated on parent Phase 3 passing the unmasked-exactness
  gate.
- **Phase B — SAM3 text→mask front-end.** `mask_from_concept` + `MASK_PROMPT` plumbing
  + the `edit_mask_prompt` bench. This is what makes it "just tell me the change."
  Gated on Phase A.
- **Phase C — ComfyUI node + UX polish.** The node (with the HDC-grad edge), the
  `--flair_mask_dilate` margin knob, `docs/inference/flair_edit.md`, tooltips. Promote
  to a standalone repo on publish like the directedit/DAVE/PiD nodes once stable.

## Risks / honest limits

- **Localizable edits only.** Mask-free global re-stylings have no linear `A` —
  that's DirectEdit's job. This is a hard boundary, stated, not a TODO.
- **Compute/memory**: HDC backprops through the VAE decoder every σ-step — the real
  cost and the OOM risk. Editing usually runs at a single supported tier (no SR-style
  upscaling), so the tier cap is naturally satisfied; cap `hdc_steps` and keep the
  decoder resident. Parent Phase 0 proved 512px fits a 16 GB card.
- **SAM3 localization is best-effort.** Open-vocabulary concept seg is strong but
  not perfect; the explicit-mask path is the correctness anchor and the SAM3 path
  degrades to "draw it yourself" when a concept isn't found. The `edit_mask_prompt`
  bench documents the reliable concept classes.
- **Domain prior bound.** Fill quality is bounded by what the Anima prior knows
  (parent caveat). In-domain anime/illustration edits are the target; out-of-domain
  fills look like Anima, not like the input's style.
- **Posterior sampling, not deterministic.** Run near `α→1` for repeatable edits
  (default `flair_alpha=0.9` here, vs SR's 0.5). Diversity across seeds is available
  but off the editing happy-path.
- **Prompt-context coherence on very sparse prompts.** The prior pull conditions the
  *whole* latent on ψ_tar while HDC pins the kept region; with an extremely terse
  prompt the masked fill is coherent with context (standard prompt-inpaint behavior)
  but global prompt adherence is weaker than a full caption would give. This is the
  intended trade — the whole point is *not* needing the full caption.

## Explicitly NOT doing

- **Replacing DirectEdit** — global mask-free semantic edits stay there. FLAIR-edit
  is additive (localized edits only).
- **Blocking `easycontrol_inpaint.md`** — the trained adapter is the other end of the
  same task; the pilot benches them against each other, it doesn't supersede it.
- **Non-linear conditioning** (depth/sketch/sanitize structural control) — that's
  EasyControl; FLAIR is linear-`A` only.
- **Any training** — FLAIR is training-free by construction (parent proposal).
- **Shipping before the unmasked-exactness gate passes** — if HDC doesn't hold the
  kept region bit-exact, the ψ_src-free advantage doesn't exist; the parent Phase 3
  gate + invariant test (c) guard this.
</content>
</invoke>
