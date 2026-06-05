# Shelve GAD + σ-FiLM for mod-guidance — architectural ceiling, both inert

**Status:** proposed closure (2026-06-05). Decision-only — no new mechanism.
**TL;DR:** GAD-for-mod-guidance and the σ-FiLM head are both **default-off already**,
and σ-FiLM is now shown **inert even when opted in**. The deficiency they target is an
*architectural ceiling*, not a trainable gap. Keep the GAD *machinery* (turbo uses it),
stop investing in σ-FiLM, and optionally strip the σ-FiLM code (bit-exact-off, so removal
changes nothing live).

Supersedes the "open question" in `docs/experimental/gad.md` (now RESOLVED) and extends
[[project_mod_guidance_sigma_film]] / [[project_mod_guidance_text_derivative_orthogonal]].

---

## Verdict: architectural ceiling, measured

Mod-guidance injects pooled text through the **AdaLN modulation** path. AdaLN `shift` is a
spatially-*uniform* per-channel constant (`shift_B_T_1_1_D` broadcast over H,W,
`models.py:887`) — i.e. a pure **DC** push; `scale`/`gate` only rescale the AC that
cross-attention already wrote. So the head's reachable output set is DC + a rescale of
present structure. It **cannot synthesize new spatial structure.**

The DC/AC-extended `text_jacobian.py` (DAVE decomposition of the response deltas: DC =
per-channel spatial mean, AC = residual) measured the consequence on synth, head-to-head:

| σ | `dT_ac_frac` (teacher response is AC) | `cos_ceiling` = √(DC frac) | full `cos` (0602 / σ-FiLM) |
|---|---|---|---|
| 0.1 | 0.997 | **0.051** | 0.001 / −0.002 |
| 0.5 | 0.995 | 0.067 | 0.002 / 0.002 |
| 0.9 | 0.967 | **0.174** | −0.004 / 0.005 |

- **The teacher's text response is ~99% AC.** So the best full-cos *any* AdaLN-modulation
  head can reach is **0.05–0.17** — and `dT_ac_frac` is **identical across both heads**
  (it's `dT`, head-independent), the consistency check that says the ceiling is real, not
  a head artifact. Full `cos` already sits at that ceiling.
- **σ-FiLM is a no-op head-to-head** (runs `20260605-1620-sigma-film-dcac-synth` vs
  `…-1627-0602-dcac-synth`): full `cos` ~0 both, `cos_dc` ~0 both, and its *own* stated
  target — the magnitude ratio ‖ΔS‖/‖ΔT‖ — barely moved (σ=0.9: 0.049→0.056; still a ~14×
  collapse across σ for both heads).

Mechanism docs: `library/anima/models.py:887` (uniform shift), `:1808` (pooled delta into
AdaLN emb). Probe: `bench/mod_guidance/text_jacobian.py`.

## Both levers are already default-off (nothing to flip)

- **GAD for mod-guidance** — `--gad_weight 0` default → bit-for-bit the MSE-only head.
- **σ-FiLM** — `--mod_sigma_film` default `False`; off ⇒ bit-exact to the plain head.
- Base DiT ships `pooled_text_proj` zero-init with `enable_pooled_text_modulation=False`,
  so mod-guidance itself is opt-in at inference.

"Make it opt-out" is therefore already the shipped state. The live question is keep-shelved
vs remove (below).

## Methodological lesson: GAD-l2 ↓ ≠ aligned — watch `cos`, not the loss

The σ-FiLM training log showed `train/loss_gad` falling, which *looks* like progress. It
isn't. `L_gad = ‖ΔS − ΔT‖² = ‖ΔS‖² − 2⟨ΔS,ΔT⟩ + ‖ΔT‖²` with `ΔT` fixed. When the reachable
`ΔS` is ~orthogonal to `ΔT` (the ceiling), the l2-optimal move is to **shrink ‖ΔS‖ toward
its tiny projection** — the floor is `ΔS=0` → loss `‖ΔT‖²`. So a falling GAD-l2 is the head
learning to *push less in the useless directions*, not to *aim better*. σ-FiLM lowers it
further only because the extra `Linear(D,2D)` is **capacity to fit that floor per-σ** — zero
steering gain. The probe confirms: `cos` flat at ~0, ratio flat-to-lower.

> **Rule:** for any directional-matching objective on a capped lever, the loss can fall by
> magnitude suppression. Gate on the **direction metric (`cos`)**, never the l2 loss.

## Keep the GAD machinery (turbo needs it)

GAD is falsified **for the pure-AdaLN mod head**, not in general. Its other home —
**turbo / DP-DMD** (initial-noise sensitivity under trajectory compression) — is a separate,
live use with real headroom. Do **not** rip out the GAD mechanism in `distill_mod`/config;
just leave the mod-guidance path at `gad_weight=0` and documented-dead.

## Recommendation: strip σ-FiLM (optional, low-risk)

σ-FiLM is the clear-cut removal candidate — proven inert *even when opted in*, not merely
off-by-default. Bit-exact-off, so removal changes nothing live. If we want a clean tree, the
touch points are:

| File | What to remove |
|---|---|
| `library/anima/models.py` | `pooled_text_sigma_film` Linear, `enable_pooled_text_sigma_film`, the σ-FiLM branch in `_pooled_text_delta`, its zero-init in `init_weights` |
| `library/anima/weights.py` | `sigma_film.`-prefix materialize / allowlist / route (auto-arm on load) |
| `scripts/distill_mod/config.py` | `--mod_sigma_film` flag + `ModConfig.mod_sigma_film` + resolve |
| `scripts/distill_mod/distill.py` | `mod_modules` sibling (init / fp32 cast / optimizer / grad-norm / save) |
| `library/inference/corrections/mod_guidance.py` | the σ-flat steering warning |
| `bench/mod_guidance/text_jacobian.py` | the `sigma_film.` auto-arm load branch |
| `~/ComfyUI-Spectrum-KSampler` (external) | `_project_film` / `_compute_t_emb` σ-FiLM branch (also drops its pending live smoke-test obligation) |

The σ-FiLM checkpoint `output/ckpt/pooled_text_proj.safetensors` carries inert `sigma_film.*`
weights that auto-arm and do nothing; prefer a plain `pooled_text_proj` for the shipped head.

**Alternative:** keep σ-FiLM as a frozen, documented dead-end (it's bit-exact-off and the
finding is fresh). Either is defensible; the cost of keeping is the Spectrum-node branch
carrying an unverified-live path that does nothing.

## Out of scope (what would actually revive this)

The only thing that breaks the ceiling is giving pooled-text an **AC-writing route** — a
pooled-text-gated *spatial* LoRA or a mini cross-attention (K/V from the pooled vector). That
abandons the cheap pure-AdaLN premise and substantially overlaps cross-attention's existing
job, so it needs its own cost/benefit proposal; deliberately **not** proposed here. If pooled
content-steering ever becomes a goal, that's the doc to write — and GAD would finally have
real headroom there (the `cos` ceiling would no longer be ~0.1).

## Evidence / reproduce

- DC/AC head-to-head (synth, authoritative): `bench/mod_guidance/results/20260605-1620-sigma-film-dcac-synth/`, `…-1627-0602-dcac-synth/`; per-σ DC/AC tables in each `result.json`.
- Off-distribution caveat: the first σ-FiLM run (`…-1603-sigma-film-dcac`) was on real latents (head is synth-trained) and modestly inflated high-σ ratio/ceiling — use the synth runs.
- Prior schedule closure (per-block / σ both falsified): `docs/findings/mod_guidance_quality_tag_axis.md` → "Schedule axis".
- Resolved open question: `docs/experimental/gad.md` → "GAD for mod-guidance".

```bash
# Re-confirm the ceiling on any head (both heads → identical dT_ac_frac):
uv run python -m bench.mod_guidance.text_jacobian \
  --pooled_text_proj output/ckpt/pooled_text_proj.safetensors \
  --synth_data_dir post_image_dataset/distill_mod_synth \
  --sigmas 0.1 0.3 0.5 0.7 0.9 --n_pairs 96 --label confirm
```

## Decisions to make

1. **Strip σ-FiLM code, or freeze it as a documented dead-end?** (Recommend strip — proven inert; bit-exact-off.)
2. **Re-export the shipped head without `sigma_film.*` weights?** (Recommend yes — the inert sibling auto-arms and confuses provenance.)
3. **GAD machinery:** confirm keep (turbo). (Recommend keep.)
