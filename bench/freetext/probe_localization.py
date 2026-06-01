"""FreeText Phase-0 (item 2): I2T cross-attention localization probe.

Question this answers: during base-Anima sampling, does the *image->text
cross-attention* localize the writing region? FreeText's whole Stage-1
("where to write") rests on that premise — it reads endogenous I2T attention,
aggregates over the target text tokens (anchored by sink-like tokens), and
turns the result into a writing mask. Before wiring any of that, we verify
the premise holds on Anima's specific attention + attention-sink behavior.

What it does
------------
1. Loads the base DiT / text encoder / VAE exactly as ``make test`` does and
   runs a normal generation (default prompt renders a sign reading "ANIMA").
2. Installs an *eager recompute* hook on every block's ``cross_attn``: Anima's
   cross-attention runs through fused flash/flex kernels that never materialize
   ``softmax(QK^T)``, so we recompute it from the same post-RMSNorm q/k the
   kernel uses (cross-attn has NO RoPE — see models.py:269,354 — so this is a
   faithful reproduction). Reduced on the fly to per-patch attention mass over
   a few token groups; nothing large is retained.
3. Maps the target string ("ANIMA") to its Qwen3 token indices via offset
   mapping; padding positions (zeroed in strategy.py:137 — the documented
   Anima sink) are the natural "sink" set.
4. Renders overlays: a timestep x layer grid of the *entity* (target-token)
   attention, plus an Entity / Special / Sink / Entity+Special comparison at
   the best step (the FreeText Table-4 question). Writes a result.json with
   per-(step,layer) concentration so the GO/NO-GO is quantitative too.

Run:
    python bench/freetext/probe_localization.py --label anima-base
    python bench/freetext/probe_localization.py --prompt '... reads "HELLO" ...' \
        --target HELLO --guidance_scale 4.0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

# bench/ is not an installed package — bootstrap the repo root onto sys.path so
# `library` / `inference` / `anima_lora` import the same way the other probes do.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import anima_lora  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402
from inference import parse_args, resolve_seed  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.inference import generate, get_generation_settings  # noqa: E402
from library.inference.models import load_dit_model, load_shared_models  # noqa: E402

# The canonical INFERENCE_BASE prompt — its sign already reads "ANIMA".
DEFAULT_PROMPT = (
    "masterpiece, best quality, score_7, safe. An anime girl wearing a black tank-top"
    " and denim shorts is standing outdoors. She's holding a rectangular sign out in"
    ' front of her that reads "ANIMA". She\'s looking at the viewer with a smile. The'
    " background features some trees and blue sky with clouds."
)
DEFAULT_NEG = (
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia"
)
DIT = "models/diffusion_models/anima-base-v1.0.safetensors"
QWEN3 = "models/text_encoders/qwen_3_06b_base.safetensors"
VAE = "models/vae/qwen_image_vae.safetensors"


# ---------------------------------------------------------------------------
# Capture state. ``fwd`` counts model forwards; with CFG each step is two
# forwards (cond then uncond — generation.py:774,797), so cond forwards are the
# even ones. ``store[(fwd, block)] = dict of reduced (L_img,) vectors``.
# ---------------------------------------------------------------------------
_STATE = {"on": False, "fwd": -1, "groups": None}
_STORE: dict[tuple[int, int], dict] = {}


def _make_capture_qkv(orig_compute_qkv, block_pos: int):
    """Wrap a cross_attn.compute_qkv to recompute + reduce the I2T softmax."""

    def wrapped(x, context, rope_cos_sin=None):
        q, k, v = orig_compute_qkv(x, context, rope_cos_sin=rope_cos_sin)
        if not _STATE["on"]:
            return q, k, v
        groups = _STATE["groups"]
        # q: (B, L_img, H, D); k: (B, L_txt, H, D). B==1 at inference.
        qf = q[0].float()
        kf = k[0].float()
        L_img, H, D = qf.shape
        scale = 1.0 / math.sqrt(D)
        dev = qf.device
        # Accumulate per-group attention mass head-by-head to keep peak memory
        # tiny ((L_img, L_txt) fp32 per head, freed each iter). Stay on GPU and
        # transfer once per (forward, block) — not once per head — to avoid
        # tens of thousands of D2H syncs across the run.
        acc = {name: torch.zeros(L_img, device=dev) for name in ("entity", "special", "pad")}
        real_mass = torch.zeros((), device=dev)
        for h in range(H):
            scores = (qf[:, h, :] @ kf[:, h, :].transpose(0, 1)) * scale  # (L_img, L_txt)
            attn = scores.softmax(dim=-1)
            for name, idx in (
                ("entity", groups["target"]),
                ("special", groups["special"]),
                ("pad", groups["pad"]),
            ):
                if idx:
                    acc[name] += attn[:, idx].mean(dim=1)
            if groups["real"]:
                real_mass += attn[:, groups["real"]].sum(dim=1).mean()
        out = {name: (acc[name] / H).half().cpu().numpy() for name in acc}
        out["real_mass"] = float(real_mass / H)
        out["L_img"] = L_img
        _STORE[(_STATE["fwd"], block_pos)] = out
        return q, k, v

    return wrapped


def _install_hooks(anima):
    handles = []
    for pos, block in enumerate(anima.blocks):
        ca = block.cross_attn
        orig = ca.compute_qkv  # bound method, captured before shadowing
        ca.compute_qkv = _make_capture_qkv(orig, pos)
        handles.append((ca, orig))

    def _pre_hook(module, args):
        _STATE["fwd"] += 1

    h = anima.register_forward_pre_hook(_pre_hook)
    return handles, h


# ---------------------------------------------------------------------------
# Token grouping: target span -> Qwen3 token indices; padding -> sink set.
# ---------------------------------------------------------------------------
def compute_token_groups(tokenizer, prompt: str, target: str, max_length: int) -> dict:
    enc = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors=None,
    )
    ids = enc["input_ids"]
    amask = enc["attention_mask"]
    offs = enc.get("offset_mapping")
    n = len(ids)
    real = [i for i in range(n) if amask[i] == 1]
    pad = [i for i in range(n) if amask[i] == 0]
    special = [i for i in real if offs is not None and offs[i][0] == offs[i][1]]

    target_idx: list[int] = []
    c0 = prompt.find(target)
    if offs is not None and c0 >= 0:
        c1 = c0 + len(target)
        for i in real:
            a, b = offs[i]
            if a == b:
                continue
            if a < c1 and b > c0:  # char-span overlap
                target_idx.append(i)
    if not target_idx:  # fallback: contiguous decoded-token match
        target_idx = _decode_match(tokenizer, ids, real, target)

    decoded = [tokenizer.decode([ids[i]]) for i in target_idx]
    return {
        "target": target_idx,
        "special": special,
        "pad": pad,
        "real": real,
        "target_tokens": decoded,
        "n_real": len(real),
        "n_pad": len(pad),
        "char_span": [c0, c0 + len(target)] if c0 >= 0 else None,
    }


def _decode_match(tokenizer, ids, real, target: str) -> list[int]:
    """Find the shortest contiguous real-token run decoding to contain target."""
    norm = target.replace(" ", "").lower()
    for i_start in range(len(real)):
        acc = ""
        for j in range(i_start, len(real)):
            acc += tokenizer.decode([ids[real[j]]])
            if norm in acc.replace(" ", "").lower():
                return real[i_start : j + 1]
            if len(acc) > len(target) + 12:
                break
    return []


# ---------------------------------------------------------------------------
# Reduction / metrics.
# ---------------------------------------------------------------------------
def patch_grid(h_latent: int, w_latent: int, l_img: int) -> tuple[int, int]:
    p = round(math.sqrt(h_latent * w_latent / l_img))
    hp, wp = h_latent // p, w_latent // p
    assert hp * wp == l_img, f"grid {hp}x{wp} != L_img {l_img} (p={p})"
    return hp, wp


def concentration(vec: np.ndarray, top_frac: float = 0.05) -> float:
    """Fraction of total mass in the top `top_frac` patches. 1.0=delta, ~top_frac=uniform."""
    v = vec.astype(np.float64)
    s = v.sum()
    if s <= 0:
        return 0.0
    k = max(1, int(len(v) * top_frac))
    return float(np.sort(v)[-k:].sum() / s)


def cond_steps() -> list[int]:
    """Sorted distinct cond forward indices (even ones if CFG ran)."""
    fwds = sorted({fwd for (fwd, _) in _STORE})
    has_odd = any(f % 2 == 1 for f in fwds)
    return [f for f in fwds if f % 2 == 0] if has_odd else fwds


def to_map(vec_half, hp: int, wp: int) -> np.ndarray:
    return np.asarray(vec_half, dtype=np.float32).reshape(hp, wp)


# ---------------------------------------------------------------------------
# Visualization.
# ---------------------------------------------------------------------------
def _overlay(ax, img_rgb, heat, title, hp, wp, H, W):
    ax.imshow(img_rgb, extent=(0, W, H, 0))
    h = heat - heat.min()
    if h.max() > 0:
        h = h / h.max()
    ax.imshow(
        h, extent=(0, W, H, 0), cmap="jet", alpha=0.5, interpolation="bilinear"
    )
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def render_timestep_layer_grid(img_rgb, blocks, steps, hp, wp, run_dir, n_blocks):
    H, W = img_rgb.shape[:2]
    fig, axes = plt.subplots(
        len(steps), len(blocks), figsize=(2.1 * len(blocks), 2.1 * len(steps))
    )
    axes = np.atleast_2d(axes)
    for r, (step, fwd) in enumerate(steps):
        for c, bp in enumerate(blocks):
            ax = axes[r, c]
            rec = _STORE.get((fwd, bp))
            if rec is None:
                ax.axis("off")
                continue
            heat = to_map(rec["entity"], hp, wp)
            conc = concentration(rec["entity"])
            _overlay(ax, img_rgb, heat, f"L{bp} t{step}\nconc={conc:.2f}", hp, wp, H, W)
    fig.suptitle("Entity (target-token) I2T attention — timestep x layer", fontsize=11)
    fig.tight_layout()
    out = run_dir / "entity_timestep_layer.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def _otsu(vec: np.ndarray) -> float:
    """Otsu threshold on a normalized [0,1] vector (FreeText Stage-1 uses Otsu)."""
    hist, edges = np.histogram(vec, bins=64, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = np.cumsum(hist[::-1])[::-1]
    tot = hist.sum()
    if tot == 0:
        return 0.5
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1e-9)
    m1 = (np.cumsum((hist * centers)[::-1])[::-1]) / np.maximum(w1, 1e-9)
    sigma_b = (w0 / tot) * (w1 / tot) * (m0 - m1) ** 2
    return float(centers[int(np.nanargmax(sigma_b))])


def render_aggregate_mask(img_rgb, steps, n_blocks, hp, wp, run_dir):
    """FreeText-style aggregate: mean entity map over informative (shallow-mid
    block x mid timestep) pairs, then an Otsu binary writing-mask preview."""
    H, W = img_rgb.shape[:2]
    b_lo, b_hi = max(1, int(n_blocks * 0.15)), int(n_blocks * 0.65)
    s_lo, s_hi = int(len(steps) * 0.10), int(len(steps) * 0.70)
    sel = []
    for si in range(s_lo, max(s_hi, s_lo + 1)):
        for bp in range(b_lo, b_hi):
            rec = _STORE.get((steps[si], bp))
            if rec is not None:
                sel.append(np.asarray(rec["entity"], dtype=np.float32))
    if not sel:
        return None
    agg = np.mean(sel, axis=0).reshape(hp, wp)
    norm = agg - agg.min()
    norm = norm / (norm.max() + 1e-9)
    thr = _otsu(norm.ravel())
    mask = (norm >= thr).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    axes[0].imshow(img_rgb, extent=(0, W, H, 0))
    axes[0].set_title("decoded", fontsize=9)
    _overlay(axes[1], img_rgb, norm,
             f"aggregate entity\n(L{b_lo}-{b_hi}, t{s_lo}-{s_hi}) conc={concentration(agg.ravel()):.2f}",
             hp, wp, H, W)
    axes[2].imshow(img_rgb, extent=(0, W, H, 0))
    axes[2].imshow(mask, extent=(0, W, H, 0), cmap="Reds", alpha=0.45,
                   interpolation="nearest")
    axes[2].set_title(f"Otsu writing mask (thr={thr:.2f})", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("FreeText Stage-1 preview — aggregated localization + mask", fontsize=11)
    fig.tight_layout()
    out = run_dir / "aggregate_mask.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def render_group_comparison(img_rgb, fwd, bp, hp, wp, run_dir, step, blockmean=None):
    H, W = img_rgb.shape[:2]
    rec = _STORE[(fwd, bp)] if blockmean is None else blockmean
    groups = [
        ("Entity", rec["entity"]),
        ("Special (BOS-like)", rec["special"]),
        ("Sink (padding)", rec["pad"]),
        ("Entity + Special", rec["entity"] + rec["special"]),
    ]
    fig, axes = plt.subplots(1, len(groups) + 1, figsize=(2.4 * (len(groups) + 1), 2.6))
    axes[0].imshow(img_rgb, extent=(0, W, H, 0))
    axes[0].set_title("decoded", fontsize=9)
    axes[0].axis("off")
    for ax, (name, vec) in zip(axes[1:], groups):
        heat = to_map(np.asarray(vec, dtype=np.float32), hp, wp)
        _overlay(ax, img_rgb, heat, f"{name}\nconc={concentration(np.asarray(vec)):.2f}", hp, wp, H, W)
    tag = "block-mean" if blockmean is not None else f"L{bp}"
    fig.suptitle(f"Token-group localization @ step {step} ({tag})", fontsize=11)
    fig.tight_layout()
    out = run_dir / "token_group_comparison.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def build_args(a) -> argparse.Namespace:
    argv = [
        "--dit", a.dit,
        "--text_encoder", a.text_encoder,
        "--vae", a.vae,
        "--vae_chunk_size", "64",
        "--vae_disable_cache",
        "--attn_mode", a.attn_mode,
        "--lora_multiplier", "1.0",
        "--prompt", a.prompt,
        "--negative_prompt", DEFAULT_NEG,
        "--image_size", str(a.image_size[0]), str(a.image_size[1]),
        "--infer_steps", str(a.infer_steps),
        "--flow_shift", "1.0",
        "--sampler", a.sampler,
        "--guidance_scale", str(a.guidance_scale),
        "--seed", str(a.seed),
        "--save_path", "output/tests",
    ]
    if a.lora_weight:
        argv += ["--lora_weight", a.lora_weight, "--lora_multiplier", "1.0"]
    return parse_args(argv)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--target", default="ANIMA", help="Substring to localize.")
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="H W")
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--sampler", default="euler")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora_weight", default=None, help="Optional adapter; default bare DiT.")
    p.add_argument("--dit", default=DIT)
    p.add_argument("--text_encoder", default=QWEN3)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--attn_mode", default="flash")
    p.add_argument("--label", default=None)
    p.add_argument("--n_show_blocks", type=int, default=6)
    p.add_argument("--n_show_steps", type=int, default=6)
    a = p.parse_args()

    args = build_args(a)
    if getattr(args, "device", None) is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.seed = resolve_seed(args)
    gen_settings = get_generation_settings(args)
    device = gen_settings.device

    print(f"[freetext-probe] target={a.target!r} cfg={a.guidance_scale} "
          f"steps={a.infer_steps} size={a.image_size} device={device}")

    # Token groups (Qwen3 tokenizer, raw prompt — no chat template, so offsets
    # align 1:1 with crossattn_emb positions).
    tokenizer = anima_utils.load_qwen3_tokenizer(args.text_encoder)
    groups = compute_token_groups(tokenizer, a.prompt, a.target, max_length=512)
    print(f"[freetext-probe] target tokens {groups['target']} -> {groups['target_tokens']}; "
          f"#real={groups['n_real']} #pad={groups['n_pad']} #special={len(groups['special'])}")
    if not groups["target"]:
        print("[freetext-probe] WARNING: target substring not found in tokenization; "
              "entity map will be empty. Check --target / --prompt.")

    # Load models. load_shared_models gives the (CPU) text encoder; we load the
    # DiT ourselves so we can hook it before generate() picks it up.
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}
    vae = anima_lora.load_vae(args.vae, device="cpu", disable_mmap=True,
                              spatial_chunk_size=64, disable_cache=True)
    vae.to(torch.bfloat16).eval().to(device)

    anima = load_dit_model(args, device, torch.bfloat16)
    handles, pre_h = _install_hooks(anima)
    shared_models["model"] = anima

    _STATE["groups"] = groups
    _STATE["fwd"] = -1
    _STATE["on"] = True
    latent = generate(args, gen_settings, shared_models)
    _STATE["on"] = False
    pre_h.remove()

    # Decode the final image for overlays.
    img = anima_lora.decode_to_pil(vae, latent, device)
    if isinstance(img, list):
        img = img[0]
    img_rgb = np.asarray(img.convert("RGB"))
    H, W = a.image_size
    h_latent, w_latent = H // 8, W // 8

    steps = cond_steps()
    n_blocks = len(anima.blocks)
    any_rec = next(iter(_STORE.values()))
    hp, wp = patch_grid(h_latent, w_latent, any_rec["L_img"])
    print(f"[freetext-probe] captured {len(_STORE)} (fwd,block) maps; "
          f"{len(steps)} cond steps; patch grid {hp}x{wp}; blocks={n_blocks}")

    # Per-(step,block) entity concentration table.
    table = {}
    for si, fwd in enumerate(steps):
        for bp in range(n_blocks):
            rec = _STORE.get((fwd, bp))
            if rec is not None and groups["target"]:
                table[(si, bp)] = concentration(np.asarray(rec["entity"], dtype=np.float32))

    run_dir = make_run_dir("freetext", a.label or f"{a.target.lower()}-cfg{a.guidance_scale:g}")
    img.save(run_dir / "decoded.png")

    artifacts = ["decoded.png"]
    best = {"step": 0, "block": n_blocks // 2, "conc": -1.0}
    if table:
        (bsi, bbp), bconc = max(table.items(), key=lambda kv: kv[1])
        best = {"step": bsi, "block": bbp, "conc": bconc}

        # timestep x layer grid (entity)
        show_steps_idx = np.linspace(0, len(steps) - 1, min(a.n_show_steps, len(steps)))
        show_steps = [(int(round(i)), steps[int(round(i))]) for i in show_steps_idx]
        show_blocks = sorted(set(
            int(round(x)) for x in np.linspace(1, n_blocks - 1, a.n_show_blocks)
        ))
        artifacts.append(render_timestep_layer_grid(
            img_rgb, show_blocks, show_steps, hp, wp, run_dir, n_blocks).name)

        # block-mean at the best step (FreeText aggregates over selected layers)
        best_fwd = steps[bsi]
        mid = range(n_blocks // 4, 3 * n_blocks // 4)
        bm = {k: np.mean([np.asarray(_STORE[(best_fwd, bp)][k], dtype=np.float32)
                          for bp in mid if (best_fwd, bp) in _STORE], axis=0)
              for k in ("entity", "special", "pad")}
        artifacts.append(render_group_comparison(
            img_rgb, best_fwd, bbp, hp, wp, run_dir, bsi, blockmean=bm).name)

        agg_out = render_aggregate_mask(img_rgb, steps, n_blocks, hp, wp, run_dir)
        if agg_out is not None:
            artifacts.append(agg_out.name)

    # Concentration of a uniform map = top_frac (~0.05); report the lift.
    metrics = {
        "target": a.target,
        "target_token_indices": groups["target"],
        "target_tokens": groups["target_tokens"],
        "n_real_tokens": groups["n_real"],
        "n_pad_tokens": groups["n_pad"],
        "patch_grid": [hp, wp],
        "n_cond_steps": len(steps),
        "n_blocks": n_blocks,
        "uniform_baseline_conc": 0.05,
        "best_entity_conc": best["conc"],
        "best_step": best["step"],
        "best_block": best["block"],
        "entity_conc_by_step_block": {f"{si}_{bp}": v for (si, bp), v in table.items()},
    }
    write_result(run_dir, script=__file__, args=a, metrics=metrics,
                 label=a.label, artifacts=artifacts, device=device)
    np.savez_compressed(
        run_dir / "reduced.npz",
        **{f"{fwd}_{bp}_{k}": np.asarray(v, dtype=np.float32)
           for (fwd, bp), rec in _STORE.items()
           for k, v in rec.items() if k in ("entity", "special", "pad")},
    )

    print(f"[freetext-probe] best entity concentration = {best['conc']:.3f} "
          f"(uniform baseline 0.05) at step {best['step']} block {best['block']}")
    print(f"[freetext-probe] results -> {run_dir}")


if __name__ == "__main__":
    main()
