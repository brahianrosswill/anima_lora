"""Phase 0.1 — spectral-guidance falsification gate (P1 + P2).

Tests whether Anima's frozen diffusion operator has the property the Spectral
Guidance paper (arXiv:2605.28900) relies on:

  P1  As noise grows, the posterior-mean operator T_t collapses onto a few
      surviving directions φ_{t,k} (leading modes of the round-trip covariance
      T_t T_t*).
  P2  That collapse happens as a *transition* over the schedule, not flatly —
      and (cross-check vs project_sigma_signal_resolves_by_045) it should land
      near σ≈0.45.

We estimate T_t T_t* directly from the frozen DiT, no f_φ trained yet. For one
clean latent x0 we draw M noise realizations, run the model to its posterior
mean x̂₀ = x_t − σ·v for each, and take the covariance of {x̂₀} *across the M
draws*. Its effective rank is exactly "how many directions does E[x₀|x_t] vary
along as the noise varies" — the object P1 names. We average that curve over a
handful of artists (one clean latent each).

Why per-noise (not the proposal's across-batch) covariance: with one latent per
artist the samples have different resolutions → no shared feature dim for an
across-content covariance, and the rank would cap at #artists−1. The round-trip
covariance caps at M instead, so a small artist set still resolves the spectrum.

bsz=1 throughout (one forward per noise draw, looped) — flat memory; nothing is
batched across artists or draws. Inference only, so there is no gradient to
accumulate.

Expected GO shape: PR(σ) high at low σ (operator ≈ identity, all directions
recoverable) → drops to a few modes at high σ, with a clear knee. NO-GO: flat
(flat-high = no collapse; flat-low = collapsed manifold already won — the ⚠️ in
the proposal). READ THE PLOT — the verdict string is a heuristic.

    python -m bench.spectral_guidance.probe_spectrum --dit ... \
        --n_artists 8 --noise_draws 32 --label smoke
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from bench._anima import add_common_args, build_anima
from bench._common import make_run_dir, write_result
from library.io.cache import (
    TE_CACHE_SUFFIX,
    load_cached_latents,
    load_cached_text_features,
)

log = logging.getLogger("bench.spectral_guidance.probe_spectrum")
logging.basicConfig(level=logging.INFO, format="%(message)s")

_NPZ_RE = re.compile(r"_\d+x\d+_anima\.npz$")


def _stem(npz: Path) -> str:
    """`{stem}_{Wpix}x{Hpix}_anima.npz` → `{stem}` (TE sidecar shares the stem)."""
    return _NPZ_RE.sub("", npz.name)


def sample_one_per_artist(
    lora_root: Path, n_artists: int, seed: int, *, require_te: bool
) -> list[tuple[str, Path, Path | None]]:
    """Pick one (latent, TE) pair from each of ``n_artists`` distinct artist dirs.

    ``post_image_dataset/lora/`` is nested one subdir per artist; we walk it
    directly (``discover_bucketed_samples`` only globs the flat top level).
    """
    rng = random.Random(seed)
    artist_dirs = sorted(p for p in lora_root.iterdir() if p.is_dir())
    rng.shuffle(artist_dirs)

    picks: list[tuple[str, Path, Path | None]] = []
    for d in artist_dirs:
        npzs = sorted(d.glob("*_anima.npz"))
        if not npzs:
            continue
        npz = rng.choice(npzs)
        te = d / f"{_stem(npz)}{TE_CACHE_SUFFIX}"
        te = te if te.exists() else None
        if require_te and te is None:
            continue
        picks.append((d.name, npz, te))
        if len(picks) >= n_artists:
            break

    if len(picks) < n_artists:
        log.warning(
            "only %d artists with usable caches (asked %d)", len(picks), n_artists
        )
    return picks


def build_context(te_path: Path | None, uncond: bool, device) -> torch.Tensor:
    """Return a ``(1, S, D)`` bf16 cross-attn context — the model's own padding."""
    if uncond:
        from library.inference.uncond import default_uncond_path, load_uncond_crossattn

        ctx = load_uncond_crossattn(str(default_uncond_path()), device, torch.bfloat16)
        return ctx.to(device, torch.bfloat16)  # (1, S, D)
    crossattn, _ = load_cached_text_features(str(te_path))
    return crossattn.unsqueeze(0).to(device, torch.bfloat16)


@torch.no_grad()
def x0_hat(
    anima, x0b: torch.Tensor, sigma: float, context: torch.Tensor, gen: torch.Generator
) -> torch.Tensor:
    """One noised forward → flattened posterior-mean estimate x̂₀ = x_t − σ·v.

    ``x0b`` is the clean latent already shaped ``(1, C, 1, H, W)`` on device.
    Mirrors the production forward call (generation.py:743): bf16 forward, fp32
    x̂₀, ``padding_mask`` of zeros, no ``crossattn_seqlens`` (do NOT mask pad).
    """
    eps = torch.randn(x0b.shape, generator=gen, device=x0b.device, dtype=x0b.dtype)
    x_t = (1.0 - sigma) * x0b + sigma * eps
    t_expand = torch.full((1,), sigma, device=x0b.device, dtype=torch.bfloat16)
    pad = torch.zeros(
        1, 1, x0b.shape[-2], x0b.shape[-1], dtype=torch.bfloat16, device=x0b.device
    )
    v = anima(x_t, t_expand, context, padding_mask=pad)
    return (x_t.float() - sigma * v.float()).reshape(-1)


def effective_rank(eigs: torch.Tensor) -> tuple[float, float, float]:
    """(participation ratio, entropy effective rank, trace) of an eigenvalue set."""
    lam = eigs.clamp_min(0).double()
    s = lam.sum()
    if s <= 0:
        return 0.0, 0.0, 0.0
    pr = (s * s / (lam * lam).sum()).item()
    p = (lam / s).clamp_min(1e-12)
    erank = math.exp(float(-(p * p.log()).sum()))
    return pr, erank, float(s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dit", default="models/diffusion_models/anima-base-v1.0.safetensors")
    p.add_argument("--data", default="post_image_dataset/lora")
    p.add_argument("--n_artists", type=int, default=8, help="distinct artists (B)")
    p.add_argument("--noise_draws", type=int, default=32, help="M: caps the rank ceiling")
    p.add_argument("--n_bins", type=int, default=24, help="σ grid points")
    p.add_argument("--sigma_lo", type=float, default=0.10)
    p.add_argument("--sigma_hi", type=float, default=0.95)
    p.add_argument("--uncond", action="store_true", help="null-text operator (paper-faithful)")
    add_common_args(p)
    opts = p.parse_args()

    device = torch.device(getattr(opts, "device", "cuda"))
    torch.manual_seed(getattr(opts, "seed", 0))

    picks = sample_one_per_artist(
        REPO_ROOT / opts.data, opts.n_artists, getattr(opts, "seed", 0),
        require_te=not opts.uncond,
    )
    if not picks:
        raise SystemExit(f"no usable artist caches under {opts.data}")
    log.info("artists: %s", ", ".join(name for name, _, _ in picks))

    bundle = build_anima(opts, dit_path=opts.dit, adapter=None, train_mode=False)
    anima = bundle.anima

    sigmas = np.linspace(opts.sigma_lo, opts.sigma_hi, opts.n_bins)
    M = opts.noise_draws
    n_a = len(picks)
    PR = np.zeros((n_a, opts.n_bins))
    ER = np.zeros((n_a, opts.n_bins))
    TR = np.zeros((n_a, opts.n_bins))
    KS = (1, 2, 4, 8)
    TK = np.zeros((n_a, opts.n_bins, len(KS)))  # top-K energy fraction

    for ai, (name, npz, te) in enumerate(picks):
        x0, res, _, _ = load_cached_latents(str(npz))
        x0b = x0.to(device, torch.bfloat16).unsqueeze(0).unsqueeze(2)  # (1,C,1,H,W)
        context = build_context(te, opts.uncond, device)
        gen = torch.Generator(device=device).manual_seed(getattr(opts, "seed", 0) + ai)
        log.info("[%d/%d] %s  latent=%s", ai + 1, n_a, name, res)

        for si, sig in enumerate(sigmas):
            sig = float(sig)
            X = None
            for m in range(M):
                x = x0_hat(anima, x0b, sig, context, gen)
                if X is None:
                    X = torch.empty(M, x.numel(), device=device)
                X[m] = x
            Xc = X - X.mean(0, keepdim=True)
            gram = (Xc @ Xc.T) / M  # (M, M) — nonzero spectrum == that of cov
            eigs = torch.linalg.eigvalsh(gram)
            PR[ai, si], ER[ai, si], TR[ai, si] = effective_rank(eigs)
            # top-K energy fraction — the P1 metric (does a few-mode subspace
            # carry the round-trip variance?). Robust to the M ceiling at the
            # collapsed end, unlike PR.
            desc = torch.sort(eigs.clamp_min(0).double(), descending=True).values
            tot = desc.sum()
            if tot > 0:
                for ki, k in enumerate(KS):
                    TK[ai, si, ki] = float(desc[:k].sum() / tot)

    pr_mean, pr_std = PR.mean(0), PR.std(0)
    er_mean = ER.mean(0)
    tk_mean = TK.mean(0)  # (n_bins, len(KS))
    tk4 = tk_mean[:, KS.index(4)]  # top-4 energy fraction vs σ

    def _cross(curve: np.ndarray, level: float) -> float:
        """First σ where ``curve`` crosses ``level`` upward (linear interp)."""
        for i in range(1, len(curve)):
            if curve[i - 1] < level <= curve[i]:
                t = (level - curve[i - 1]) / (curve[i] - curve[i - 1] + 1e-12)
                return float(sigmas[i - 1] + t * (sigmas[i] - sigmas[i - 1]))
        return float("nan")

    # P1/P2 are read off the top-4 energy fraction (does a few-mode subspace
    # carry the round-trip variance, and *where* does that concentration set in).
    # tk4 rises 0→1 as σ grows; the guidance window is where it crosses ~0.5.
    transition_sigma = _cross(tk4, 0.5)
    i45 = int(np.argmin(np.abs(sigmas - 0.45)))
    tk4_at_45 = float(tk4[i45])

    p1_subspace_at_mid = tk4_at_45 >= 0.5  # few modes dominate by σ≈0.45
    p2_transition_near = 0.35 <= transition_sigma <= 0.60 if math.isfinite(
        transition_sigma
    ) else False
    if p1_subspace_at_mid and p2_transition_near:
        verdict = "GO"
    elif p1_subspace_at_mid:
        verdict = f"GO-but-transition-off-0.45 (cross σ={transition_sigma:.2f})"
    elif not math.isfinite(transition_sigma) or transition_sigma > 0.80:
        verdict = "NO-GO (no mid window: subspace only collapses in hopeless tail)"
    elif tk4[-1] < 0.4:
        verdict = "NO-GO (flat: operator stays high-rank, no dominant subspace)"
    else:
        verdict = "MARGINAL — read the plot"

    run_dir = make_run_dir("spectral_guidance", label=getattr(opts, "label", None))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        # left: effective rank (P2 — where/whether it collapses)
        ax.plot(sigmas, pr_mean, "-o", ms=3, color="C0", label="participation ratio")
        ax.fill_between(sigmas, pr_mean - pr_std, pr_mean + pr_std, color="C0", alpha=0.2)
        ax.plot(sigmas, er_mean, "--s", ms=3, color="C1", label="entropy eff-rank")
        ax.axhline(M - 1, color="grey", ls=":", lw=1, label=f"M−1 ceiling ({M - 1})")
        ax.axvline(0.45, color="k", ls="--", lw=1, label="σ≈0.45 (sigma_signal)")
        ax.set_xlabel("σ"); ax.set_ylabel("effective rank of T_t T_t*")
        ax.set_title("effective rank")
        ax.legend(fontsize=7, loc="lower left")
        # right: top-K energy fraction (P1 — does a few-mode subspace dominate)
        for ki, k in enumerate(KS):
            ax2.plot(sigmas, tk_mean[:, ki], "-o", ms=3, label=f"top-{k} energy")
        ax2.axhline(0.5, color="grey", ls=":", lw=1)
        ax2.axvline(0.45, color="k", ls="--", lw=1, label="σ≈0.45")
        if math.isfinite(transition_sigma):
            ax2.axvline(transition_sigma, color="r", ls="-.", lw=1,
                        label=f"tk4=0.5 @ σ={transition_sigma:.2f}")
        ax2.set_xlabel("σ"); ax2.set_ylabel("energy fraction in top-K modes")
        ax2.set_ylim(0, 1); ax2.set_title("top-K subspace dominance")
        ax2.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"spectral-guidance Phase 0.1 [{ 'uncond' if opts.uncond else 'cond' }] — {verdict}")
        fig.tight_layout()
        fig.savefig(run_dir / "effective_rank.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log.warning("plot skipped: %s", e)

    np.savez(
        run_dir / "spectrum.npz",
        sigmas=sigmas, PR=PR, ER=ER, TR=TR, TK=TK, KS=np.array(KS),
        artists=np.array([n for n, _, _ in picks]),
    )

    write_result(
        run_dir,
        script=__file__,
        args=opts,
        device=device,
        metrics={
            "verdict": verdict,
            "transition_sigma": transition_sigma,  # σ where top-4 energy crosses 0.5
            "tk4_at_0.45": tk4_at_45,
            "tk4_high_sigma": float(tk4[-1]),
            "rank_ceiling": M - 1,
            "conditioning": "uncond" if opts.uncond else "cond",
            "n_artists": n_a,
            "artists": [n for n, _, _ in picks],
            "sigmas": sigmas.tolist(),
            "pr_mean": pr_mean.tolist(),
            "pr_std": pr_std.tolist(),
            "erank_mean": er_mean.tolist(),
            "tk_mean": {str(k): tk_mean[:, ki].tolist() for ki, k in enumerate(KS)},
        },
        artifacts=["effective_rank.png", "spectrum.npz"],
    )
    log.info("\nverdict: %s  (tk4 crosses 0.5 @ σ=%.2f, tk4@0.45=%.2f)  → %s",
             verdict, transition_sigma, tk4_at_45, run_dir / "result.json")


if __name__ == "__main__":
    main()
