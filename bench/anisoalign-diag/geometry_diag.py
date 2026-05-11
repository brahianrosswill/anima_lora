#!/usr/bin/env python
"""AnisoAlign-style geometric diagnostics for paired (image, text) embeddings.

Reproduces §3 diagnostics of Yu et al. 2026 (arXiv:2605.07825) on our
cached PE / pooled-T5 features so we can decide whether AnisoAlign-style
correction is worth wiring into IP-Adapter or DirectEdit.

What it measures (paired image side X = mean-pooled PE, text side Y =
pooled T5):

* Per-modality covariance spectrum: participation ratio (PR), effective
  dim (d_eff/d), anisotropy ratio A.  Sanity-check against the
  ``pe_feature_diagnostics`` memory (PR=6.2 on our PE pool).
* Cross-modal compatibility: log-spectral correlation C_lambda and
  top-q principal subspace overlap O_q.  Random baseline is q/d.
* Centroid bias: G_mu = ||mu_x - mu_y||_2.
* Residual after global mean correction: paired distance ratio
  D_residual / D_initial; residual-covariance anisotropy A_r and
  effective dim d_eff(Sigma_r)/d.
* Cumulative residual energy E(K) = sum_{j<=K} lambda_j(Sigma_r) /
  tr(Sigma_r); plotted against the K/d isotropic baseline.

Default sample is mean-pooled PE features and pooled T5 v0 from
``post_image_dataset/lora/``.  Both are 1024-d, both come from
fundamentally different encoders -- so this script is **not** a drop-in
reproduction of the paper's CLIP-space numbers; it tests whether the
*shape of the gap* (centroid + low-effective-dim residual) shows up
across our two unrelated encoders too.

Usage::

    uv run python bench/anisoalign-diag/geometry_diag.py
    uv run python bench/anisoalign-diag/geometry_diag.py --max_samples 1000 --label first-pass
    uv run python bench/anisoalign-diag/geometry_diag.py --no_l2_normalize --label raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402


def _list_paired_stems(cache_dir: Path, text_variant: str) -> list[str]:
    """Stems with both PE and pooled-<variant> sidecars present."""
    pe_stems = {p.name.removesuffix("_anima_pe.safetensors") for p in cache_dir.glob("*_anima_pe.safetensors")}
    pool_stems = []
    for stem in sorted(pe_stems):
        pool_path = cache_dir / f"{stem}_anima_pooled.safetensors"
        if not pool_path.exists():
            continue
        with safe_open(pool_path, framework="pt") as f:
            keys = set(f.keys())
            if f"pooled_{text_variant}" in keys:
                pool_stems.append(stem)
    return pool_stems


def _load_pairs(
    cache_dir: Path,
    stems: list[str],
    text_variant: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, Y) as fp32 tensors of shape (N, 1024).

    X = mean-pooled PE features (one vector per image).
    Y = pooled T5 embedding for the requested caption variant.
    """
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for stem in stems:
        with safe_open(cache_dir / f"{stem}_anima_pe.safetensors", framework="pt") as f:
            pe = f.get_tensor("image_features").to(torch.float32)  # (T, 1024)
        with safe_open(cache_dir / f"{stem}_anima_pooled.safetensors", framework="pt") as f:
            pooled = f.get_tensor(f"pooled_{text_variant}").to(torch.float32)  # (1024,)
        xs.append(pe.mean(dim=0))
        ys.append(pooled)
    X = torch.stack(xs, dim=0)
    Y = torch.stack(ys, dim=0)
    return X, Y


def _l2_normalize(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return z / (z.norm(dim=-1, keepdim=True).clamp_min(eps))


def _covariance_spectrum(z: torch.Tensor, ridge: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (eigvals_desc, eigvecs_desc) of centered covariance.

    Returns ``(d,)`` eigenvalues sorted descending and the matching
    ``(d, d)`` orthonormal eigenvector matrix (columns).
    """
    zc = z - z.mean(dim=0, keepdim=True)
    n = zc.shape[0]
    cov = (zc.T @ zc) / max(n - 1, 1)
    cov = cov + ridge * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    evals = torch.flip(evals, dims=[0])
    evecs = torch.flip(evecs, dims=[1])
    evals = evals.clamp_min(0.0)
    return evals, evecs


def _participation_ratio(evals: torch.Tensor) -> float:
    """tr(Sigma)^2 / tr(Sigma^2) -- effective dim / Hoyer PR."""
    s1 = float(evals.sum())
    s2 = float((evals * evals).sum())
    if s2 <= 0.0:
        return 0.0
    return s1 * s1 / s2


def _anisotropy_ratio(evals: torch.Tensor) -> float:
    """lambda_max / mean(lambda).  Isotropic gauss -> ~1."""
    d = evals.numel()
    s = float(evals.sum())
    if s <= 0.0:
        return 0.0
    return float(evals[0]) / (s / d)


def _subspace_overlap(U_x: torch.Tensor, U_y: torch.Tensor) -> float:
    """O_q = (1/q) ||U_x^T U_y||_F^2 in [0, 1]."""
    q = U_x.shape[1]
    M = U_x.T @ U_y  # (q, q)
    return float((M * M).sum()) / q


def _spectral_correlation(evals_x: torch.Tensor, evals_y: torch.Tensor, eps: float = 1e-12) -> float:
    """corr(log lambda_x, log lambda_y) over matched ranks."""
    a = torch.log(evals_x.clamp_min(eps))
    b = torch.log(evals_y.clamp_min(eps))
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a * b).sum() / denom)


def _residual_diagnostics(
    X: torch.Tensor,
    Y: torch.Tensor,
    ridge: float,
) -> dict:
    """Paired residual diagnostics after global mean correction.

    Following the paper: keep image features fixed, shift text to image
    centroid, measure paired residual r_i = (x_i - mu_x) - (y_i - mu_y).
    """
    mu_x = X.mean(dim=0)
    mu_y = Y.mean(dim=0)
    xc = X - mu_x
    yc = Y - mu_y

    centroid_bias = float((mu_x - mu_y).norm())
    d_initial = float((X - Y).norm(dim=1).mean())
    R = xc - yc  # paired residual after centering
    d_residual = float(R.norm(dim=1).mean())
    residual_ratio = d_residual / max(d_initial, 1e-12)

    n = R.shape[0]
    cov_r = (R.T @ R) / max(n - 1, 1)
    cov_r = cov_r + ridge * torch.eye(cov_r.shape[0], dtype=cov_r.dtype, device=cov_r.device)
    evals_r, _ = torch.linalg.eigh(cov_r)
    evals_r = torch.flip(evals_r, dims=[0]).clamp_min(0.0)

    pr_r = _participation_ratio(evals_r)
    A_r = _anisotropy_ratio(evals_r)

    # cumulative energy curve E(K)
    cum = torch.cumsum(evals_r, dim=0)
    cum_norm = cum / cum[-1].clamp_min(1e-12)
    return {
        "centroid_bias": centroid_bias,
        "d_initial": d_initial,
        "d_residual": d_residual,
        "residual_ratio": residual_ratio,
        "residual_eff_dim": pr_r,
        "residual_eff_dim_frac": pr_r / evals_r.numel(),
        "residual_anisotropy_A_r": A_r,
        "_residual_evals": evals_r.cpu().numpy(),
        "_residual_cum_energy": cum_norm.cpu().numpy(),
    }


def _per_modality_metrics(name: str, z: torch.Tensor, ridge: float) -> dict:
    evals, _ = _covariance_spectrum(z, ridge=ridge)
    pr = _participation_ratio(evals)
    return {
        f"{name}_n": int(z.shape[0]),
        f"{name}_d": int(z.shape[1]),
        f"{name}_eff_dim": pr,
        f"{name}_eff_dim_frac": pr / z.shape[1],
        f"{name}_anisotropy": _anisotropy_ratio(evals),
        f"{name}_top1_frac": float(evals[0] / evals.sum().clamp_min(1e-12)),
        f"{name}_top10_frac": float(evals[:10].sum() / evals.sum().clamp_min(1e-12)),
        f"_{name}_evals": evals.cpu().numpy(),
    }


def _save_plots(run_dir: Path, results: dict, top_q_list: list[int]) -> list[str]:
    artifacts: list[str] = []

    # 1) Per-modality + residual eigenvalue spectra (log-log, normalised by trace)
    fig, ax = plt.subplots(figsize=(6, 4))
    for tag, color in [("X", "tab:red"), ("Y", "tab:blue"), ("residual", "tab:purple")]:
        evals = results[f"_{tag}_evals" if tag != "residual" else "_residual_evals"]
        evals = np.asarray(evals)
        s = float(evals.sum())
        if s <= 0:
            continue
        norm = evals / s
        ax.loglog(np.arange(1, len(norm) + 1), norm.clip(min=1e-12), label=tag, color=color)
    d = int(results["X_d"])
    ax.axhline(1.0 / d, ls="--", color="gray", label=f"isotropic 1/d (d={d})")
    ax.set_xlabel("eigenvalue rank j")
    ax.set_ylabel("lambda_j / tr(Sigma)")
    ax.set_title("Covariance spectra (X=PE-pooled, Y=T5-pooled, residual after mean-correction)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    p = run_dir / "spectra.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    artifacts.append(p.name)

    # 2) Cumulative residual energy E(K) vs isotropic K/d
    fig, ax = plt.subplots(figsize=(6, 4))
    cum = np.asarray(results["_residual_cum_energy"])
    K = np.arange(1, len(cum) + 1)
    ax.plot(K, cum, label="empirical E(K)", color="tab:purple")
    ax.plot(K, K / d, ls="--", color="gray", label="isotropic K/d")
    ax.set_xscale("log")
    ax.set_xlabel("top-K residual components")
    ax.set_ylabel("cumulative energy")
    ax.set_title(
        f"Residual energy concentration  (A_r={results['residual_anisotropy_A_r']:.1f}, "
        f"d_eff/d={results['residual_eff_dim_frac']:.3f})"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = run_dir / "residual_energy.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    artifacts.append(p.name)

    # 3) Subspace overlap O_q vs random baseline q/d
    fig, ax = plt.subplots(figsize=(6, 4))
    qs = np.asarray(top_q_list, dtype=float)
    obs = np.asarray([results["overlaps"][str(int(q))] for q in qs])
    ax.plot(qs, obs, marker="o", color="tab:red", label="observed O_q")
    ax.plot(qs, qs / d, ls="--", color="gray", label="random q/d")
    ax.set_xscale("log")
    ax.set_xlabel("subspace size q")
    ax.set_ylabel("overlap")
    ax.set_title(f"Top-q subspace overlap  (C_lambda={results['spectral_correlation_C_lambda']:.3f})")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    p = run_dir / "subspace_overlap.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    artifacts.append(p.name)

    return artifacts


def _write_summary(run_dir: Path, results: dict, top_q_list: list[int]) -> str:
    lines: list[str] = []
    lines.append(f"AnisoAlign-style geometric diagnostics  (n={results['X_n']}, d={results['X_d']})")
    lines.append("")
    lines.append("Per-modality (centered, ridge-stabilised covariance):")
    lines.append(f"  X (mean-pooled PE)   eff_dim={results['X_eff_dim']:.2f}  ({results['X_eff_dim_frac']:.3f}*d)"
                 f"   anisotropy={results['X_anisotropy']:.2f}"
                 f"   top1={results['X_top1_frac']:.3f}  top10={results['X_top10_frac']:.3f}")
    lines.append(f"  Y (pooled T5 v0)     eff_dim={results['Y_eff_dim']:.2f}  ({results['Y_eff_dim_frac']:.3f}*d)"
                 f"   anisotropy={results['Y_anisotropy']:.2f}"
                 f"   top1={results['Y_top1_frac']:.3f}  top10={results['Y_top10_frac']:.3f}")
    lines.append("")
    lines.append("Cross-modal compatibility (paper §3.1):")
    lines.append(f"  spectral correlation C_lambda  = {results['spectral_correlation_C_lambda']:+.3f}   "
                 f"(paper CLIP: 0.845)")
    lines.append("  top-q subspace overlap O_q vs q/d random baseline:")
    for q in top_q_list:
        obs = results["overlaps"][str(q)]
        lines.append(f"    q={q:5d}  O_q={obs:.3f}   random={q/results['X_d']:.3f}   ratio={obs/(q/results['X_d']):.2f}x")
    lines.append("")
    lines.append("Modality gap (paper §3.2):")
    lines.append(f"  centroid bias       G_mu        = {results['centroid_bias']:.3f}")
    lines.append(f"  paired distance pre-correction  = {results['d_initial']:.3f}")
    lines.append(f"  paired distance post-correction = {results['d_residual']:.3f}")
    lines.append(f"  residual ratio  D_res / D_init  = {results['residual_ratio']:.3f}   "
                 f"(paper CLIP: 0.89 -- centroid alone is insufficient)")
    lines.append(f"  residual anisotropy A_r         = {results['residual_anisotropy_A_r']:.2f}   "
                 f"(paper CLIP: 28.6)")
    lines.append(f"  residual effective dim          = {results['residual_eff_dim']:.2f}  "
                 f"({results['residual_eff_dim_frac']:.3f}*d)   (paper CLIP: 0.284*d)")
    lines.append("")
    lines.append("Interpretation:")
    G_mu = results["centroid_bias"]
    d_init = results["d_initial"]
    rr = results["residual_ratio"]
    A_r = results["residual_anisotropy_A_r"]
    C_lam = results["spectral_correlation_C_lambda"]
    # Mean overlap-vs-random ratio across the requested q list. =1 means eigen-directions are random w.r.t. each other.
    overlap_ratios = [results["overlaps"][str(q)] / max(q / results["X_d"], 1e-12) for q in top_q_list]
    mean_overlap_ratio = float(np.mean(overlap_ratios)) if overlap_ratios else 1.0

    # Centroid vs residual decomposition
    centroid_share = G_mu / max(d_init, 1e-12)
    if centroid_share > 0.7:
        lines.append(f"  - centroid bias dominates ({G_mu:.2f}/{d_init:.2f} = {centroid_share:.0%} of pre-correction gap); "
                     f"mean-centering alone removes {1.0 - rr:.0%} of paired distance")
    elif centroid_share > 0.3:
        lines.append(f"  - centroid is meaningful but not dominant ({centroid_share:.0%} of gap); "
                     f"mean-centering removes {1.0 - rr:.0%}")
    else:
        lines.append(f"  - centroid bias is small ({centroid_share:.0%} of gap); the gap is mostly residual")

    # AnisoAlign precondition gate: shared dominant geometry needs BOTH high C_lambda AND O_q >> q/d.
    if C_lam > 0.5 and mean_overlap_ratio > 1.5:
        lines.append(f"  - SHARED dominant geometry (C_lambda={C_lam:.2f}, O_q/(q/d) avg={mean_overlap_ratio:.2f}x): "
                     f"AnisoAlign Stage-II 'transport in shared subspace' is well-defined")
    elif C_lam > 0.5 and mean_overlap_ratio < 1.2:
        lines.append(f"  - SPECTRAL SHAPE compatible (C_lambda={C_lam:.2f}) but DIRECTIONS DISJOINT "
                     f"(O_q/(q/d) avg={mean_overlap_ratio:.2f}x ~ random baseline): "
                     f"both encoders are heavy-tailed but their dominant axes do not coincide. "
                     f"The paper's 'compatible dominant geometry' precondition is NOT met by this pair.")
    else:
        lines.append(f"  - C_lambda={C_lam:.2f}, O_q/(q/d) avg={mean_overlap_ratio:.2f}x: "
                     f"no compatibility either in spectral shape or directions")

    # Residual anisotropy
    if A_r > 5.0:
        lines.append(f"  - residual is strongly anisotropic (A_r={A_r:.1f}, eff_dim={results['residual_eff_dim']:.1f} "
                     f"of d={results['X_d']}); a low-rank transform in *the residual's own eigenbasis* (NOT the "
                     f"shared X/Y subspace) would capture most of it")
    elif A_r > 2.0:
        lines.append(f"  - residual mildly anisotropic (A_r={A_r:.1f}); subspace correction *may* help")
    else:
        lines.append(f"  - residual close to isotropic (A_r={A_r:.1f}); a full anisotropic fix unlikely to win over centroid-only")

    # Bottom line
    lines.append("")
    lines.append("Bottom line:")
    if mean_overlap_ratio < 1.2 and C_lam > 0.5:
        lines.append("  AnisoAlign as written assumes a CLIP-style joint embedding where dominant subspaces overlap.")
        lines.append("  This pair (PE encoder, T5 encoder -- never trained jointly) does not satisfy that. The high")
        lines.append("  C_lambda is just both being heavy-tailed -- not evidence of shared semantic axes. So:")
        lines.append("    - centroid recentre is justified and cheap (~{:.0%} of the paired gap)".format(1.0 - rr))
        lines.append("    - low-rank radial CDF transport in residual eigenbasis MAY help further (A_r={:.0f})".format(A_r))
        lines.append("    - Stage-II 'shared subspace polar transport' is NOT applicable here as-is")
    elif mean_overlap_ratio > 1.5 and C_lam > 0.5:
        lines.append("  Pair satisfies the AnisoAlign precondition. Full Stage-II transport is on the table.")
    else:
        lines.append("  Neither precondition holds -- AnisoAlign-style correction unlikely to be a clean fit.")

    text = "\n".join(lines) + "\n"
    p = run_dir / "summary.txt"
    p.write_text(text)
    return text


def _drop_private(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AnisoAlign §3 diagnostics on (mean-pooled PE, pooled T5) pairs."
    )
    parser.add_argument("--cache_dir", type=Path, default=Path("post_image_dataset/lora"))
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Cap on paired stems used (0 = use all available).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_variant", default="v0", choices=["v0", "v1", "v2", "v3"])
    parser.add_argument(
        "--no_l2_normalize",
        action="store_true",
        help="Skip L2 normalisation of pooled vectors before covariance analysis. "
             "Default normalises so we are in AnisoAlign's S^{d-1} regime.",
    )
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument(
        "--top_q",
        default="16,32,64,128,256,512",
        help="Comma-separated subspace sizes for O_q.",
    )
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    if not cache_dir.exists():
        print(f"[error] cache dir not found: {cache_dir}", file=sys.stderr)
        return 2

    print(f"[info] scanning {cache_dir} for paired (PE, pooled-{args.text_variant}) sidecars...")
    stems = _list_paired_stems(cache_dir, args.text_variant)
    print(f"[info] {len(stems)} paired stems available")
    if len(stems) < 32:
        print("[error] not enough paired samples for meaningful covariance diagnostics", file=sys.stderr)
        return 2

    if args.max_samples and args.max_samples < len(stems):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(stems), size=args.max_samples, replace=False)
        stems = [stems[int(i)] for i in sorted(idx)]
        print(f"[info] sampled {len(stems)} stems (seed={args.seed})")

    print("[info] loading paired (PE-pool, T5-pool) tensors...")
    X, Y = _load_pairs(cache_dir, stems, args.text_variant)
    print(f"[info] X={tuple(X.shape)}  Y={tuple(Y.shape)}")

    if not args.no_l2_normalize:
        X = _l2_normalize(X)
        Y = _l2_normalize(Y)
        print("[info] L2-normalised both pools to the unit sphere")

    top_q_list = [int(q) for q in args.top_q.split(",") if q.strip()]
    d = X.shape[1]
    top_q_list = [q for q in top_q_list if 0 < q <= d]

    print("[info] computing per-modality spectra...")
    metrics_x = _per_modality_metrics("X", X, args.ridge)
    metrics_y = _per_modality_metrics("Y", Y, args.ridge)
    evals_x, U_x = _covariance_spectrum(X, ridge=args.ridge)
    evals_y, U_y = _covariance_spectrum(Y, ridge=args.ridge)

    print("[info] computing cross-modal compatibility (C_lambda, O_q)...")
    C_lambda = _spectral_correlation(evals_x, evals_y)
    overlaps: dict[str, float] = {}
    for q in top_q_list:
        overlaps[str(q)] = _subspace_overlap(U_x[:, :q], U_y[:, :q])

    print("[info] computing paired residual diagnostics (centroid + anisotropy)...")
    res = _residual_diagnostics(X, Y, args.ridge)

    results: dict = {}
    results.update(metrics_x)
    results.update(metrics_y)
    results["spectral_correlation_C_lambda"] = C_lambda
    results["overlaps"] = overlaps
    results.update(res)

    run_dir = make_run_dir("anisoalign-diag", label=args.label)
    print(f"[info] writing artifacts to {run_dir}")

    artifacts = _save_plots(run_dir, results, top_q_list)
    summary_text = _write_summary(run_dir, results, top_q_list)
    artifacts.append("summary.txt")

    public_metrics = _drop_private(results)
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=public_metrics,
        label=args.label,
        artifacts=artifacts,
        device="cpu",
        extra={"stem_count": len(stems)},
    )
    print()
    print(summary_text)
    print(f"[done] {run_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
