#!/usr/bin/env python
"""Dynamic-Spectrum probe — phase 1: is per-step "step-fattening" error
predictable, and is it predictable from a *single* forward's DiT features?

Motivation
----------
Spectrum forecasts on a *fixed* cadence. A dynamic schedule would instead spend
forwards where the trajectory is actually curving. The cleanest version of that
is an error-controlled step: at each step estimate the local truncation error of
a fattened step and step coarsely where it's small, finely where it's large.

For deterministic Euler on this flow-matching schedule the step is
    x_{i+1} = x_i - (sigma_i - sigma_{i+1}) * v_i        (sampling.step)
so collapsing steps i and i+1 into one big Euler step from x_i lands at an error

    e_i = || x_hat_{i+2} - x_{i+2} ||
        = (sigma_{i+1} - sigma_{i+2}) * || v_{i+1} - v_i ||                 (exact)

i.e. the local fattening error is exactly the sigma-weighted velocity *change*.
That makes the whole target computable from a single reference Euler run with
ZERO extra forwards, and pins the question precisely:

  Q1 (is it worth it?)  Is e_i non-uniform along the trajectory? If flat, a fixed
                        schedule is already optimal and adaptivity buys nothing.
  Q2 (sigma enough?)    Does sigma_i alone rank e_i? If yes, you don't need a head
                        — just reshape the fixed schedule by sigma.
  Q3 (one-shot head?)   Can a tiny head on step-i DiT features (the final_layer
                        input — same tap Spectrum uses) predict e_i, beating the
                        sigma baseline and approaching the one-step *history*
                        baseline ||v_i - v_{i-1}|| (which needs a second eval)?

Success criterion is NOT low MSE (cf. the mod-guidance text-derivative head, which
hit low MSE but cos~=0). A step-size controller consumes a *scalar* error estimate,
so the right metric is rank correlation: Spearman(prediction, true e_i) on
held-out trajectories.

This is CFG=1 / euler only (one forward per step, deterministic, exact target).
CFG and er_sde composability are deliberately out of scope for the first read.

Usage
-----
  # quick smoke (correctness check, ~1 min)
  uv run python -m bench.dynamic_spectrum.run_bench --num_trajectories 2 --ref_steps 24 --label smoke

  # real probe (backgroundable)
  uv run python -m bench.dynamic_spectrum.run_bench --num_trajectories 24 --ref_steps 100
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anima_lora import GenerationRequest, generate, get_generation_settings  # noqa: E402
from bench._anima import DEFAULT_DIT, DEFAULT_TEXT_ENCODER, DEFAULT_VAE  # noqa: E402
from bench._common import make_run_dir, write_result  # noqa: E402

# A spread of subjects / styles so the trajectory geometry isn't a single mode.
BUILTIN_PROMPTS = [
    "a red fox curled asleep in a snowy pine forest, soft morning light",
    "portrait of an elderly fisherman, weathered face, golden hour, photorealistic",
    "a neon-lit cyberpunk alley in the rain, reflections on wet asphalt",
    "still life of lemons and a ceramic jug on a linen cloth, oil painting",
    "a sweeping mountain valley with a river, dramatic clouds, wide landscape",
    "a calico cat sitting on a windowsill beside a potted succulent",
    "an astronaut floating above a coral-colored planet, cinematic",
    "a bustling night market with paper lanterns and steam from food stalls",
    "watercolor of cherry blossoms over a quiet stone bridge",
    "a vintage red motorcycle parked outside a seaside cafe, sunny",
    "macro shot of a dragonfly on a dewy reed at dawn",
    "a cozy library with tall wooden shelves and warm lamplight",
    "a lighthouse on a rocky cliff during a stormy sunset",
    "a bowl of ramen with soft-boiled egg, chopsticks, top-down, moody",
    "a hot air balloon drifting over patchwork farmland at sunrise",
    "an ornate brass pocket watch on dark velvet, studio lighting",
]


def _load_dataset_prompts(dataset_dir: str, n: int, seed: int) -> list[str]:
    """Sample n real captions from the dataset's .txt sidecars (the caption master).

    image_dataset/ is a symlink to nested artist dirs, so os.walk(followlinks=True)
    (rglob/plain find miss them — see project_image_dataset_symlink_nested).
    Captions are Danbooru-style tag strings; we feed them verbatim (rating + artist
    tag + tags) so the conditioning matches real training/use, not hand-written prose.
    """
    root = Path(dataset_dir)
    if not root.exists():
        return []
    txts: list[str] = []
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for fn in files:
            if fn.endswith(".txt"):
                txts.append(os.path.join(dirpath, fn))
    txts.sort()  # deterministic before shuffle
    if not txts:
        return []
    random.Random(seed).shuffle(txts)
    prompts: list[str] = []
    for path in txts:
        try:
            cap = Path(path).read_text(errors="ignore").strip()
        except OSError:
            continue
        cap = " ".join(cap.split())  # collapse whitespace/newlines
        if cap:
            prompts.append(cap)
        if len(prompts) >= n:
            break
    return prompts


# ----------------------------------------------------------------------------- capture
_CAP: dict[str, list] = {"feat": [], "v": []}


def _final_pre_hook(module, args):
    # args[0] = feature into final_layer, shape (B, T, Hp, Wp, D). Pool patch dims.
    feat = args[0].detach().float()
    pooled = feat.mean(dim=tuple(range(1, feat.ndim - 1)))  # (B, D)
    _CAP["feat"].append(pooled.cpu())


def _fwd_hook(module, inp, out):
    v = out.detach().float()  # (B, C, 1, H, W) velocity prediction
    _CAP["v"].append(v.reshape(v.shape[0], -1).cpu())


# ----------------------------------------------------------------------------- spearman
def _spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rho via Pearson on ranks (no scipy dependency)."""
    if a.numel() < 3:
        return float("nan")
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if denom == 0:
        return float("nan")
    return float((ra @ rb) / denom)


# ----------------------------------------------------------------------------- head
class TinyHead(torch.nn.Module):
    def __init__(self, d_in: int, hidden: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_head(Xtr, ytr, Xte, *, epochs: int, device: str) -> torch.Tensor:
    torch.manual_seed(0)
    head = TinyHead(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, ytr, Xte = Xtr.to(device), ytr.to(device), Xte.to(device)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(head(Xtr), ytr)
        loss.backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        return head(Xte).cpu()


# ----------------------------------------------------------------------------- run one
def _run_trajectory(prompt: str, seed: int, args, gen_settings, shared_models):
    _CAP["feat"].clear()
    _CAP["v"].clear()
    req = GenerationRequest(
        prompt=prompt,
        negative_prompt="",
        image_size=(args.image_size[0], args.image_size[1]),
        infer_steps=args.ref_steps,
        guidance_scale=1.0,  # CFG off -> exactly one forward per step
        flow_shift=args.flow_shift,
        sampler="euler",
        seed=seed,
        dit=args.dit,
        vae=args.vae,
        text_encoder=args.text_encoder,
        device=args.device,
        attn_mode=args.attn_mode,
    )
    a = req.to_args()
    a.device = args.device
    with torch.no_grad():
        generate(a, gen_settings, shared_models=shared_models)
    feats = _CAP["feat"]  # list of (1, D)
    vs = _CAP["v"]  # list of (1, F)
    n = min(len(feats), len(vs))
    return feats[:n], vs[:n]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dit", default=DEFAULT_DIT)
    p.add_argument("--vae", default=DEFAULT_VAE)
    p.add_argument("--text_encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn_mode", default="flash")
    p.add_argument(
        "--compile", action="store_true",
        help="block-compile the DiT (--compile_blocks) for faster forwards. Bit-exact "
        "(compiles each block's _forward; final_layer + top forward stay eager, so the "
        "capture hooks are unaffected). First trajectory pays the inductor warmup.",
    )
    p.add_argument("--num_trajectories", type=int, default=24)
    p.add_argument("--ref_steps", type=int, default=100)
    p.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="H W")
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--seed0", type=int, default=1234)
    p.add_argument("--holdout_frac", type=float, default=0.3)
    p.add_argument("--head_epochs", type=int, default=400)
    p.add_argument(
        "--dataset_dir", default="image_dataset",
        help="sample real captions from .txt sidecars under here (the caption master); "
        "'' disables and falls back to the built-in prompt list",
    )
    p.add_argument("--prompts_file", default=None, help="one prompt per line; overrides --dataset_dir")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    # Precedence: --prompts_file > real dataset captions > built-in list.
    prompt_source = "builtin"
    prompts = BUILTIN_PROMPTS
    if args.prompts_file:
        lines = [ln.strip() for ln in Path(args.prompts_file).read_text().splitlines() if ln.strip()]
        if lines:
            prompts, prompt_source = lines, f"file:{args.prompts_file}"
    elif args.dataset_dir:
        sampled = _load_dataset_prompts(args.dataset_dir, args.num_trajectories, args.seed0)
        if sampled:
            prompts, prompt_source = sampled, f"dataset:{args.dataset_dir}({len(sampled)})"
        else:
            print(f"[warn] no captions under {args.dataset_dir}; using built-in prompts")
    print(f"[prompts] source={prompt_source}  n={len(prompts)}")

    # Warm-up call loads DiT into shared_models (text encoder seeded by
    # load_shared_models — the shared_models path does NOT auto-load it); THEN
    # hook so the final_layer / module hooks land on the live, shared instance.
    from library.inference.models import load_shared_models

    # The model is loaded ONCE here and reused across trajectories via shared_models,
    # so block-compile happens once at this warm-up load (same resolution as the real
    # trajectories -> the inductor graph is reused, no per-trajectory recompile).
    warm = GenerationRequest(
        prompt=prompts[0], image_size=(args.image_size[0], args.image_size[1]),
        infer_steps=4, guidance_scale=1.0, flow_shift=args.flow_shift, sampler="euler",
        seed=args.seed0, dit=args.dit, vae=args.vae, text_encoder=args.text_encoder,
        device=args.device, attn_mode=args.attn_mode,
        extra_argv=("--compile_blocks",) if args.compile else (),
    )
    wa = warm.to_args()
    wa.device = args.device
    gen_settings = get_generation_settings(wa)
    shared_models: dict = load_shared_models(wa)  # seeds text_encoder (CPU)
    with torch.no_grad():
        generate(wa, gen_settings, shared_models=shared_models)
    anima = shared_models["model"]
    h1 = anima.register_forward_hook(_fwd_hook)
    h2 = anima.final_layer.register_forward_pre_hook(_final_pre_hook)

    # ------------------------------------------------------------------ collect rows
    # per row: traj id, step i, sigma_i, ||v_i||, ||feat_i||, history dv=||v_i-v_{i-1}||,
    #          target e_i = (sigma_{i+1}-sigma_{i+2})*||v_{i+1}-v_i||, feat vector
    from library.inference.sampling import get_timesteps_sigmas

    _, sigmas = get_timesteps_sigmas(args.ref_steps, args.flow_shift, torch.device("cpu"))
    sigmas = sigmas.float()  # length ref_steps+1

    rows_traj, rows_i, rows_sigma = [], [], []
    rows_vnorm, rows_featnorm, rows_hist, rows_target = [], [], [], []
    feat_mat = []

    try:
        for k in range(args.num_trajectories):
            prompt = prompts[k % len(prompts)]
            seed = args.seed0 + k
            feats, vs = _run_trajectory(prompt, seed, args, gen_settings, shared_models)
            n = len(vs)
            if n < 3:
                print(f"[warn] trajectory {k} captured only {n} steps; skipping")
                continue
            vnorm = torch.stack([v.norm() for v in vs])  # (n,)
            # valid target indices: need v_{i+1}, v_{i+2}, sigma_{i+2}
            for i in range(n - 2):
                dv = (vs[i + 1] - vs[i]).norm()
                e_i = (sigmas[i + 1] - sigmas[i + 2]).abs() * dv
                hist = (vs[i] - vs[i - 1]).norm() if i >= 1 else float("nan")
                rows_traj.append(k)
                rows_i.append(i)
                rows_sigma.append(float(sigmas[i]))
                rows_vnorm.append(float(vnorm[i]))
                rows_featnorm.append(float(feats[i].norm()))
                rows_hist.append(float(hist))
                rows_target.append(float(e_i))
                feat_mat.append(feats[i].squeeze(0))
            print(f"[traj {k + 1}/{args.num_trajectories}] '{prompt[:40]}...' steps={n}")
    finally:
        h1.remove()
        h2.remove()

    traj = torch.tensor(rows_traj)
    sig = torch.tensor(rows_sigma)
    vn = torch.tensor(rows_vnorm)
    fn = torch.tensor(rows_featnorm)
    hist = torch.tensor(rows_hist)
    tgt = torch.tensor(rows_target)
    logt = torch.log(tgt.clamp_min(1e-8))
    X = torch.stack(feat_mat).float()  # (R, D)

    # ------------------------------------------------------------------ split by traj
    g = torch.Generator().manual_seed(0)
    uniq = traj.unique()
    perm = uniq[torch.randperm(len(uniq), generator=g)]
    n_test = max(1, int(round(args.holdout_frac * len(uniq))))
    test_trajs = set(perm[:n_test].tolist())
    is_test = torch.tensor([int(t) in test_trajs for t in traj.tolist()])
    tr, te = ~is_test, is_test

    # standardize features on train
    mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mu) / sd
    head_pred_te = _train_head(Xn[tr], logt[tr], Xn[te], epochs=args.head_epochs, device=args.device)

    # ------------------------------------------------------------------ metrics
    def sp(pred, mask):
        return _spearman(pred[mask], logt[mask])

    # within-trajectory averaged spearman on test (controls for cross-traj scale)
    def sp_within(pred):
        vals = []
        for t in test_trajs:
            m = traj == t
            if m.sum() >= 3:
                vals.append(_spearman(pred[m], logt[m]))
        vals = [v for v in vals if v == v]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    head_full = torch.full_like(logt, float("nan"))
    head_full[te] = head_pred_te

    # history baseline only defined where hist is finite
    hmask = te & torch.isfinite(hist)

    metrics = {
        "config": {
            "num_trajectories": args.num_trajectories,
            "ref_steps": args.ref_steps,
            "image_size": list(args.image_size),
            "flow_shift": args.flow_shift,
            "prompt_source": prompt_source,
            "n_rows": int(X.shape[0]),
            "n_test_trajs": n_test,
            "feat_dim": int(X.shape[1]),
        },
        "spearman_pooled_test": {
            "sigma": sp(sig, te),
            "neg_sigma": sp(-sig, te),
            "v_norm": sp(vn, te),
            "feat_norm": sp(fn, te),
            "history_dv": _spearman(hist[hmask], logt[hmask]),
            "learned_head": sp(head_full, te),
        },
        "spearman_within_traj_test": {
            "sigma": sp_within(sig),
            "neg_sigma": sp_within(-sig),
            "v_norm": sp_within(vn),
            "feat_norm": sp_within(fn),
            "learned_head": sp_within(head_full),
        },
        "error_profile": {
            "median_e": float(tgt.median()),
            "p90_over_p10": float(tgt.quantile(0.9) / tgt.quantile(0.1).clamp_min(1e-8)),
            "frac_mass_sigma_lt_0.45": float(tgt[sig < 0.45].sum() / tgt.sum()),
            "frac_rows_sigma_lt_0.45": float((sig < 0.45).float().mean()),
        },
    }

    # ------------------------------------------------------------------ verdict
    # Decision metric is WITHIN-trajectory Spearman: an online step-size controller
    # ranks steps inside one trajectory, so cross-trajectory scale (which inflates
    # pooled sigma) is not signal it can exploit. abs() since sigma ranks negatively.
    sp_head = abs(metrics["spearman_within_traj_test"]["learned_head"])
    sp_sig = max(
        abs(metrics["spearman_within_traj_test"]["sigma"]),
        abs(metrics["spearman_within_traj_test"]["neg_sigma"]),
    )
    sp_vn = abs(metrics["spearman_within_traj_test"]["v_norm"])  # free one-shot scalar
    p90_10 = metrics["error_profile"]["p90_over_p10"]
    if p90_10 < 2.0:
        verdict = f"FLAT: local error nearly uniform (p90/p10={p90_10:.2f}); adaptivity unlikely to help."
    elif sp_head - sp_sig > 0.10:
        free = (
            f" NB free ||v|| scalar already ranks at {sp_vn:.3f}"
            if sp_vn >= sp_sig
            else ""
        )
        verdict = (
            f"HEAD WINS (within-traj): one-shot head Spearman={sp_head:.3f} beats sigma={sp_sig:.3f}; "
            f"trajectory-adaptive trigger has signal.{free} Proceed to phase-2 gating."
        )
    elif sp_sig >= 0.6:
        verdict = (
            f"SIGMA SUFFICES (within-traj): sigma ranks error {sp_sig:.3f} vs head {sp_head:.3f}; "
            f"reshape the fixed schedule by sigma, no head needed."
        )
    else:
        verdict = (
            f"INCONCLUSIVE (within-traj): head={sp_head:.3f}, sigma={sp_sig:.3f}, v_norm={sp_vn:.3f}; "
            f"weak signal — neither a clear win nor clearly flat."
        )
    metrics["verdict"] = verdict

    # ------------------------------------------------------------------ artifacts
    run_dir = make_run_dir("dynamic_spectrum", label=args.label)

    csv = run_dir / "per_step.csv"
    with csv.open("w") as f:
        f.write("traj,step,sigma,v_norm,feat_norm,history_dv,target_e,is_test,head_pred\n")
        for j in range(len(rows_i)):
            hp = f"{float(head_full[j]):.6g}" if te[j] else ""
            f.write(
                f"{rows_traj[j]},{rows_i[j]},{rows_sigma[j]:.6f},{rows_vnorm[j]:.6f},"
                f"{rows_featnorm[j]:.6f},{rows_hist[j]:.6f},{rows_target[j]:.6f},"
                f"{int(bool(te[j]))},{hp}\n"
            )

    artifacts = ["per_step.csv"]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].scatter(sig.numpy(), tgt.clamp_min(1e-8).log10().numpy(), s=6, alpha=0.3)
        # binned median profile
        edges = torch.linspace(0, 1, 21)
        cx, cy = [], []
        for b in range(20):
            m = (sig >= edges[b]) & (sig < edges[b + 1])
            if m.sum() > 0:
                cx.append(float((edges[b] + edges[b + 1]) / 2))
                cy.append(float(tgt[m].clamp_min(1e-8).log10().median()))
        ax[0].plot(cx, cy, "r-o", ms=3, label="binned median")
        ax[0].axvline(0.45, color="k", ls="--", lw=0.8, label="sigma=0.45")
        ax[0].set_xlabel("sigma_i")
        ax[0].set_ylabel("log10 local fattening error e_i")
        ax[0].set_title("Q1/Q2: error profile vs sigma")
        ax[0].legend(fontsize=8)

        names = ["sigma", "neg_sigma", "v_norm", "feat_norm", "history_dv", "learned_head"]
        vals = [metrics["spearman_pooled_test"][n] for n in names]
        colors = ["#888"] * 5 + ["#d62728"]
        ax[1].bar(range(len(names)), [abs(v) if v == v else 0 for v in vals], color=colors)
        ax[1].set_xticks(range(len(names)))
        ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax[1].set_ylabel("|Spearman| vs true e_i (test)")
        ax[1].set_title("Q3: predictor ranking power")
        ax[1].set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(run_dir / "profile.png", dpi=130)
        artifacts.append("profile.png")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] plot failed: {e}")

    write_result(
        run_dir, script=__file__, args=args, metrics=metrics,
        artifacts=artifacts, device=torch.device(args.device), label=args.label,
    )

    print("\n=== Dynamic-Spectrum probe ===")
    print(f"rows={X.shape[0]}  test_trajs={n_test}/{len(uniq)}")
    print(f"Spearman(within-traj): sigma={sp_sig:.3f}  v_norm={sp_vn:.3f}  HEAD={sp_head:.3f}")
    print(f"error p90/p10={p90_10:.2f}  mass(sigma<0.45)={metrics['error_profile']['frac_mass_sigma_lt_0.45']:.2f}")
    print(f"VERDICT: {verdict}")
    print(f"-> {run_dir}")


if __name__ == "__main__":
    main()
