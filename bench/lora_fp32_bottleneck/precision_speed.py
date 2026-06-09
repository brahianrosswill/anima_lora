"""Does the LoRA fp32-bottleneck matmul policy buy real precision, and what
does it cost?

The LoRA training forwards (networks/lora_modules/lora.py:89, ortho.py:320,
chimera.py) upcast the rank matmuls to fp32 — "recovers mantissa precision
that bf16 sheds across the large-embed_dim accumulation". Two claims to test:

1. **Accumulation**: cuBLAS bf16 GEMMs already accumulate in fp32 internally,
   so the fp32 path's only real win is skipping ONE final rounding of the
   ``(B, L, r)`` bottleneck (and of the up output). If true, pure-bf16 LoRA
   matmuls should sit at the bf16 *rounding floor*, not meaningfully worse.
   Caveat probed explicitly: ``torch.backends.cuda.matmul
   .allow_bf16_reduced_precision_reduction`` defaults True, which permits
   bf16 split-K reductions — the one mechanism by which a bf16 GEMM could
   genuinely lose accumulation precision.

2. **Autocast inertness** (found while designing this bench): train.py:898
   wraps the training forward in ``accelerator.autocast()`` with the default
   ``mixed_precision="bf16"``. Autocast re-casts fp32 inputs of ``F.linear``
   back to bf16, so the ``.float()`` casts are *undone* before the GEMM —
   the live training numerics should already be pure-bf16 GEMMs plus dead
   cast traffic. Verified here by comparing the live-autocast path against
   explicit bf16 (expected ~bit-identical fwd) and against the no-autocast
   fp32 path (expected measurably different).

Regimes (all from the same bf16 leaf params; reference = fp64):

    bf16_pure       explicit bf16 matmuls, no casts (the candidate policy)
    live_autocast   the actual shipped code path (custom_down autograd, as
                    base.toml sets use_custom_down_autograd=true) under
                    torch.autocast(bf16) — what training runs TODAY
    fp32_custom     same code path, NO autocast — what the comments intend
    fp32_legacy     the x.float() legacy branch, NO autocast
    fp32_tf32       fp32_legacy with TF32 matmul enabled — a possible
                    middle ground (10-bit-mantissa inputs, fp32 accumulate,
                    tensor-core speed)

Measured per regime × shape (qkv 2048→6144, mlp1 2048→8192, mlp2 8192→2048;
B=1, L=4200, r=32 — live bucket/dim defaults):

  * forward delta rel-L2 error vs fp64 (+ the bf16-rounding floor baseline)
  * grad rel-L2 error vs fp64 for d(down), d(up), d(x)
  * fwd+bwd wall time (base GEMM included for share-of-step context) and
    peak memory, eager and torch.compile'd
  * optional ``--train_steps N``: N Adam steps on a synthetic regression,
    same seed/init across regimes, loss curves + final ΔW deviation vs the
    fp64 run — does bottleneck precision matter against gradient noise?

Usage::

    uv run python bench/lora_fp32_bottleneck/precision_speed.py
    uv run python bench/lora_fp32_bottleneck/precision_speed.py --no-compile --train_steps 0
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402


class _LoRADownProjectFn(torch.autograd.Function):
    """Inline replica of the retired ``networks/lora_modules/custom_autograd``
    Function (removed 2026-06-10 as a result of this bench) so the
    ``live_autocast`` / ``fp32_custom`` regimes stay reproducible."""

    @staticmethod
    def forward(ctx, x, weight):
        out = F.linear(x.float(), weight.float())
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        go = grad_out.float()
        grad_x = go.matmul(weight.float()).to(x.dtype)
        grad_weight = (
            go.reshape(-1, go.shape[-1])
            .transpose(0, 1)
            .matmul(x.float().reshape(-1, x.shape[-1]))
        )
        return grad_x, grad_weight.to(weight.dtype)


def lora_down_project(x, weight, inv_scale):
    assert inv_scale is None, "bench replica covers the unscaled path only"
    return _LoRADownProjectFn.apply(x, weight)


SHAPES = {
    "qkv_proj": (2048, 6144),
    "mlp_layer1": (2048, 8192),
    "mlp_layer2": (8192, 2048),
}

REGIMES = ["bf16_pure", "live_autocast", "fp32_custom", "fp32_legacy", "fp32_tf32"]


@contextmanager
def tf32(enabled: bool):
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def make_problem(in_dim, out_dim, args, device, seed=0):
    """bf16 leaf params + input with DiT-like channel outliers."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(1, args.seq_len, in_dim, generator=g, device=device)
    n_out = max(1, int(in_dim * 0.01))
    idx = torch.randperm(in_dim, generator=g, device=device)[:n_out]
    # DiT inputs carry 80–96x DC-bias channel outliers (hydra-lora.md §Fixes);
    # they stress exactly the accumulation path the fp32 policy worries about.
    x[..., idx] *= args.outlier_scale
    x = x.to(torch.bfloat16)

    base_w = (
        torch.randn(out_dim, in_dim, generator=g, device=device) / in_dim**0.5
    ).to(torch.bfloat16)

    down = torch.randn(args.rank, in_dim, generator=g, device=device) * (
        1.0 / in_dim**0.5
    )
    # Mid-training magnitude (zero-init up would make every error metric 0/0).
    up = torch.randn(out_dim, args.rank, generator=g, device=device) * 1e-3
    down = down.to(torch.bfloat16)
    up = up.to(torch.bfloat16)
    go = torch.randn(1, args.seq_len, out_dim, generator=g, device=device).to(
        torch.bfloat16
    )
    return x, base_w, down, up, go


def delta_fwd(x, down, up, mask, regime):
    """The LoRAModule.forward training-branch delta math, per regime.

    Mirrors lora.py:94-113 (scale=alpha/dim=1 at the live 32/32 default,
    multiplier=1, no dropout). ``mask`` is the all-ones ``_timestep_mask``
    default buffer (fp32, like base.py registers it).
    """
    if regime == "bf16_pure":
        lx = F.linear(x, down)
        lx = lx * mask.to(torch.bfloat16)
        return F.linear(lx, up)
    if regime == "live_autocast":
        with torch.autocast("cuda", torch.bfloat16):
            lx = lora_down_project(x, down, None)
            lx = lx * mask
            return F.linear(lx, up.float())
    if regime == "fp32_custom":
        lx = lora_down_project(x, down, None)
        lx = lx * mask
        return F.linear(lx, up.float())
    if regime in ("fp32_legacy", "fp32_tf32"):
        lx = F.linear(x.float(), down.float())
        lx = lx * mask
        return F.linear(lx, up.float())
    raise ValueError(regime)


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.double()
    denom = ref.norm().clamp_min(1e-30)
    return float((a.double() - ref).norm() / denom)


# ---------------------------------------------------------------------------
# 1. Precision: fwd + grads vs fp64 reference
# ---------------------------------------------------------------------------


def bench_precision(args, device):
    results = {}
    mask = torch.ones(1, args.rank, dtype=torch.float32, device=device)
    for shape_name, (in_dim, out_dim) in SHAPES.items():
        x, _base_w, down0, up0, go = make_problem(in_dim, out_dim, args, device)

        # fp64 reference (exact bf16 -> fp64 upcasts of the same values).
        x64 = x.double().requires_grad_(True)
        d64 = down0.double().requires_grad_(True)
        u64 = up0.double().requires_grad_(True)
        ref = F.linear(F.linear(x64, d64) * mask.double(), u64)
        ref.backward(go.double())
        ref_out = ref.detach()
        ref_gd, ref_gu, ref_gx = d64.grad, u64.grad, x64.grad

        shape_res = {
            # Rounding floor: the fp64-exact delta simply stored as bf16.
            "floor_round_ref_to_bf16": rel_err(ref_out.to(torch.bfloat16), ref_out)
        }
        for regime in REGIMES:
            xg = x.clone().requires_grad_(True)
            dg = down0.clone().requires_grad_(True)
            ug = up0.clone().requires_grad_(True)
            ctx = tf32(True) if regime == "fp32_tf32" else nullcontext()
            with ctx:
                out = delta_fwd(xg, dg, ug, mask, regime)
                out.backward(go.to(out.dtype))
            shape_res[regime] = {
                "fwd": rel_err(out.detach(), ref_out),
                "grad_down": rel_err(dg.grad, ref_gd),
                "grad_up": rel_err(ug.grad, ref_gu),
                "grad_x": rel_err(xg.grad, ref_gx),
                "fwd_dtype": str(out.dtype),
            }
        # Inertness check: live autocast vs explicit bf16.
        xg = x.clone().requires_grad_(True)
        out_live = delta_fwd(xg, down0.clone(), up0.clone(), mask, "live_autocast")
        out_bf16 = delta_fwd(
            x.clone().requires_grad_(True),
            down0.clone(),
            up0.clone(),
            mask,
            "bf16_pure",
        )
        shape_res["live_autocast_vs_bf16_pure_max_abs_diff"] = float(
            (out_live.detach().float() - out_bf16.detach().float()).abs().max()
        )
        results[shape_name] = shape_res
    return results


# ---------------------------------------------------------------------------
# 2. Accumulation probe: is a bf16 GEMM's accumulation already fp32?
# ---------------------------------------------------------------------------


def bench_accumulation(args, device):
    """err(bf16 GEMM) ~= err(round(fp32 GEMM -> bf16)) ==> accumulation is
    effectively fp32 and the only loss is final rounding."""
    out = {}
    for shape_name, (in_dim, out_dim) in SHAPES.items():
        x, _w, _d, _u, _go = make_problem(in_dim, out_dim, args, device)
        g = torch.Generator(device=device).manual_seed(7)
        w = (torch.randn(out_dim, in_dim, generator=g, device=device) / in_dim**0.5).to(
            torch.bfloat16
        )
        ref = F.linear(x.double(), w.double())
        prev = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            bf16_reduced = F.linear(x, w)
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            bf16_full = F.linear(x, w)
        finally:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = prev
        fp32 = F.linear(x.float(), w.float())
        out[shape_name] = {
            "k_dim": in_dim,
            "bf16_gemm_reduced_precision_reduction": rel_err(bf16_reduced, ref),
            "bf16_gemm_full_fp32_reduction": rel_err(bf16_full, ref),
            "fp32_gemm": rel_err(fp32, ref),
            "fp32_gemm_rounded_to_bf16": rel_err(fp32.to(torch.bfloat16), ref),
        }
    return out


# ---------------------------------------------------------------------------
# 3. Speed + memory: fwd+bwd per regime, eager and compiled
# ---------------------------------------------------------------------------


def make_step(base_w, mask, regime):
    def step(x, down, up, go):
        y = F.linear(x, base_w) + delta_fwd(x, down, up, mask, regime).to(
            torch.bfloat16
        )
        y.backward(go)
        return y

    return step


def time_step(step, x, down, up, go, iters, warmup):
    for _ in range(warmup):
        down.grad = up.grad = x.grad = None
        step(x, down, up, go)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        down.grad = up.grad = x.grad = None
        step(x, down, up, go)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def measure_peak(step, x, down, up, go):
    down.grad = up.grad = x.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    step(x, down, up, go)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def bench_speed(args, device, compile_mode):
    results = {}
    mask = torch.ones(1, args.rank, dtype=torch.float32, device=device)
    for shape_name, (in_dim, out_dim) in SHAPES.items():
        x0, base_w, down0, up0, go = make_problem(in_dim, out_dim, args, device)
        shape_res = {}

        # Base GEMM alone, for share-of-step context.
        def base_only(x, down, up, go):
            y = F.linear(x, base_w)
            y.backward(go)
            return y

        xb = x0.clone().requires_grad_(True)
        shape_res["base_only_ms"] = time_step(
            base_only, xb, down0.clone(), up0.clone(), go, args.iters, args.warmup
        )

        for regime in REGIMES:
            xg = x0.clone().requires_grad_(True)
            dg = down0.clone().requires_grad_(True)
            ug = up0.clone().requires_grad_(True)
            step = make_step(base_w, mask, regime)
            entry = {}
            ctx = tf32(True) if regime == "fp32_tf32" else nullcontext()
            with ctx:
                entry["eager_ms"] = time_step(
                    step, xg, dg, ug, go, args.iters, args.warmup
                )
                entry["eager_peak_mib"] = measure_peak(step, xg, dg, ug, go)
                if compile_mode is not None:
                    try:
                        cstep = torch.compile(step, dynamic=False, mode=compile_mode)
                        entry["compiled_ms"] = time_step(
                            cstep, xg, dg, ug, go, args.iters, args.warmup
                        )
                        entry["compiled_peak_mib"] = measure_peak(cstep, xg, dg, ug, go)
                    except (
                        Exception
                    ) as e:  # compile failures are a finding, not a crash
                        entry["compiled_error"] = f"{type(e).__name__}: {e}"
            shape_res[regime] = entry
        results[shape_name] = shape_res
    return results


# ---------------------------------------------------------------------------
# 4. Optional: Adam trajectory divergence on a synthetic regression
# ---------------------------------------------------------------------------


def bench_train_probe(args, device, run_dir):
    """N Adam steps fitting base(x)+target_delta; identical seed/init across
    regimes. fp64 run = ground truth trajectory. Answers: does bottleneck
    precision shift optimization beyond its own rounding floor?"""
    in_dim, out_dim = SHAPES["qkv_proj"]
    x, base_w, down0, up0, go = make_problem(in_dim, out_dim, args, device, seed=3)
    del go
    g = torch.Generator(device=device).manual_seed(11)
    # Target: base output + a reachable rank-r delta of trained-LoRA magnitude.
    td = torch.randn(args.rank, in_dim, generator=g, device=device) / in_dim**0.5
    tu = torch.randn(out_dim, args.rank, generator=g, device=device) * 3e-3
    with torch.no_grad():
        target = F.linear(x.float(), base_w.float()) + F.linear(
            F.linear(x.float(), td), tu
        )

    mask = torch.ones(1, args.rank, dtype=torch.float32, device=device)
    curves, finals = {}, {}
    delta_w_ref = None
    for regime in ["fp64_ref"] + REGIMES:
        if regime == "fp64_ref":
            d = down0.double().clone().requires_grad_(True)
            u = up0.double().clone().requires_grad_(True)
        else:
            d = down0.clone().requires_grad_(True)
            u = up0.clone().requires_grad_(True)
        opt = torch.optim.Adam([d, u], lr=1e-3)
        losses = []
        ctx = tf32(True) if regime == "fp32_tf32" else nullcontext()
        with ctx:
            for _ in range(args.train_steps):
                opt.zero_grad(set_to_none=True)
                if regime == "fp64_ref":
                    delta = F.linear(F.linear(x.double(), d) * mask.double(), u)
                else:
                    delta = delta_fwd(x, d, u, mask, regime)
                y = F.linear(x.float(), base_w.float()) + delta.float()
                loss = F.mse_loss(y, target)
                loss.backward()
                opt.step()
                losses.append(float(loss))
        curves[regime] = losses
        with torch.no_grad():
            dw = (u.double() @ d.double()).cpu()
        if regime == "fp64_ref":
            delta_w_ref = dw
            finals[regime] = {"final_loss": losses[-1]}
        else:
            finals[regime] = {
                "final_loss": losses[-1],
                "delta_w_rel_dev_vs_fp64_run": float(
                    (dw - delta_w_ref).norm() / delta_w_ref.norm().clamp_min(1e-30)
                ),
            }
    (run_dir / "train_probe_curves.json").write_text(json.dumps(curves))
    return finals


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--seq_len", type=int, default=4200)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--outlier_scale", type=float, default=90.0)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--train_steps", type=int, default=200)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument(
        "--compile_mode", default=None, help='inductor mode (e.g. "max-autotune")'
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    run_dir = make_run_dir("lora_fp32_bottleneck", label=args.label)

    metrics = {
        "settings": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32_default": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction_default": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
        }
    }
    print("== precision (rel-L2 vs fp64) ==")
    metrics["precision"] = bench_precision(args, device)
    print(json.dumps(metrics["precision"], indent=2))

    print("== accumulation probe ==")
    metrics["accumulation"] = bench_accumulation(args, device)
    print(json.dumps(metrics["accumulation"], indent=2))

    print("== speed / memory ==")
    metrics["speed"] = bench_speed(
        args, device, (args.compile_mode or "default") if args.compile else None
    )
    print(json.dumps(metrics["speed"], indent=2))

    if args.train_steps > 0:
        print(f"== train probe ({args.train_steps} Adam steps) ==")
        metrics["train_probe"] = bench_train_probe(args, device, run_dir)
        print(json.dumps(metrics["train_probe"], indent=2))

    artifacts = ["train_probe_curves.json"] if args.train_steps > 0 else []
    out = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=artifacts,
        device=device,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
