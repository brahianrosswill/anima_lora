#!/usr/bin/env python
"""EasyControl Position-Aware Interpolation (PAI) — RoPE equivalence + alignment.

Verifies the two correctness claims behind the ``cond_res_scale`` knob
(``networks/methods/easycontrol.py`` + ``VideoRopePosition3DEmb
.generate_embeddings_scaled``):

Test 1 — IDENTITY at scale 1.0
    ``generate_embeddings_scaled(shape, h_scale=1, w_scale=1)`` must be
    *bit-exact* to ``generate_embeddings(shape)``. This is what guarantees
    ``cond_res_scale = 1.0`` leaves every existing EasyControl checkpoint
    unchanged (``self.seq[:H] * 1.0 == self.seq[:H]``).

Test 2 — ALIGNMENT for an integer downscale ratio
    With a downscaled cond grid and an integer rescale ``S_h = S_w = r``, each
    cond patch ``i`` must land at RoPE position ``i·r`` — i.e. exactly on the
    target grid's coordinate ``i·r``. We build the full-resolution target RoPE,
    build the scaled cond RoPE on the ``r×``-smaller grid, and assert each cond
    row equals the corresponding *subsampled* target row. If the scaled
    positions land on the target grid, a downsampled cond stays pixel-aligned
    with the full-res target — which is the whole point of PAI (paper §3.3,
    Eq. 11-12).

Test 3 — TOKEN-COUNT / RANGE sanity
    The scaled grid has fewer tokens (cheaper cond stream) and no position
    exceeds the trained grid extent.

All tests run on CPU with a tiny RoPE module — no model download.

Usage
-----
    uv run python bench/easycontrol/pai_equivalence.py
    uv run python bench/easycontrol/pai_equivalence.py --dtype bf16
"""

from __future__ import annotations

import argparse
import logging

import torch

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima.models import VideoRopePosition3DEmb  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def _make_rope(head_dim: int, len_hwt: int, device, dtype) -> VideoRopePosition3DEmb:
    emb = VideoRopePosition3DEmb(
        head_dim=head_dim,
        len_h=len_hwt,
        len_w=len_hwt,
        len_t=len_hwt,
        enable_fps_modulation=True,
    )
    return emb.to(device=device, dtype=dtype)


def _shape(T: int, H: int, W: int, D: int) -> torch.Size:
    # generate_embeddings consumes a (B, T, H, W, C) shape; only T/H/W are read.
    return torch.Size((1, T, H, W, D))


def test_identity(emb, T, H, W, head_dim) -> dict:
    """scale=1.0 must be bit-exact to the native-position embeddings."""
    cos_n, sin_n = emb.generate_embeddings(_shape(T, H, W, head_dim), fps=None)
    cos_s, sin_s = emb.generate_embeddings_scaled(
        _shape(T, H, W, head_dim), h_scale=1.0, w_scale=1.0, fps=None
    )
    dcos = (cos_n - cos_s).abs().max().item()
    dsin = (sin_n - sin_s).abs().max().item()
    return {
        "max_abs_diff_cos": dcos,
        "max_abs_diff_sin": dsin,
        "bit_exact": dcos == 0.0 and dsin == 0.0,
    }


def test_alignment(emb, T, full_hw, ratio, head_dim) -> dict:
    """Integer downscale ratio: cond row i must equal target row i*ratio."""
    H_full = W_full = full_hw
    H_c = W_c = full_hw // ratio

    cos_full, sin_full = emb.generate_embeddings(
        _shape(T, H_full, W_full, head_dim), fps=None
    )
    cos_sc, sin_sc = emb.generate_embeddings_scaled(
        _shape(T, H_c, W_c, head_dim),
        h_scale=float(ratio),
        w_scale=float(ratio),
        fps=None,
    )

    # Flattened index in (T, H, W) row-major: idx = (t*H + h)*W + w. The cond
    # grid (H_c, W_c) maps patch (h_c, w_c) -> full grid (h_c*ratio, w_c*ratio).
    max_cos = 0.0
    max_sin = 0.0
    for t in range(T):
        for h_c in range(H_c):
            for w_c in range(W_c):
                sc_idx = (t * H_c + h_c) * W_c + w_c
                full_idx = (t * H_full + h_c * ratio) * W_full + (w_c * ratio)
                max_cos = max(
                    max_cos,
                    (cos_sc[sc_idx] - cos_full[full_idx]).abs().max().item(),
                )
                max_sin = max(
                    max_sin,
                    (sin_sc[sc_idx] - sin_full[full_idx]).abs().max().item(),
                )
    return {
        "ratio": ratio,
        "full_grid": [H_full, W_full],
        "cond_grid": [H_c, W_c],
        "max_abs_diff_cos": max_cos,
        "max_abs_diff_sin": max_sin,
        "aligned": max_cos == 0.0 and max_sin == 0.0,
    }


def test_token_count(emb, T, full_hw, ratio, head_dim) -> dict:
    H_c = W_c = full_hw // ratio
    cos_sc, _ = emb.generate_embeddings_scaled(
        _shape(T, H_c, W_c, head_dim),
        h_scale=float(ratio),
        w_scale=float(ratio),
        fps=None,
    )
    cos_full, _ = emb.generate_embeddings(_shape(T, full_hw, full_hw, head_dim), fps=None)
    s_c = cos_sc.shape[0]
    s_full = cos_full.shape[0]
    max_pos = (H_c - 1) * ratio  # largest scaled position along an axis
    return {
        "cond_tokens": s_c,
        "full_tokens": s_full,
        "token_ratio": s_c / s_full,
        "max_scaled_pos": max_pos,
        "within_grid": max_pos < emb.max_h,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--len", type=int, default=64, help="RoPE table extent (len_h/w/t)")
    p.add_argument("--full_hw", type=int, default=8, help="full target grid (H=W)")
    p.add_argument("--ratio", type=int, default=2, help="integer downscale ratio")
    p.add_argument("--t", type=int, default=1, help="temporal patches (1 = image)")
    p.add_argument(
        "--dtype", choices=["fp32", "bf16"], default="fp32",
        help="fp32 for the exactness claim; bf16 to sanity-check train precision.",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--label", default="pai")
    args = p.parse_args()

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    emb = _make_rope(args.head_dim, args.len, device, dtype)

    identity = test_identity(emb, args.t, args.full_hw, args.full_hw, args.head_dim)
    alignment = test_alignment(emb, args.t, args.full_hw, args.ratio, args.head_dim)
    tokens = test_token_count(emb, args.t, args.full_hw, args.ratio, args.head_dim)

    print()
    print("=== Test 1: identity at scale 1.0 (bit-exact to native) ===")
    print(
        f"  max|Δcos|={identity['max_abs_diff_cos']:.3e}  "
        f"max|Δsin|={identity['max_abs_diff_sin']:.3e}  "
        f"-> {'PASS' if identity['bit_exact'] else 'FAIL'}"
    )
    print()
    print(f"=== Test 2: alignment, ratio={args.ratio} "
          f"({alignment['full_grid']} <- {alignment['cond_grid']}) ===")
    print(
        f"  max|Δcos|={alignment['max_abs_diff_cos']:.3e}  "
        f"max|Δsin|={alignment['max_abs_diff_sin']:.3e}  "
        f"-> {'PASS (cond lands on target grid)' if alignment['aligned'] else 'FAIL'}"
    )
    print()
    print("=== Test 3: token-count / range sanity ===")
    print(
        f"  cond_tokens={tokens['cond_tokens']} / full={tokens['full_tokens']} "
        f"(ratio {tokens['token_ratio']:.3f})  "
        f"max_pos={tokens['max_scaled_pos']} within_grid={tokens['within_grid']}"
    )

    all_pass = (
        identity["bit_exact"] and alignment["aligned"] and tokens["within_grid"]
    )
    print()
    print(f"Verdict: {'ALL PASS' if all_pass else 'FAILURE'}")

    metrics = {
        "identity": identity,
        "alignment": alignment,
        "tokens": tokens,
        "all_pass": all_pass,
    }
    out_dir = make_run_dir("easycontrol", label=args.label)
    result_path = write_result(
        out_dir,
        script=__file__,
        args=args,
        label=args.label,
        metrics=metrics,
        device=device,
    )
    logger.info(f"result → {result_path}")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
