"""Invariant tests for FLAIR-edit (docs/proposal/flair_edit.md).

CPU-only — no DiT/VAE weights. Covers the load-bearing structural guarantees:
(a) the inpaint operator is self-adjoint and passes the dot-product test;
(b) a full-keep mask + α=1 + zero prior weight is a near-identity (the loop does
    not corrupt data it's handed) — driven by tiny identity fakes;
(e) a ``flair_task=edit`` GenerationRequest round-trips through the parser to the
    FLAIR branch, and the exactly-one-mask validation fires.

The real-model guarantees — unmasked-region bit-exactness after a solve and the
dim-2 round-trip — live in the bench (``bench/flair/``), which loads weights.
"""

from __future__ import annotations

import pytest
import torch

from library.inference.corrections.flair import FlairConfig, flair_solve
from library.inference.corrections.flair_operators import (
    InpaintOperator,
    build_operator,
)


# --- (a) operator adjoint + self-adjointness -------------------------------- #


def test_inpaint_operator_dot_product_adjoint():
    """⟨A x, y⟩ == ⟨x, A^T y⟩ for the binary-mask operator (A = A^T)."""
    torch.manual_seed(0)
    mask = (torch.rand(1, 1, 8, 8) > 0.5).float()
    op = InpaintOperator(mask=mask)
    x = torch.randn(1, 3, 8, 8)
    y = torch.randn(1, 3, 8, 8)
    lhs = (op.degrade(x) * y).sum()
    rhs = (x * op.degrade(y)).sum()  # degrade IS A^T (self-adjoint)
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_inpaint_operator_self_adjoint_idempotent():
    """A is a 0/1 projection: A == A^T and A∘A == A."""
    mask = (torch.rand(1, 1, 6, 6) > 0.3).float()
    op = InpaintOperator(mask=mask)
    x = torch.randn(1, 3, 6, 6)
    once = op.degrade(x)
    twice = op.degrade(once)
    assert torch.allclose(once, twice, atol=1e-6)  # idempotent projection
    # adjoint_init is the self-adjoint A^T y = m ⊙ y
    assert torch.allclose(op.adjoint_init(x, target_hw=(6, 6)), op.degrade(x))


def test_build_operator_inpaint_requires_mask():
    with pytest.raises(ValueError, match="requires a keep-mask"):
        build_operator("inpaint")


# --- (b) full-keep + α=1 + zero prior ⇒ near-identity ----------------------- #


class _IdentityVAE:
    """A 3-channel identity VAE so latents == pixels (differentiable decode)."""

    dtype = torch.float32

    @torch.no_grad()
    def encode_pixels_to_latents(self, pixels):
        return pixels.clone()

    def decode_to_pixels(self, latents):
        return latents * 1.0  # identity, keeps the autograd edge for HDC


class _ZeroAnima:
    """A velocity field that returns 0 — the prior pull is zeroed anyway."""

    def __call__(self, x5, te, embed, padding_mask=None):
        return torch.zeros_like(x5)


def test_full_keep_zero_prior_is_near_identity():
    """A full-keep mask with reg=0 + α=1 leaves a clean latent untouched."""
    torch.manual_seed(0)
    x_gt = torch.randn(1, 3, 8, 8)  # stand-in "clean latent" (== pixels here)
    keep = torch.ones(1, 1, 8, 8)
    op = build_operator("inpaint", mask=keep)
    y = op.degrade(x_gt)  # full-keep ⇒ y == x_gt

    cfg = FlairConfig(
        infer_steps=8,
        t_stop=0.2,
        reg_scale=0.0,  # zero prior pull
        alpha=1.0,  # deterministic DTA
        hdc_steps=12,
        hdc_lr=0.02,
        guidance_scale=1.0,
    )
    mu = flair_solve(
        _ZeroAnima(),
        _IdentityVAE(),
        embed=torch.zeros(1, 4, 8),
        neg_embed=torch.zeros(1, 4, 8),
        y=y,
        operator=op,
        target_hw=(8, 8),
        cfg=cfg,
        device=torch.device("cpu"),
    )
    assert mu.shape == x_gt.shape
    # The kept (== whole) region must survive the loop ~unchanged. Not bit-exact:
    # the adjoint init rounds through bf16 and Adam's normalized HDC step has a
    # small noise floor even at ~zero gradient. The invariant is "doesn't corrupt
    # the data" — a tight relative error and near-perfect correlation, not drift.
    err = (mu.float() - x_gt).norm() / x_gt.norm()
    assert err < 0.05, f"full-keep reconstruction drifted: rel-L2 {err:.4f}"
    corr = torch.corrcoef(torch.stack([mu.float().flatten(), x_gt.flatten()]))[0, 1]
    assert corr > 0.995, f"reconstruction decorrelated from input: corr {corr:.4f}"


# --- (e) request → parser → FLAIR branch + validation ----------------------- #


def _req(**kw):
    from library.inference.request import GenerationRequest

    base = dict(
        prompt="blue hair",
        flair_task="edit",
        flair_edit_image="src.png",
        text_encoder="te.safetensors",
        save_path="out.png",
    )
    base.update(kw)
    return GenerationRequest(**base)


def test_request_reaches_flair_branch():
    args = _req(flair_mask="mask.png").to_args()
    assert args.flair_task == "edit"
    assert args.flair_edit_image == "src.png"
    assert args.flair_mask == "mask.png"
    # Faithful DTA: α scales the paper's `inv_alpha: 1-t` schedule (1.0 = unscaled).
    assert args.flair_alpha == 1.0
    # Edit defaults to the stable linear λ_R; the calibrated table diverges the hole.
    assert args.flair_calib == "off"


def test_request_mask_prompt_path():
    args = _req(flair_mask_prompt="eyes").to_args()
    assert args.flair_mask_prompt == "eyes"
    assert args.flair_mask is None


def test_flair_edit_requires_exactly_one_mask():
    with pytest.raises(ValueError, match="exactly one of"):
        _req(flair_mask="m.png", flair_mask_prompt="eyes").to_args()
    with pytest.raises(ValueError, match="exactly one of"):
        _req().to_args()  # neither mask given


def test_flair_edit_requires_source_image():
    with pytest.raises(ValueError, match="flair_edit_image"):
        _req(flair_edit_image=None, flair_mask="m.png").to_args()
