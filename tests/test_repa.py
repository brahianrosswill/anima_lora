"""REPA v2 adapter + wiring tests (docs/proposal/repa_v2_patchwise_pe_spatial.md).

Covers the load-bearing invariants:
  - native_flatten layout-agnosticism: the captured (B,1,seq,1,D) and eager
    (B,1,H,W,D) block outputs produce a bit-identical alignment loss.
  - grad flows back through the captured feature (into the LoRA blocks).
  - PE-grid orientation disambiguation (aspect-symmetric token counts).
  - the composer gate is off by default (flag-off ⇒ "repa" never active).
"""

from __future__ import annotations

import types

import pytest
import torch

from library.training.repa import REPAHead, REPAMethodAdapter
from library.vision.buckets import get_bucket_spec


def _make_adapter(mode: str, *, patch: int = 2) -> REPAMethodAdapter:
    a = REPAMethodAdapter()
    a._mode = mode
    a._patch = patch
    a._spec = get_bucket_spec("pe_spatial")
    return a


def _primary(latents: torch.Tensor, *, is_train: bool = True):
    # extra_forwards only reads .is_train and .latents.
    return types.SimpleNamespace(is_train=is_train, latents=latents)


def _ctx(network=None):
    return types.SimpleNamespace(network=network)


def _square_inputs(b=2, d_enc=768):
    """Square 32x32 encoder grid → 1024 patches + CLS. Latent 64x64, patch 2."""
    spec = get_bucket_spec("pe_spatial")
    # (32,32) bucket → 1024 patches, T_pe = 1025.
    pe = torch.randn(b, 32 * 32 + 1, d_enc)
    latents = torch.zeros(b, 16, 64, 64)  # h_dit=w_dit=32 → 1024 DiT tokens
    return spec, pe, latents


def test_native_flatten_layout_agnostic():
    """Eager (B,1,H,W,D) and native-flatten (B,1,seq,1,D) → identical loss."""
    _spec, pe, latents = _square_inputs()
    b, d = 2, 64
    # Shared underlying token data in row-major (B, 1024, D).
    tokens = torch.randn(b, 1024, d)
    eager = tokens.reshape(b, 1, 32, 32, d).clone()
    flat = tokens.reshape(b, 1, 1024, 1, d).clone()

    for mode in ("relational", "absolute"):
        net = None
        if mode == "absolute":
            net = types.SimpleNamespace(repa_head=REPAHead(d, d, 768))
        a = _make_adapter(mode)

        a._captured, a._pe_features, a._latent_hw = eager, pe, (64, 64)
        loss_eager = a.extra_forwards(_ctx(net), _primary(latents))["repa"]

        a._captured, a._pe_features, a._latent_hw = flat, pe, (64, 64)
        loss_flat = a.extra_forwards(_ctx(net), _primary(latents))["repa"]

        assert torch.allclose(loss_eager, loss_flat, atol=0, rtol=0), mode
        assert torch.isfinite(loss_eager)


def test_grad_flows_to_captured():
    _spec, pe, latents = _square_inputs()
    b, d = 2, 64
    cap = torch.randn(b, 1, 32, 32, d, requires_grad=True)
    a = _make_adapter("relational")
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    loss = a.extra_forwards(_ctx(), _primary(latents))["repa"]
    loss.backward()
    assert cap.grad is not None and torch.isfinite(cap.grad).all()
    assert cap.grad.abs().sum() > 0


def test_relational_zero_when_identical_structure():
    """Same per-token directions on both sides → Gram match → ~0 loss."""
    a = _make_adapter("relational")
    b, n, d = 2, 1024, 768
    feats = torch.randn(b, n, d)
    # DiT side D == PE d here so the same directions give identical Grams.
    cap = feats.reshape(b, 1, 32, 32, d)
    pe = torch.cat([torch.zeros(b, 1, d), feats], dim=1)  # prepend CLS
    a._captured, a._pe_features, a._latent_hw = cap, pe, (64, 64)
    loss = a.extra_forwards(_ctx(), _primary(torch.zeros(b, 16, 64, 64)))["repa"]
    assert loss.item() < 1e-6


def test_pe_grid_orientation_disambiguation():
    a = _make_adapter("relational")
    # 1058 patches is shared by (46,23) portrait and (23,46) landscape.
    assert 46 * 23 == 23 * 46 == 1058
    gh, gw = a._pe_grid(1058, h_lat=144, w_lat=72)  # portrait
    assert (gh, gw) == (46, 23)
    gh, gw = a._pe_grid(1058, h_lat=72, w_lat=144)  # landscape
    assert (gh, gw) == (23, 46)
    # Square is unambiguous.
    assert a._pe_grid(1024, 64, 64) == (32, 32)


def test_skips_when_not_train_or_missing():
    a = _make_adapter("relational")
    _spec, pe, latents = _square_inputs()
    a._captured, a._pe_features, a._latent_hw = (
        torch.randn(2, 1, 32, 32, 64),
        pe,
        (64, 64),
    )
    # Validation pass → no REPA term.
    assert a.extra_forwards(_ctx(), _primary(latents, is_train=False)) is None
    # Missing PE features → skip.
    a._pe_features = None
    assert a.extra_forwards(_ctx(), _primary(latents)) is None


def test_composer_gate_off_by_default():
    from library.training.losses import build_loss_composer

    args = types.SimpleNamespace(
        vr_loss_weight=0.0, functional_loss_weight=0.0, multiscale_loss_weight=0.0
    )
    # A network with no _repa_weight attribute → repa must not be active.
    net = types.SimpleNamespace()
    comp = build_loss_composer(args, net)
    assert "repa" not in comp.active_losses

    net._repa_weight = 0.05
    comp = build_loss_composer(args, net)
    assert "repa" in comp.active_losses


def test_repa_loss_handler_weighting():
    from library.training.losses import LossContext, _repa_loss

    pred = torch.zeros(2, 16, 1, 8, 8)
    base = dict(
        model_pred=pred,
        target=pred,
        timesteps=None,
        weighting=None,
        huber_c=None,
        loss_weights=None,
        batch={},
        args=None,
        is_train=True,
    )
    net = types.SimpleNamespace(_repa_weight=0.05)
    ctx = LossContext(network=net, aux={"repa": torch.tensor(2.0)}, **base)
    assert _repa_loss(ctx).item() == pytest.approx(0.1)
    # weight 0 → zero
    net0 = types.SimpleNamespace(_repa_weight=0.0)
    ctx0 = LossContext(network=net0, aux={"repa": torch.tensor(2.0)}, **base)
    assert _repa_loss(ctx0).item() == 0.0
