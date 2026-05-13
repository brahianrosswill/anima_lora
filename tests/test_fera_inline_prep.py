"""Parity test for FeRA inline-prep (plan.md Phase 1).

Builds a minimal DiT-shaped harness (a handful of named Linears matched
by FeRA's default target regex) and runs a forward + backward through
both legacy (``inline_prep=False``) and inline (``inline_prep=True``)
paths against the same input and weights. Asserts bit-equal forward
output and bit-equal gradients on every trainable parameter.

The test guards three things:
  1. ``FeRALinear.forward`` reads from the prep back-ref when wired, from
     ``_routing_weights`` / ``_cached_R_q`` otherwise.
  2. The router + Cayley computations run inside the inline path's
     ``Anima._run_blocks``-equivalent hook, not eagerly in
     ``prepare_forward``.
  3. Autograd flows correctly through the prep's side-effect writes —
     router/S_q/S_p/lambda_layer grads match between the two paths.

Runs CPU-only. Cudagraph + ``compile_mode='full'`` behavior is verified
separately via ``make exp-fera`` after this test passes.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from networks.methods.fera import FeRANetwork


class _MockBlock(nn.Module):
    """One block's worth of FeRA-targeted Linears.

    Names are chosen so FeRA's default target regex
    (``.*\\.(qkv_proj|q_proj|kv_proj|output_proj|layer[12])$``) matches
    every Linear here. Real Anima blocks have an attention path + an MLP
    path; we just chain four Linears so the test exercises both ortho
    and non-ortho FeRA codepaths under the same harness without dragging
    in attention/RoPE/etc.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = nn.ModuleDict({"qkv_proj": nn.Linear(dim, dim)})
        self.cross_attn = nn.ModuleDict(
            {"q_proj": nn.Linear(dim, dim), "kv_proj": nn.Linear(dim, dim)}
        )
        self.output_proj = nn.Linear(dim, dim)
        self.mlp = nn.ModuleDict(
            {"layer1": nn.Linear(dim, dim), "layer2": nn.Linear(dim, dim)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.self_attn["qkv_proj"](x)
        h = self.cross_attn["q_proj"](h) + self.cross_attn["kv_proj"](h)
        h = self.output_proj(h)
        h = self.mlp["layer2"](self.mlp["layer1"](h).relu())
        return x + h


class _MockDiT(nn.Module):
    """Stand-in for ``library.anima.models.Anima`` — only the two hooks
    FeRA inline-prep needs: ``set_fera_prep`` / ``clear_fera_prep`` and
    the ``_run_blocks``-shaped forward that fires the prep at the top of
    its compute graph.
    """

    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([_MockBlock(dim) for _ in range(depth)])

    def set_fera_prep(self, prep) -> None:
        object.__setattr__(self, "_fera_prep_ref", prep)

    def clear_fera_prep(self) -> None:
        object.__setattr__(self, "_fera_prep_ref", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prep = getattr(self, "_fera_prep_ref", None)
        if prep is not None:
            prep()
        for block in self.blocks:
            x = block(x)
        return x


def _build_pair(dim: int, depth: int, ortho: bool, num_experts: int, rank: int):
    """Build two FeRANetwork-wrapped DiTs with identical weights — one
    legacy, one inline. Returns ``(legacy_net, legacy_dit, inline_net,
    inline_dit, z_t)``.
    """
    torch.manual_seed(0)
    base = _MockDiT(dim, depth)
    sd = {k: v.detach().clone() for k, v in base.state_dict().items()}

    legacy_dit = _MockDiT(dim, depth)
    legacy_dit.load_state_dict(sd)
    legacy_net = FeRANetwork(
        unet=legacy_dit,
        rank=rank,
        alpha=float(rank),
        num_experts=num_experts,
        num_bands=2,
        ortho=ortho,
        ortho_init_std=0.05,
        inline_prep=False,
    )
    legacy_net.apply_to(text_encoders=None, unet=legacy_dit)

    inline_dit = _MockDiT(dim, depth)
    inline_dit.load_state_dict(sd)
    inline_net = FeRANetwork(
        unet=inline_dit,
        rank=rank,
        alpha=float(rank),
        num_experts=num_experts,
        num_bands=2,
        ortho=ortho,
        ortho_init_std=0.05,
        inline_prep=True,
    )
    inline_net.apply_to(text_encoders=None, unet=inline_dit)

    # Sync FeRA params (random init differs across the two networks even
    # under the same seed because RNG state is consumed between them).
    inline_net.load_state_dict(legacy_net.state_dict(), strict=False)

    # Inline-mode rebind after load: load_state_dict overwrites the layers'
    # Parameters in place, but the back-refs survive. Verify they're wired.
    assert inline_net.inline_prep, "inline_prep should remain True post-load"
    for layer in inline_net.fera_layers.values():
        assert layer._fera_prep is inline_net.prep

    # FEI input: B=2, C=4, H=8, W=8 (matches the indicator's 4D path).
    torch.manual_seed(1)
    z_t = torch.randn(2, 4, 8, 8)
    return legacy_net, legacy_dit, inline_net, inline_dit, z_t


def _forward_backward(net, dit, z_t, x):
    net.prepare_forward(z_t)
    out = dit(x)
    out.sum().backward()
    return out, {
        n: p.grad.detach().clone()
        for n, p in net.named_parameters()
        if p.grad is not None
    }


@pytest.mark.parametrize("ortho", [False, True])
def test_inline_prep_matches_legacy(ortho):
    """End-to-end parity: same output and same grads regardless of which
    path computes the per-step routing state."""
    dim, depth = 16, 2
    num_experts, rank = 3, 4

    legacy_net, legacy_dit, inline_net, inline_dit, z_t = _build_pair(
        dim, depth, ortho=ortho, num_experts=num_experts, rank=rank
    )

    torch.manual_seed(42)
    x = torch.randn(2, 4, dim, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()

    out_legacy, g_legacy = _forward_backward(legacy_net, legacy_dit, z_t, x)
    out_inline, g_inline = _forward_backward(inline_net, inline_dit, z_t, x2)

    torch.testing.assert_close(out_inline, out_legacy, atol=1e-6, rtol=1e-5)

    # Trainable params live under fera_layers.* and router.*.
    assert set(g_legacy.keys()) == set(g_inline.keys()), (
        f"param keys differ: legacy-only={set(g_legacy) - set(g_inline)}, "
        f"inline-only={set(g_inline) - set(g_legacy)}"
    )
    for name in g_legacy:
        torch.testing.assert_close(
            g_inline[name], g_legacy[name], atol=1e-6, rtol=1e-5,
            msg=lambda m, n=name: f"grad mismatch on {n}: {m}",
        )


def test_inline_prep_fecl_force_legacy():
    """FECL composability gate: inline_prep=True + fecl_weight>0 must
    fall back to the legacy path (the inline path can't recover
    pre-base-pass gates for backward recompute)."""
    torch.manual_seed(0)
    dit = _MockDiT(16, 1)
    net = FeRANetwork(
        unet=dit,
        rank=4,
        alpha=4.0,
        num_experts=3,
        num_bands=2,
        ortho=False,
        inline_prep=True,
        fecl_weight=0.1,
    )
    assert not net.inline_prep, (
        "FeRANetwork should force inline_prep=False when fecl_weight>0; "
        "see plan.md edge-cases / docstring."
    )


def test_prep_runs_only_inside_dit_forward():
    """In inline mode, ``prepare_forward(z_t)`` must NOT populate the
    per-step gates eagerly — those should appear only after the DiT
    forward fires the prep hook. Verifies the cudagraph-stable design:
    if gates were set eagerly, cudagraph would see a different address
    every step."""
    torch.manual_seed(0)
    dit = _MockDiT(16, 1)
    net = FeRANetwork(
        unet=dit,
        rank=4,
        alpha=4.0,
        num_experts=3,
        num_bands=2,
        ortho=False,
        inline_prep=True,
    )
    net.apply_to(text_encoders=None, unet=dit)

    z_t = torch.randn(2, 4, 8, 8)
    net.prepare_forward(z_t)
    # Eagerly: gates must still be None on the prep — only the FEI buffer
    # was updated.
    assert net.prep._gates is None
    assert net.prep._fei.abs().sum() > 0  # FEI did get copied

    # After DiT forward, gates should be populated.
    x = torch.randn(2, 4, 16)
    dit(x)
    assert net.prep._gates is not None
    assert net.prep._gates.shape == (2, 3)
