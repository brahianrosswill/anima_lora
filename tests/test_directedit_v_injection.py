"""Smoke tests for DirectEdit V-injection (paper Eq. 13).

Doesn't exercise the real DiT — just verifies the patching mechanism, the
state-machine semantics, and the CFG-batch broadcasting in
``_VInjectionState``.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from library.inference.directedit import (
    _resolve_t_inj_blocks,
    _v_injection_scope,
    _VInjectionState,
)


class _FakeAttention(nn.Module):
    """Surface-compatible stand-in for ``library.anima.models.Attention``."""

    def __init__(self) -> None:
        super().__init__()
        self.output_proj = nn.Identity()
        self.output_dropout = nn.Identity()

    def compute_qkv(self, x, context, rope_cos_sin=None):  # noqa: ARG002
        return x, x, x

    def forward(self, x, attn_params, context, rope_cos_sin=None):  # noqa: ARG002
        return x


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeAnima(nn.Module):
    def __init__(self, n: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock() for _ in range(n)])


# ─── _resolve_t_inj_blocks ──────────────────────────────────────────────────


def test_resolve_default_skips_final_block():
    assert _resolve_t_inj_blocks(_FakeAnima(n=4), None) == {0, 1, 2}


def test_resolve_explicit_indices():
    assert _resolve_t_inj_blocks(_FakeAnima(n=4), [1, 3]) == {1, 3}


def test_resolve_rejects_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        _resolve_t_inj_blocks(_FakeAnima(n=4), [0, 4])


# ─── _VInjectionState ───────────────────────────────────────────────────────


def test_capture_then_inject_swaps_v():
    state = _VInjectionState({0})
    src_v = torch.tensor([1.0, 2.0, 3.0])
    tar_v = torch.tensor([10.0, 20.0, 30.0])

    state.mode = state.CAPTURE
    pass_through = state.hook(0, src_v)
    assert torch.equal(pass_through, src_v)
    assert torch.equal(state.cache[0], src_v)

    state.mode = state.INJECT
    injected = state.hook(0, tar_v)
    assert torch.equal(injected, src_v)


def test_block_outside_index_set_is_noop():
    state = _VInjectionState({0})
    state.mode = state.CAPTURE
    state.hook(99, torch.tensor([1.0]))
    assert 99 not in state.cache


def test_inject_broadcasts_across_cfg_batch_doubling():
    state = _VInjectionState({0})
    src_v = torch.randn(1, 16, 8, 64)  # [B=1, L, H, D] (cond src)
    state.mode = state.CAPTURE
    state.hook(0, src_v)

    state.mode = state.INJECT
    tar_v = torch.randn(2, 16, 8, 64)  # [B=2: CFG cond+uncond on tar]
    injected = state.hook(0, tar_v)

    assert injected.shape == tar_v.shape
    # Both batch rows should equal the single src row.
    assert torch.equal(injected[0], src_v[0])
    assert torch.equal(injected[1], src_v[0])


def test_inject_with_no_cache_passes_through():
    """Tar step at i < t_inj but src capture didn't touch this block: leave v alone."""
    state = _VInjectionState({0})
    state.mode = state.INJECT
    v = torch.tensor([1.0])
    out = state.hook(0, v)
    assert torch.equal(out, v)


def test_mode_none_is_pure_passthrough():
    state = _VInjectionState({0})
    state.mode = None
    v = torch.tensor([1.0])
    state.hook(0, v)
    assert 0 not in state.cache  # capture didn't fire


# ─── _v_injection_scope (patch + restore) ───────────────────────────────────


def test_scope_patches_only_selected_blocks():
    """Patching adds an instance-level `forward`; restore removes it."""
    m = _FakeAnima(n=4)

    # Pre-patch: every attn resolves `forward` via the class method.
    for i in range(4):
        assert "forward" not in m.blocks[i].self_attn.__dict__

    with _v_injection_scope(m, {0, 2}):
        assert "forward" in m.blocks[0].self_attn.__dict__
        assert "forward" not in m.blocks[1].self_attn.__dict__
        assert "forward" in m.blocks[2].self_attn.__dict__
        assert "forward" not in m.blocks[3].self_attn.__dict__

    # Instance attribute removed after scope — back to class-method resolution.
    for i in range(4):
        assert "forward" not in m.blocks[i].self_attn.__dict__


def test_scope_restores_on_exception():
    m = _FakeAnima(n=2)

    with pytest.raises(RuntimeError, match="boom"):
        with _v_injection_scope(m, {0, 1}):
            raise RuntimeError("boom")

    for i in range(2):
        assert "forward" not in m.blocks[i].self_attn.__dict__


def test_scope_clears_state_on_exit():
    m = _FakeAnima(n=2)
    with _v_injection_scope(m, {0}) as state:
        state.mode = state.CAPTURE
        state.hook(0, torch.tensor([1.0]))
        assert state.cache  # populated mid-scope

    assert not state.cache
    assert state.mode is None


def test_empty_block_set_is_inert():
    """t_inj=0 path: scope receives an empty set — nothing patched, nothing breaks."""
    m = _FakeAnima(n=2)
    with _v_injection_scope(m, set()):
        for i in range(2):
            assert "forward" not in m.blocks[i].self_attn.__dict__
