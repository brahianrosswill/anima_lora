"""DirectEdit (Yang & Ye, arXiv:2605.02417v1) — flow-based image editing primitive.

Two-pass training-free editor for flow-matching DiTs:

1. **Inversion** (clean -> noise): step backward through the same Euler ODE
   the generator runs forward, querying v_θ at each step's input. Record
   per-step residuals ``ΔZ_i = Z_inv[i+1] − Z_inv[i]`` — these are the
   "anchor" the paper uses to make reconstruction bit-exact instead of
   trying to rectify the inversion path itself.

2. **Editing** (noise -> clean): standard generation loop, but every model
   call is queried at ``Z[i] + ΔZ[i]`` instead of ``Z[i]``. The cross-attn
   prompt is the edit target ψ_tar; the residual ΔZ pins the trajectory to
   the source. For ``t_inj > 0`` we also run a parallel src stream with
   ψ_src and inject its self-attn V into the tar stream for the first
   ``t_inj`` steps (paper Eq. 13). Mask blending (paper Eq. 12) is still
   v3 — left as a stub here.

Anima conventions used:
  * sigmas[0] = 1 (pure noise), sigmas[T] = 0 (clean), per
    ``library/inference/sampling.py::get_timesteps_sigmas``.
  * Latents: 5D ``[B, C, 1, H/8, W/8]`` (frame dim of 1 — image, not video).
  * The model's call signature matches what ``generate_body`` uses:
    ``anima(latents, t_expand, embed, padding_mask=...)`` where ``embed`` is
    already-preprocessed crossattn (post-T5, 512-padded).

This module is self-contained: ``invert`` and ``edit_forward`` accept the
already-loaded ``Anima`` model and pre-encoded ``ψ_src`` / ``ψ_tar`` embeds,
so the calling script (``scripts/edit.py``) can reuse the existing TE/VAE/DiT
loaders from ``library.inference.{models,text}``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Iterable, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

from library.anima import models as anima_models

logger = logging.getLogger(__name__)


def _padding_mask_for(latents: torch.Tensor) -> torch.Tensor:
    """Anima expects a (B, 1, H_lat, W_lat) zero mask for non-padded inputs."""
    bs = latents.shape[0]
    h_lat = latents.shape[-2]
    w_lat = latents.shape[-1]
    return torch.zeros(bs, 1, h_lat, w_lat, dtype=torch.bfloat16, device=latents.device)


@torch.no_grad()
def _v_pred(
    anima: anima_models.Anima,
    latents: torch.Tensor,
    sigma: torch.Tensor,
    embed: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    guidance_scale: float,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """One model forward (with optional CFG). Returns velocity prediction."""
    # Anima's t_embedder was trained on timesteps in [0, 1000] (= sigma * num_train_timesteps);
    # see library/runtime/noise.py and library/inference/sampling.py::get_timesteps_sigmas.
    # Passing raw sigma in [0, 1] gives near-constant AdaLN across the schedule, which keeps
    # the residual-anchored Euler step on the source manifold and kills edit leverage at t_inj=0.
    t_expand = (sigma * 1000.0).expand(latents.shape[0]).to(latents.device, dtype=torch.bfloat16)
    noise_pred = anima(latents, t_expand, embed, padding_mask=padding_mask)
    if guidance_scale != 1.0 and embed_neg is not None:
        uncond = anima(latents, t_expand, embed_neg, padding_mask=padding_mask)
        noise_pred = uncond + guidance_scale * (noise_pred - uncond)
    return noise_pred


# ─────────────────────────────────────────────────────────────────────────────
# V-injection (paper Eq. 13)
# ─────────────────────────────────────────────────────────────────────────────


class _VInjectionState:
    """Per-block self-attn V cache + mode flag.

    Lifecycle for one editing step with ``i < t_inj``:
      1. ``mode = "capture"``, run src forward → V tensors stashed per block.
      2. ``mode = "inject"``, run tar forward → cached V replaces the freshly
         computed V inside the patched ``Attention.forward``.

    The cache is overwritten each step (28 entries, no growth).
    """

    CAPTURE = "capture"
    INJECT = "inject"

    def __init__(self, block_indices: Set[int]) -> None:
        self.block_indices = block_indices
        self.cache: dict[int, torch.Tensor] = {}
        self.mode: Optional[str] = None

    def hook(self, block_idx: int, v: torch.Tensor) -> torch.Tensor:
        if self.mode is None or block_idx not in self.block_indices:
            return v
        if self.mode == self.CAPTURE:
            # Detach so dynamo can't tie this into a graph; clone is unnecessary
            # since the source forward is the only writer this step and we
            # consume the cache before the next overwrite.
            self.cache[block_idx] = v.detach()
            return v
        if self.mode == self.INJECT:
            cached = self.cache.get(block_idx)
            if cached is None:
                return v
            # Cached V is from src (cond, no CFG). Tar may run CFG, so the
            # batch dim of `v` could be 2× the cached one — broadcast on dim 0.
            if cached.shape[0] != v.shape[0] and v.shape[0] % cached.shape[0] == 0:
                cached = cached.repeat_interleave(v.shape[0] // cached.shape[0], dim=0)
            return cached.to(dtype=v.dtype, device=v.device)
        return v

    def clear(self) -> None:
        self.cache.clear()
        self.mode = None


def _make_patched_self_attn_forward(attn, block_idx: int, state: _VInjectionState):
    """Build a replacement ``Attention.forward`` that routes V through ``state.hook``.

    Two backends share this code path:
      * ``library/anima/models.py::Attention`` — used by the standalone CLI
        (``scripts/edit.py``). Forward signature is
        ``(x, attn_params, context, rope_cos_sin=None)`` and dispatches via
        ``attention_dispatch.dispatch_attention``.
      * ``comfy/comfy/ldm/cosmos/predict2.py::Attention`` — used inside
        ComfyUI (the bundled DiT impl). Forward signature is
        ``(x, context=None, rope_emb=None, transformer_options={})`` and
        dispatches via ``self.compute_attention``.

    We detect via ``compute_attention`` (comfy-only) and emit a patched
    function whose signature matches the actual call site. Patching with the
    wrong signature raises ``TypeError: ... unexpected keyword argument
    'rope_emb'`` on the first edit step.
    """
    compute_qkv = attn.compute_qkv

    # Comfy cosmos Attention path.
    if hasattr(attn, "compute_attention"):
        compute_attention = attn.compute_attention

        def patched_comfy(x, context=None, rope_emb=None, transformer_options=None, **_kwargs):
            q, k, v = compute_qkv(x, context, rope_emb=rope_emb)
            v = state.hook(block_idx, v)
            return compute_attention(
                q, k, v, transformer_options=transformer_options or {}
            )

        return patched_comfy

    # Library Anima Attention path.
    output_proj = attn.output_proj
    output_dropout = attn.output_dropout
    dispatcher = anima_models.attention_dispatch.dispatch_attention

    def patched_library(x, attn_params, context, rope_cos_sin=None):
        q, k, v = compute_qkv(x, context, rope_cos_sin=rope_cos_sin)
        v = state.hook(block_idx, v)
        if q.dtype != v.dtype:
            if (
                not attn_params.supports_fp32 or attn_params.requires_same_dtype
            ) and torch.is_autocast_enabled():
                target_dtype = v.dtype
                q = q.to(target_dtype)
                k = k.to(target_dtype)
        qkv = [q, k, v]
        del q, k, v
        result = dispatcher(qkv, attn_params=attn_params)
        return output_dropout(output_proj(result))

    return patched_library


@contextmanager
def _v_injection_scope(anima: anima_models.Anima, block_indices: Set[int]):
    """Monkey-patch ``self_attn.forward`` on selected blocks for the scope.

    Yields the shared ``_VInjectionState``; outer code toggles
    ``state.mode`` per src/tar pass.
    """
    state = _VInjectionState(block_indices)
    # Track which attns had a pre-existing instance-level `forward` so we can
    # restore by either reassigning or `del`ing — a plain reassign would leave
    # an instance attribute behind (closing a refcycle through the bound
    # method), which is functionally fine but leaks state across scopes.
    patched: list[tuple[torch.nn.Module, bool, object]] = []
    for idx, block in enumerate(anima.blocks):
        if idx not in block_indices:
            continue
        attn = block.self_attn
        had_instance_forward = "forward" in attn.__dict__
        prior = attn.__dict__.get("forward")
        patched.append((attn, had_instance_forward, prior))
        # Assigning to the instance attribute shadows the class method; nn.Module
        # __call__ resolves `self.forward` via normal attribute lookup, so this
        # works without descriptor binding.
        attn.forward = _make_patched_self_attn_forward(attn, idx, state)
    try:
        yield state
    finally:
        for attn, had_instance_forward, prior in patched:
            if had_instance_forward:
                attn.forward = prior
            else:
                # Remove the instance attribute so attribute lookup falls back
                # to the class method — exactly the pre-patch state.
                del attn.forward
        state.clear()


def _resolve_t_inj_blocks(
    anima: anima_models.Anima,
    t_inj_blocks: Optional[Iterable[int]],
) -> Set[int]:
    """Default to "all blocks except the final one" (SD3.5-style) when None."""
    n = len(anima.blocks)
    if t_inj_blocks is None:
        return set(range(n - 1))
    out = {int(i) for i in t_inj_blocks}
    if not out:
        return out
    if min(out) < 0 or max(out) >= n:
        raise ValueError(
            f"t_inj_blocks {sorted(out)} out of range for model with {n} blocks "
            f"(valid: 0..{n - 1})."
        )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inversion + edit forward
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def invert(
    anima: anima_models.Anima,
    z_clean: torch.Tensor,
    embed_src: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    sigmas: torch.Tensor,
    guidance_scale: float = 1.0,
    step_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Invert ``z_clean`` (= VAE-encoded source image) along the Anima ODE.

    Returns ``(z_inv, delta_z)``:
      * ``z_inv``: list of length T+1, where ``z_inv[T] == z_clean`` (cast/dtype
        match the input) and ``z_inv[0]`` is the maximally-noised inversion.
      * ``delta_z``: list of length T, with ``delta_z[i] = z_inv[i+1] − z_inv[i]``
        — the anchor residuals consumed by ``edit_forward``.

    Inversion convention (paper §3.2 in our index):
        ``z_inv[i] = z_inv[i+1] + (sigmas[i] − sigmas[i+1]) · v_θ(z_inv[i+1], σ=sigmas[i+1])``
    iterated for ``i = T-1 .. 0``.

    CFG during inversion is usually a wash (the source has no negative concept
    to push away from). Default ``guidance_scale=1.0`` skips it. Pass >1.0
    only if you want the inverted noise to land where re-generation with the
    same CFG would put it.
    """
    device = z_clean.device
    T = sigmas.shape[0] - 1
    padding_mask = _padding_mask_for(z_clean)

    z_inv: List[torch.Tensor] = [None] * (T + 1)  # type: ignore[list-item]
    z_inv[T] = z_clean.to(torch.bfloat16)

    delta_z: List[torch.Tensor] = [None] * T  # type: ignore[list-item]

    iterator = tqdm(range(T - 1, -1, -1), desc="DirectEdit inversion", total=T)
    for step_idx, i in enumerate(iterator, start=1):
        sigma_in = sigmas[i + 1].to(device)  # σ of the input we feed v_θ
        v = _v_pred(
            anima,
            z_inv[i + 1],
            sigma_in,
            embed_src,
            embed_neg,
            guidance_scale,
            padding_mask,
        )
        # z_inv[i] is at higher noise; (sigmas[i] - sigmas[i+1]) > 0 in our index.
        coeff = (sigmas[i] - sigmas[i + 1]).to(device, dtype=torch.float32)
        z_inv[i] = (z_inv[i + 1].float() + coeff * v.float()).to(torch.bfloat16)
        delta_z[i] = (z_inv[i + 1].float() - z_inv[i].float()).to(torch.bfloat16)
        if step_callback is not None:
            step_callback(step_idx, T)

    return z_inv, delta_z


@torch.no_grad()
def edit_forward(
    anima: anima_models.Anima,
    z_init: torch.Tensor,
    delta_z: List[torch.Tensor],
    embed_tar: torch.Tensor,
    embed_neg: Optional[torch.Tensor],
    sigmas: torch.Tensor,
    guidance_scale: float = 4.0,
    embed_src: Optional[torch.Tensor] = None,
    t_inj: int = 0,
    t_inj_blocks: Optional[Iterable[int]] = None,
    mask: Optional[torch.Tensor] = None,  # noqa: ARG001 — Eq. 12 mask blend (v3)
    step_callback: Optional[Callable[[int, int], None]] = None,
) -> torch.Tensor:
    """Forward (noise -> clean) edit pass anchored to the inversion residuals.

    Step rule (paper §3.2 in our index):
        ``ẑ_i = z[i] + delta_z[i]                        # anchor``
        ``v_i = v_θ(ẑ_i, σ=sigmas[i], ψ_tar)             # query at anchored pt``
        ``z[i+1] = z[i] − (sigmas[i] − sigmas[i+1]) · v_i # standard Euler step``

    For ``t_inj > 0`` (paper Eq. 13), the first ``t_inj`` steps additionally:
      1. Evolve a parallel src branch ``Z_t^src`` (init = ``z_init``) with
         ``v_θ(Ẑ_t^src, ψ_src)``, capturing each block's self-attn V.
      2. Replace V in the tar self-attn with the cached src V before the
         dispatcher (``F̂_t^tar = Attention(Q_t^tar, K_t^tar, V_t^src)``).

    Args:
      z_init: should be ``z_inv[0]`` from ``invert(...)`` for the residual
        trick to fire correctly.
      embed_src: required when ``t_inj > 0`` (drives the src capture forward).
      t_inj: number of early steps to inject src V into the tar self-attn.
        ``0`` reproduces the v1 paper-baseline behavior (pure ΔZ-anchored
        edit, no parallel src stream).
      t_inj_blocks: which block indices to inject at. Default ``None`` →
        all blocks except the final one (SD3.5-style default; Anima is
        single-stream cross-attn DiT, so this is the conservative analog).
      mask: paper Eq. 12 background-lock blend — still v3, ignored here.

    Notes on src CFG: the src capture pass is always run at CFG=1 (no
    embed_neg); paper Algorithm 1 doesn't apply CFG to the src branch, and
    capturing V from a CFG-mixed branch would conflate ψ_src and the
    negative concept.
    """
    device = z_init.device
    T = sigmas.shape[0] - 1
    if len(delta_z) != T:
        raise ValueError(
            f"delta_z has length {len(delta_z)} but sigmas implies T={T} steps "
            "— inversion and editing must use the same sigma schedule."
        )
    if t_inj < 0:
        raise ValueError(f"t_inj must be >= 0, got {t_inj}.")
    if t_inj > T:
        logger.warning(
            "t_inj=%d clamped to T=%d (full-trajectory injection).", t_inj, T
        )
        t_inj = T
    if t_inj > 0 and embed_src is None:
        raise ValueError("V-injection (t_inj > 0) requires embed_src (ψ_src).")
    if mask is not None:
        logger.warning(
            "mask= ignored: per-step background-lock blending (paper Eq. 12) "
            "is v3 work; only V-injection is wired."
        )

    padding_mask = _padding_mask_for(z_init)
    block_indices = _resolve_t_inj_blocks(anima, t_inj_blocks) if t_inj > 0 else set()

    z_tar = z_init.to(torch.bfloat16)
    z_src = z_init.to(torch.bfloat16) if t_inj > 0 else None

    if t_inj > 0:
        logger.info(
            "DirectEdit V-injection: t_inj=%d / T=%d, injecting at %d / %d blocks",
            t_inj, T, len(block_indices), len(anima.blocks),
        )

    with _v_injection_scope(anima, block_indices) as state:
        iterator = tqdm(range(T), desc="DirectEdit editing", total=T)
        for i in iterator:
            d = delta_z[i].to(device).float()
            z_hat_tar = (z_tar.float() + d).to(torch.bfloat16)
            sigma_in = sigmas[i].to(device)
            coeff = (sigmas[i] - sigmas[i + 1]).to(device, dtype=torch.float32)

            if i < t_inj:
                # ── 1. Capture src V at this step (CFG=1, no neg).
                z_hat_src = (z_src.float() + d).to(torch.bfloat16)
                state.mode = state.CAPTURE
                v_src = _v_pred(
                    anima, z_hat_src, sigma_in,
                    embed_src, None, 1.0, padding_mask,
                )
                # Evolve src branch (Algorithm 1 line: Z_{t+1}^src ← ...).
                z_src = (z_src.float() - coeff * v_src.float()).to(torch.bfloat16)

                # ── 2. Inject into tar self-attn for both CFG branches.
                state.mode = state.INJECT
                v_tar = _v_pred(
                    anima, z_hat_tar, sigma_in,
                    embed_tar, embed_neg, guidance_scale, padding_mask,
                )
                state.mode = None
            else:
                # Past t_inj: pure tar forward, no parallel src.
                v_tar = _v_pred(
                    anima, z_hat_tar, sigma_in,
                    embed_tar, embed_neg, guidance_scale, padding_mask,
                )

            z_tar = (z_tar.float() - coeff * v_tar.float()).to(torch.bfloat16)
            if step_callback is not None:
                step_callback(i + 1, T)

    return z_tar
