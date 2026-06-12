"""REPA v2 — patchwise / relational alignment against a vision encoder.

Revives the archived v1 REPA (Yu et al., arXiv:2410.06940; ``_archive/repa/``)
at the granularity its own docstring pre-planned, with two loss forms selectable
per run for the Phase-0 three-arm A/B (see
``docs/experimental/repa.md``):

- **Arm A — absolute patchwise**: project DiT block tokens → encoder dim with a
  small MLP head, cosine per token in fp32, mean. Paper-faithful REPA at the
  right grid (no global pooling).
- **Arm B — relational (Gram)**: L2-normalize tokens on each side independently,
  ``G = F̂F̂ᵀ``, ``loss = MSE(G_dit, G_pe)``. No head — similarities live within
  each space so dimensions never need to match, and any global domain direction
  cancels out of the pairwise structure (relational KD, Park et al. 2019).

Both arms align a noisy-input mid-block feature to the *clean*-image encoder
target at every sampled σ, captured from the **primary** forward (no second DiT
forward → no block-swap offloader desync).

Granularity / capture, the two non-obvious mechanics:

1. **Grid match.** The encoder tokens live on a ``(gh, gw)`` patch grid that
   varies per aspect bucket (PE-Spatial: 32×32 at square, ~46×23 at 2:1). The
   DiT patch grid is ``(H_lat//patch, W_lat//patch)``. We adaptive-avg-pool the
   DiT side down to the encoder grid so both sides are ``N = gh*gw`` tokens in
   the same row-major order. ``(gh, gw)`` is recovered from the *encoder feature's
   own token count* (ground truth, no metadata threading) disambiguated by the
   latent's orientation (portrait vs landscape) — token count alone is ambiguous
   because the bucket table is aspect-symmetric.

2. **native_flatten layout.** Under ``compile_blocks()`` the captured block
   output is the fake-5D ``(B, 1, seq, 1, D)`` shape, not the eager
   ``(B, 1, H, W, D)``. The hook fires regardless (it sits on ``block.__call__``,
   outside the compiled ``block._forward``), and we reshape ``(B, …, D) →
   (B, N_dit, D)`` from the latent's patch grid, which is layout-agnostic: both
   layouts flatten to the same row-major ``(B, N_dit, D)``.

Config rides the LoRA network kwargs (``use_repa`` / ``repa_mode`` /
``repa_weight`` / ``repa_layer`` / ``repa_encoder``), parsed by the factory and
stashed on the network; the adapter reads them off ``ctx.network`` so no new
``args`` plumbing is needed. The scalar alignment loss is returned under
``aux["repa"]`` and weighted by ``LossComposer`` stage 2 (``losses._repa_loss``).

Phase-1 operating-point levers (``docs/proposal/repa_phase1_operating_point.md``,
both default-off):

- ``repa_anneal_steps`` — hard cutoff (HASTE, arXiv:2505.16792: alignment helps
  early, degrades late). Value in (0, 1] = fraction of ``max_train_steps``;
  value > 1 = absolute optimizer steps. The adapter keeps its own train
  micro-batch counter and converts via ``gradient_accumulation_steps`` (the
  ``step_contrastive_warmup`` pattern) — past the cutoff the term is skipped
  entirely (no PE transfer, no Gram).
- ``repa_spatial_norm`` — iREPA-style (arXiv:2512.10794) spatial
  standardization of the *target* tokens, ``(pe − mean_tok) / (std_tok + ε)``
  before per-token L2-norm + Gram. Cancels the shared global component that
  compresses pairwise cosines. Relational mode only.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from library.training.method_adapter import (
    ForwardArtifacts,
    MethodAdapter,
    SetupCtx,
    StepCtx,
)
from library.vision.buckets import get_bucket_spec

logger = logging.getLogger(__name__)


class REPAHead(nn.Module):
    """3-layer MLP projecting DiT hidden dim → vision-encoder feature dim.

    Matches ``h_phi`` from the REPA paper (3 linear + SiLU). Last layer init
    near-zero so step-0 cosine is unbiased and the head learns the projection
    from a small-norm output. Only used by Arm A (absolute); Arm B has no head.
    """

    def __init__(self, dit_dim: int, hidden_dim: int, encoder_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dit_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encoder_dim)
        nn.init.normal_(self.fc3.weight, std=1e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        return self.fc3(x)


class REPAMethodAdapter(MethodAdapter):
    """Bridges REPA into AnimaTrainer's adapter dispatch.

    Setup: install a forward post-hook on ``unet.blocks[repa_layer]`` and read
    the run's REPA config off the network.
    Prime: stash this step's cached PE-Spatial features from the batch.
    Extra forward: pool DiT tokens to the encoder grid, compute the absolute /
    relational alignment scalar, return it under ``aux["repa"]``.
    """

    name = "repa"

    def __init__(self) -> None:
        self._captured: Optional[torch.Tensor] = None
        self._pe_features: Optional[torch.Tensor] = None
        self._latent_hw: Optional[tuple[int, int]] = None
        self._hook_handle = None
        # Filled in on_network_built.
        self._mode = "relational"
        self._layer = 8
        self._patch = 2
        self._spec = None
        self._anneal_steps = 0.0
        self._spatial_norm = False
        # Optimizer-step clock: train micro-batches seen, converted with
        # gradient_accumulation_steps at the anneal gate.
        self._train_micro_steps = 0

    # ------------------------------------------------------------------ setup
    def on_network_built(self, ctx: SetupCtx) -> None:
        net = ctx.network
        self._mode = str(getattr(net, "_repa_mode", "relational")).lower()
        self._layer = int(getattr(net, "_repa_layer", 8))
        self._anneal_steps = float(getattr(net, "_repa_anneal_steps", 0.0) or 0.0)
        self._spatial_norm = bool(getattr(net, "_repa_spatial_norm", False))
        encoder = str(getattr(net, "_repa_encoder", "pe_spatial"))
        self._spec = get_bucket_spec(encoder)
        self._patch = int(ctx.unet.patch_spatial)

        if self._mode == "absolute" and getattr(net, "repa_head", None) is None:
            raise ValueError(
                "repa_mode='absolute' requires the network to expose a "
                "'repa_head' submodule (attached by networks.lora_anima.factory "
                "when use_repa=true and repa_mode=absolute)."
            )

        blocks = ctx.unet.blocks
        if not (0 <= self._layer < len(blocks)):
            raise ValueError(
                f"repa_layer={self._layer} out of range (DiT has {len(blocks)} blocks)"
            )

        def _hook(_module, _inputs, output: torch.Tensor) -> None:
            # Block output: eager (B,1,H,W,D) or native_flatten (B,1,seq,1,D).
            # Keep grad — we want it to flow back into LoRA modules in blocks
            # <= repa_layer. The hook runs in block.__call__, outside the
            # compiled block._forward, so it fires under compile_blocks().
            self._captured = output

        self._hook_handle = blocks[self._layer].register_forward_hook(_hook)

        weight = float(getattr(net, "_repa_weight", 0.0))
        if self._mode == "absolute":
            head = net.repa_head
            head_desc = (
                f"; head {head.fc1.in_features}→{head.fc2.out_features}"
                f"→{head.fc3.out_features}"
            )
        else:
            head_desc = "; no head (Gram)"
        anneal_desc = (
            f", anneal={self._anneal_steps:g}"
            f"{' (fraction)' if 0 < self._anneal_steps <= 1.0 else ' steps' if self._anneal_steps > 1 else ''}"
            if self._anneal_steps > 0
            else ""
        )
        ctx.accelerator.print(
            f"REPA[{self._mode}]: hook on block {self._layer}/{len(blocks)}, "
            f"encoder={encoder} grid≤{self._spec.t_max_patches}tok, "
            f"weight={weight}{anneal_desc}"
            f"{', spatial_norm' if self._spatial_norm else ''}{head_desc}"
        )

    # ------------------------------------------------------------------- step
    def prime_for_forward(
        self, ctx: StepCtx, batch, latents: torch.Tensor, *, is_train: bool
    ) -> None:
        # Drop any stash from a step that didn't trigger the hook.
        self._captured = None
        self._pe_features = None
        self._latent_hw = None
        if not is_train:
            return
        # Advance the optimizer-step clock on every train micro-batch (even
        # ones missing PE features) so it stays in lockstep with the trainer's
        # global_step; validation passes don't tick it.
        micro_step = self._train_micro_steps
        self._train_micro_steps += 1
        if self._past_anneal_cutoff(ctx.args, micro_step):
            return
        feats = batch.get("repa_pe_features") if isinstance(batch, dict) else None
        if feats is None:
            # Batch lacks PE features (some sample missing its sidecar, or the
            # loader was off this step) — skip the term rather than crash.
            return
        self._pe_features = feats.to(ctx.accelerator.device, dtype=torch.float32)
        self._latent_hw = (int(latents.shape[-2]), int(latents.shape[-1]))

    def _past_anneal_cutoff(self, args, micro_step: int) -> bool:
        """Hard anneal cutoff (lever 1): True once the optimizer-step clock
        passes ``repa_anneal_steps`` — (0, 1] is a fraction of
        ``max_train_steps``, > 1 is absolute optimizer steps. 0 = off."""
        if self._anneal_steps <= 0:
            return False
        cutoff = self._anneal_steps
        if cutoff <= 1.0:
            max_steps = int(getattr(args, "max_train_steps", 0) or 0)
            if max_steps <= 0:
                if not getattr(self, "_warned_no_max_steps", False):
                    logger.warning(
                        "REPA: repa_anneal_steps=%g is a fraction but "
                        "max_train_steps is unset — anneal disabled.",
                        self._anneal_steps,
                    )
                    self._warned_no_max_steps = True
                return False
            cutoff = cutoff * max_steps
        accum = int(getattr(args, "gradient_accumulation_steps", 1) or 1)
        if micro_step // accum < cutoff:
            return False
        if not getattr(self, "_anneal_cutoff_logged", False):
            logger.info(
                "REPA: anneal cutoff reached at optimizer step %d "
                "(repa_anneal_steps=%g) — alignment loss off for the rest of the run.",
                micro_step // accum,
                self._anneal_steps,
            )
            self._anneal_cutoff_logged = True
        return True

    def _pe_grid(self, n_pe: int, h_lat: int, w_lat: int) -> tuple[int, int]:
        """Resolve the encoder ``(gh, gw)`` patch grid for ``n_pe`` patch tokens.

        Token count alone is ambiguous (the bucket table is aspect-symmetric:
        46×23 and 23×46 both have 1058 patches), so disambiguate by the latent's
        orientation. Returns the unique bucket whose patch product == ``n_pe``.
        """
        cands = [(h, w) for (h, w) in self._spec.buckets if h * w == n_pe]
        if not cands:
            raise RuntimeError(
                f"REPA: no {self._spec.encoder} bucket has {n_pe} patches "
                f"(buckets={self._spec.buckets}). Cache encoder/grid mismatch?"
            )
        if len(cands) > 1:
            portrait = h_lat >= w_lat
            cands = [(h, w) for (h, w) in cands if (h >= w) == portrait] or cands
        return cands[0]

    def extra_forwards(self, ctx: StepCtx, primary: ForwardArtifacts) -> Optional[dict]:
        if not primary.is_train or self._pe_features is None or self._latent_hw is None:
            return None
        if self._captured is None:
            # PE features loaded + train step, but the block hook never fired —
            # REPA would be silently inert. Warn once so a mis-wired hook (e.g.
            # a future compile change that swallows it) is visible, not silent.
            if not getattr(self, "_warned_no_capture", False):
                logger.warning(
                    "REPA: block %d hook did not fire on a train step — alignment "
                    "loss is inert. Check the forward hook under compile_blocks().",
                    self._layer,
                )
                self._warned_no_capture = True
            return None

        cap = self._captured
        b = cap.shape[0]
        d_dit = cap.shape[-1]
        # Layout-agnostic flatten → (B, N_dit, D) row-major.
        tokens = cap.reshape(b, -1, d_dit)

        h_lat, w_lat = self._latent_hw
        h_dit, w_dit = h_lat // self._patch, w_lat // self._patch
        if tokens.shape[1] != h_dit * w_dit:
            raise RuntimeError(
                f"REPA: captured {tokens.shape[1]} DiT tokens but latent grid is "
                f"{h_dit}x{w_dit}={h_dit * w_dit} (patch={self._patch}). "
                "Block/patch-grid mismatch."
            )

        pe = self._pe_features  # (B, T_pe, d_enc) fp32
        n_pe = pe.shape[1] - (1 if self._spec.use_cls else 0)
        gh, gw = self._pe_grid(n_pe, h_lat, w_lat)
        if self._spec.use_cls:
            pe = pe[:, 1:, :]  # drop CLS → (B, gh*gw, d_enc)

        # Pool DiT grid down to the encoder grid: (B,D,h,w) → (B,D,gh,gw).
        dit_grid = tokens.reshape(b, h_dit, w_dit, d_dit).permute(0, 3, 1, 2)
        dit_pooled = F.adaptive_avg_pool2d(dit_grid.float(), (gh, gw))
        dit_tok = dit_pooled.flatten(2).transpose(1, 2)  # (B, gh*gw, D) fp32

        if self._mode == "absolute":
            head = ctx.network.repa_head
            head_dtype = next(head.parameters()).dtype
            proj = head(dit_tok.to(head_dtype)).float()  # (B, N, d_enc)
            cos = F.cosine_similarity(proj, pe, dim=-1)  # (B, N)
            loss = (1.0 - cos).mean()
        else:
            # Relational (Gram): per-token L2-norm within each space, then match
            # the N×N affinity structure. Dimensions never need to align.
            if self._spatial_norm:
                # Lever 2 (iREPA): standardize the target across the token axis
                # before per-token L2-norm. Pretrained patch tokens share a
                # large global component that the Gram cancels only imperfectly
                # (it still sits inside every token's normalization); removing
                # it sharpens the target affinity contrast.
                pe = (pe - pe.mean(dim=1, keepdim=True)) / (
                    pe.std(dim=1, keepdim=True) + 1e-6
                )
            dit_hat = F.normalize(dit_tok, dim=-1)
            pe_hat = F.normalize(pe, dim=-1)
            g_dit = torch.bmm(dit_hat, dit_hat.transpose(1, 2))  # (B, N, N)
            g_pe = torch.bmm(pe_hat, pe_hat.transpose(1, 2))
            loss = F.mse_loss(g_dit, g_pe)

        return {"repa": loss}
