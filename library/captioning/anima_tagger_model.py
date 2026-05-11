"""Anima tagger head — multi-label tags + 3-class rating + 8-class people-count, off frozen PE-Core.

Architecture (``pool_kind == "map"``, default for new runs):

::

    tokens [T, d_in]                                # PE patch tokens, CLS at [0]
        ├─ MAPHead(K queries, H heads)  → [K, d_in]
        ├─ CLS  = tokens[:, 0]          → [1, d_in]
        └─ mean = tokens.mean(dim=1)    → [1, d_in]
              concat → [(K+use_cls+use_mean) * d_in]
        ↓ LayerNorm + Linear(d_pool, d_hidden) + GELU + Dropout
    trunk_h [d_hidden]
        ├─→ Linear(d_hidden, n_tags)          → tag_logits
        ├─→ Linear(d_hidden, n_ratings)       → rating_logits
        └─→ Linear(d_hidden, n_people_counts) → people_logits  (omitted when n_people_counts == 0)

Legacy ``pool_kind == "mean"`` path: ``forward(feat[B, d_in])`` skips the
pool and feeds the pre-pooled feature directly into the trunk. Used by
checkpoints trained before the MAP pool landed (their ``config.json`` lacks
``pool_kind`` and so resolves to ``"mean"`` via :meth:`AnimaTaggerConfig.from_dict`).

Forward dispatches on input rank: rank-2 → legacy mean path; rank-3 → MAP
path. Either path errors if the config and input shape disagree.

The trunk is shared between heads so the auxiliary signals (rating /
people-count) nudge the same representation that's predicting tags.
``n_tags``/``n_ratings``/``n_people_counts``/``d_in`` all come from
``vocab.json`` + the cached PE token dimension (always ``d_enc``, not the
pooled feature dim — pool-output dim is derived from ``d_in`` × the active
pool channels).

Inference receives all heads in one forward; training computes per-head
losses and combines with ``λ_rating`` / ``λ_people``. ``n_people_counts=0``
in the config means "no people head was trained" — used to load legacy
checkpoints; ``forward`` returns ``None`` in that slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AnimaTaggerConfig:
    d_in: int                        # PE token dim (d_enc), not the pool-output dim.
    n_tags: int
    n_ratings: int = 3
    # 0 = no people head (legacy checkpoint). Trainer always sets this from
    # the manifest (currently len(PEOPLE_COUNT_LABELS) == 8) when in use.
    n_people_counts: int = 0
    d_hidden: int = 1024
    dropout: float = 0.1
    # Pool config. ``"mean"`` (legacy) consumes a pre-pooled [B, d_in]
    # feature; ``"map"`` consumes a [B, T, d_in] token sequence and runs
    # MAPHead + (optional) CLS + (optional) mean inside the head. Default
    # is "mean" so legacy config.json files load unchanged.
    pool_kind: str = "mean"
    pool_n_queries: int = 4
    pool_n_heads: int = 8
    pool_use_cls: bool = True
    pool_use_mean: bool = True

    @property
    def trunk_in_dim(self) -> int:
        """Width of the trunk's first Linear — depends on pool channels."""
        if self.pool_kind == "mean":
            return self.d_in
        if self.pool_kind == "map":
            n_chan = self.pool_n_queries + int(self.pool_use_cls) + int(self.pool_use_mean)
            return self.d_in * n_chan
        raise ValueError(f"unknown pool_kind={self.pool_kind!r}")

    def to_dict(self) -> dict:
        return {
            "d_in": self.d_in,
            "n_tags": self.n_tags,
            "n_ratings": self.n_ratings,
            "n_people_counts": self.n_people_counts,
            "d_hidden": self.d_hidden,
            "dropout": self.dropout,
            "pool_kind": self.pool_kind,
            "pool_n_queries": self.pool_n_queries,
            "pool_n_heads": self.pool_n_heads,
            "pool_use_cls": self.pool_use_cls,
            "pool_use_mean": self.pool_use_mean,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnimaTaggerConfig":
        return cls(
            d_in=int(d["d_in"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d.get("n_ratings", 3)),
            n_people_counts=int(d.get("n_people_counts", 0)),
            d_hidden=int(d.get("d_hidden", 1024)),
            dropout=float(d.get("dropout", 0.1)),
            pool_kind=str(d.get("pool_kind", "mean")),
            pool_n_queries=int(d.get("pool_n_queries", 4)),
            pool_n_heads=int(d.get("pool_n_heads", 8)),
            pool_use_cls=bool(d.get("pool_use_cls", True)),
            pool_use_mean=bool(d.get("pool_use_mean", True)),
        )


class MAPHead(nn.Module):
    """Multi-query attention pool — K learnable queries attend over the token grid.

    Shape: ``[B, T, D] → [B, K, D]``. Pre-norm on K/V (the queries are
    learnable parameters and don't need it). Uses :class:`nn.MultiheadAttention`
    with ``batch_first=True``; PyTorch routes through SDPA so this is a
    single fused kernel on CUDA.

    Initialization: queries drawn from N(0, 1/√D) so the dot-product scale
    matches the post-LayerNorm key/value scale and the initial attention
    map is roughly uniform (no early collapse onto a single token).
    """

    def __init__(self, d: int, n_queries: int = 4, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"MAPHead: d={d} must be divisible by n_heads={n_heads}")
        self.q = nn.Parameter(torch.randn(n_queries, d) * (d ** -0.5))
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        B = tokens.shape[0]
        q = self.q.unsqueeze(0).expand(B, -1, -1)        # [B, K, D]
        kv = self.norm_kv(tokens)                        # [B, T, D]
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out                                       # [B, K, D]


class AnimaTaggerHead(nn.Module):
    def __init__(self, cfg: AnimaTaggerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.pool_kind == "mean":
            self.pool: Optional[MAPHead] = None
        elif cfg.pool_kind == "map":
            self.pool = MAPHead(
                d=cfg.d_in,
                n_queries=cfg.pool_n_queries,
                n_heads=cfg.pool_n_heads,
                dropout=0.0,
            )
        else:
            raise ValueError(f"unknown pool_kind={cfg.pool_kind!r}")

        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.trunk_in_dim),
            nn.Linear(cfg.trunk_in_dim, cfg.d_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.tag_head = nn.Linear(cfg.d_hidden, cfg.n_tags)
        self.rating_head = nn.Linear(cfg.d_hidden, cfg.n_ratings)
        # Optional — older checkpoints have n_people_counts=0 and no people
        # head in the state_dict. Keeping the attribute as None lets `forward`
        # return a stable 3-tuple shape in both cases.
        self.people_head: Optional[nn.Linear] = (
            nn.Linear(cfg.d_hidden, cfg.n_people_counts)
            if cfg.n_people_counts > 0 else None
        )

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, trunk_in_dim] via MAP + (optional) CLS / mean concat."""
        assert self.pool is not None, "MAP pool path called without configured pool"
        cfg = self.cfg
        chans = [self.pool(tokens).flatten(1)]                  # [B, K*D]
        if cfg.pool_use_cls:
            chans.append(tokens[:, 0])                          # [B, D]
        if cfg.pool_use_mean:
            chans.append(tokens.mean(dim=1))                    # [B, D]
        return torch.cat(chans, dim=-1)

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # Dispatch on input rank: [B, D] → legacy mean path; [B, T, D] → MAP.
        if feat.dim() == 2:
            if self.cfg.pool_kind != "mean":
                raise ValueError(
                    f"got pre-pooled feature [B, D] but pool_kind={self.cfg.pool_kind!r} "
                    f"expects [B, T, D] tokens"
                )
            x = feat
        elif feat.dim() == 3:
            if self.cfg.pool_kind != "map":
                raise ValueError(
                    f"got token sequence [B, T, D] but pool_kind={self.cfg.pool_kind!r} "
                    f"expects pre-pooled [B, D]"
                )
            x = self._pool_tokens(feat)
        else:
            raise ValueError(f"AnimaTaggerHead.forward: unexpected feat.dim()={feat.dim()}")
        h = self.trunk(x)
        people_logits = self.people_head(h) if self.people_head is not None else None
        return self.tag_head(h), self.rating_head(h), people_logits
