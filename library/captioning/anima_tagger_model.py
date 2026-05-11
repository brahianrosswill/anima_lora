"""Anima tagger head — multi-label tags + 3-class rating + 8-class people-count, off frozen PE-Core.

Architecture:

::

    feature [d_in]
        ↓ LayerNorm + Linear(d_in, d_hidden) + GELU + Dropout
    trunk_h [d_hidden]
        ├─→ Linear(d_hidden, n_tags)          → tag_logits
        ├─→ Linear(d_hidden, n_ratings)       → rating_logits
        └─→ Linear(d_hidden, n_people_counts) → people_logits  (omitted when n_people_counts == 0)

The trunk is shared between heads so the auxiliary signals (rating /
people-count) nudge the same representation that's predicting tags.
``n_tags``/``n_ratings``/``n_people_counts``/``d_in`` all come from
``vocab.json`` + the cached PE feature shape.

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
    d_in: int
    n_tags: int
    n_ratings: int = 3
    # 0 = no people head (legacy checkpoint). Trainer always sets this from
    # the manifest (currently len(PEOPLE_COUNT_LABELS) == 8) when in use.
    n_people_counts: int = 0
    d_hidden: int = 1024
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {
            "d_in": self.d_in,
            "n_tags": self.n_tags,
            "n_ratings": self.n_ratings,
            "n_people_counts": self.n_people_counts,
            "d_hidden": self.d_hidden,
            "dropout": self.dropout,
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
        )


class AnimaTaggerHead(nn.Module):
    def __init__(self, cfg: AnimaTaggerConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.LayerNorm(cfg.d_in),
            nn.Linear(cfg.d_in, cfg.d_hidden),
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

    def forward(
        self, feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = self.trunk(feat)
        people_logits = self.people_head(h) if self.people_head is not None else None
        return self.tag_head(h), self.rating_head(h), people_logits
