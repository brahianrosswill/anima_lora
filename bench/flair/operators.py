"""Shim — FLAIR forward operators were promoted to the inference engine.

Now live at ``library/inference/corrections/flair_operators.py`` (the editing
application needs the ``inpaint`` operator at inference time). Re-exported here
so ``bench/flair/sanity_sr.py`` keeps importing ``from bench.flair.operators
import build_operator`` unchanged.
"""

from __future__ import annotations

from library.inference.corrections.flair_operators import (  # noqa: F401
    InpaintOperator,
    SROperator,
    build_operator,
)
