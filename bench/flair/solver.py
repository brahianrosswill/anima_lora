"""Shim — the FLAIR solver was promoted to the inference engine.

Phase 0/1 validated this port in the bench; it now lives at
``library/inference/corrections/flair.py`` so FLAIR-edit can ship it from
``generate()`` (``docs/proposal/flair_edit.md``). This module re-exports the
solver so ``bench/flair/{sanity_sr,calibrate_lambda}.py`` keep importing
``from bench.flair.solver import ...`` unchanged.
"""

from __future__ import annotations

from library.inference.corrections.flair import (  # noqa: F401
    DEFAULT_LAMBDA_PATH,
    FlairConfig,
    _hdc_project,
    _init_mu,
    _lambda_of_t,
    _velocity,
    flair_solve,
    load_lambda_table,
)
