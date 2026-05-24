"""Anima — programmatic front door.

A thin façade that re-exports the handful of real entry points an embedder
needs, so driving the pipeline is "read these exports" instead of
"reverse-engineer ``inference.py`` / ``train.py`` ``main()``"::

    import anima_lora

    settings = anima_lora.get_generation_settings(args)
    latent = anima_lora.generate(args, settings)
    image = anima_lora.decode_to_pil(vae, latent, device)

Each name resolves lazily (PEP 562) the first time it's accessed, so
``import anima_lora`` itself stays cheap and avoids the circular-import chains
the underlying packages guard against.

The canonical homes are unchanged — this module only re-exports them:

| export | canonical home |
|--------|----------------|
| ``generate`` / ``get_generation_settings`` / ``save_output`` / ``decode_to_pil`` / ``GenerationRequest`` / ``prepare_text_inputs`` / ``ensure_text_strategies`` | ``library.inference`` |
| ``load_method_preset`` / ``read_config_from_file`` | ``library.config.io`` |
| ``load_anima_model`` | ``library.anima.weights`` |
| ``load_dit_model`` | ``library.inference.models`` |
| ``load_vae`` | ``library.models.qwen_vae`` |

Note: model/config paths are still resolved relative to the current working
directory — run from the repo root (``anima_lora/``), same as the CLI.
"""

from __future__ import annotations

import importlib as _importlib

# export name -> dotted module that defines it
_ATTR_TO_MODULE: dict[str, str] = {
    # generation + output (library.inference)
    "generate": "library.inference",
    "get_generation_settings": "library.inference",
    "save_output": "library.inference",
    "decode_to_pil": "library.inference",
    "GenerationRequest": "library.inference",
    "prepare_text_inputs": "library.inference",
    "ensure_text_strategies": "library.inference",
    # config merge chain (library.config.io)
    "load_method_preset": "library.config.io",
    "read_config_from_file": "library.config.io",
    # model loaders
    "load_anima_model": "library.anima.weights",
    "load_dit_model": "library.inference.models",
    "load_vae": "library.models.qwen_vae",
}


def __getattr__(name: str):
    module = _ATTR_TO_MODULE.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(_importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = list(_ATTR_TO_MODULE.keys())
