"""Training entry-points for shipped methods (lora family + lora-gui + EasyControl).

Each ``cmd_*`` is a thin shim that translates env vars + extra argv into the
right ``train.py`` (via ``accelerate launch``) call. Experimental methods
(postfix, ip-adapter) live in ``scripts/experimental_tasks/training.py`` and are
wired up under ``make exp-*`` in ``tasks.py``.
"""

from __future__ import annotations

import os
import sys

from ._common import PY, ROOT, run, train

# EasyControl control-task projects under easycontrol_adapters/. Each maps to a
# configs/methods/<name>.toml that swaps the cond source / caption policy and an
# easycontrol_adapters/<name>/ project (mangafy/prep etc.). Selected at runtime
# via the EASYADAPTER env var (exported by the Makefile), e.g.
# ``make easycontrol EASYADAPTER=colorize``.
_EASYADAPTERS = {"colorize"}


def _easyadapter() -> str:
    """Resolve the EASYADAPTER env var (validated). "" → default easycontrol."""
    adapter = (os.environ.get("EASYADAPTER") or "").strip()
    if adapter and adapter not in _EASYADAPTERS:
        raise SystemExit(
            f"Unknown EASYADAPTER={adapter!r}. Known: {sorted(_EASYADAPTERS)}."
        )
    return adapter


def cmd_lora(extra):
    train("lora", extra)


def cmd_lora_gui(extra):
    """Train from configs/gui-methods/<variant>.toml.

    Variant is taken from GUI_PRESETS env var, falling back to the first
    positional extra arg (``python tasks.py lora-gui tlora ...``), then to
    ``lora`` (plain). Extra args after the variant are forwarded as usual.
    """
    variant = os.environ.get("GUI_PRESETS")
    if not variant and extra and not extra[0].startswith("-"):
        variant = extra[0]
        extra = extra[1:]
    variant = variant or "lora"

    expected = ROOT / "configs" / "gui-methods" / f"{variant}.toml"
    if not expected.exists():
        available = sorted(
            p.stem for p in (ROOT / "configs" / "gui-methods").glob("*.toml")
        )
        print(
            f"Unknown gui-methods variant: {variant!r}\n"
            f"Available: {', '.join(available)}",
            file=sys.stderr,
        )
        sys.exit(1)

    train(variant, extra, methods_subdir="gui-methods")


def cmd_easycontrol(extra):
    """EasyControl. ``EASYADAPTER=<name>`` selects a control-task project under
    easycontrol_adapters/ (e.g. ``colorize``) → runs configs/methods/<name>.toml;
    unset → the default ref==target easycontrol.toml."""
    train(_easyadapter() or "easycontrol", extra)


def cmd_easycontrol_download(extra):
    """Download an EasyControl control-task project's extra weights.

    ``EASYADAPTER=colorize`` fetches the Sketch2Manga screening weights
    (``models/sketch2manga/``) used by the learned Phase B condition synthesizer
    (``easycontrol_adapters/colorization/prep.py --engine sd``). The default
    EasyControl (no adapter) needs no extra weights beyond the Anima base.
    """
    from scripts.tasks import downloads as _downloads

    adapter = _easyadapter()
    if adapter == "colorize":
        _downloads.cmd_download_sketch2manga(extra)
        return
    print(
        "Default EasyControl needs no extra weights (uses the Anima base from "
        "`make download-models`). Set EASYADAPTER=colorize for the Sketch2Manga "
        "screening weights."
    )


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.

    ``EASYADAPTER=colorize`` instead builds the colorization *condition* cache
    (mangafy the existing color images → VAE-encode into
    ``post_image_dataset/easycontrol/colorize/cond/``); the color target latents + TE are
    reused from the LoRA cache, so no target re-encode is needed. See
    ``easycontrol_adapters/colorization/prep.py``.
    """
    adapter = _easyadapter()
    if adapter == "colorize":
        run([PY, "easycontrol_adapters/colorization/prep.py", *extra])
        return

    src = "easycontrol-dataset"
    dst = "post_image_dataset/easycontrol"
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--vae",
            "models/vae/qwen_image_vae.safetensors",
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
        ]
    )
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            src,
            "--cache_dir",
            dst,
            "--qwen3",
            "models/text_encoders/qwen_3_06b_base.safetensors",
            "--dit",
            "models/diffusion_models/anima-base-v1.0.safetensors",
            "--caption_shuffle_variants",
            "4",
            "--caption_tag_dropout_rate",
            "0.1",
        ]
    )
