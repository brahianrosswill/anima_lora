"""Training entry-points for shipped methods (lora family + lora-gui + EasyControl).

Each ``cmd_*`` is a thin shim that translates env vars + extra argv into the
right ``train.py`` (via ``accelerate launch``) call. Experimental methods
(postfix, ip-adapter) live in ``scripts/experimental_tasks/training.py`` and are
wired up under ``make exp-*`` in ``tasks.py``.
"""

from __future__ import annotations

import os
import sys
import tomllib

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


def _near_twin_preprocess() -> None:
    """Resize + VAE/TE caching for the mined near-twin pair tree.

    Every knob is read from the ``[preprocess]`` table of
    ``configs/easycontrol/near_twins.toml`` (written by the staging step), so this
    stays in lockstep with the dataset blueprint that step also rewrites.

    The mined ``staging/`` tree holds native-resolution images (symlinks to the
    corpus), so the pass first resizes them into constant-token buckets under
    ``resized/`` (``target_res`` tiers) — that resized tree is the training
    ``image_dir`` — and only then VAE/TE-encodes it into ``cache/``.
    """
    cfg_path = ROOT / "configs" / "easycontrol" / "near_twins.toml"
    if not cfg_path.is_file():
        raise SystemExit(
            f"{cfg_path} not found — run `make easycontrol-staging "
            "EASYADAPTER=near_twin` first to mine the pair tree."
        )
    pp = tomllib.loads(cfg_path.read_text(encoding="utf-8")).get("preprocess") or {}
    base = "post_image_dataset/easycontrol/near_twins"
    staging = pp.get("image_dir", f"{base}/staging")
    resized = pp.get("resized_dir", f"{base}/resized")
    cache = pp.get("cache_dir", f"{base}/cache")
    recursive = ["--recursive"] if pp.get("recursive", True) else []
    target_res = pp.get("target_res", [1024])
    target_res_flag = (
        ["--target_res", *[str(e) for e in target_res]] if target_res else []
    )

    # 1. Resize the native-res staging tree into constant-token buckets. min_pixels
    #    defaults to 0 here (not 0.5MP) so a small member can't be dropped and
    #    orphan its pair partner. Captions ride along (copy_captions default).
    run(
        [
            PY,
            "scripts/preprocess/resize_images.py",
            "--src",
            staging,
            "--dst",
            resized,
            "--min_pixels",
            str(pp.get("min_pixels", 0)),
            *target_res_flag,
            *recursive,
        ]
    )
    # 2. VAE latents from the bucket-resized tree.
    run(
        [
            PY,
            "scripts/preprocess/cache_latents.py",
            "--dir",
            resized,
            "--cache_dir",
            cache,
            "--vae",
            pp.get("vae", "models/vae/qwen_image_vae.safetensors"),
            "--batch_size",
            str(pp.get("batch_size", 4)),
            "--chunk_size",
            str(pp.get("chunk_size", 64)),
            *recursive,
        ]
    )
    # 3. Text-encoder outputs from the same tree (captions copied during resize).
    run(
        [
            PY,
            "scripts/preprocess/cache_text_embeddings.py",
            "--dir",
            resized,
            "--cache_dir",
            cache,
            "--qwen3",
            pp.get("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
            "--dit",
            pp.get("dit", "models/diffusion_models/anima-base-v1.0.safetensors"),
            "--caption_shuffle_variants",
            str(pp.get("caption_shuffle_variants", 4)),
            "--caption_tag_dropout_rate",
            str(pp.get("caption_tag_dropout_rate", 0.1)),
            *recursive,
        ]
    )


def cmd_easycontrol_preprocess(extra):
    """Full EasyControl preprocess: VAE latents + text-encoder outputs.

    Source: ``easycontrol-dataset/``  Caches: ``post_image_dataset/easycontrol/``.

    ``EASYADAPTER=colorize`` instead builds the colorization *condition* cache
    (mangafy the existing color images → VAE-encode into
    ``post_image_dataset/easycontrol/colorize/cond/``); the color target latents + TE are
    reused from the LoRA cache, so no target re-encode is needed. See
    ``easycontrol_adapters/colorization/prep.py``.

    ``EASYADAPTER=near_twin`` caches the mined pair tree, with every knob (source,
    cache dir, model paths, batch/chunk, caption policy) read from the
    ``[preprocess]`` table of ``configs/easycontrol/near_twins.toml``.
    """
    if (os.environ.get("EASYADAPTER") or "").strip() == "near_twin":
        _near_twin_preprocess()
        return
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


# EasyControl adapters that ship a *staging* step (data generation that
# materializes the training tree, before the VAE/TE preprocess pass) → the CLI
# that produces it. ``near_twin`` mines the in-artist pair tree.
_EASY_STAGERS = {
    "near_twin": [PY, "-m", "easycontrol_adapters.tools.near_twin"],
}


def cmd_easycontrol_staging(extra):
    """Generate an EasyControl adapter's *staging* dataset (no VAE/TE caching).

    The adapter-specific data-generation step that materializes the training
    tree — analogous to colorize's cond synthesis — kept separate from the later
    ``easycontrol-preprocess`` VAE/TE caching pass.

    ``EASYADAPTER=near_twin`` mines the in-artist near-twin pair tree into
    ``post_image_dataset/easycontrol/near_twins/staging/`` and (re)writes the
    dataset blueprint ``configs/easycontrol/near_twins.toml``. Run knobs come from
    that file's ``[staging]`` table; extra CLI args override it, e.g.::

        make easycontrol-staging EASYADAPTER=near_twin \\
            ARGS="--region --artists ama_mitsuki"
    """
    adapter = (os.environ.get("EASYADAPTER") or "").strip()
    cmd = _EASY_STAGERS.get(adapter)
    if cmd is None:
        raise SystemExit(
            f"easycontrol-staging needs a staging-capable EASYADAPTER. "
            f"Known: {sorted(_EASY_STAGERS)}.\n"
            "(The default EasyControl reads easycontrol-dataset/ directly; "
            "colorize's cond synthesis runs under "
            "`easycontrol-preprocess EASYADAPTER=colorize`.)"
        )
    run([*cmd, *extra])
