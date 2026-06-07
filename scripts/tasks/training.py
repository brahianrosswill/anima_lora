"""Training entry-points for shipped methods (lora family + lora-gui + EasyControl).

Each ``cmd_*`` is a thin shim that translates env vars + extra argv into the
right ``train.py`` (via ``accelerate launch``) call. Experimental methods
(postfix, ip-adapter) live in ``scripts/experimental_tasks/training.py`` and are
wired up under ``make exp-*`` in ``tasks.py``.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tomllib
from pathlib import Path

import toml

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


def _toml_table_to_argv(table: dict) -> list[str]:
    """Flatten a flat TOML table into ``--key value`` train.py argv.

    Bools become bare ``--flag`` when true (omitted when false); lists spread
    into ``--key v1 v2``; scalars become ``--key str(value)``. Used to fold a
    near_twins.toml ``[training]`` table into CLI overrides (CLI wins the merge
    chain, so these override the easycontrol method config).
    """
    argv: list[str] = []
    for key, val in table.items():
        flag = f"--{key}"
        if isinstance(val, bool):
            if val:
                argv.append(flag)
        elif isinstance(val, (list, tuple)):
            argv.append(flag)
            argv.extend(str(v) for v in val)
        else:
            argv.append(flag)
            argv.append(str(val))
    return argv


def _near_twin_train_extra(extra) -> list[str]:
    """Build train.py extra-argv for an ``EASYADAPTER=near_twin`` run.

    ``near_twins.toml`` is a multi-purpose file (``[staging]`` / ``[preprocess]``
    / ``[training]`` knob tables above the generated ``[general]`` + ``[[datasets]]``
    blueprint), but train.py's dataset-config validator rejects any top-level key
    outside the blueprint. So we extract just the blueprint sections into a clean
    generated sidecar and point ``--dataset_config`` at that, then fold the
    optional ``[training]`` table into CLI overrides on top of the easycontrol
    method config. User-supplied ``extra`` argv is appended last so it still wins.

    Note: the generated blueprint exposes each mined member as an independent
    ref==target subset (no ``cond_cache_dir`` pairing), so this is a vanilla
    EasyControl run over the mined images — not yet a clean→tagged control task.
    """
    cfg_path = ROOT / "configs" / "easycontrol" / "near_twins.toml"
    if not cfg_path.is_file():
        raise SystemExit(
            f"{cfg_path} not found — run `make easycontrol-staging "
            "EASYADAPTER=near_twin` then `make easycontrol-preprocess "
            "EASYADAPTER=near_twin` first."
        )
    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    blueprint = {k: cfg[k] for k in ("general", "datasets") if k in cfg}
    if not blueprint.get("datasets"):
        raise SystemExit(
            f"{cfg_path} has no [[datasets]] blueprint yet — run "
            "`make easycontrol-staging EASYADAPTER=near_twin` to mine the pair "
            "tree (it writes the blueprint tail)."
        )

    # Write the blueprint-only dataset config beside the caches (a gitignored data
    # dir that exists once preprocess has run). Regenerated each invocation so it
    # tracks the source file, and stable-pathed so the --queue daemon path can
    # re-read it later.
    subset = next(
        (
            s
            for ds in blueprint["datasets"]
            for s in ds.get("subsets", [])
            if s.get("image_dir")
        ),
        None,
    )
    base_dir = (
        ROOT / Path(subset["image_dir"]).parent
        if subset
        else ROOT / "post_image_dataset" / "easycontrol" / "near_twins"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    ds_path = base_dir / "dataset_config.toml"
    ds_path.write_text(
        "# AUTO-GENERATED from configs/easycontrol/near_twins.toml — do not edit.\n"
        "# Blueprint-only copy (train.py's dataset-config validator rejects the\n"
        "# [staging]/[preprocess]/[training] knob tables in the source file).\n\n"
        + toml.dumps(blueprint),
        encoding="utf-8",
    )

    return [
        "--dataset_config",
        str(ds_path),
        *_toml_table_to_argv(cfg.get("training") or {}),
        *list(extra or []),
    ]


def cmd_easycontrol(extra):
    """EasyControl. ``EASYADAPTER=<name>`` selects a control-task project under
    easycontrol_adapters/ (e.g. ``colorize``) → runs configs/methods/<name>.toml;
    unset → the default ref==target easycontrol.toml.

    ``EASYADAPTER=near_twin`` runs the easycontrol method against the mined
    near-twin blueprint (``configs/easycontrol/near_twins.toml``), folding that
    file's optional ``[training]`` table in as CLI overrides."""
    adapter = (os.environ.get("EASYADAPTER") or "").strip()
    if adapter in ("near_twin", "near_twins"):  # accept the easy plural typo
        train("easycontrol", _near_twin_train_extra(extra))
        return
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
    # 4. Pair the cond/ tree (the _tags reference latent for each _no_tags target).
    _near_twin_build_cond(pp)


def _near_twin_build_cond(pp: dict) -> None:
    """Materialize the ``cond/`` latent tree for the near-twin *removal* task.

    Pairing convention (matches the generated blueprint): the denoising target is
    the clean ``_no_tags`` member; its ``_tags`` twin is the EasyControl condition
    reference. The loader resolves the cond latent by the *target* stem+bucket
    under ``cond_cache_dir``, so for each ``{id}_no_tags_{WxH}_anima.npz`` target
    latent we symlink the sibling ``{id}_tags_{WxH}_anima.npz`` into
    ``cond/<artist>/{id}_no_tags_{WxH}_anima.npz`` (cond content = the _tags
    latent, filed under the _no_tags name). Same-bucket twins only — a member that
    bucketed to a different resolution has no latent at the target's bucket and is
    skipped with a warning.

    Pure symlinks over the existing cache; the tree is rebuilt from scratch each
    run so a dropped pair can't leave a stale link behind.
    """
    base = "post_image_dataset/easycontrol/near_twins"
    cache_dir = ROOT / pp.get("cache_dir", f"{base}/cache")
    cond_dir = ROOT / pp.get("cond_dir", f"{base}/cond")
    if not cache_dir.is_dir():
        raise SystemExit(
            f"{cache_dir} not found — run the VAE/TE caching pass first "
            "(`make easycontrol-preprocess EASYADAPTER=near_twin`)."
        )
    if cond_dir.exists():
        shutil.rmtree(cond_dir)

    pat = re.compile(r"^(?P<id>.+)_no_tags_(?P<bucket>\d{4}x\d{4})_anima\.npz$")
    linked = skipped = 0
    for npz in sorted(cache_dir.rglob("*_no_tags_*_anima.npz")):
        m = pat.match(npz.name)
        if not m:
            continue
        twin = npz.with_name(f"{m['id']}_tags_{m['bucket']}_anima.npz")
        if not twin.is_file():
            print(
                f"  [near_twin cond] no _tags twin at bucket {m['bucket']} for "
                f"{npz.relative_to(cache_dir)} — skipping (unpaired / diff bucket).",
                file=sys.stderr,
            )
            skipped += 1
            continue
        link = cond_dir / npz.relative_to(cache_dir)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(twin.resolve())
        linked += 1
    print(
        f"[near_twin cond] linked {linked} cond latents into {cond_dir}"
        + (f" ({skipped} skipped)" if skipped else "")
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
