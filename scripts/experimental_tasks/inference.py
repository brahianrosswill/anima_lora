"""Experimental inference entry-points (exp-test-* commands).

Covers the unstable methods kept under ``make exp-*``: soft tokens, BYG, plus
the DirectEdit + postfix-tail inversion probes. Reference-image variants accept
REF_IMAGE env or first positional arg, copy the ref alongside the generated
output. (EasyControl graduated to the shipped ``test-easycontrol`` — see
``scripts/tasks/inference.py``; IP-Adapter was downgraded to ``bench/ip_adapter/``.)
"""

from __future__ import annotations

import os
import random
import shutil
import sys
from pathlib import Path

from scripts.tasks._common import (
    INFERENCE_BASE,
    ROOT,
    _random_ref_image,
    _REF_IMAGE_EXTS,
    latest_output,
    run,
)

_TE_SUFFIX = "_anima_te.safetensors"


def _te_cache_candidates(ref_image: str | os.PathLike) -> list[Path]:
    """TE cache locations to probe for a resized reference image.

    Preprocessing writes caches under ``post_image_dataset/lora/`` mirroring
    the per-artist subdir layout of ``post_image_dataset/resized/`` (see
    ``resolve_cache_path`` with ``image_dir``). So for
    ``resized/mejikara_scene/10083096.png`` the cache lands at
    ``lora/mejikara_scene/10083096_anima_te.safetensors`` — not flat under
    ``lora/``. Probe, in order: the nested mirror, the flat ``lora/`` root,
    and the legacy sidecar next to the image.
    """
    from library.io.cache import resolve_cache_path  # noqa: PLC0415

    ref = Path(ref_image)
    stem = ref.stem
    resized_root = ROOT / "post_image_dataset" / "resized"
    lora_root = ROOT / "post_image_dataset" / "lora"
    nested = Path(
        resolve_cache_path(
            str(ref),
            _TE_SUFFIX,
            cache_dir=str(lora_root),
            image_dir=str(resized_root),
        )
    )
    # Deduplicate while preserving order (nested == flat when no subdir).
    candidates = [
        nested,
        lora_root / f"{stem}{_TE_SUFFIX}",
        ref.parent / f"{stem}{_TE_SUFFIX}",
    ]
    seen: set[Path] = set()
    return [p for p in candidates if not (p in seen or seen.add(p))]


def _resolve_te_cache(ref_image: str | os.PathLike) -> Path | None:
    """First existing TE cache for ``ref_image``, or ``None``."""
    return next((p for p in _te_cache_candidates(ref_image) if p.is_file()), None)


def _resolve_ref_image(ref_image: str) -> str:
    """Resolve a possibly-partial ``REF_IMAGE`` to a real file under ``resized/``.

    Accepts a path that already exists as given, or a partial path relative to
    ``post_image_dataset/resized/`` with or without an extension (e.g.
    ``sushispin/10186995`` → ``.../resized/sushispin/10186995.png``). Returning
    the full nested path is what lets ``_te_cache_candidates`` /
    ``resolve_cache_path`` mirror the per-artist subdir into the cache lookup —
    a bare ``artist/stem`` makes ``relpath`` escape the resized root and fall
    back to the (wrong) flat ``lora/stem`` candidate. Returns ``ref_image``
    untouched when nothing matches, so downstream "not found" messaging fires.
    """
    if Path(ref_image).is_file():
        return ref_image
    resized_root = ROOT / "post_image_dataset" / "resized"
    base = resized_root / ref_image
    if base.is_file():
        return str(base)
    for ext in _REF_IMAGE_EXTS:
        for cand in (Path(f"{base}{ext}"), base.with_suffix(ext)):
            if cand.is_file():
                return str(cand)
    return ref_image


def cmd_test_soft(extra):
    """Inference with latest soft_tokens weight (SoftREPA-style per-layer × per-t bank).

    Resolves the newest ``anima_soft_tokens*.safetensors`` under ``output/ckpt/``
    and passes it via ``--soft_tokens_weight``. The network is built in
    ``library/inference/generation.py``, ``apply_to`` monkey-patches the first
    ``n_layers`` ``Block.forward``s, and ``append_postfix(..., timesteps=t)``
    fires per CFG branch inside the denoising loop (mirrored in the Spectrum
    runner). Composes freely with ``--spectrum``; cached spectrum steps skip
    blocks so soft_tokens silently no-ops on those steps.
    """
    run(
        [
            *INFERENCE_BASE,
            "--soft_tokens_weight",
            str(latest_output("anima_soft_tokens")),
            *extra,
        ]
    )


def cmd_test_turbo(extra):
    """Inference with the latest turbo student LoRA at 2 steps, cfg=1.0.

    CFG is baked into the student during distillation, so production inference
    runs cfg=1.0 (no double-CFG). Step count defaults to 2 — matching the
    DP-DMD student's `student_steps=2` rollout — but extra args can override.
    """
    weight = latest_output("anima_turbo")
    base = list(INFERENCE_BASE)
    # Replace defaults so `--infer_steps`/`--guidance_scale` reflect the turbo
    # contract (2 steps, cfg=1.0). User extra args still win since they come last.
    base = _override_arg(base, "--sampler", "euler")
    # Per-step-expert checkpoints bind head k to denoise step k, so infer_steps
    # MUST equal the trained head count K (= student_steps). Read it from the
    # metadata and pin infer_steps to K; overshoot would repeat the last head
    # (the inference helper clamps) and undershoot would skip the quality head.
    infer_steps = "2"
    try:
        from safetensors import safe_open

        with safe_open(str(weight), framework="pt") as f:
            md = f.metadata() or {}
        if str(md.get("ss_turbo_per_step_expert", "")).strip() in ("1", "true", "True"):
            K = int(md.get("ss_turbo_step_expert_K", "2") or "2")
            infer_steps = str(K)
            print(
                f"[test-turbo] per-step-expert checkpoint: pinning "
                f"--infer_steps {K} (= trained head count). Override at your own "
                "risk — heads beyond K repeat the last (quality) head."
            )
    except Exception:
        pass
    base = _override_arg(base, "--infer_steps", infer_steps)
    base = _override_arg(base, "--guidance_scale", "1.0")
    run(
        [
            *base,
            "--lora_weight",
            str(weight),
            *extra,
        ]
    )


def _override_arg(argv: list[str], flag: str, value: str) -> list[str]:
    """Replace a ``--flag VALUE`` (or ``--flag V1 V2``) pair in argv with a
    fresh ``--flag value`` pair. Used to retarget INFERENCE_BASE defaults
    for the turbo contract (2 steps, cfg=1.0) without rewriting the whole list.
    """
    if flag not in argv:
        return argv + [flag, value]
    i = argv.index(flag)
    # Drop the flag and its single value; INFERENCE_BASE doesn't use multi-arg
    # flags for these two keys.
    return argv[:i] + [flag, value] + argv[i + 2 :]


def cmd_test_spd(extra):
    """Inference with the latest SPD fine-tune LoRA on the SPD sampler.

    Runs at the *schedule the LoRA was trained on* — read from the safetensors
    metadata (``ss_spd_stages`` / ``ss_spd_transition_sigmas``, stamped by
    ``scripts/distill_spd.py``) so the trajectory geometry can't silently
    mismatch what was trained (proposal R2). CFG stays at the production
    default (4.0); ``--spd`` forces Euler internally.

        make exp-test-spd
        make exp-test-spd ARGS="--spd_stages 0.5 0.75 1.0 --spd_transition_sigmas 0.6 0.4"
        make exp-test-spd ARGS="--seed 1234 --image_size 832 1248"

    User ``ARGS`` win: passing ``--spd_stages`` / ``--spd_transition_sigmas``
    in ARGS overrides the metadata schedule.
    """
    import json

    from safetensors import safe_open

    weight = latest_output("anima_spd")
    md: dict[str, str] = {}
    try:
        with safe_open(str(weight), "pt") as f:
            md = f.metadata() or {}
    except Exception as e:  # noqa: BLE001
        print(f"  warn: could not read SPD schedule from {weight}: {e}")

    base = _override_arg(list(INFERENCE_BASE), "--sampler", "euler")  # SPD forces Euler
    cmd = [*base, "--lora_weight", str(weight), "--spd"]

    stages = md.get("ss_spd_stages")
    trans = md.get("ss_spd_transition_sigmas")
    label = md.get("ss_spd_schedule_label", "?")
    if stages and "--spd_stages" not in extra:
        cmd += ["--spd_stages", *(str(s) for s in json.loads(stages))]
    if trans and "--spd_transition_sigmas" not in extra:
        cmd += ["--spd_transition_sigmas", *(str(s) for s in json.loads(trans))]
    print(f"  > SPD LoRA: {weight}  schedule='{label}' stages={stages} σ={trans}")
    run([*cmd, *extra])


def cmd_test_directedit(extra):
    """DirectEdit on a random source image, seeded by wd-swinv2-tagger-v3.

    Pipeline:
      1. Pick source image (REF_IMAGE env, first positional arg, or random
         from ``post_image_dataset/resized/``).
      2. Run wd-swinv2-tagger-v3 on the source -> ``src_tags`` caption
         (downloaded on first use to ``models/captioners/wd-swinv2-tagger-v3/``).
      3. Build edit prompts:
            prompt_src = src_tags
            prompt_tar = src_tags + ", " + PROMPT
         (PROMPT env or ``--prompt`` extra arg supplies the edit instruction.
         Defaults to ``"double peace"``.)
      4. Call ``scripts/edit.py`` (DirectEdit invert + edit) using the same
         DiT/VAE/TE trio as the other inference targets.
      5. Save under ``output/tests/directedit/`` and copy the source image
         alongside as ``<name>_src.png``.

    Examples:
      make exp-test-directedit PROMPT='double peace'
      REF_IMAGE=foo.png make exp-test-directedit PROMPT='glasses'
      python tasks.py exp-test-directedit foo.png --prompt 'smile'
    """
    # 1. Resolve source image — same logic as the other reference-image tests.
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-directedit [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-directedit\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_image = _resolve_ref_image(ref_image)

    # 2. Pull the user-supplied edit instruction. PROMPT env wins; fall back
    #    to a ``--prompt`` flag in extra; final default = "double peace".
    edit_prompt = os.environ.get("PROMPT", "").strip()
    cleaned_extra: list[str] = []
    skip_next = False
    for j, tok in enumerate(extra):
        if skip_next:
            skip_next = False
            continue
        if tok == "--prompt" and j + 1 < len(extra):
            if not edit_prompt:
                edit_prompt = extra[j + 1]
            skip_next = True
            continue
        cleaned_extra.append(tok)
    extra = cleaned_extra
    if not edit_prompt:
        edit_prompt = "double peace, v v. She is showing double peace"

    # 3. Run Anima Tagger on the source.
    from PIL import Image  # noqa: PLC0415

    anima_ckpt = (
        ROOT / "models" / "captioners" / "anima-tagger-v1" / "model.safetensors"
    )
    if not anima_ckpt.exists():
        raise SystemExit(
            f"Anima Tagger checkpoint missing at {anima_ckpt} — "
            "train via `python -m scripts.anima_tagger.cli`."
        )

    print(f"  > tagging source: {ref_image}")
    from library.captioning.anima_tagger import AnimaTagger  # noqa: PLC0415

    tagger = AnimaTagger(ckpt_dir=anima_ckpt.parent)

    src_caption = tagger.predict_caption(Image.open(ref_image))
    if not src_caption:
        print(
            "  ! tagger produced no tags above threshold; using empty source "
            "prompt — DirectEdit reconstruction will be weaker than usual.",
            file=sys.stderr,
        )
    print(
        f"  > src caption: {src_caption[:120]}{'...' if len(src_caption) > 120 else ''}"
    )

    # 4. Save dir + edit.py invocation. Reuse INFERENCE_BASE for the model
    #    path trio (--dit / --text_encoder / --vae) so this stays in sync with
    #    the other test commands automatically.
    #
    #    Hand the edit instruction to edit.py via --edit_instruction so the
    #    dispatcher (Qwen3 last-token cosine + threshold/gap gate; see
    #    library/inference/edit_dispatcher.py and plan.md) runs in-process —
    #    REPLACE on confident matches, REMOVE on explicit `-X` / `no X`,
    #    APPEND otherwise. Running the dispatcher in this wrapper would load
    #    Qwen3 a second time; we'd rather edit.py do it once.
    save_dir = ROOT / "output" / "tests" / "directedit"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop "inference.py"
    leftover_base = list(base_iter)
    args = [py, "scripts/edit.py", *_filter_inference_base_for_edit(leftover_base)]
    args += [
        "--image",
        str(ref_image),
        "--prompt_src",
        src_caption,
        "--edit_instruction",
        edit_prompt,
        "--save_path",
        str(save_dir),
    ]
    args += list(extra)
    run(args)

    # 5. Copy the source alongside the edited output for side-by-side review.
    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
        shutil.copy(ref_image, src_dst)
        print(f"  > Source pasted: {src_dst}")


def cmd_test_directedit_dry(extra):
    """DirectEdit functional sanity check using preprocessed cross-emb variants.

    Bypasses the tagger and the text encoder. Auto-resolves the source image's
    `_anima_te.safetensors` cache (the file `cache_text_embeddings.py` writes
    — same format the trainer consumes) and runs one invert + edit pass per
    stored variant with ψ_tar == ψ_src. With `--caption_shuffle_variants N`
    caches, this sweeps v0 (pristine) + v1..v{N-1} (tag-shuffled). Each pass
    should reconstruct the source; divergence flags numeric drift in
    invert/edit_forward against that variant's cross-emb representation.

    Add ``--fm_score`` to also rank each variant's ψ_src by its intrinsic
    flow-matching error (AGSM-style reward; lower = more on-manifold) and
    correlate that ranking against each variant's reconstruction MSE — a
    quantitative replacement for eyeballing the side-by-side divergence.

    Examples:
      make exp-test-directedit-dry
      REF_IMAGE=foo.png make exp-test-directedit-dry
      python tasks.py exp-test-directedit-dry foo.png --seed 7
      python tasks.py exp-test-directedit-dry foo.png --fm_score
    """
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-directedit-dry [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-directedit-dry\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_image = _resolve_ref_image(ref_image)

    # Auto-resolve the matching TE cache file. Preprocessing mirrors the
    # resized/ subdir layout under post_image_dataset/lora/, so probe the
    # nested mirror first, then the flat lora/ root, then the legacy sidecar.
    candidates = _te_cache_candidates(ref_image)
    cache_path = _resolve_te_cache(ref_image)
    if cache_path is None:
        looked = "\n".join(f"      {p}" for p in candidates)
        print(
            f"  ! No TE cache found for {ref_image}.\n"
            f"    Looked in:\n{looked}\n"
            "    Run `make preprocess-te` first (with --caption_shuffle_variants N "
            "to get a multi-variant cache).",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"  > TE cache: {cache_path}")

    save_dir = ROOT / "output" / "tests" / "directedit_dry"
    save_dir.mkdir(parents=True, exist_ok=True)

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop "inference.py"
    leftover_base = list(base_iter)
    args = [py, "scripts/edit.py", *_filter_inference_base_for_edit(leftover_base)]
    args += [
        "--image",
        str(ref_image),
        "--cached_embed",
        str(cache_path),
        "--save_path",
        str(save_dir),
    ]
    args += list(extra)
    run(args)

    # Copy the source alongside the reconstruction for side-by-side review.
    pngs = sorted(
        (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
        shutil.copy(ref_image, src_dst)
        print(f"  > Source pasted: {src_dst}")


def _resolve_inference_base_flag(name: str) -> str | None:
    """Read a ``--name <value>`` pair out of ``INFERENCE_BASE``.

    Lets ``cmd_invert_directedit`` reuse the same ``--dit`` path as the
    other test commands without duplicating the model-path string.
    """
    base = list(INFERENCE_BASE)
    for i, tok in enumerate(base):
        if tok == name and i + 1 < len(base):
            return base[i + 1]
    return None


def _resolve_ref_image_pool(directory: Path, n: int) -> list[str]:
    """Pick ``n`` distinct random images from ``directory``.

    Same convention as ``_random_ref_image`` but returns a list (no
    replacement) for the N_IMAGES > 1 path.
    """
    if not directory.is_dir():
        return []
    pool = [p for p in directory.rglob("*") if p.suffix.lower() in _REF_IMAGE_EXTS]
    if not pool:
        return []
    if n >= len(pool):
        return [str(p) for p in pool]
    return [str(p) for p in random.sample(pool, n)]


def cmd_invert_directedit(extra):
    """Probe: does an inverted per-image postfix tail make DirectEdit dry-mode
    reconstruction (ψ_tar == ψ_src) more robust than the bare T5(tags) prefix?

    For each of ``N_IMAGES`` source images:

      1. Invert the orthogonal postfix tail (scripts/inversion/invert_postfix_tail.py)
         — K trainable scales over a frozen Q@diag(s) basis, optimized against
         flow-matching loss through the frozen DiT.
      2. Run ``scripts/edit.py --cached_embed`` in dry mode against the
         **baseline** v0 prefix (T5(tags)).
      3. Splice ``Q @ diag(s)`` into the last K positions of that prefix and
         run dry mode again. This is the **postfix-augmented** ψ_src.

    Both runs write to ``output/tests/invert_directedit/<stem>/{baseline,postfix}/``
    with the source pasted alongside, so the postfix's reconstruction lift can
    be eyeballed without going through the full bench analyzer.

    Env vars (all optional, all match invert_postfix_tail.py defaults):
      N_IMAGES (default 1)   — number of images to process
      REF_IMAGE              — single explicit image path (sets N_IMAGES=1)
      K (default 48)         — tail length
      INVERT_STEPS (100)     — inversion optimization steps
      INVERT_LR (0.01)       — AdamW lr
      LAMBDA_ZERO (0.0)      — ‖s‖² regularization
      SIGMA_MIN (0.0)        — P-GRAFT low-σ skip (lower σ bound)
      SIGMA_MAX (1.0)        — upper σ bound; <1.0 restricts to low-σ supervision
      BASIS (svd_te)         — basis kind (svd_te|random)
      SEED (0)               — per-image inversion seed
      TIMESTEPS_PER_STEP (1) — batched sigmas per forward (× GRAD_ACCUM = total)
      GRAD_ACCUM (1)         — grad accum steps

    Examples:
      make exp-invert-directedit
      REF_IMAGE=post_image_dataset/resized/12345.png make exp-invert-directedit
      N_IMAGES=3 make exp-invert-directedit
      K=8 INVERT_STEPS=50 make exp-invert-directedit
    """

    # 1. Resolve image pool.
    ref_image_override = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image_override and extra and not extra[0].startswith("-"):
        ref_image_override = extra[0]
        extra = extra[1:]

    n_images_env = os.environ.get("N_IMAGES", "").strip()
    try:
        n_images = max(1, int(n_images_env)) if n_images_env else 1
    except ValueError:
        print(f"  ! N_IMAGES={n_images_env!r} is not an int, using 1", file=sys.stderr)
        n_images = 1
    if ref_image_override:
        n_images = 1

    if ref_image_override:
        images = [_resolve_ref_image(ref_image_override)]
    else:
        images = _resolve_ref_image_pool(
            ROOT / "post_image_dataset" / "resized", n_images
        )
    if not images:
        print(
            "Usage: python tasks.py exp-invert-directedit [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-invert-directedit\n"
            "   or: N_IMAGES=3 python tasks.py exp-invert-directedit\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Inversion knobs — env overrides for the common dials, defaults match
    #    the proposal (and the invert_postfix_tail.py CLI defaults).
    K = int(os.environ.get("K", "8"))
    invert_steps = int(os.environ.get("INVERT_STEPS", "100"))
    invert_lr = float(os.environ.get("INVERT_LR", "1e-1"))
    lambda_zero = float(os.environ.get("LAMBDA_ZERO", "0.0"))
    sigma_min = float(os.environ.get("SIGMA_MIN", "0"))
    sigma_max = float(os.environ.get("SIGMA_MAX", "1.0"))
    basis_kind = os.environ.get("BASIS", "random").strip()
    seed = int(os.environ.get("SEED", "0"))
    timesteps_per_step = int(os.environ.get("TIMESTEPS_PER_STEP", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "2"))

    run_root = ROOT / "output" / "tests" / "invert_directedit"
    run_root.mkdir(parents=True, exist_ok=True)
    basis_path = run_root / f"svd_basis_K{K}.pt"

    # 3. Resolve the shared --dit / etc. flags from INFERENCE_BASE so this
    #    target follows the same model trio as the rest of the test family.
    dit_path = _resolve_inference_base_flag("--dit")
    if dit_path is None:
        print("  ! INFERENCE_BASE has no --dit value", file=sys.stderr)
        sys.exit(1)
    attn_mode = _resolve_inference_base_flag("--attn_mode") or "flash"

    # Lazy import — keep the task module light when this command isn't run.
    from library.inference.editing.postfix_inversion import (  # noqa: PLC0415
        load_or_build_basis,
        load_tail_s,
        splice_tail_into_te_cache,
    )

    base_iter = iter(INFERENCE_BASE)
    py = next(base_iter)
    next(base_iter)  # drop 'inference.py'
    leftover_base = list(base_iter)
    edit_base_args = [
        py,
        "scripts/edit.py",
        *_filter_inference_base_for_edit(leftover_base),
    ]

    for i, ref_image in enumerate(images):
        stem = Path(ref_image).stem
        print(f"\n=== [{i + 1}/{len(images)}] {stem} ===")

        # 4. Find the cached TE for this image (the baseline v0 prefix).
        #    Caches mirror the resized/ subdir layout under lora/, so probe
        #    the nested mirror before the flat root and the legacy sidecar.
        te_path = _resolve_te_cache(ref_image)
        if te_path is None:
            looked = "\n".join(f"    {p}" for p in _te_cache_candidates(ref_image))
            print(
                f"  ! No TE cache found for {stem}. Looked in:\n{looked}\n"
                "    Run `make preprocess-te` first.",
                file=sys.stderr,
            )
            continue

        # 5. Invert the postfix tail via the standalone CLI.
        img_dir = run_root / stem
        invert_out = img_dir / "inversion"
        s_path = invert_out / "s" / f"{stem}_s.safetensors"
        if s_path.exists():
            print(f"  > Reusing existing inversion: {s_path}")
        else:
            invert_cmd = [
                py,
                "scripts/inversion/invert_postfix_tail.py",
                # directedit needs the ortho_tail s-vector (spliced via basis Q
                # below); the probe script now defaults to soft_tokens, which
                # writes a bank/ file instead — pin the mode explicitly.
                "--parameterization",
                "ortho_tail",
                "--dit",
                str(dit_path),
                "--attn_mode",
                str(attn_mode),
                "--image_dir",
                str(te_path.parent),
                "--image_stem",
                stem,
                "--K",
                str(K),
                "--basis",
                basis_kind,
                "--basis_path",
                str(basis_path),
                "--steps",
                str(invert_steps),
                "--lr",
                str(invert_lr),
                "--lambda_zero",
                str(lambda_zero),
                "--sigma_min",
                str(sigma_min),
                "--sigma_max",
                str(sigma_max),
                "--seed",
                str(seed),
                "--timesteps_per_step",
                str(timesteps_per_step),
                "--grad_accum",
                str(grad_accum),
                "--output_dir",
                str(invert_out),
            ]
            run(invert_cmd)
            if not s_path.exists():
                print(
                    f"  ! Inversion did not produce {s_path}; skipping {stem}",
                    file=sys.stderr,
                )
                continue

        # 6. Build the spliced TE cache — load s + Q on CPU, splice, write.
        #    D=1024 matches Qwen3's hidden size (the only one Anima ships
        #    against); load_or_build_basis verifies cached shape and fails
        #    loud if the on-disk basis doesn't match.
        s, _ = load_tail_s(str(s_path))
        Q = load_or_build_basis(
            K=K,
            D=1024,
            kind=basis_kind,
            te_cache_dir=str(te_path.parent),
            basis_path=str(basis_path),
            seed=seed,
        )

        spliced_te = img_dir / "te_postfix.safetensors"
        splice_tail_into_te_cache(
            str(te_path),
            str(spliced_te),
            s=s,
            Q=Q,
            variant_index=0,
        )
        print(f"  > Spliced TE cache: {spliced_te}")

        # 7. Two dry-mode edit.py invocations: baseline + postfix.
        for label, cache_for_run in (
            ("baseline", te_path),
            ("postfix", spliced_te),
        ):
            save_dir = img_dir / label
            save_dir.mkdir(parents=True, exist_ok=True)
            edit_cmd = [
                *edit_base_args,
                "--image",
                str(ref_image),
                "--cached_embed",
                str(cache_for_run),
                "--cached_embed_variants",
                "0",
                "--save_path",
                str(save_dir),
                *list(extra),
            ]
            run(edit_cmd)

            # Paste the source for side-by-side eyeballing.
            pngs = sorted(
                (p for p in save_dir.glob("*.png") if not p.name.endswith("_src.png")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if pngs:
                src_dst = pngs[0].with_name(pngs[0].stem + "_src.png")
                shutil.copy(ref_image, src_dst)
                print(f"  > [{label}] source pasted: {src_dst}")


def _filter_inference_base_for_edit(args: list[str]) -> list[str]:
    """Drop INFERENCE_BASE flags that ``scripts/edit.py`` doesn't accept.

    INFERENCE_BASE bundles plenty of generation-only flags (--prompt, --seed,
    --image_size, --infer_steps, --sampler, etc.) that overlap with or
    conflict with edit.py's own. Keep only the model/path flags we actually
    want to forward; let edit.py supply its own defaults for the rest.
    """
    keep_flags = {
        "--dit",
        "--text_encoder",
        "--vae",
        "--vae_chunk_size",
        "--attn_mode",
    }
    boolean_flags = {"--vae_disable_cache"}
    out: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in keep_flags and i + 1 < len(args):
            out.extend([tok, args[i + 1]])
            i += 2
        elif tok in boolean_flags:
            out.append(tok)
            i += 1
        else:
            i += 1
    return out


def _flair_caption(ref_image: str, fallback: str) -> str:
    """Canonical prompt for FLAIR reconstruction — the image's ``.txt`` sidecar.

    Tries the caption next to the resized image, then the master caption under
    ``image_dataset/`` (mirroring the per-artist subdir), then ``fallback``.
    """
    ref = Path(ref_image)
    sidecar = ref.with_suffix(".txt")
    if sidecar.is_file():
        cap = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
        if cap:
            return cap
    # Master caption: image_dataset/<rel>/<stem>.txt mirrors resized/<rel>/<stem>.png
    resized_root = ROOT / "post_image_dataset" / "resized"
    try:
        rel = ref.resolve().relative_to(resized_root.resolve())
        master = (ROOT / "image_dataset" / rel).with_suffix(".txt")
        if master.is_file():
            cap = master.read_text(encoding="utf-8", errors="ignore").strip()
            if cap:
                return cap
    except ValueError:
        pass
    return fallback


def _canonical_hw(ref_image: str, edge: int = 512) -> tuple[int, int]:
    """(H, W) preserving the source aspect ratio at ~``edge``² area, each ÷16.

    Square 512 forces a crop/stretch on non-square sources; instead match the
    source's canonical ratio while holding total pixels near ``edge``² so HDC
    (which backprops through the VAE decoder every σ-step) stays inside a 16 GB
    budget. Both dims are snapped to multiples of 16 (the DiT patch constraint).
    """
    from PIL import Image  # noqa: PLC0415

    with Image.open(ref_image) as im:
        w, h = im.size
    ar = w / h if h else 1.0
    area = float(edge * edge)
    out_w = max(16, int(round((area * ar) ** 0.5 / 16)) * 16)
    out_h = max(16, int(round((area / ar) ** 0.5 / 16)) * 16)
    return out_h, out_w


def _random_edit_mask(w: int, h: int, seed: int):
    """A random rectangular *edit* mask (white = region to fill) → PIL 'L' image.

    Covers ~12–30% of the frame with 1–2 boxes — enough to observe the prior's
    fill while leaving most of the image for Hard Data Consistency to lock.
    """
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(int(rng.integers(1, 3))):
        bw = int(rng.integers(int(0.20 * w), int(0.45 * w) + 1))
        bh = int(rng.integers(int(0.20 * h), int(0.45 * h) + 1))
        x0 = int(rng.integers(0, max(1, w - bw)))
        y0 = int(rng.integers(0, max(1, h - bh)))
        mask[y0 : y0 + bh, x0 : x0 + bw] = 255
    return Image.fromarray(mask, mode="L")


def cmd_test_flair(extra):
    """FLAIR-edit / reconstruction observer (training-free, no source prompt).

    Default (no MASK/MASK_PROMPT/PROMPT) is a **reconstruction observation**:
    pick a source image, knock out a *random* region, and let the FLAIR inpaint
    solver fill it back conditioned on the image's own **canonical caption** —
    Hard Data Consistency locks everything outside the hole bit-exact. This is
    the cheap way to eyeball whether the prior reconstructs a masked region
    before trusting it for real edits.

    Three knobs turn it into the productized FLAIR-edit
    (``docs/proposal/flair_edit.md``):
      - ``PROMPT='blue hair'``  — the delta-only edit prompt (replaces the caption).
      - ``MASK=hair.png``       — an explicit edit-region mask (white = fill).
      - ``MASK_PROMPT='eyes'``  — SAM3 derives the region from a concept phrase.

    Source resolution mirrors the other reference-image tests (REF_IMAGE env,
    first positional arg, or a random pick from ``post_image_dataset/resized/``).
    Runs at 512px by default (HDC backprops through the VAE decoder every σ-step —
    fits a 16 GB card per the parent Phase-0 result); override with extra args.

    Examples:
      make exp-test-flair                                  # random-mask reconstruction
      REF_IMAGE=girl.png make exp-test-flair               # observe a specific image
      REF_IMAGE=girl.png MASK=hair.png PROMPT='blue hair' make exp-test-flair
      REF_IMAGE=girl.png MASK_PROMPT='eyes' PROMPT='glasses' make exp-test-flair
      python tasks.py exp-test-flair girl.png --image_size 768 768 --flair_alpha 1.0
    """
    import os  # noqa: PLC0415

    # 1. Source image — same resolution as the other reference-image tests.
    ref_image = os.environ.get("REF_IMAGE", "").strip()
    if not ref_image and extra and not extra[0].startswith("-"):
        ref_image = extra[0]
        extra = extra[1:]
    if not ref_image:
        ref_image = _random_ref_image(ROOT / "post_image_dataset" / "resized") or ""
    if not ref_image:
        print(
            "Usage: python tasks.py exp-test-flair [<ref_image>] [extra...]\n"
            "   or: REF_IMAGE=path/to/ref.png python tasks.py exp-test-flair\n"
            "   (no ref given and post_image_dataset/resized/ is empty)",
            file=sys.stderr,
        )
        sys.exit(1)
    ref_image = _resolve_ref_image(ref_image)

    # 2. Resolution + seed. Default to the source's **canonical aspect ratio** at
    #    ~512² area (HDC-safe on 16 GB); an explicit --image_size in extra wins and
    #    suppresses the auto-derive (argparse last-wins still applies downstream).
    seed = 0
    user_image_size = False
    user_hw: tuple[int, int] | None = None
    for j, tok in enumerate(extra):
        if tok == "--image_size" and j + 2 < len(extra):
            user_image_size = True
            user_hw = (int(extra[j + 1]), int(extra[j + 2]))
        if tok == "--seed" and j + 1 < len(extra):
            seed = int(extra[j + 1])

    if user_image_size:
        out_h, out_w = user_hw  # type: ignore[misc]
    else:
        out_h, out_w = _canonical_hw(ref_image)
    print(f"  > resolution: {out_w}x{out_h} (HxW, canonical ratio of source)")

    save_dir = ROOT / "output" / "tests" / "flair"
    mask_dir = save_dir / "_masks"
    save_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    # 3. Mask source — MASK (explicit) > MASK_PROMPT (SAM3) > random (default).
    mask_path = os.environ.get("MASK", "").strip()
    mask_prompt = os.environ.get("MASK_PROMPT", "").strip()
    mask_args: list[str] = []
    if mask_path:
        mask_args = ["--flair_mask", _resolve_ref_image(mask_path)]
        print(f"  > explicit mask: {mask_args[1]}")
    elif mask_prompt:
        mask_args = ["--flair_mask_prompt", mask_prompt]
        print(f"  > SAM3 mask from concept: {mask_prompt!r}")
    else:
        random_mask = mask_dir / f"{Path(ref_image).stem}_editmask.png"
        _random_edit_mask(out_w, out_h, seed).save(random_mask)
        mask_args = ["--flair_mask", str(random_mask)]
        print(f"  > random reconstruction mask: {random_mask}")

    # 4. Prompt — PROMPT (delta) overrides; default = the canonical caption.
    prompt = os.environ.get("PROMPT", "").strip()
    if not prompt:
        prompt = _flair_caption(ref_image, "masterpiece, best quality, score_7")
        print(f"  > canonical reconstruction prompt: '{prompt[:80]}'")
    else:
        print(f"  > delta edit prompt: '{prompt}'")

    # 5. Build the inference.py invocation. Overrides come AFTER INFERENCE_BASE so
    #    argparse's last-wins resolves --prompt/--image_size/--guidance_scale/--save_path.
    args = [
        *INFERENCE_BASE,
        "--prompt",
        prompt,
        "--guidance_scale",
        "4.0",
        "--save_path",
        str(save_dir),
        "--flair_task",
        "edit",
        "--flair_edit_image",
        str(ref_image),
        *mask_args,
    ]
    # Auto-derived size goes before extra (so an explicit --image_size in extra
    # still wins via argparse last-wins); when the user supplied one we don't add
    # ours at all.
    if not user_image_size:
        args += ["--image_size", str(out_h), str(out_w)]
    args += list(extra)
    run(args)

    # 6. Copy source (+ mask) alongside the newest output for side-by-side review.
    pngs = sorted(
        (
            p
            for p in save_dir.glob("*.png")
            if not p.name.endswith(("_src.png", "_mask.png"))
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if pngs:
        shutil.copy(ref_image, pngs[0].with_name(pngs[0].stem + "_src.png"))
        if mask_args[0] == "--flair_mask":
            shutil.copy(mask_args[1], pngs[0].with_name(pngs[0].stem + "_mask.png"))
        print(f"  > Source pasted: {pngs[0].with_name(pngs[0].stem + '_src.png')}")


def cmd_test_byg(extra):
    """Inference with the latest BYG editing LoRA (source image + instruction).

    NOTE (v1): BYG ships as a *plain LoRA*, so the trained weights load via the
    standard ``--lora_weight`` path; the only missing inference piece is the
    parameter-free source-concat conditioning patch (``BYGConditioning`` in
    ``networks/methods/byg.py``) being installed at generation time and primed
    with the VAE-encoded reference. That wiring into ``library/inference/`` is
    the next phase (mirrors the EasyControl KV-prefill node). Until then this
    command is a placeholder so the collapse-watch validation can be run once
    inference is wired.
    """
    raise SystemExit(
        "exp-test-byg: BYG inference (source-concat patch install + ref encode) "
        "is not wired yet — see the P2 inference step in "
        "docs/proposal/byg_unpaired_editing.md. Training (exp-byg) is functional; "
        "the trained checkpoint is a plain LoRA loadable via --lora_weight."
    )
