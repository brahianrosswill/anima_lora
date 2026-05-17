"""
Distillation prep — Phase 1 (uncond sidecar) + Phase 2 (synthetic latents).

Pre-generates auxiliary artifacts consumed by ``scripts/distill_modulation.py``.

Phase 1 — uncond TE sidecar:
    Emits ``<cache_dir>/_anima_uncond_te.safetensors`` — the ``T5("")``
    cross-attention baseline used as the student's *unconditional* text input
    AND as CFG-negative during Phase 2 synthesis. Replaces the
    ``torch.zeros_like(crossattn_emb)`` shortcut, which is neither paper-
    faithful (Starodubcev et al., ICLR 2026, arXiv:2602.09268v1 §5: "we
    propagate the textual prompt solely through the pooled text embedding,
    using an unconditional prompt for T5") nor what Anima's own CFG inference
    path uses (``library/inference/text.py:99-127``).

Phase 2 — teacher-driven synthetic clean latents:
    Walks each existing ``*_anima_te.safetensors`` in ``--cache_dir``, picks
    the sibling latent NPZ's resolution, runs the frozen teacher
    (base DiT, ``skip_pooled_text_proj=True``) from fresh noise through full
    CFG denoising (positive = cached crossattn_emb v0, negative = T5("") from
    the Phase 1 sidecar), saves the resulting clean latent under
    ``--synth_dir`` using the same NPZ layout as
    ``preprocess/cache_latents.py``. The trainer can then point at
    ``--synth_dir`` instead of (or alongside) the real-image cache to fit on
    the teacher's own manifold, removing the real-vs-teacher distribution gap
    that inflates the irreducible MSE floor.

Usage:
    # both phases (default — runs Phase 1 first if sidecar missing, then Phase 2)
    python scripts/distill_mod_prep.py

    # Phase 1 only (fast — staging only the uncond sidecar)
    python scripts/distill_mod_prep.py --skip_synth

    # Phase 2 only (assumes uncond sidecar exists)
    python scripts/distill_mod_prep.py --skip_uncond

    # cap synthesis for a smoke test
    python scripts/distill_mod_prep.py --max_samples 16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as _load_safetensors
from safetensors.torch import save_file
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

UNCOND_TE_FILENAME = "_anima_uncond_te.safetensors"
DEFAULT_SEQ_LEN = 512  # matches library/inference/text.py CFG-uncond padding


# =============================================================================
# Phase 1: T5("") unconditional sidecar
# =============================================================================


def encode_uncond_crossattn(
    qwen3_path: str,
    dit_path: str,
    *,
    t5_tokenizer_path: str | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``T5("")`` through Qwen3 + LLM adapter, zero padding positions,
    pad/truncate to ``seq_len``. Returns ``(crossattn_emb, pooled)``, both bf16
    on CPU. Shape: ``(seq_len, 1024)`` and ``(1024,)``.

    Mirrors the negative-prompt path in ``library/inference/text.py:99-127``
    and the encode path in ``preprocess/cache_text_embeddings.py:71-105``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

    logger.info(f"Loading LLM adapter from {dit_path} ...")
    llm_adapter = anima_utils.load_llm_adapter(
        dit_path, dtype=torch.bfloat16, device=str(device)
    )

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    with torch.no_grad():
        tokens_and_masks = tokenize_strategy.tokenize([""])
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = (
            encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens_and_masks
            )
        )
        crossattn_emb = llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids.to(device, dtype=torch.long),
            target_attention_mask=t5_attn_mask.to(device),
            source_attention_mask=attn_mask,
        )
        # Zero padding positions — attention sinks in cross-attention softmax.
        crossattn_emb[~t5_attn_mask.to(device).bool()] = 0

    # crossattn_emb: (1, S, 1024). Pad or truncate to seq_len.
    cur_seq = crossattn_emb.shape[1]
    if cur_seq < seq_len:
        crossattn_emb = F.pad(crossattn_emb, (0, 0, 0, seq_len - cur_seq))
    elif cur_seq > seq_len:
        crossattn_emb = crossattn_emb[:, :seq_len, :]

    crossattn_emb = crossattn_emb.squeeze(0).to(dtype=torch.bfloat16).cpu()
    pooled = crossattn_emb.amax(dim=0)  # matches load_cached_text_features fallback

    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return crossattn_emb, pooled


def stage_uncond_sidecar(
    cache_dir: Path,
    qwen3_path: str,
    dit_path: str,
    *,
    t5_tokenizer_path: str | None,
    seq_len: int,
    overwrite: bool,
) -> Path:
    """Phase 1 entry point. Writes ``<cache_dir>/_anima_uncond_te.safetensors``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / UNCOND_TE_FILENAME

    if out_path.exists() and not overwrite:
        logger.info(
            f"Uncond sidecar already exists at {out_path}; pass --overwrite to regenerate."
        )
        return out_path

    crossattn_emb, pooled = encode_uncond_crossattn(
        qwen3_path,
        dit_path,
        t5_tokenizer_path=t5_tokenizer_path,
        seq_len=seq_len,
    )
    save_file({"crossattn_emb": crossattn_emb, "pooled": pooled}, str(out_path))
    logger.info(
        f"Wrote {out_path}  (crossattn_emb={tuple(crossattn_emb.shape)}, "
        f"pooled={tuple(pooled.shape)}, dtype={crossattn_emb.dtype})"
    )
    return out_path


# =============================================================================
# Phase 2: teacher-driven synthetic clean latents
# =============================================================================


def _load_uncond_for_synth(
    uncond_path: Path, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Load the Phase 1 sidecar as a ``(1, seq, 1024)`` tensor for CFG-negative."""
    sd = _load_safetensors(str(uncond_path))
    uncond = sd["crossattn_emb"]
    return uncond.to(device=device, dtype=dtype).unsqueeze(0).contiguous()


def _denoise_one(
    model,
    crossattn_pos: torch.Tensor,
    crossattn_neg: torch.Tensor,
    *,
    H_lat: int,
    W_lat: int,
    num_steps: int,
    cfg_scale: float,
    flow_shift: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the frozen teacher through CFG denoising from fresh noise.

    Returns the clean latent ``(1, 16, H_lat, W_lat)`` in float32 on CPU.
    Mirrors the dense-path branch of ``library/inference/generation.py:529-706``
    minus all the extras (spectrum / dcw / mod-guidance / postfix / hydra) —
    the teacher here is the bare base DiT, so none of those apply.
    """
    from library.inference import sampling as inference_utils

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        num_steps, flow_shift, device
    )
    timesteps = (timesteps / 1000.0).to(device, dtype=dtype)

    sampler = inference_utils.ERSDESampler(sigmas, seed=seed, device=device)

    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, 16, 1, H_lat, W_lat),
        dtype=dtype,
        device=device,
        generator=gen,
    )
    padding_mask = torch.zeros(1, 1, H_lat, W_lat, dtype=dtype, device=device)

    do_cfg = abs(cfg_scale - 1.0) > 1e-6

    for i, t in enumerate(timesteps):
        t_expand = t.expand(latents.shape[0])
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            noise_pred = model.forward_mini_train_dit(
                latents,
                t_expand,
                crossattn_pos,
                padding_mask=padding_mask,
                skip_pooled_text_proj=True,
            )
            if do_cfg:
                uncond_noise_pred = model.forward_mini_train_dit(
                    latents,
                    t_expand,
                    crossattn_neg,
                    padding_mask=padding_mask,
                    skip_pooled_text_proj=True,
                )
                noise_pred = uncond_noise_pred + cfg_scale * (
                    noise_pred - uncond_noise_pred
                )

        denoised = latents.float() - sigmas[i] * noise_pred.float()
        latents = sampler.step(latents, denoised, i).to(latents.dtype)

    # latents: (1, 16, 1, H_lat, W_lat) → drop temporal dim
    return latents.float().squeeze(2).cpu()


def generate_synthetic_latents(
    cache_dir: Path,
    synth_dir: Path,
    *,
    dit_path: str,
    uncond_path: Path,
    attn_mode: str,
    num_steps: int,
    cfg_scale: float,
    flow_shift: float,
    seed: int,
    variant: int,
    max_samples: int | None,
    blocks_to_swap: int,
    overwrite: bool,
) -> None:
    """Phase 2 entry point. Iterates TE caches, runs teacher denoising, dumps NPZs."""
    from library.anima import weights as anima_utils
    from library.anima.models import Anima
    from library.io.cache import (
        discover_cached_pairs,
        get_latent_resolution,
        load_cached_text_features,
    )

    pairs = discover_cached_pairs(str(cache_dir))
    if not pairs:
        logger.warning(
            f"No (latent.npz, TE) pairs discovered in {cache_dir}. Run preprocess first."
        )
        return

    synth_dir.mkdir(parents=True, exist_ok=True)

    if max_samples is not None:
        pairs = pairs[: int(max_samples)]
    logger.info(
        f"Phase 2: synthesizing {len(pairs)} clean latents from teacher "
        f"(steps={num_steps}, cfg={cfg_scale}, flow_shift={flow_shift}, seed={seed})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    logger.info(f"Loading base DiT (teacher) from {dit_path} ...")
    model: Anima = anima_utils.load_anima_model(
        device,
        dit_path,
        attn_mode=attn_mode,
        split_attn=False,
        loading_device="cpu" if blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )
    if blocks_to_swap > 0:
        model.enable_block_swap(blocks_to_swap, device)
        model.move_to_device_except_swap_blocks(device)
    else:
        model.to(device)
    model.set_static_token_count(4096)
    model.eval()

    crossattn_neg = _load_uncond_for_synth(uncond_path, device, dtype)

    pbar = tqdm(pairs, desc="synth latents")
    n_written = 0
    n_skipped = 0
    for sample_idx, pair in enumerate(pbar):
        # Resolution from sibling real-image latent NPZ — keeps synthetic
        # aspect distribution matched to the real dataset and guarantees
        # the constant-token-bucketing invariant is satisfied.
        try:
            res_str = get_latent_resolution(pair.npz_path)  # e.g. "64x64"
            H_lat, W_lat = (int(x) for x in res_str.split("x"))
        except Exception as e:
            logger.warning(f"  skip {pair.stem}: bad latent NPZ ({e})")
            continue

        out_path = synth_dir / f"{pair.stem}_{H_lat}x{W_lat}_anima.npz"
        if out_path.exists() and not overwrite:
            n_skipped += 1
            pbar.set_postfix_str(f"skip {pair.stem}")
            continue

        crossattn_pos, _pooled = load_cached_text_features(
            pair.te_path, variant=variant
        )
        if crossattn_pos is None:
            logger.warning(f"  skip {pair.stem}: no crossattn_emb in TE cache")
            continue
        crossattn_pos = crossattn_pos.to(device=device, dtype=dtype).unsqueeze(0)

        # Per-sample seed deterministic-but-varied across samples — same noise
        # for the same stem on re-runs makes the pool stable for ablations.
        per_seed = (int(seed) * 1_000_003 + sample_idx) & 0x7FFFFFFF

        clean = _denoise_one(
            model,
            crossattn_pos,
            crossattn_neg,
            H_lat=H_lat,
            W_lat=W_lat,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            seed=per_seed,
            device=device,
            dtype=dtype,
        )  # (1, 16, H_lat, W_lat) float32 CPU

        # Original pixel size: H_lat*8, W_lat*8 (matches the VAE downsample).
        H_pix, W_pix = H_lat * 8, W_lat * 8
        key_suffix = f"_{H_lat}x{W_lat}"
        np.savez(
            out_path,
            **{
                f"latents{key_suffix}": clean.squeeze(0).numpy(),  # (16, H_lat, W_lat)
                f"original_size{key_suffix}": np.array([W_pix, H_pix]),
                f"crop_ltrb{key_suffix}": np.array([0, 0, W_pix, H_pix]),
            },
        )
        n_written += 1
        pbar.set_postfix_str(f"{pair.stem} {H_lat}x{W_lat}")

    pbar.close()
    logger.info(
        f"Phase 2 done: wrote {n_written}, skipped {n_skipped} (already cached). "
        f"Output → {synth_dir}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# Driver
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="post_image_dataset/lora",
        help="LoRA cache dir (source TE + real-image latents).",
    )
    parser.add_argument(
        "--synth_dir",
        type=str,
        default="post_image_dataset/distill_mod_synth",
        help="Output dir for synthetic clean latents.",
    )
    parser.add_argument(
        "--qwen3",
        type=str,
        default="models/text_encoders/qwen_3_06b_base.safetensors",
    )
    parser.add_argument(
        "--dit",
        type=str,
        default="models/diffusion_models/anima-base-v1.0.safetensors",
    )
    parser.add_argument("--t5_tokenizer_path", type=str, default=None)
    parser.add_argument(
        "--seq_len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="Uncond TE seq length (default 512; matches CFG-uncond convention).",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="flash",
        help="DiT attention mode for Phase 2 teacher forwards.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=28,
        help="Denoising steps for synthesis (default 28 = Anima production).",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=4.0,
        help="CFG scale for synthesis (default 4.0 = Anima production).",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=1.0,
        help=(
            "Flow-matching sigma shift. Default 1.0 = Anima production env "
            "(configs/base.toml `discrete_flow_shift=1.0`; every DCW/FeRA bench "
            "and `scripts/dcw/measure_bias_args.py`). `inference.py`'s 5.0 default "
            "is upstream cruft that production callers override."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed; per-sample seed = seed * 1_000_003 + sample_idx.",
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=0,
        help="TE cache variant index to use as the conditioning prompt (default v0).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the number of synthetic latents (None = all discovered pairs).",
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Offload N transformer blocks to CPU during synthesis (low-VRAM).",
    )
    parser.add_argument(
        "--skip_uncond",
        action="store_true",
        help="Skip Phase 1 (assume the uncond sidecar already exists).",
    )
    parser.add_argument(
        "--skip_synth",
        action="store_true",
        help="Skip Phase 2 (stage only the uncond sidecar).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-encode the uncond sidecar AND re-synthesize already-present latents.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    synth_dir = Path(args.synth_dir)

    # ── Phase 1 ────────────────────────────────────────────────────────
    uncond_path = cache_dir / UNCOND_TE_FILENAME
    if not args.skip_uncond:
        uncond_path = stage_uncond_sidecar(
            cache_dir,
            args.qwen3,
            args.dit,
            t5_tokenizer_path=args.t5_tokenizer_path,
            seq_len=args.seq_len,
            overwrite=args.overwrite,
        )
    elif not uncond_path.exists():
        raise FileNotFoundError(
            f"--skip_uncond was passed but {uncond_path} doesn't exist. "
            f"Run without --skip_uncond first."
        )

    # ── Phase 2 ────────────────────────────────────────────────────────
    if args.skip_synth:
        logger.info("--skip_synth set; not generating synthetic latents.")
        return

    generate_synthetic_latents(
        cache_dir,
        synth_dir,
        dit_path=args.dit,
        uncond_path=uncond_path,
        attn_mode=args.attn_mode,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        flow_shift=args.flow_shift,
        seed=args.seed,
        variant=args.variant,
        max_samples=args.max_samples,
        blocks_to_swap=args.blocks_to_swap,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
