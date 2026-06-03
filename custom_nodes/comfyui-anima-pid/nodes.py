"""ComfyUI nodes: NVIDIA PiD pixel-diffusion decoder for Anima / Qwen-Image latents.

Two nodes:
  * AnimaPiDLoader  — load a PiD qwenimage checkpoint -> ANIMA_PID socket
  * AnimaPiDDecode  — LATENT (+ PiD model) -> IMAGE, 4x super-resolved

PiD REPLACES VAE Decode: it consumes the (normalized) Qwen latent and emits RGB
pixels directly, upscaling 4x in the same pass (latent_grid*8 -> *4). The gemma
text encoder is skipped entirely (zero caption embeddings — see pid_core), so no
~5GB download and no prompt input. Drop AnimaPiDDecode where VAEDecode was:

    checkpoint -> KSampler -> LATENT ─┐
                                      ├─► AnimaPiDDecode ─► IMAGE (4x) -> SaveImage
    AnimaPiDLoader (PiD .pth) ────────┘

The official 2k->4k 4-step checkpoint auto-downloads from the public nvidia/PiD
repo on first use (select the "(auto-download)" entry in AnimaPiDLoader) into
ComfyUI/models/pid/. To use your own, drop a .pth/.safetensors there and pick it
from the dropdown. Weights are NVIDIA NSCLv1 (non-commercial).
"""

import os
import shutil

import torch

import comfy.model_management as mm
import folder_paths
from comfy.utils import ProgressBar

from .pid_core import (
    SR_SCALE,
    VAE_DOWN,
    build_pid_net,
    comfy_latent_to_lq,
    count_tiles,
    load_pid_weights,
    pid_decode_latent,
    pid_decode_latent_tiled,
)

# Register a ComfyUI models/pid folder for PiD checkpoints (.pth / .safetensors).
_PID_DIR = os.path.join(folder_paths.models_dir, "pid")
os.makedirs(_PID_DIR, exist_ok=True)
folder_paths.add_model_folder_path("pid", _PID_DIR)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Official 4-step qwenimage 2k->4k checkpoint on the (ungated, public) nvidia/PiD
# repo. Auto-fetched into models/pid/ on first use, flattened + renamed to a flat
# filename so the dropdown contract stays uniform.
_HF_PID_REPO = "nvidia/PiD"
_HF_PID_FILE = "checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/model_ema_bf16.pth"
_OFFICIAL_CKPT = "PiD_res2kto4k_sr4x_official_qwenimage_distill_4step.pth"
_AUTODL_SUFFIX = " (auto-download)"


def _download_official_ckpt() -> str:
    """Fetch the official PiD qwenimage checkpoint into models/pid/ (one-time).

    Mirrors the tagger node's ``hf_hub_download`` + flatten pattern: the source
    lives under ``checkpoints/.../model_ema_bf16.pth`` on the repo but is moved to
    a flat, descriptive filename in ``models/pid/`` so the loader's dropdown lists
    it like any hand-placed checkpoint. Returns the local path."""
    dest = os.path.join(_PID_DIR, _OFFICIAL_CKPT)
    if os.path.exists(dest):
        return dest
    from huggingface_hub import hf_hub_download

    print(
        f"[AnimaPiD] fetching {_HF_PID_REPO}/{_HF_PID_FILE} -> {dest} (one-time, ~public download).\n"
        f"[AnimaPiD] NOTE: PiD weights are NVIDIA NSCLv1 — non-commercial (research/evaluation) use only."
    )
    downloaded = hf_hub_download(repo_id=_HF_PID_REPO, filename=_HF_PID_FILE, local_dir=_PID_DIR)
    if os.path.realpath(downloaded) != os.path.realpath(dest):
        shutil.move(downloaded, dest)
    return dest


class AnimaPiDModel:
    """Holder for a loaded PiD net + its compute dtype (ANIMA_PID socket)."""

    def __init__(self, net, dtype):
        self.net = net
        self.dtype = dtype


class AnimaPiDLoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = list(folder_paths.get_filename_list("pid"))
        # Surface the official checkpoint as a selectable entry even when the
        # folder is empty, so a fresh install has something to pick (and trigger
        # the one-time auto-download). Hand-placed files still list normally.
        if _OFFICIAL_CKPT not in files:
            files.insert(0, _OFFICIAL_CKPT + _AUTODL_SUFFIX)
        return {
            "required": {
                "ckpt_name": (files,),
                "dtype": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("ANIMA_PID",)
    RETURN_NAMES = ("pid_model",)
    FUNCTION = "load"
    CATEGORY = "Anima/PiD"

    def load(self, ckpt_name, dtype):
        # The auto-download sentinel ("<official> (auto-download)") resolves to
        # the official checkpoint, fetching it on first use.
        if ckpt_name.endswith(_AUTODL_SUFFIX) or ckpt_name == _OFFICIAL_CKPT:
            path = _download_official_ckpt()
            ckpt_name = _OFFICIAL_CKPT
        else:
            path = folder_paths.get_full_path("pid", ckpt_name)
        if path is None:
            raise FileNotFoundError(
                f"PiD checkpoint {ckpt_name!r} not found under {_PID_DIR}. "
                f"Download nvidia/PiD checkpoints/PiD_res2kto4k_sr4x_official_qwenimage_distill_4step/"
                f"model_ema_bf16.pth and place it there, or select the auto-download entry."
            )
        dt = _DTYPES[dtype]
        device = mm.get_torch_device()
        net = build_pid_net(device, dt)
        missing, unexpected = load_pid_weights(net, path)
        if missing:
            print(f"[AnimaPiD] WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
        if unexpected:
            print(f"[AnimaPiD] note: {len(unexpected)} unexpected keys ignored (e.g. {unexpected[:3]})")
        print(f"[AnimaPiD] loaded {ckpt_name} as {dtype} on {device}")
        return (AnimaPiDModel(net, dt),)


class AnimaPiDDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pid_model": ("ANIMA_PID",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 4, "min": 1, "max": 8}),
                "sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                    "tooltip": "Latent degradation level PiD assumes. 0.0 = clean decode; "
                                               "higher lets PiD synthesize/hallucinate more detail."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "tile_latent": ("INT", {"default": 64, "min": 0, "max": 256, "step": 8,
                                        "tooltip": "0 = decode whole image at once (4K output may OOM on <=16GB). "
                                                   ">0 = tile the latent (each tile -> tile*32 px) with feather "
                                                   "blending. 64 -> 2048px tiles."}),
                "tile_overlap": ("INT", {"default": 16, "min": 0, "max": 64, "step": 4,
                                         "tooltip": "Latent-space overlap between tiles (pixels = overlap*32). "
                                                    "Larger = fewer seams, slower."}),
                "compile": ("BOOLEAN", {"default": False,
                                        "tooltip": "Per-block torch.compile of the PiD net: each transformer block "
                                                   "is compiled as its own small graph (faster compile, fewer graph "
                                                   "breaks than whole-net — mirrors Anima Block Compile). First run "
                                                   "per output size is slow (compilation), then fast; with tiling on "
                                                   "all tiles share one size so the blocks compile once."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "Anima/PiD"

    def decode(self, pid_model, latent, steps, sigma, seed, tile_latent, tile_overlap, compile=False):
        net = pid_model.net
        dt = pid_model.dtype
        device = mm.get_torch_device()

        lq = comfy_latent_to_lq(latent["samples"], device, dt)  # (B,16,h,w) normalized
        lh, lw = lq.shape[-2], lq.shape[-1]
        out_h, out_w = lh * VAE_DOWN * SR_SCALE, lw * VAE_DOWN * SR_SCALE
        print(f"[AnimaPiD] decode latent {lh}x{lw} -> {out_h}x{out_w} ({SR_SCALE}x), "
              f"steps={steps} sigma={sigma} tile={tile_latent or 'off'} compile={compile}")

        use_tiling = bool(tile_latent) and (lh > tile_latent or lw > tile_latent)
        # Drive ComfyUI's node progress bar: one tick per SDE step, summed over
        # tiles. (PiD runs its own sampler loop, so without this the node shows
        # no progress.)
        n_tiles = count_tiles(lq, tile_latent, tile_overlap) if use_tiling else 1
        pbar = ProgressBar(steps * n_tiles)
        step_cb = lambda: pbar.update(1)  # noqa: E731
        if use_tiling:
            px = pid_decode_latent_tiled(
                net, lq, steps=steps, sigma=sigma, seed=seed,
                tile=tile_latent, overlap=tile_overlap, dtype=dt, compile=compile,
                step_cb=step_cb,
            )
        else:
            px = pid_decode_latent(net, lq, steps=steps, sigma=sigma, seed=seed,
                                   dtype=dt, compile=compile, step_cb=step_cb)

        # (B,3,H,W) in [-1,1] -> ComfyUI IMAGE (B,H,W,3) in [0,1]
        img = ((px.float() + 1.0) / 2.0).clamp(0, 1).permute(0, 2, 3, 1).contiguous().cpu()
        return (img,)


NODE_CLASS_MAPPINGS = {
    "AnimaPiDLoader": AnimaPiDLoader,
    "AnimaPiDDecode": AnimaPiDDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaPiDLoader": "Anima PiD Loader",
    "AnimaPiDDecode": "Anima PiD Decode (4x SR)",
}
