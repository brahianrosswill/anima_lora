"""Self-contained PiD-Qwen 4-step pixel-diffusion decoder core.

No hydra / imaginaire / gemma. Everything here is derived from the live
nv-tlabs PiD `qwenimage` 2kto4k checkpoint config (captured by introspection):

  * NET_KWARGS         — exact PidNet constructor args
  * STUDENT_T_LIST etc — the distilled 4-step SDE schedule
  * gemma is replaced by ZERO caption embeddings: the distill path uses no CFG
    and the net's y_embedder is RMSNorm-fronted, so zeros are safe and the
    ~5GB gemma download is skipped entirely.

The PiD net consumes a *normalized* Qwen latent (LQ_latent) and emits RGB pixels
directly — there is NO VAE decode at the end, and the Qwen VAE is not needed at
all (we feed the latent straight in). Output spatial size = latent_grid * 8 *
sr_scale(=4).
"""

from __future__ import annotations

import torch

from .pid_net import PidNet

# ---- Exact PidNet constructor kwargs (introspected from the live qwenimage ckpt) ----
NET_KWARGS = dict(
    in_channels=3, num_groups=24, hidden_size=1536, pixel_hidden_size=16,
    pixel_attn_hidden_size=1152, pixel_num_groups=16, patch_depth=14, pixel_depth=2,
    num_text_blocks=4, patch_size=16, txt_embed_dim=2304, txt_max_length=300,
    use_text_rope=True, text_rope_theta=10000.0, rope_mode="ntk_aware",
    rope_ref_h=1024, rope_ref_w=1024, repa_encoder_index=6, enable_ed=False,
    ed_compress_ratio=1, ed_depth_per_stage=1, ed_window_size=2, ed_num_heads=None,
    ed_hidden_size=None, ed_use_token_shuffle=True, lq_inject_mode="controlnet",
    lq_in_channels=0, lq_latent_channels=16, lq_hidden_dim=512, lq_num_res_blocks=4,
    lq_gate_type="sigma_aware_per_token_per_dim", lq_interval=2, zero_init_lq=True,
    train_lq_proj_only=False, sr_scale=4, latent_spatial_down_factor=8,
    pit_lq_inject=False, pit_lq_gate_type="sigma_aware_per_token_per_dim",
)

SR_SCALE = 4
VAE_DOWN = 8           # latent grid -> vae-native pixels
MODEL_MAX_LENGTH = 300
CAPTION_CHANNELS = 2304
FM_TIMESCALE = 1000.0
STUDENT_T_LIST = [0.999, 0.866, 0.634, 0.342, 0.0]  # 4-step SDE schedule
STUDENT_SAMPLE_STEPS = 4

# Qwen-Image VAE per-channel latent normalization (== ComfyUI Wan21 format,
# scale_factor 1.0; == anima_lora qwen_vae.latents_mean/std).
QWEN_LATENTS_MEAN = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
QWEN_LATENTS_STD = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]


def build_pid_net(device="cuda", dtype=torch.bfloat16) -> PidNet:
    net = PidNet(**NET_KWARGS)
    net = net.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return net


def load_pid_weights(net: PidNet, ckpt_path: str) -> None:
    """Load consolidated PiD checkpoint. The official `model_ema_bf16.pth` stores
    keys under a `net.` prefix (PixelDiTModel.state_dict(prefix='net.')); strip it.
    Also tolerates a bare-key state dict and a {'state_dict'/'model': ...} wrapper."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if any(k.startswith("net.") for k in sd):
        sd = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    return missing, unexpected


def comfy_latent_to_lq(samples: torch.Tensor, device, dtype=torch.bfloat16) -> torch.Tensor:
    """ComfyUI LATENT['samples'] (raw Qwen VAE latent, 4D or 5D) -> PiD LQ_latent
    (per-channel normalized (mu-mean)/std), 4D (B,16,h,w)."""
    x = samples
    if x.ndim == 5:               # (B,C,T,h,w) -> drop singleton frame
        x = x[:, :, 0]
    mean = torch.tensor(QWEN_LATENTS_MEAN, device=x.device, dtype=torch.float32).view(1, 16, 1, 1)
    std = torch.tensor(QWEN_LATENTS_STD, device=x.device, dtype=torch.float32).view(1, 16, 1, 1)
    lq = (x.float() - mean) / std
    return lq.to(device=device, dtype=dtype)


# torch.compile cache: (id(net), H, W) -> compiled module. One graph per output
# resolution; with tiling on, all tiles share one (H, W) so it compiles once.
_COMPILE_CACHE: dict = {}


def get_runner(net: PidNet, H: int, W: int, dtype, enable: bool):
    """Return the net to call in the sample loop: a torch.compile-wrapped net
    (built once per (H, W) and cached) when `enable`, else the eager net.

    Mirrors PiD's official `_maybe_compile_net`: precompute all positional caches
    for this fixed (H, W, text_length) FIRST so the compiled forward only hits
    cache-return branches (no RoPE recompute / dict mutation -> no graph breaks)."""
    if not enable:
        return net
    key = (id(net), int(H), int(W))
    runner = _COMPILE_CACHE.get(key)
    if runner is None:
        device = next(net.parameters()).device
        net.precompute_positional_caches(
            image_height=int(H), image_width=int(W),
            text_length=MODEL_MAX_LENGTH, device=device, pixel_dtype=dtype,
        )
        runner = torch.compile(net, mode="default", dynamic=False)
        _COMPILE_CACHE[key] = runner
    return runner


def _t_list(num_steps: int, device) -> torch.Tensor:
    full = torch.tensor(STUDENT_T_LIST, device=device, dtype=torch.float32)
    if num_steps == STUDENT_SAMPLE_STEPS:
        return full
    idx = torch.linspace(0, len(full) - 1, num_steps + 1).round().long()
    return full[idx]


@torch.no_grad()
def pid_decode_latent(net: PidNet, lq_latent: torch.Tensor, *, steps: int = 4,
                      sigma: float = 0.0, seed: int = 0, dtype=torch.bfloat16,
                      compile: bool = False, _runner=None) -> torch.Tensor:
    """Run the 4-step SDE student. lq_latent: (B,16,h,w) normalized.
    Returns pixels (B,3,H,W) in [-1,1] with H=h*8*4, W=w*8*4.

    `compile=True` torch.compiles the net (cached per output resolution; first
    call per (H,W) is slow). `_runner` lets callers (e.g. the tiled path) pass a
    pre-built compiled net so all same-size tiles share one graph."""
    device = lq_latent.device
    B, _, lh, lw = lq_latent.shape
    H, W = lh * VAE_DOWN * SR_SCALE, lw * VAE_DOWN * SR_SCALE
    run = _runner if _runner is not None else get_runner(net, H, W, dtype, compile)

    cap = torch.zeros(B, MODEL_MAX_LENGTH, CAPTION_CHANNELS, device=device, dtype=dtype)
    deg = torch.full((B,), float(sigma), device=device, dtype=torch.float32)
    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(B, 3, H, W, device=device, dtype=torch.float32, generator=gen)

    tl = _t_list(steps, device)
    autocast = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else torch.autocast("cpu", dtype=dtype)
    with autocast:
        for t_cur, t_next in zip(tl[:-1], tl[1:]):
            t_scaled = t_cur.expand(B) * FM_TIMESCALE
            v = run(x.to(dtype), t_scaled, cap, lq_latent=lq_latent, degrade_sigma=deg)
            s = [B] + [1] * (x.ndim - 1)
            t_c = t_cur.double().view(*s)
            x0 = (x.double() - t_c * v.double())  # velocity -> x0
            if t_next.item() > 0:
                eps = torch.randn(x0.shape, device=device, dtype=torch.float64, generator=gen)
                t_n = t_next.double().view(1).expand(s)
                x = ((1.0 - t_n) * x0 + t_n * eps).float()
            else:
                x = x0.float()
    return x.clamp(-1, 1)


def _tile_positions(dim: int, tile: int, stride: int):
    """Start indices so every tile is exactly `tile` wide (last snapped to edge)."""
    if dim <= tile:
        return [0], dim
    pos = list(range(0, dim - tile + 1, stride))
    if pos[-1] != dim - tile:
        pos.append(dim - tile)
    return pos, tile


def _feather_1d(n: int, overlap_px: int, device) -> torch.Tensor:
    """Linear ramp from a small floor->1 over `overlap_px` at each end, 1 in the
    middle. Floor>0 so single-coverage border pixels normalize cleanly."""
    w = torch.ones(n, device=device, dtype=torch.float32)
    if overlap_px > 0:
        ramp = torch.linspace(1.0 / (overlap_px + 1), 1.0, overlap_px, device=device)
        k = min(overlap_px, n)
        w[:k] = torch.minimum(w[:k], ramp[:k])
        w[-k:] = torch.minimum(w[-k:], ramp.flip(0)[:k])
    return w


@torch.no_grad()
def pid_decode_latent_tiled(net: PidNet, lq_latent: torch.Tensor, *, steps: int = 4,
                            sigma: float = 0.0, seed: int = 0, tile: int = 64,
                            overlap: int = 16, dtype=torch.bfloat16,
                            compile: bool = False) -> torch.Tensor:
    """Tiled SR decode for latents larger than `tile` (memory bound). Decodes
    overlapping latent tiles and feather-blends them in pixel space. Output
    (B,3, h*32, w*32) in [-1,1]."""
    device = lq_latent.device
    B, _, Hh, Ww = lq_latent.shape
    up = VAE_DOWN * SR_SCALE  # 32
    Hout, Wout = Hh * up, Ww * up
    stride = max(1, tile - overlap)
    ys, th = _tile_positions(Hh, tile, stride)
    xs, tw = _tile_positions(Ww, tile, stride)
    ov_px = overlap * up

    acc = torch.zeros(B, 3, Hout, Wout, device=device, dtype=torch.float32)
    wsum = torch.zeros(1, 1, Hout, Wout, device=device, dtype=torch.float32)
    wy = _feather_1d(th * up, ov_px, device)
    wx = _feather_1d(tw * up, ov_px, device)
    wmask = (wy[:, None] * wx[None, :])[None, None]  # (1,1,th*32,tw*32)

    # All tiles share one fixed output size -> compile once, reuse for every tile.
    runner = get_runner(net, th * up, tw * up, dtype, compile)

    n = 0
    for yi in ys:
        for xi in xs:
            tile_lq = lq_latent[..., yi:yi + th, xi:xi + tw]
            px = pid_decode_latent(net, tile_lq, steps=steps, sigma=sigma, seed=seed + n,
                                   dtype=dtype, _runner=runner)
            py, px_ = yi * up, xi * up
            acc[..., py:py + th * up, px_:px_ + tw * up] += px.float() * wmask
            wsum[..., py:py + th * up, px_:px_ + tw * up] += wmask
            n += 1
    return (acc / wsum.clamp(min=1e-6)).clamp(-1, 1)
