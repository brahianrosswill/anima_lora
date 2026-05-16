# Chimera / Custom Autograd Problems

Audit date: 2026-05-16

Scope: `networks/lora_modules/chimera.py`, `networks/lora_modules/custom_autograd.py`, and the surrounding LoRA factory/config/test wiring.

## P1: Chimera inference materializes large fp32 branch outputs

`ChimeraHydraInferenceModule.forward` runs both down projections in fp32 and combines both up pools in fp32:

- `x_lora.float() @ lora_down_c.weight.float()`
- `x_lora.float() @ lora_down_f.weight.float()`
- `einsum(..., lora_up_*_weight.float())`
- two `bmm` calls producing `out_c` and `out_f`

Relevant code:

- `networks/lora_modules/chimera.py:517-523`
- `networks/lora_modules/chimera.py:537-543`
- `networks/lora_modules/chimera.py:545-553`

The transient output tensors are full hidden-size tensors. On the default DiT shape, an MLP `layer1` branch output is roughly:

```text
B * 4096 tokens * 8192 out_features * 4 bytes ~= 128 MiB per batch item
```

The inference path creates two such branch outputs before summing. This is not saved-for-backward activation VRAM, but it is still peak memory and bandwidth pressure.

Fix options:

1. Accumulate one branch into the other with `baddbmm` or an equivalent in-place accumulation shape, avoiding simultaneous `out_c` and `out_f` full tensors.
2. Benchmark bf16 branch math for inference after load. The training path already casts the downstream Chimera chain to bf16; inference currently forces fp32 more broadly.
3. Keep fp32 as a compatibility fallback if image quality changes.

## P1: Chimera training still pays dual full-output temporaries

Custom down autograd reduces the saved input cost, but Chimera still computes two independent branch outputs:

```python
out_c = torch.bmm(lx_c_3d, P_combined_c.transpose(1, 2))
out_f = torch.bmm(lx_f_3d, P_combined_f.transpose(1, 2))
out = (out_c * scale_c + out_f * scale_f).reshape(...)
```

Relevant code:

- `networks/lora_modules/chimera.py:381-390`

The branch outputs are full hidden-size tensors. This is the next likely adapter-side peak after the down-projection saved-input issue.

Fix:

Use an accumulation form so only one full branch output is live at a time. Validate forward and gradient equality against the current implementation before profiling.

## P2: Content router pooling is repeated per adapted Linear

The content router pools rank-space activations with:

```python
pooled = lx_c.reshape(B, -1, r).pow(2).mean(dim=1).sqrt()
```

Relevant code:

- `networks/lora_modules/chimera.py:269-285`

This is rank-sized, so it is not the main VRAM issue. Still, on the default Chimera target regex, every MLP `layer1` / `layer2` Chimera module repeats a full-token reduction. If router overhead shows up in profiler traces, this is the small repeated compute to inspect.

Fix:

Profile before changing. If it matters, test whether pooling can be fused with the down-projection output lifecycle or computed in a cheaper dtype without changing router behavior.

## P2: Docs/config comments are stale

`use_custom_down_autograd` is now default-on:

- `configs/base.toml:47`

But `docs/optimizations/for_compile.md` still says the flag is default-off and only lists LoRA, Hydra, Ortho, and OrthoHydra as wired:

- `docs/optimizations/for_compile.md:275-283`

The GUI Chimera config also says the FreqRouter has 16-dim sigma features, but sets `sigma_feature_dim = 0`:

- `configs/gui-methods/chimera_hydra.toml:58-63`

Fix:

Update the docs and comments so they reflect the current scaled-path behavior (channel-scale fix landed; Chimera is now covered by `use_custom_down_autograd`).

## Validation already run

```text
uv run pytest tests/test_lora_custom_autograd.py tests/test_chimera_router_stats.py tests/test_chimera_node_loader.py -q
30 passed
```

The custom-autograd suite now covers the scaled (channel-scale) path and Chimera custom-autograd equality (with and without channel_scale).
