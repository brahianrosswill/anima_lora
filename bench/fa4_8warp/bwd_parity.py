"""Minimal FA4 SM120 backward smoke + parity gate.

Runs ONE small shape through the cute (FA4) backward, compares dQ/dK/dV against
an fp32 SDPA reference (torch.autograd.grad). Fast iteration target while
debugging the sm120 bwd launch — single JIT, not the 7-shape bench sweep.

  PYTHONPATH=<fork> /tmp/fa4bench/bin/python bench/fa4_8warp/bwd_parity.py
"""

import torch
import torch.nn.functional as F
from flash_attn.cute import flash_attn_func

torch.manual_seed(0)
DEV = "cuda"
B, S, H, D = 2, 512, 16, 128  # small seq for fast JIT; real Anima is S=4096

q = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
g = torch.randn(B, S, H, D, device=DEV, dtype=torch.bfloat16)

# FA4 forward + backward
out = flash_attn_func(q, k, v, causal=False)
out = out[0] if isinstance(out, tuple) else out
dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
print("FA4 backward ran.")
print("  out   nan?", torch.isnan(out).any().item())
print("  dq/dk/dv nan?", [torch.isnan(t).any().item() for t in (dq, dk, dv)])

# fp32 SDPA reference (B,H,S,D layout)
qf = q.detach().float().transpose(1, 2).requires_grad_(True)
kf = k.detach().float().transpose(1, 2).requires_grad_(True)
vf = v.detach().float().transpose(1, 2).requires_grad_(True)
ref = F.scaled_dot_product_attention(qf, kf, vf, is_causal=False)
gf = g.float().transpose(1, 2)
rdq, rdk, rdv = torch.autograd.grad(ref, (qf, kf, vf), gf)
rdq, rdk, rdv = (t.transpose(1, 2) for t in (rdq, rdk, rdv))

for name, a, b in [("dQ", dq, rdq), ("dK", dk, rdk), ("dV", dv, rdv)]:
    a = a.float()
    err = (a - b).abs()
    print(f"  {name}: max_abs={err.max().item():.4f}  mean_abs={err.mean().item():.5f}")

ofwd = out.float()
oref = F.scaled_dot_product_attention(
    q.detach().float().transpose(1, 2), k.detach().float().transpose(1, 2),
    v.detach().float().transpose(1, 2), is_causal=False).transpose(1, 2)
print(f"  fwd out: max_abs={(ofwd - oref).abs().max().item():.4f}")
