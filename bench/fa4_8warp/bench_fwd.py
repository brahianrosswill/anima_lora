"""FA4 SM120 8-MMA-warp microbench vs 4-warp vs FA2, at Anima's real shapes.

Context: LingYeAI flash-attention PR #2599 scales the SM120 TMA forward kernel
from 4->8 MMA warps to close the causal/non-causal perf gap. Their sweep was
B=1 H=32 D=128, seqlen 8k..128k. Anima runs H=16 D=128 at seqlen 4096 and is
**non-causal** (image DiT) — below their entire sweep, so the win is untested
on our workload. This isolates the kernel-throughput question from the
torch.compile graph-break friction that sank FA4 for us in 2026-04
(docs/optimizations/fa4.md).

Backend selected by env BACKEND={fa2,fa4}:
  fa2 -> flash_attn.flash_attn_func (installed 2.8.3 C++ path; the FA2 baseline)
  fa4 -> flash_attn.cute.flash_attn_func (CuTe DSL; run with PYTHONPATH=<fork>
         on the 8-warp branch vs its 4-warp parent)

Results appended to bench/fa4_8warp/results.jsonl keyed by RESULT_LABEL.
"""

import os
import json
import time
import torch

BACKEND = os.environ.get("BACKEND", "fa2").lower()
MODE = os.environ.get("MODE", "fwd").lower()   # fwd | bwd
LABEL = os.environ.get("RESULT_LABEL", BACKEND)
DTYPE = torch.bfloat16
DEV = "cuda"
WARMUP = int(os.environ.get("WARMUP", "20"))   # FA4 JITs per shape on first call
ITERS = int(os.environ.get("ITERS", "100"))

# Anima DiT: 16 heads x head_dim 128 (model_channels 2048). Self-attn is 4096
# image tokens, non-causal. Cross-attn q=4096 against 64..512 text tokens.
N_HEADS = 16
HEAD_DIM = 128

# (name, B, seq_q, seq_kv, causal)
CONFIGS = [
    ("self_4096_noncausal_b2", 2, 4096, 4096, False),  # the real hot path
    ("self_4096_noncausal_b1", 1, 4096, 4096, False),
    ("self_4096_causal_b2",    2, 4096, 4096, True),    # ref: gap the PR targets
    ("cross_4096x512_b2",      2, 4096, 512,  False),
    ("cross_4096x256_b2",      2, 4096, 256,  False),
    ("cross_4096x128_b2",      2, 4096, 128,  False),
    ("cross_4096x64_b2",       2, 4096, 64,   False),
]


def get_attn():
    if BACKEND == "fa2":
        from flash_attn import flash_attn_func

        def run(q, k, v, causal):
            return flash_attn_func(q, k, v, causal=causal)

        return run
    elif BACKEND == "fa4":
        from flash_attn.cute import flash_attn_func

        def run(q, k, v, causal):
            out = flash_attn_func(q, k, v, causal=causal)
            return out[0] if isinstance(out, tuple) else out

        return run
    raise SystemExit(f"unknown BACKEND={BACKEND}")


def _time(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def bench_one(run, B, sq, skv, causal):
    rg = MODE == "bwd"
    q = torch.randn(B, sq, N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV, requires_grad=rg)
    k = torch.randn(B, skv, N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV, requires_grad=rg)
    v = torch.randn(B, skv, N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV, requires_grad=rg)

    # forward FLOPs (causal ~halves the score/AV work)
    fwd_flops = 2 * 2 * B * N_HEADS * sq * skv * HEAD_DIM
    if causal:
        fwd_flops *= 0.5

    if MODE == "fwd":
        ms = _time(lambda: run(q, k, v, causal))
        return ms, fwd_flops / (ms / 1000) / 1e12

    # bwd: forward once (retain graph), then time pure backward via autograd.grad
    out = run(q, k, v, causal)
    grad_out = torch.randn_like(out)
    ms = _time(lambda: torch.autograd.grad(
        out, (q, k, v), grad_out, retain_graph=True))
    # backward attention is ~2.5x forward FLOPs (dQ, dK, dV)
    tflops = (2.5 * fwd_flops) / (ms / 1000) / 1e12
    return ms, tflops


def main():
    print(f"BACKEND={BACKEND}  MODE={MODE}  LABEL={LABEL}  "
          f"dev={torch.cuda.get_device_name()}  cap={torch.cuda.get_device_capability()}")
    run = get_attn()
    rows = []
    for name, B, sq, skv, causal in CONFIGS:
        ms, tflops = bench_one(run, B, sq, skv, causal)
        print(f"  {name:28s}  {ms:8.4f} ms   {tflops:7.1f} TFLOPS")
        rows.append(dict(config=name, B=B, seq_q=sq, seq_kv=skv, causal=causal,
                         ms=round(ms, 5), tflops=round(tflops, 2)))

    rec = dict(label=LABEL, backend=BACKEND, mode=MODE,
               device=torch.cuda.get_device_name(),
               heads=N_HEADS, head_dim=HEAD_DIM, dtype="bf16",
               warmup=WARMUP, iters=ITERS, ts=time.time(), rows=rows)
    out = os.path.join(os.path.dirname(__file__), "results.jsonl")
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"appended -> {out}")


if __name__ == "__main__":
    main()
