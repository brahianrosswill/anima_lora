# FA4 SM120 8-MMA-warp — backward bench plan

Goal: get the FA4 (CuTe DSL) **backward** kernel running on consumer Blackwell
(sm_120, RTX 5070 Ti) under the maintained `nvidia-cutlass-dsl` (4.5.2), then
bench it against FA2 — completing the picture the forward microbench started.

## What we already know (forward, done)

Measured on RTX 5070 Ti, Anima shapes (16 heads × head_dim 128, **non-causal**),
`bench/fa4_8warp/bench_fwd.py`, results in `results.jsonl`:

- **Forward, self-attn 4096 non-causal B2:** FA2 3.24 ms → FA4 4-warp 3.09 → **FA4 8-warp 2.91 ms (−10.3% vs FA2)**. The PR's own delta (8 vs 4 warps) is ~6–8%.
- Forward parity vs fp32 SDPA: clean (max_abs 0.001–0.008, no NaN).
- **Backward: blocked.** FA2 bwd works (self 4096 nc B2 = 7.50 ms, ~2.3× fwd). FA4 bwd does **not** run on sm_120 + cutlass-dsl 4.5.2 — two independent failures (below).

Caveat to keep in mind: backward is ~2.3× forward, so even an 11% forward win
dilutes to ~3% of a training step before the torch.compile graph-break tax.
The realistic payoff is **inference** (forward only). Backward bench is to
*confirm* whether training is worth revisiting at all.

## Repos / homes

- **LingYeAI fork (the PR, newer upstream base):** `https://github.com/LingYeAI/flash-attention.git`
  - branch `sm120-tma-num-mma-warps-8`, HEAD `44779ed` = **8 MMA warps** (the PR)
  - parent `b5c0ffc` = 4 MMA warps (pre-PR baseline)
  - Forward already works + is correct on sm_120. Backward hits failure (2) below.
- **Our old dz fork (has the working-ish SM120 backward, older upstream base):**
  `https://github.com/sorryhyun/flash-attention-sm120-fix.git`, branch
  `dz/sm120_tma_optimized`, checked out locally at `/home/sorryhyun/anima/flash-attention`.
  - Key SM120 fix commits to mine for backward logic:
    - `c828d4d` — SM120 fwd TMA O-store crash **+ backward `dQ_single_wg` unbound** fix
    - `aa99a98` — SM120 NaN loss (exclude stmatrix for SM80 MMA register layout)
  - Backward hits failure (1) below on current cutlass-dsl.

Current state of the local clone: it is the dz fork **with the `lingye` remote
already added and `sm120-tma-num-mma-warps-8` fetched**. So both lineages are
reachable from `/home/sorryhyun/anima/flash-attention` — no second clone needed.

## The two backward failures (root-caused)

**(1) dz fork — `atomicrmw() missing 1 required positional argument: 'res'`**
- Site: `flash_attn/cute/utils.py::atomic_add_fp32` (~line 394), call
  `nvvm.atomicrmw(op=..., ptr=..., a=Float32(a).ir_value())`.
- cutlass-dsl 4.5.2 signature changed to:
  `atomicrmw(res, op, ptr, a, *, b=None, mem_order=None, syncscope=None, ...)`
  (`.../cutlass/_mlir/dialects/_nvvm_ops_gen.py:178`). `res` is the **MLIR result
  type** of the atomic (f32 here — atomicrmw returns the prior value).
- **Fix:** pass `res=<f32 IR type>` as the first positional arg, e.g. the type of
  `Float32(a).ir_value()`. Cross-check how cutlass-dsl 4.5.2's own code / examples
  construct the result-type arg for `nvvm.atomicrmw` and mirror it. ~1 line.

**(2) lingye branch — `None to Float conversion is not supported`**
- Only surfaces after porting the dz `dQ_single_wg = False` fix (see below);
  before that it's `UnboundLocalError: dQ_single_wg`.
- Site: `flash_attn/cute/flash_bwd.py:~473` `.launch(...)`, during kernel-arg
  marshalling — a scalar passed into the bwd kernel is `None` where a `Float` is
  required. Prime suspects: `softmax_scale`, `softcap`, or a window/`local` scalar
  not threaded through `FlashAttnFunc.backward` → `_flash_attn_bwd` for the sm120
  path. (Forward sets `scale = softmax_scale or head_dim**-0.5`; the bwd path may
  not re-derive it.)
- **Debug:** in `_flash_attn_bwd` (interface.py) print the scalar args just before
  the sm120 launch, find which is `None`, trace where the other arches set it, set
  it for `arch // 10 == 12`. Likely 1–3 lines.

## Plan

Pick **lingye `sm120-tma-num-mma-warps-8` as the base** (current upstream, forward
works + correct, carries the 8-warp PR) and port the dz SM120 backward fixes onto
it. Rationale: the dz fork is on much older upstream and would need more API-drift
fixes than just `atomicrmw`; lingye is one `dQ_single_wg` line + one None-scalar
away from running.

1. **Branch.** In `/home/sorryhyun/anima/flash-attention`:
   `git stash` (preserve the local `utils.py` mod), `git checkout -b
   fa4-sm120-bwd-fix lingye/sm120-tma-num-mma-warps-8`. Commit fixes here so the
   work is recoverable (the bench so far used throwaway working-tree edits).
2. **Re-apply the import guard** so `import flash_attn` works without the built C++
   ext (the cute path doesn't need it):
   wrap the `from flash_attn.flash_attn_interface import (...)` block in
   `flash_attn/__init__.py` in `try/except Exception: pass`.
3. **Port fix (a): `dQ_single_wg`.** In `interface.py`, in the `arch // 10 == 12`
   backward block (right after the `deterministic`/`mask_mod` asserts, ~line 1314),
   add `dQ_single_wg = False`. (Mirrors dz `c828d4d`.)
4. **Debug fix (b): None→Float.** Find the None scalar at the `flash_bwd.py:473`
   launch (print args in `_flash_attn_bwd`), set it for the sm120 path. Re-derive
   `softmax_scale = head_dim**-0.5` if that's the culprit.
5. **If the lingye bwd still uses `atomic_add_fp32`** and hits failure (1), apply
   the `res=` fix from §(1) to `utils.py` too. (lingye errored at launch, not
   atomicrmw, so this may not be needed — confirm.)
6. **Parity-gate the backward** before trusting timings: compare dQ/dK/dV against
   an fp32 SDPA reference (`torch.autograd.grad`) — max_abs at bf16 ULP level,
   no NaN. The old SM120 fork had a NaN-loss history (dz `aa99a98`); do **not**
   report bwd ms without this check. If NaN/large error, port `aa99a98` (stmatrix
   exclusion) too.
7. **Bench backward** (8-warp vs 4-warp vs FA2). 8-warp should ≈ 4-warp for bwd
   (the PR only touches forward) — running both confirms that and catches regressions.

## Environment recipe (already built; reuse `/tmp/fa4bench`)

The painful bits, captured so this isn't re-derived:

- The project `.venv` (torch 2.12+cu132, flash-attn 2.8.3) is the **FA2 baseline** —
  run FA2 directly with `uv run --no-sync`.
- For FA4: `uv venv /tmp/fa4bench --python 3.13` then
  `uv pip install --python /tmp/fa4bench/bin/python "nvidia-cutlass-dsl[cu13]>=4.4.2" quack-kernels`.
  `quack-kernels` pulls a **working torch 2.12+cu130** into that venv, so it is
  self-sufficient (don't rely on `--system-site-packages`; it inherits the *base*
  interpreter, not the project venv).
- Run FA4 with the fork on the path:
  `PYTHONPATH=/home/sorryhyun/anima/flash-attention /tmp/fa4bench/bin/python ...`
  (PYTHONPATH shadows the installed flash_attn; cutlass resolves from the venv's
  nested `nvidia_cutlass_dsl/python_packages` via its `.pth`).
- cutlass JIT-compiles per shape on first call → keep `WARMUP` ≥ 15–30.

## Bench commands

```bash
cd /home/sorryhyun/anima/anima_lora
FORK=/home/sorryhyun/anima/flash-attention

# FA2 backward (project venv)
BACKEND=fa2 MODE=bwd RESULT_LABEL=fa2_2.8.3 ITERS=50 WARMUP=15 \
  uv run --no-sync python bench/fa4_8warp/bench_fwd.py

# FA4 backward (on the fa4-sm120-bwd-fix branch, guard applied)
PYTHONPATH=$FORK BACKEND=fa4 MODE=bwd RESULT_LABEL=fa4_8warp ITERS=50 WARMUP=15 \
  /tmp/fa4bench/bin/python bench/fa4_8warp/bench_fwd.py
# then checkout b5c0ffc (4-warp) and re-run with RESULT_LABEL=fa4_4warp
```

## Success criteria

- FA4 sm120 backward runs without crashing, on the 8-warp and 4-warp branches.
- Backward dQ/dK/dV parity vs fp32 SDPA: no NaN, max_abs at bf16 ULP level.
- `results.jsonl` has `mode=bwd` rows for `fa2_2.8.3`, `fa4_8warp`, `fa4_4warp`.
- Conclusion: FA4-vs-FA2 **full step** (fwd+bwd) delta on the hot path → decides
  whether training-time FA4 is worth the (separate) torch.compile integration work.

## Results (DONE — 2026-06-02, RTX 5070 Ti, bf16)

All success criteria met. Backward now runs on sm_120 + cutlass-dsl 4.5.2.

**Fixes (committed to branch `fa4-sm120-bwd-fix`, off `lingye/sm120-tma-num-mma-warps-8`):**
- `interface.py`: `dQ_single_wg = False` in the `arch//10==12` bwd block (was unbound).
- `flash_bwd.py`: stop clobbering the raw `softmax_scale` — `compute_softmax_scale_log2`
  nulls it for the `score_mod=None` path, but the SM80/SM120 bwd kernel scales dK by the
  raw scale in its epilogue (line ~845), so `None` crashed the launch (None→Float).
- `__init__.py`: try/except guard around the C++ `flash_attn_interface` import.
- The dz `atomicrmw res=` fix (failure §1) was **already present** on lingye — not re-applied.
- No NaN → the `aa99a98` stmatrix exclusion was **not** needed.
- 4-warp parity branch: `fa4-sm120-bwd-fix-4warp` (off `b5c0ffc`, cherry-picked the fix).

**Parity** (B2 S512 H16 D128 nc, vs fp32 SDPA): dQ/dK/dV max_abs 0.003–0.005, no NaN.

**Backward ms** (self_4096_nc_b2, hot path): FA2 **7.50** · FA4 4-warp **8.54** · FA4 8-warp **9.18**.
- FA4 bwd is **+18–24% slower than FA2** across self-attn shapes; 8w≈4w (bwd kernel is
  byte-identical — the PR diff is forward-only — the 7% spread is run-to-run/clock noise).

**Conclusion — full step (fwd+bwd), self_4096_nc_b2:**
- FA2: 3.24 + 7.50 = **10.74 ms**
- FA4 8-warp: 2.91 + 9.18 = **12.09 ms** → **+12.6% slower per step.**
- The −10% forward win is more than erased by the +22% backward loss. **Training-time FA4 is
  not worth the torch.compile integration work.** The forward-only inference win stands as the
  sole payoff (a separate integration question).

## Cleanup discipline

- The local fork is a scratch tree with a real local mod (`flash_attn/cute/utils.py`).
  Preserve it: `git stash` before checkouts, `git stash pop` after; restore
  `flash_attn/__init__.py` (drop the guard) when done.
- Commit the actual fixes to the `fa4-sm120-bwd-fix` branch so they survive (push
  to a fork if worth keeping). `/tmp/fa4bench` is ephemeral — recreate from the
  recipe above if gone.
