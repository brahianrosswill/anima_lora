#!/usr/bin/env python
"""DAVE Phase 1 — derive the per-block DC-attenuation mask from the probe.

Desk exercise, no GPU. Reads the Phase-0 ``per_block.npz`` (the (S, L) matrices
of DC cross-seed similarity, AC cross-seed similarity and DC power ratio) and
turns it into a per-block attenuation *weight* ``w(l) ∈ [0, 1]`` — the figure's
"Power Ratio" gate, fit from our own data:

    w_raw(l) = power_ratio(l) · max(0, DC_sim(l) − AC_sim(l))      (early window)
    w(l)     = w_raw(l) / max_l w_raw(l)                            (normalize)

Both factors are averaged over the early σ window (first ⌊S/3⌋ steps, matching
the probe's verdict window). The product self-zeros the two block families DAVE
must *not* touch:

  * **block 0** — DC-heavy by power but its AC is also cross-seed-locked, so the
    gap ≈ 0 kills its weight (attenuating there perturbs a shared signal).
  * **early-mid AC-shared blocks (~1–8)** — large DC−AC gap but ~zero DC power,
    so there is no DC energy to attenuate.

What survives is the late blocks (~19–27) and mid blocks (~9–11), exactly where
the probe localized the targets.

At inference ``--dave`` reads this ``weight`` vector and forms the per-block DC
attenuation factor ``(1 − α_l) = strength · w(l)`` (``--dave_strength`` is the
live sweep knob), so the most-implicated block (w=1) attenuates by ``strength``
and self-zeroed blocks stay no-ops.

    uv run python bench/dave/derive_alpha_mask.py            # latest run → shipped npz
    uv run python bench/dave/derive_alpha_mask.py --probe bench/dave/results/<ts>/per_block.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (anima_lora/)

from library.env import resolve_under_home  # noqa: E402

DEFAULT_OUT = "networks/calibration/dave_alpha.npz"


def _latest_probe() -> Path:
    """Newest ``per_block.npz`` under bench/dave/results/."""
    results = Path(__file__).resolve().parent / "results"
    cands = sorted(results.glob("*/per_block.npz"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise SystemExit(
            "No per_block.npz found under bench/dave/results/. Run "
            "probe_dc_convergence.py first."
        )
    return cands[-1]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--probe",
        default=None,
        help="Path to per_block.npz (default: latest under bench/dave/results/).",
    )
    p.add_argument(
        "--out",
        default=None,
        help=f"Output npz path (default: shipped {DEFAULT_OUT}).",
    )
    p.add_argument(
        "--early_frac",
        type=float,
        default=1.0 / 3.0,
        help="Fraction of leading steps that count as the early window.",
    )
    opts = p.parse_args()

    probe_path = Path(opts.probe) if opts.probe else _latest_probe()
    d = np.load(probe_path)
    dc, ac, pr = d["dc_sim"], d["ac_sim"], d["power_ratio"]
    heavy = set(int(b) for b in d["heavy_blocks"])
    S, L = dc.shape
    early = max(1, int(round(S * opts.early_frac)))

    pow_e = np.nanmean(pr[:early], axis=0)              # (L,)
    gap_e = np.nanmean(dc[:early] - ac[:early], axis=0)  # (L,)
    w_raw = pow_e * np.clip(gap_e, 0.0, None)
    wmax = float(w_raw.max())
    weight = w_raw / wmax if wmax > 0 else np.zeros_like(w_raw)

    print(f"probe : {probe_path}")
    print(f"shape : S={S} steps, L={L} blocks, early window = first {early} steps\n")
    print("block | pow_e | gap_e  | weight  | note")
    print("-" * 52)
    for b in range(L):
        note = []
        if b in heavy:
            note.append("DC-heavy")
        if weight[b] < 0.05:
            note.append("self-zeroed")
        print(
            f"{b:5d} | {pow_e[b]:.3f} | {gap_e[b]:+.3f} | {weight[b]:.4f} "
            f"| {', '.join(note)}"
        )

    top = np.argsort(weight)[::-1][:6]
    print(f"\ntop-6 targets (block:weight): "
          f"{', '.join(f'{int(b)}:{weight[b]:.2f}' for b in top)}")

    # Sanity gates the README calls out: block 0 must self-zero (gap≈0), and the
    # peak should land in the late blocks, not block 0.
    if weight[0] >= 0.1:
        print(f"\n⚠️  block 0 weight = {weight[0]:.3f} (expected ≈0 — gap should kill it)")
    peak = int(np.argmax(weight))
    if peak < L // 2:
        print(f"\n⚠️  peak block {peak} is in the first half — expected a late block")
    else:
        print(f"\npeak block = {peak} (late ✓), block 0 = {weight[0]:.4f} (self-zeroed ✓)")

    out = Path(resolve_under_home(opts.out)) if opts.out else Path(resolve_under_home(DEFAULT_OUT))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        weight=weight.astype(np.float32),
        power_ratio_early=pow_e.astype(np.float32),
        gap_early=gap_e.astype(np.float32),
        heavy_blocks=np.array(sorted(heavy), dtype=np.int64),
        early_steps=np.int64(early),
        num_blocks=np.int64(L),
        source_probe=str(probe_path),
    )
    print(f"\n→ wrote {out}")


if __name__ == "__main__":
    main()
