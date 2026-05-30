"""GPU-side accumulators + single-sync flush for the turbo training loop.

All accumulators live on-device; they're flushed in one stacked ``.tolist()``
at every ``log_interval`` so per-step CUDA syncs go to zero.

Health-scalar semantics (see proposal log for context):

* ``grad``      — overall DMD2 gradient magnitude into x_pred
* ``dm``        — DM regularizer strength (v_real − v_fake)
* ``cfg``       — CA branch strength (CFG bake direction)
* ``xpred``     — x_pred dispersion: → 0 means collapse to mean, drifting upward
  means student is exploding.
* ``v_student`` — direct student velocity magnitude; runaway student manifests
  here before x_pred_std catches up (x_pred = x_t − t·v_student).

Fake-tracking ratios (real DMD2 health signals — ``loss_student`` is a
sign-random gradient vehicle, not a real loss):

* ``rel_gap``  — rms(τ·Δ_dm) / rms(τ·v_real_dm): fraction of teacher score the DM
  gap still represents. ↑ = fake lagging → bump fake.
* ``mag_ratio`` — rms(v_fake_dm) / rms(v_real_dm): ≈1 healthy; collapse/blow-up bad.
* ``cos``       — cosine(v_fake_dm, v_real_dm): ↓ = fake pointing the wrong way.
* ``dm_to_ca``  — effective DM vs CA magnitude. Decoupled DMD wants CA as the
  engine and DM as the shield, so DM ≳ CA for long stretches is a red flag.
  Accumulated only on do_ca steps (own denominator).

Mean-variance reg (lever B / paper Eq. 7; 0 when disabled):

* ``mv`` — the per-step Eq.7 KL value (pre-weight). Higher = the student's
  per-image stats are further from the real-latent target.

DP-DMD diversity loss (objective=dpdmd only; 0 under dmd2):

* ``div`` — the first-step diversity MSE ‖v_first − v_target‖² (pre-weight).
  Falling = the student's step-1 velocity is converging on the teacher's
  K-step anchor (the diverse landing point).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FlushedMetrics:
    fake: float
    grad: float
    dm: float
    cfg: float
    xpred: float
    v_student: float
    rel_gap: float
    mag_ratio: float
    cos: float
    dm_to_ca: float
    ca_steps: float
    mv: float
    div: float
    alpha: float


class TurboMetrics:
    """GPU-resident accumulators with a single-sync stacked flush."""

    def __init__(self, device: torch.device):
        z = lambda: torch.zeros((), device=device)  # noqa: E731
        # Always-on rms scalars.
        self.fake = z()
        self.grad = z()
        self.dm = z()
        self.cfg = z()
        self.xpred = z()
        self.v_student = z()
        # Fake-tracking.
        self.rel_gap = z()
        self.mag_ratio = z()
        self.cos = z()
        # CA-conditional (own denom).
        self.dm_to_ca = z()
        self.ca_steps = z()
        # Mean-variance reg (lever B / Eq.7); 0 when disabled.
        self.mv = z()
        # DP-DMD first-step diversity loss; 0 under the dmd2 objective.
        self.div = z()
        # Pure-Python (no GPU work).
        self.alpha = 0.0

    @torch.no_grad()
    def accumulate_per_step(
        self,
        *,
        fake_loss_mean_t: torch.Tensor,
        grad_signal: torch.Tensor,
        delta_dm: torch.Tensor,
        delta_cfg: torch.Tensor,
        x_pred: torch.Tensor,
        v_student: torch.Tensor,
        tau_dm_e: torch.Tensor,
        v_real_cond_dm: torch.Tensor,
        v_fake_cond_dm: torch.Tensor,
        mv_loss: torch.Tensor,
    ) -> None:
        eps_r = 1e-8
        self.fake.add_(fake_loss_mean_t.float())
        self.mv.add_(mv_loss.detach().float())
        self.grad.add_(grad_signal.float().pow(2).mean().sqrt())
        self.dm.add_(delta_dm.float().pow(2).mean().sqrt())
        self.cfg.add_(delta_cfg.float().pow(2).mean().sqrt())
        self.xpred.add_(x_pred.detach().float().std())
        self.v_student.add_(v_student.detach().float().pow(2).mean().sqrt())
        # Fake-tracking diagnostics at the DM eval point.
        vr = v_real_cond_dm.float()
        vf = v_fake_cond_dm.float()
        dm_w = (tau_dm_e * delta_dm.float()).pow(2).mean().sqrt()
        self.rel_gap.add_(dm_w / ((tau_dm_e * vr).pow(2).mean().sqrt() + eps_r))
        self.mag_ratio.add_(
            vf.pow(2).mean().sqrt() / (vr.pow(2).mean().sqrt() + eps_r)
        )
        self.cos.add_((vf * vr).sum() / (vf.norm() * vr.norm() + eps_r))

    @torch.no_grad()
    def accumulate_dm_to_ca(
        self,
        *,
        tau_ca_e: torch.Tensor,
        alpha_eff: float,
        delta_cfg: torch.Tensor,
        delta_dm: torch.Tensor,
        tau_dm_e: torch.Tensor,
    ) -> None:
        eps_r = 1e-8
        dm_w = (tau_dm_e * delta_dm.float()).pow(2).mean().sqrt()
        ca_w = (tau_ca_e * (alpha_eff - 1.0) * delta_cfg.float()).pow(2).mean().sqrt()
        self.dm_to_ca.add_(dm_w / (ca_w + eps_r))
        self.ca_steps.add_(1.0)

    @torch.no_grad()
    def add_div(self, div_loss_t: torch.Tensor) -> None:
        """Accumulate the DP-DMD first-step diversity loss (pre-weight)."""
        self.div.add_(div_loss_t.detach().float())

    def add_alpha(self, alpha_eff: float) -> None:
        self.alpha += alpha_eff

    def flush(self, log_interval: int) -> FlushedMetrics:
        """One CUDA sync per log boundary: stack everything, read once."""
        stacked = (
            torch.stack(
                [
                    self.fake,
                    self.grad,
                    self.dm,
                    self.cfg,
                    self.xpred,
                    self.v_student,
                    self.rel_gap,
                    self.mag_ratio,
                    self.cos,
                    self.mv,
                    self.div,
                ]
            )
            / log_interval
        )
        # dm_to_ca has its own denominator (only do_ca steps contribute).
        dm_to_ca = self.dm_to_ca / self.ca_steps.clamp(min=1.0)
        packed = torch.cat(
            [
                stacked,
                dm_to_ca.reshape(1),
                self.ca_steps.reshape(1),
            ]
        ).tolist()
        return FlushedMetrics(
            fake=packed[0],
            grad=packed[1],
            dm=packed[2],
            cfg=packed[3],
            xpred=packed[4],
            v_student=packed[5],
            rel_gap=packed[6],
            mag_ratio=packed[7],
            cos=packed[8],
            mv=packed[9],
            div=packed[10],
            dm_to_ca=packed[11],
            ca_steps=packed[12],
            alpha=self.alpha / log_interval,
        )

    def reset(self) -> None:
        for t in (
            self.fake, self.grad, self.dm, self.cfg, self.xpred, self.v_student,
            self.rel_gap, self.mag_ratio, self.cos, self.mv, self.div,
            self.dm_to_ca, self.ca_steps,
        ):
            t.zero_()
        self.alpha = 0.0


def write_scalars(writer, m: FlushedMetrics, step: int) -> None:
    """Push every available scalar to TensorBoard at the canonical key names."""
    writer.add_scalar("train/fake_loss", m.fake, step)
    writer.add_scalar("train/alpha_eff", m.alpha, step)
    writer.add_scalar("train/grad_signal_rms", m.grad, step)
    writer.add_scalar("train/delta_dm_rms", m.dm, step)
    writer.add_scalar("train/delta_cfg_rms", m.cfg, step)
    writer.add_scalar("train/x_pred_std", m.xpred, step)
    writer.add_scalar("train/v_student_rms", m.v_student, step)
    writer.add_scalar("train/dm_rel_gap", m.rel_gap, step)
    writer.add_scalar("train/dm_mag_ratio", m.mag_ratio, step)
    writer.add_scalar("train/dm_cos", m.cos, step)
    if m.ca_steps > 0:
        writer.add_scalar("train/dm_to_ca", m.dm_to_ca, step)
    writer.add_scalar("train/mean_var_kl", m.mv, step)
    writer.add_scalar("train/div_loss", m.div, step)


def tqdm_postfix(m: FlushedMetrics) -> dict:
    """tqdm postfix dict — short keys for the live progress line."""
    postfix = {
        "g": f"{m.grad:.2e}",
        "relg": f"{m.rel_gap:.3f}",
        "cos": f"{m.cos:.3f}",
        "dmca": f"{m.dm_to_ca:.2f}",
        "xp": f"{m.xpred:.3f}",
        "fake": f"{m.fake:.2e}",
    }
    if m.mv > 0:
        postfix["mv"] = f"{m.mv:.3f}"
    if m.div > 0:
        postfix["div"] = f"{m.div:.3f}"
    return postfix
