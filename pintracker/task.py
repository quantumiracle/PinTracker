"""Motion-tracking POMDP shared by the simulator and the world model (paper Appendix A).

State s (71, origin-free world frame):
  root_pos(3) root_quat(4) root_lin_vel_w(3) root_ang_vel_w(3) joint_pos(29) joint_vel(29)
Sim context xi (33): foot friction(1), joint encoder bias(29), torso COM offset(3).
Every quantity below is a differentiable function of (s, xi, reference), so the reward and
observation heads of the world model are analytic; only dynamics and termination are learned.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pintracker.rot import (
  gravity_b,
  quat_apply,
  quat_apply_inv,
  quat_conj,
  quat_error_sq,
  quat_mul,
  rot6d,
  yaw_quat,
)

NJ = 29
STATE_DIM = 71
XI_DIM = 33
OBS_DIM = 3 * NJ + 6  # 93
HIST = 10  # observation history
K_FUT = 10  # future reference frames
GOAL_FRAME_DIM = 2 * NJ + 9
GOAL_DIM = K_FUT * GOAL_FRAME_DIM  # 670
DT = 0.02
R_MAX = (1.0 + 1.0 + 1.0 + 1.0 + 0.5 + 0.5) * DT  # largest per-step reward

OBS_NOISE = torch.cat(
  [torch.full((NJ,), 0.01), torch.full((NJ,), 0.5), torch.full((3,), 0.2), torch.full((3,), 0.05), torch.zeros(NJ)]
)
XI_OFFSET = torch.cat([torch.tensor([0.75]), torch.zeros(NJ), torch.zeros(3)])
XI_SCALE = torch.cat([torch.tensor([0.45]), torch.full((NJ,), 0.01), torch.tensor([0.025, 0.05, 0.05])])


def split(s: torch.Tensor):
  return s[..., 0:3], s[..., 3:7], s[..., 7:10], s[..., 10:13], s[..., 13:42], s[..., 42:71]


def obs_clean(s: torch.Tensor, xi: torch.Tensor, prev_action: torch.Tensor, default_q: torch.Tensor) -> torch.Tensor:
  """Noise-free proprioception (Table 1): biased joint pos rel. default, joint vel, IMU gyro, gravity, last action."""
  _, quat, _, w, q, qd = split(s)
  bias = xi[..., 1 : 1 + NJ]
  return torch.cat([q + bias - default_q, qd, quat_apply_inv(quat, w), gravity_b(quat), prev_action], dim=-1)


def canonical(s: torch.Tensor) -> torch.Tensor:
  """Heading- and xy-invariant features of s (71) used as network input."""
  p, quat, v, w, q, qd = split(s)
  h = yaw_quat(quat)
  return torch.cat(
    [p[..., 2:3], rot6d(quat_mul(quat_conj(h), quat)), quat_apply_inv(h, v), quat_apply_inv(quat, w), q, qd / 5.0],
    dim=-1,
  )


@dataclass
class Ref:
  """Reference quantities at one frame for a batch: tracked-body kinematics; anchor index into them."""

  body_pos: torch.Tensor  # (..., nb, 3)
  body_quat: torch.Tensor
  body_v: torch.Tensor
  body_w: torch.Tensor
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor


class Tracking:
  """Reward / termination / goal of the motion-tracking interface on top of a TreeFK."""

  def __init__(self, fk, tracked_ids, anchor_idx: int, lib, default_q, soft_lo, soft_hi, joint_limit_w=-2.0,
               fk_anchor=None):
    self.fk = fk
    self.fk_anchor = fk_anchor  # FK over the root -> anchor chain only (last body = anchor)
    self.tracked = torch.as_tensor(tracked_ids, device=default_q.device)
    self.anchor = anchor_idx  # index within tracked bodies
    self.lib = lib
    self.default_q = default_q
    self.lo, self.hi = soft_lo, soft_hi
    self.joint_limit_w = joint_limit_w

  def ref(self, g: torch.Tensor) -> Ref:
    L = self.lib
    return Ref(L.body_pos[g], L.body_quat[g], L.body_v[g], L.body_w[g], L.joint_pos[g], L.joint_vel[g])

  def bodies(self, s: torch.Tensor):
    p, quat, v, w, q, qd = split(s)
    bp, bq, bv, bw = self.fk.forward(p, quat, v, w, q, qd)
    t = self.tracked
    return bp[..., t, :], bq[..., t, :], bv[..., t, :], bw[..., t, :]

  def reward(self, s_next, ref: Ref, a, a_prev, bodies=None):
    """BeyondMimic tracking reward without self-collision, joint-limit weight -2, all terms x dt.
    ``bodies`` may pass the simulator's own body kinematics instead of FK of ``s_next``."""
    bp, bq, bv, bw = bodies if bodies is not None else self.bodies(s_next)
    an = self.anchor
    r_ap, r_aq = ref.body_pos[..., an, :], ref.body_quat[..., an, :]
    a_p, a_q = bp[..., an, :], bq[..., an, :]
    # Reference bodies re-anchored at the robot's anchor xy and heading (BeyondMimic "relative" bodies).
    delta_pos = torch.cat([a_p[..., :2], r_ap[..., 2:3]], dim=-1).unsqueeze(-2)
    delta_ori = yaw_quat(quat_mul(a_q, quat_conj(r_aq))).unsqueeze(-2).expand_as(ref.body_quat)
    rel_pos = delta_pos + quat_apply(delta_ori, ref.body_pos - r_ap.unsqueeze(-2))
    rel_quat = quat_mul(delta_ori, ref.body_quat)

    e_bp = ((rel_pos - bp) ** 2).sum(-1).mean(-1)
    e_bq = quat_error_sq(rel_quat, bq).mean(-1)
    e_bv = ((ref.body_v - bv) ** 2).sum(-1).mean(-1)
    e_bw = ((ref.body_w - bw) ** 2).sum(-1).mean(-1)
    e_ap = ((r_ap - a_p) ** 2).sum(-1)
    e_aq = quat_error_sq(r_aq, a_q)
    q = split(s_next)[4]
    r = (
      torch.exp(-e_bp / 0.3**2)
      + torch.exp(-e_bq / 0.4**2)
      + torch.exp(-e_bv / 1.0**2)
      + torch.exp(-e_bw / 3.14**2)
      + 0.5 * torch.exp(-e_ap / 0.3**2)
      + 0.5 * torch.exp(-e_aq / 0.4**2)
      - 0.1 * ((a - a_prev) ** 2).sum(-1)
      + self.joint_limit_w * ((q - self.hi).clamp_min(0) + (self.lo - q).clamp_min(0)).sum(-1)
    )
    return r * DT, (bp, bq)

  def term_features(self, anchor_pos, anchor_quat, ref: Ref):
    """z error and tilt error of the anchor, the two quantities the termination rule thresholds."""
    an = self.anchor
    dz = anchor_pos[..., 2] - ref.body_pos[..., an, 2]
    tilt = gravity_b(ref.body_quat[..., an, :])[..., 2] - gravity_b(anchor_quat)[..., 2]
    return dz, tilt

  def terminated(self, anchor_pos, anchor_quat, ref: Ref):
    dz, tilt = self.term_features(anchor_pos, anchor_quat, ref)
    return (dz.abs() > 0.25) | (tilt.abs() > 0.8)

  def anchor_pose(self, s):
    p, quat, v, w, q, qd = split(s)
    if self.fk_anchor is not None:
      bp, bq, _, _ = self.fk_anchor.forward(p, quat, v, w, q, qd, with_vel=False)
      return bp[..., -1, :], bq[..., -1, :]
    bp, bq, _, _ = self.fk.forward(p, quat, v, w, q, qd, with_vel=False)
    a_id = self.tracked[self.anchor]
    return bp[..., a_id, :], bq[..., a_id, :]

  def goal(self, clip, frame, ap, aq):
    """K future reference frames (f+1..f+K): joint pos/vel and the reference anchor pose in the
    robot anchor frame (ap, aq = robot anchor pose)."""
    L = self.lib
    ks = torch.arange(1, K_FUT + 1, device=clip.device)
    g = L.gidx(clip.unsqueeze(-1), frame.unsqueeze(-1) + ks)  # (B, K)
    ap, aq = ap.unsqueeze(-2), aq.unsqueeze(-2)
    r_ap = L.body_pos[g][..., self.anchor, :]
    r_aq = L.body_quat[g][..., self.anchor, :]
    rel_p = quat_apply_inv(aq.expand_as(r_aq), r_ap - ap)
    rel_q = rot6d(quat_mul(quat_conj(aq).expand_as(r_aq), r_aq))
    feat = torch.cat([L.joint_pos[g], L.joint_vel[g] / 5.0, rel_p, rel_q], dim=-1)
    return feat.flatten(-2)
