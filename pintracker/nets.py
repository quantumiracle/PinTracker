"""Networks (paper Table 3): world-model dynamics + termination head, policy (encoder/decoder), critic."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from pintracker.rot import quat_apply, quat_apply_inv, quat_conj, quat_from_rotvec, quat_mul, quat_normalize, rotvec_from_quat, yaw_quat
from pintracker.task import DT, GOAL_DIM, HIST, NJ, OBS_DIM, STATE_DIM, XI_DIM, canonical, split


def mlp(i: int, hidden: tuple[int, ...], o: int, act=nn.ELU) -> nn.Sequential:
  layers, d = [], i
  for h in hidden:
    layers += [nn.Linear(d, h), act()]
    d = h
  layers.append(nn.Linear(d, o))
  return nn.Sequential(*layers)


class Normalizer(nn.Module):
  """Running mean / std (Welford, merged per batch); frozen inside gradients."""

  def __init__(self, dim: int, clip: float = 10.0):
    super().__init__()
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.tensor(1e-4))
    self.clip = clip

  @torch.no_grad()
  def update(self, x: torch.Tensor):
    # Strictly in place: compiled graphs / CUDA graphs hold these buffers by address.
    # Under data parallelism the batch moments are pooled over all ranks so every copy stays identical.
    x = x.reshape(-1, x.shape[-1]).float()
    b = x.shape[0]
    m, v = x.mean(0), x.var(0, unbiased=False)
    if dist.is_initialized() and dist.get_world_size() > 1:
      mom = torch.cat([x.new_tensor([b]), x.sum(0), x.pow(2).sum(0)])
      dist.all_reduce(mom)
      b = mom[0]
      m = mom[1:1 + x.shape[1]] / b
      v = (mom[1 + x.shape[1]:] / b - m**2).clamp_min(0)
    tot = self.count + b
    delta = m - self.mean
    self.var.copy_((self.var * self.count + v * b + delta**2 * self.count * b / tot) / tot)
    self.mean.add_(delta * b / tot)
    self.count.copy_(tot)

  def forward(self, x):
    return ((x - self.mean) / torch.sqrt(self.var + 1e-6)).clamp(-self.clip, self.clip)


# --------------------------------------------------------------------------- world model

# Output = [dv_h(3), dw_b(3), dqd(29), dp_h(3), dth_b(3), dq(29)] scaled by OUT_SCALE.
_sizes = (3, 3, NJ, 3, 3, NJ)
OUT_SCALE = torch.cat([torch.full((n,), s) for n, s in zip(_sizes, (0.5, 1.0, 3.0, 0.01, 0.02, 0.02))])
# Nominal per-quantity error scales for the loss: [p, rot, v, w, q, qd].
ERR_SCALE = torch.cat([torch.full((n,), s) for n, s in zip((3, 3, 3, 3, NJ, NJ), (0.02, 0.05, 0.2, 0.5, 0.02, 1.0))])
TERM_IN = STATE_DIM + 4


class WorldModel(nn.Module):
  """Deterministic dynamics f(s, a, xi) as a residual on semi-implicit integration, plus a termination head."""

  def __init__(self, hidden=(2048, 2048, 2048)):
    super().__init__()
    self.dyn = mlp(STATE_DIM + NJ + XI_DIM, hidden, sum(_sizes))
    nn.init.zeros_(self.dyn[-1].weight)
    nn.init.zeros_(self.dyn[-1].bias)
    self.term = mlp(TERM_IN, (256, 256), 1)
    self.register_buffer("out_scale", OUT_SCALE.clone())
    self.register_buffer("err_scale", ERR_SCALE.clone())

  def step(self, s, a, xi_n):
    p, quat, v, w, q, qd = split(s)
    h = yaw_quat(quat)
    d = self.dyn(torch.cat([canonical(s), a, xi_n], -1)) * self.out_scale
    dv, dw, dqd, dp, dth, dq = torch.split(d, _sizes, -1)
    v2 = v + quat_apply(h, dv)
    w_b2 = quat_apply_inv(quat, w) + dw
    qd2 = qd + dqd
    p2 = p + DT * v2 + quat_apply(h, dp)
    quat2 = quat_normalize(quat_mul(quat, quat_from_rotvec(DT * w_b2 + dth)))
    w2 = quat_apply(quat2, w_b2)
    q2 = q + DT * qd2 + dq
    return torch.cat([p2, quat2, v2, w2, q2, qd2], -1)

  def error(self, s_pred, s_true):
    """Normalized tangent-space error (..., 70)."""
    pp, qp, vp, wp, jp, jvp = split(s_pred)
    pt, qt, vt, wt, jt, jvt = split(s_true)
    e = torch.cat([pp - pt, rotvec_from_quat(quat_mul(qt, quat_conj(qp))), vp - vt, wp - wt, jp - jt, jvp - jvt], -1)
    return e / self.err_scale

  def term_logit(self, s, dz, tilt):
    x = torch.cat([canonical(s), dz.unsqueeze(-1) / 0.25, tilt.unsqueeze(-1) / 0.8,
                   dz.abs().unsqueeze(-1) / 0.25, tilt.abs().unsqueeze(-1) / 0.8], -1)
    return self.term(x).squeeze(-1)


# --------------------------------------------------------------------------- policy / critic

class Actor(nn.Module):
  """a = dec(enc(g), o_{t-9:t}); latent z is 32-d, tanh-bounded.

  With ``ref_q0`` and ``ref_scale`` (default joint pose and action scale), the action is built on the reference pose
  (BlinkMimic --pi_ref_skip): a = dec(...) + (q_ref(f+1) - q0) / scale, and the last layer starts near zero."""

  def __init__(self, ref_q0=None, ref_scale=None, hidden=(2048, 1024, 512, 256)):
    super().__init__()
    self.obs_norm = Normalizer(OBS_DIM)
    self.goal_norm = Normalizer(GOAL_DIM)
    self.enc = mlp(GOAL_DIM, hidden, 32)
    self.dec = mlp(32 + HIST * OBS_DIM, hidden, NJ)
    self.ref_skip = ref_q0 is not None
    if self.ref_skip:
      self.register_buffer("ref_q0", ref_q0.clone().float())
      self.register_buffer("ref_scale", ref_scale.clone().float())
      with torch.no_grad():
        self.dec[-1].weight.mul_(0.01)
        self.dec[-1].bias.zero_()

  @classmethod
  def from_state_dict(cls, sd):
    hidden = tuple(v.shape[0] for k, v in sorted(((k, v) for k, v in sd.items() if k.startswith("enc.") and k.endswith(
      ".weight")), key=lambda kv: int(kv[0].split(".")[1])))[:-1]  # widths from the weights
    actor = cls(sd["ref_q0"], sd["ref_scale"], hidden) if "ref_q0" in sd else cls(hidden=hidden)
    actor.load_state_dict(sd)
    return actor

  def forward(self, hist, goal):
    z = torch.tanh(self.enc(self.goal_norm(goal)))
    a = self.dec(torch.cat([z, self.obs_norm(hist).flatten(-2)], -1))
    if self.ref_skip:  # the goal starts with the joint positions of frame f+1
      a = a + (goal[..., :NJ] - self.ref_q0) / self.ref_scale
    return a


class Critic(nn.Module):
  def __init__(self):
    super().__init__()
    d = STATE_DIM + GOAL_DIM + XI_DIM
    self.norm = Normalizer(d)
    self.net = mlp(d, (256, 128), 1)

  def inputs(self, s, goal, xi_n):
    return torch.cat([canonical(s), goal, xi_n], -1)

  def forward(self, x):
    return self.net(self.norm(x)).squeeze(-1)
