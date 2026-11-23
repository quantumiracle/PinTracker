"""Differentiable forward kinematics (poses and link velocities) for a floating-base tree.

Built from a compiled ``mujoco.MjModel`` so it matches the simulator exactly. Bodies at the same
tree depth are processed together; a body's parent is always in the previous level, so each level
only gathers from the level before it (no scatters, which are slow to differentiate).
Velocities are those of the body frame origin (mjlab's ``body_link_*_vel_w``).
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from pintracker.rot import quat_apply, quat_from_axis_angle, quat_mul


class TreeFK:
  def __init__(self, mj_model, body_names: list[str], joint_names: list[str], device):
    """body_names: entity bodies in output order, root first (full mj names).
    joint_names: the hinge joints in the order of the joint vectors passed to ``forward``."""
    m = mj_model
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
    assert all(i >= 0 for i in ids), body_names
    local = {mj: i for i, mj in enumerate(ids)}
    jidx = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n): i for i, n in enumerate(joint_names)}
    nb = len(ids)
    depth, parent = [0] * nb, [-1] * nb
    pos, quat, axis, qi = np.zeros((nb, 3)), np.zeros((nb, 4)), np.zeros((nb, 3)), np.zeros(nb, dtype=int)
    for i, mj in enumerate(ids):
      if i == 0:
        continue
      p = int(m.body_parentid[mj])
      assert p in local, f"parent of {body_names[i]} not in entity"
      parent[i], depth[i] = local[p], depth[local[p]] + 1
      pos[i], quat[i] = m.body_pos[mj], m.body_quat[mj]
      assert int(m.body_jntnum[mj]) == 1, f"{body_names[i]}: expected exactly one hinge"
      j = int(m.body_jntadr[mj])
      assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and np.allclose(m.jnt_pos[j], 0), body_names[i]
      axis[i], qi[i] = m.jnt_axis[j], jidx[j]
    self.nb, self.body_names = nb, body_names
    t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
    levels = [[0]] + [[i for i in range(nb) if depth[i] == d] for d in range(1, max(depth) + 1)]
    self.levels = []
    for d in range(1, len(levels)):
      idx, prev = levels[d], levels[d - 1]
      self.levels.append(dict(
        parent=t([prev.index(parent[i]) for i in idx], torch.long),
        pos=t(pos[idx]), quat=t(quat[idx]), axis=t(axis[idx]), qi=t(qi[idx], torch.long)))
    order = [i for lv in levels for i in lv]
    self.inv = t(np.argsort(order), torch.long)  # level-major -> body order

  def forward(self, root_pos, root_quat, root_v, root_w, q, qd, with_vel: bool = True):
    """All inputs (..., d). Returns body pos (..., nb, 3), quat (..., nb, 4), lin vel, ang vel (world)."""
    P, Q = [root_pos.unsqueeze(-2)], [root_quat.unsqueeze(-2)]
    V, W = [root_v.unsqueeze(-2)], [root_w.unsqueeze(-2)]
    for L in self.levels:
      pp, pq = P[-1][..., L["parent"], :], Q[-1][..., L["parent"], :]
      q_pre = quat_mul(pq, L["quat"].expand_as(pq))
      x = pp + quat_apply(pq, L["pos"].expand_as(pp))
      qn = quat_mul(q_pre, quat_from_axis_angle(L["axis"].expand_as(pp), q[..., L["qi"]]))
      P.append(x)
      Q.append(qn)
      if with_vel:
        pv, pw = V[-1][..., L["parent"], :], W[-1][..., L["parent"], :]
        jw = quat_apply(q_pre, L["axis"].expand_as(pp)) * qd[..., L["qi"]].unsqueeze(-1)
        W.append(pw + jw)
        V.append(pv + torch.cross(pw, x - pp, dim=-1))
    cat = lambda xs: torch.cat(xs, -2)[..., self.inv, :]  # noqa: E731
    return cat(P), cat(Q), (cat(V) if with_vel else None), (cat(W) if with_vel else None)
