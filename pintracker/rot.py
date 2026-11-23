"""Quaternion / rotation helpers. Quaternions are (w, x, y, z), matching MuJoCo and mjlab."""

from __future__ import annotations

import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
  return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    dim=-1,
  )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate v by q."""
  w, xyz = q[..., :1], q[..., 1:]
  t = 2.0 * torch.cross(xyz, v, dim=-1)
  return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  return quat_apply(quat_conj(q), v)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  half = 0.5 * angle.unsqueeze(-1)
  return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_from_rotvec(rv: torch.Tensor) -> torch.Tensor:
  """exp map. Smooth at zero (uses a Taylor expansion for small angles)."""
  theta2 = (rv * rv).sum(-1, keepdim=True)
  theta = torch.sqrt(theta2 + 1e-12)
  small = theta2 < 1e-6
  half = 0.5 * theta
  k = torch.where(small, 0.5 - theta2 / 48.0, torch.sin(half) / theta)
  w = torch.where(small, 1.0 - theta2 / 8.0, torch.cos(half))
  return torch.cat([w, rv * k], dim=-1)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
  """log map, returns the shortest rotation vector."""
  q = torch.where(q[..., :1] < 0, -q, q)
  xyz = q[..., 1:]
  s = xyz.norm(dim=-1, keepdim=True)
  w = q[..., :1].clamp(-1.0, 1.0)
  angle = 2.0 * torch.atan2(s, w)
  k = torch.where(s < 1e-6, 2.0 / w.clamp_min(1e-6), angle / s.clamp_min(1e-12))
  return xyz * k


def quat_error_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Squared geodesic angle between two orientations (|log(a b^-1)|^2)."""
  d = quat_mul(a, quat_conj(b))
  rv = rotvec_from_quat(d)
  return (rv * rv).sum(-1)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  """Heading (yaw-only) part of q."""
  w, x, y, z = q.unbind(-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  zero = torch.zeros_like(yaw)
  half = 0.5 * yaw
  return torch.stack([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    [
      1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ],
    dim=-1,
  ).reshape(q.shape[:-1] + (3, 3))


def rot6d(q: torch.Tensor) -> torch.Tensor:
  """First two columns of the rotation matrix, flattened (continuous representation)."""
  m = quat_to_mat(q)
  return m[..., :, :2].reshape(q.shape[:-1] + (6,))


def gravity_b(q: torch.Tensor) -> torch.Tensor:
  g = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
  g[..., 2] = -1.0
  return quat_apply_inv(q, g)


def quat_slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
  dot = (a * b).sum(-1, keepdim=True)
  b = torch.where(dot < 0, -b, b)
  dot = dot.abs().clamp(max=1.0)
  t = t.unsqueeze(-1)
  theta = torch.acos(dot)
  sin = torch.sin(theta)
  lin = sin < 1e-6
  wa = torch.where(lin, 1 - t, torch.sin((1 - t) * theta) / sin.clamp_min(1e-9))
  wb = torch.where(lin, t, torch.sin(t * theta) / sin.clamp_min(1e-9))
  return quat_normalize(wa * a + wb * b)
