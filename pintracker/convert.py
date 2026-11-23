"""Reference motions -> PinTracker's 50 Hz clip format.

A clip is an .npz with float32 arrays at 50 Hz:
  root_pos (T, 3)   m, world frame, z up
  root_quat (T, 4)  wxyz
  root_v (T, 3), root_w (T, 3)   root linear / angular velocity, world frame
  joint_pos (T, 29), joint_vel (T, 29)   rad, rad/s, in pintracker.motion.G1_JOINTS order
and fps = 50. Velocities are computed here by central differences after resampling.
"""

from __future__ import annotations

import pickle

import numpy as np
import torch

from pintracker.motion import G1_JOINTS, resample_clip

_NUMPY = {("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
          ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"), ("numpy", "ndarray"),
          ("numpy", "dtype")}


class _NumpyOnly(pickle.Unpickler):
  """Pickles from elsewhere may run code on load; this one only builds numpy arrays and plain containers."""

  def find_class(self, module, name):
    if (module, name) not in _NUMPY:
      raise pickle.UnpicklingError(f"refusing to load {module}.{name}")
    return super().find_class(module, name)


def load_pickle(path: str) -> dict:
  with open(path, "rb") as f:
    return _NumpyOnly(f).load()


def g1_23_to_29(q23: np.ndarray) -> np.ndarray:
  """The 23-DoF G1 order (legs 12, waist 3, per arm shoulder pitch/roll/yaw + elbow) -> 29, wrists at 0."""
  q = np.zeros((len(q23), 29), np.float64)
  q[:, :19] = q23[:, :19]
  q[:, 22:26] = q23[:, 19:23]
  return q


def to_clip(root_pos, root_quat, joint_pos, fps: float, quat_order: str = "wxyz") -> dict:
  """Arrays at any frame rate -> the 50 Hz clip dict. joint_pos is (T, 29) in G1_JOINTS order, or (T, 23)."""
  root_quat = np.asarray(root_quat, np.float64)
  if quat_order == "xyzw":
    root_quat = root_quat[:, [3, 0, 1, 2]]
  joint_pos = np.asarray(joint_pos, np.float64)
  if joint_pos.shape[1] == 23:
    joint_pos = g1_23_to_29(joint_pos)
  assert joint_pos.shape[1] == len(G1_JOINTS), f"expected 29 or 23 joints, got {joint_pos.shape[1]}"
  t = lambda x: torch.from_numpy(np.asarray(x, np.float64))  # noqa: E731
  return resample_clip(t(root_pos), t(root_quat), t(joint_pos), float(fps))


def convert_pickle(path: str, out: str, quat_order: str = "xyzw"):
  """GMR / TWIST / Movens style pickle (fps, root_pos, root_rot, dof_pos) -> clip npz."""
  d = load_pickle(path)
  clip = to_clip(d["root_pos"], d["root_rot"], d["dof_pos"], float(d["fps"]), quat_order)
  np.savez(out, **clip)
  return clip
