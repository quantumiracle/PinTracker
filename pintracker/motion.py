"""Motion data: CSV (retargeted G1, 30 Hz) -> 50 Hz clips, and a packed multi-clip GPU library."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from pintracker.rot import quat_conj, quat_mul, quat_normalize, quat_slerp, rotvec_from_quat

# Joint order of the LAFAN1/BONES G1 CSVs (and of every joint vector in this repo).
G1_JOINTS = [
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
  "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
  "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# BeyondMimic's tracked bodies for G1; the anchor is torso_link.
TRACKED_BODIES = [
  "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
ANCHOR_BODY = "torso_link"


def csv_to_clip(path: str | Path, in_fps: float = 30.0, out_fps: float = 50.0) -> dict[str, np.ndarray]:
  """LAFAN1 G1 CSV (pos m, quat xyzw, joints rad) -> clip."""
  raw = torch.from_numpy(np.loadtxt(path, delimiter=",")).double()
  return resample_clip(raw[:, :3], raw[:, [6, 3, 4, 5]], raw[:, 7:], in_fps, out_fps)  # xyzw -> wxyz


def resample_clip(pos_in: torch.Tensor, quat_in: torch.Tensor, q_in: torch.Tensor, in_fps: float,
                  out_fps: float = 50.0) -> dict[str, np.ndarray]:
  """Same resampling as mjlab's csv_to_npz: lerp/slerp to out_fps, central-difference velocities.
  Inputs are float64 tensors: root position (m), root quaternion (wxyz), joint angles (rad, G1_JOINTS order)."""
  quat_in = quat_normalize(quat_in)
  n_in = pos_in.shape[0]
  duration = (n_in - 1) / in_fps
  n_out = math.floor(duration * out_fps + 1e-6) + 1
  times = torch.arange(n_out, dtype=torch.float64) / out_fps
  phase = times / duration * (n_in - 1)
  i0 = phase.floor().long().clamp(max=n_in - 1)
  i1 = (i0 + 1).clamp(max=n_in - 1)
  b = phase - i0
  pos = pos_in[i0] * (1 - b[:, None]) + pos_in[i1] * b[:, None]
  q = q_in[i0] * (1 - b[:, None]) + q_in[i1] * b[:, None]
  quat = quat_slerp(quat_in[i0], quat_in[i1], b)
  dt = 1.0 / out_fps
  v = torch.gradient(pos, spacing=dt, dim=0)[0]
  qd = torch.gradient(q, spacing=dt, dim=0)[0]
  rel = quat_mul(quat[2:], quat_conj(quat[:-2]))
  w = rotvec_from_quat(rel) / (2 * dt)
  w = torch.cat([w[:1], w, w[-1:]], dim=0)
  f = lambda x: x.float().numpy()  # noqa: E731
  return dict(root_pos=f(pos), root_quat=f(quat), root_v=f(v), root_w=f(w), joint_pos=f(q), joint_vel=f(qd), fps=np.array(out_fps))


class MotionLibrary:
  """All clips concatenated along time. A frame is addressed by (clip id, frame-in-clip)."""

  def __init__(self, clip_files: list[Path], device, fk, tracked_ids: list[int]):
    """A file holds one clip, or several concatenated (a pack: same arrays plus ``lens`` and ``names``)."""
    clips, self.names = [], []
    for p in clip_files:
      d = dict(np.load(p))
      if "lens" not in d:
        clips.append(d)
        self.names.append(Path(p).stem)
        continue
      cut = np.cumsum(d["lens"])[:-1]
      parts = {k: np.split(d[k], cut) for k in ("root_pos", "root_quat", "root_v", "root_w", "joint_pos", "joint_vel")}
      clips += [{k: v[j] for k, v in parts.items()} for j in range(len(d["lens"]))]
      self.names += [str(n) for n in d["names"]]
    cat = lambda k: torch.as_tensor(np.concatenate([c[k] for c in clips]), device=device)  # noqa: E731
    self.root_pos = cat("root_pos")
    self.root_quat = cat("root_quat")
    self.root_v = cat("root_v")
    self.root_w = cat("root_w")
    self.joint_pos = cat("joint_pos")
    self.joint_vel = cat("joint_vel")
    lens = [c["joint_pos"].shape[0] for c in clips]
    self.clip_len = torch.tensor(lens, device=device, dtype=torch.long)
    self.clip_start = torch.cumsum(self.clip_len, 0) - self.clip_len
    self.num_clips = len(clips)
    self.num_frames = int(self.clip_len.sum())
    # Reference body kinematics of the tracked bodies (same FK as the world model's reward head).
    bp, bq, bv, bw = [], [], [], []
    for s in range(0, self.num_frames, 65536):
      sl = slice(s, s + 65536)
      p, q, v, w = fk.forward(self.root_pos[sl], self.root_quat[sl], self.root_v[sl], self.root_w[sl],
                              self.joint_pos[sl], self.joint_vel[sl])
      bp.append(p[:, tracked_ids]); bq.append(q[:, tracked_ids]); bv.append(v[:, tracked_ids]); bw.append(w[:, tracked_ids])
    self.body_pos, self.body_quat = torch.cat(bp), torch.cat(bq)
    self.body_v, self.body_w = torch.cat(bv), torch.cat(bw)

  def gidx(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Global frame index, clamping the frame to the clip (holds the last frame past the end)."""
    frame = torch.minimum(frame.clamp_min(0), self.clip_len[clip] - 1)
    return self.clip_start[clip] + frame

  @property
  def hours(self) -> float:
    return self.num_frames / 50.0 / 3600.0
