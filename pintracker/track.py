"""Play reference clips with a tracker in simulation (evaluation protocol: every clip from frame 0, no domain
randomization, no noise) and report per-clip metrics, optionally with videos of the robot and the reference."""

from __future__ import annotations

import os

import numpy as np
import torch

from pintracker.env import TrackingEnv
from pintracker.evaluate import evaluate_actors
from pintracker.render import Renderer, rollout
from pintracker.rot import quat_conj, quat_mul


def track(policy, files: list, device: str = "cuda:0"):
  """Returns (env, rows, rollout). One row per clip:

  fell / survived_s      our termination: anchor height off by > 0.25 m or tilt off by > 0.8
  norm_return            reward sum over the clip / its largest possible value (1 = perfect)
  body_err_local_cm      tracked bodies relative to the pelvis, mean over frames before a fall
  body_err_cm            same in the world frame (includes drift)
  sonic_fail             SONIC paper criterion: root height off by > 0.25 m or root orientation off by > 1 rad
  mpjpe_mm, vel_err_mm, acc_err_mm   root-relative MPJPE and velocity / acceleration errors (mm, mm/frame, mm/frame^2)
  """
  env = TrackingEnv(files, num_envs=len(files), device=device, randomize=False, episode_steps=10**9, reset_noise=False)
  lib = env.lib
  norm, _ = evaluate_actors(env, [policy])
  S, F, fell, end = rollout(env, policy)
  # The rollout runs until the longest clip is over; a fall after a shorter clip's last frame is not a fall on that clip.
  last = (lib.clip_len.cpu() - 1).clamp(max=S.shape[1] - 1)
  late = fell >= last
  fell, end = torch.where(late, torch.full_like(fell, -1), fell), torch.where(late, last, end)
  rows = []
  for i, name in enumerate(lib.names):
    n = int(fell[i]) if fell[i] >= 0 else int(end[i])
    s = S[i, :n + 1].to(device)
    g = lib.gidx(torch.full((n + 1,), i, device=device), F[i, :n + 1].to(device))
    p = env.fk.forward(s[:, 0:3], s[:, 3:7], s[:, 7:10], s[:, 10:13], s[:, 13:42], s[:, 42:71], with_vel=False)[0]
    p, r = p[:, env.tracked_ids], lib.body_pos[g]
    dz = (s[:, 2] - lib.root_pos[g][:, 2]).abs()
    w = quat_mul(quat_conj(lib.root_quat[g]), s[:, 3:7])[:, 0].abs().clamp(max=1)
    off = (dz > 0.25) | (2 * torch.acos(w) > 1.0)
    dv, dr = p.diff(dim=0), r.diff(dim=0)
    local = ((p - p[:, :1]) - (r - r[:, :1])).norm(dim=-1).mean()
    rows.append(dict(
      name=name, seconds=float(lib.clip_len[i]) / 50, fell=bool(fell[i] >= 0), survived_s=n / 50,
      norm_return=float(norm[0, i]),
      body_err_cm=100 * float((p - r).norm(dim=-1).mean()),
      body_err_local_cm=100 * float(local),
      joint_err_rad=float((s[:, 13:42] - lib.joint_pos[g]).abs().mean()),
      sonic_fail=bool(fell[i] >= 0) or bool(off.any()),
      mpjpe_mm=1000 * float(local),
      vel_err_mm=1000 * float((dv - dr).norm(dim=-1).mean()) if n > 0 else 0.0,
      acc_err_mm=1000 * float((dv.diff(dim=0) - dr.diff(dim=0)).norm(dim=-1).mean()) if n > 1 else 0.0))
  return env, rows, (S, F, fell, end)


def render(env, roll, out_dir: str, max_seconds: float = 40.0, after_fall_s: float = 2.0, fps: int = 25):
  """One mp4 per clip: the simulated robot (solid) and the reference (blue ghost), tracking camera."""
  import imageio.v2 as imageio
  os.makedirs(out_dir, exist_ok=True)
  S, F, fell, end = roll
  lib = env.lib
  rend = Renderer(env.env.sim.mj_model)
  step = max(1, 50 // fps)
  paths = []
  for i, name in enumerate(lib.names):
    t1 = min(int(end[i]), int(max_seconds * 50))
    if fell[i] >= 0:
      t1 = min(t1, int(fell[i]) + int(after_fall_s * 50))
    path = f"{out_dir}/{name}{'_FALL' if fell[i] >= 0 else ''}.mp4"
    w = imageio.get_writer(path, fps=50 // step, codec="libx264", quality=6, macro_block_size=8)
    for t in range(0, t1 + 1, step):
      g = int(lib.gidx(torch.tensor([i], device=lib.clip_len.device), F[i, t:t + 1].to(lib.clip_len.device)))
      ref = (lib.root_pos[g].cpu().numpy(), lib.root_quat[g].cpu().numpy(), lib.joint_pos[g].cpu().numpy())
      w.append_data(rend.frame(S[i, t].numpy(), ref))
    w.close()
    paths.append(path)
  return paths


def summarize(rows: list[dict]) -> dict:
  n = len(rows)
  T = sum(r["seconds"] for r in rows)
  return dict(clips=n, tracked=sum(not r["fell"] for r in rows), success_rate=sum(not r["fell"] for r in rows) / n,
              sonic_success_rate=sum(not r["sonic_fail"] for r in rows) / n,
              norm_return=float(np.mean([r["norm_return"] for r in rows])),
              kept_up=sum(r["survived_s"] for r in rows) / T,
              mpjpe_mm=float(np.mean([r["mpjpe_mm"] for r in rows])))
