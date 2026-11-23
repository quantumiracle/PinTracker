"""Render policy rollouts (evaluation protocol) as MP4: simulated robot solid, reference motion as a blue ghost.

The rollout runs in the MuJoCo Warp env exactly like ``evaluate_actor``; the recorded root pose and joint
angles are then replayed through CPU MuJoCo for rendering.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import torch

from pintracker.env import TrackingEnv
from pintracker.motion import G1_JOINTS
from pintracker.nets import Actor
from pintracker.task import HIST, obs_clean

GHOST_RGBA = np.array([0.3, 0.55, 1.0, 0.35], dtype=np.float32)


@torch.no_grad()
def rollout(env: TrackingEnv, actor: Actor, after_fall: int = 100, start_frame: int = 0, max_steps: int | None = None):
  """Returns per-env lists of robot state rows and the step at which each env fell (-1 if it did not)."""
  n, dev, lib = env.n, env.device, env.lib
  ids = torch.arange(n, device=dev)
  clip = ids % lib.num_clips
  env.reset(ids, clip, torch.full_like(clip, start_frame), noise=False)
  o = obs_clean(env.state(), env.xi, env.prev_action, env.default_q)
  hist = o.unsqueeze(1).repeat(1, HIST, 1)
  states, frames = [env.state().cpu()], [env.frame.cpu().clone()]
  fell = torch.full((n,), -1, dtype=torch.long, device=dev)
  t = 0
  while True:
    s = env.state()
    ap, aq = env.task.anchor_pose(s)
    a = actor(hist, env.task.goal(env.clip, env.frame, ap, aq))
    _, _, term, _ = env.step(a)
    t += 1
    fell = torch.where((fell < 0) & term, torch.full_like(fell, t), fell)
    states.append(env.state().cpu())
    frames.append(env.frame.cpu().clone())
    finished = env.frame >= lib.clip_len[env.clip] - 1
    done = finished | ((fell >= 0) & (t - fell >= after_fall))
    if done.all() or (max_steps is not None and t >= max_steps):
      break
    o = obs_clean(env.state(), env.xi, env.prev_action, env.default_q)
    hist = torch.cat([hist[:, 1:], o.unsqueeze(1)], 1)
  S, F = torch.stack(states, 1), torch.stack(frames, 1)  # (n, T, 71), (n, T)
  fell = fell.cpu()
  end = torch.where(fell >= 0, torch.clamp(fell + after_fall, max=S.shape[1] - 1),
                    (lib.clip_len[clip].cpu() - 1 - start_frame).clamp(max=S.shape[1] - 1))
  return S, F, fell, end


class Renderer:
  def __init__(self, mj: mujoco.MjModel, width=960, height=544):
    self.m = mj
    self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, width)
    self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, height)
    free = [j for j in range(mj.njnt) if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    self.root_adr = mj.jnt_qposadr[free]
    prefix = mj.joint(free).name.rsplit("/", 1)[0] + "/" if "/" in mj.joint(free).name else ""
    self.q_adr = np.array([mj.joint(prefix + j).qposadr[0] for j in G1_JOINTS])
    self.d, self.d_ref = mujoco.MjData(mj), mujoco.MjData(mj)
    self.r = mujoco.Renderer(mj, height, width)
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    self.cam.trackbodyid = int(mj.jnt_bodyid[free])
    self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 135.0, -12.0
    self.opt = mujoco.MjvOption()
    self.pert = mujoco.MjvPerturb()

  def _set(self, d, pos, quat, q):
    d.qpos[self.root_adr:self.root_adr + 3] = pos
    d.qpos[self.root_adr + 3:self.root_adr + 7] = quat
    d.qpos[self.q_adr] = q
    mujoco.mj_kinematics(self.m, d)
    mujoco.mj_comPos(self.m, d)  # subtree COM, used by the tracking camera

  def frame(self, s: np.ndarray, ref: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    self._set(self.d, s[0:3], s[3:7], s[13:42])
    self._set(self.d_ref, *ref)
    self.r.update_scene(self.d, self.cam, self.opt)
    scn = self.r.scene
    n0 = scn.ngeom
    mujoco.mjv_addGeoms(self.m, self.d_ref, self.opt, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
    for i in range(n0, scn.ngeom):
      scn.geoms[i].rgba[:] = GHOST_RGBA
    return self.r.render()


def render_clips(ckpt_path: str, files, clip_names: list[str], out_dir: str, max_seconds: float = 45.0,
                 fall_context_s: float = 20.0, start_s: float = 0.0):
  """start_s > 0 starts every clip at that time (reference state, no noise) and shows a fixed window from there;
  a checkpoint path ending in "init" renders a randomly initialized (untrained) actor."""
  import imageio.v2 as imageio

  os.makedirs(out_dir, exist_ok=True)
  sel = [f for f in files if os.path.splitext(os.path.basename(f))[0] in clip_names]
  env = TrackingEnv(sel, num_envs=len(sel), device="cuda:0", randomize=False, episode_steps=10**9, reset_noise=False)
  torch.manual_seed(0)
  actor = Actor().to("cuda:0")
  if not ckpt_path.endswith("init"):
    actor = Actor.from_state_dict(torch.load(ckpt_path, map_location="cuda:0")["actor"]).to("cuda:0")
  start = int(start_s * 50)
  S, F, fell, end = rollout(env, actor, start_frame=start,
                            max_steps=int(max_seconds * 50) + 100 if start else None)
  lib = env.lib
  rend = Renderer(env.env.sim.mj_model)
  out = []
  for i, name in enumerate(lib.names):
    if start:  # fixed window from the start frame
      t0 = 0
    elif fell[i] >= 0:  # show the run-up to the fall and its aftermath
      t0 = max(0, int(fell[i]) - int(fall_context_s * 50))
    else:
      t0 = 0
    t1 = min(int(end[i]), t0 + int(max_seconds * 50) + int(2 * 50 * (fell[i] >= 0)))
    path = f"{out_dir}/{name}{'_FALL' if fell[i] >= 0 else ''}.mp4"
    w = imageio.get_writer(path, fps=50, codec="libx264", quality=7, macro_block_size=8)
    for t in range(t0, t1 + 1):
      g = int(lib.gidx(torch.tensor([i], device=lib.clip_len.device), F[i, t:t + 1].to(lib.clip_len.device)))
      ref = (lib.root_pos[g].cpu().numpy(), lib.root_quat[g].cpu().numpy(), lib.joint_pos[g].cpu().numpy())
      w.append_data(rend.frame(S[i, t].numpy(), ref))
    w.close()
    msg = f"{name}: frames {t0}-{t1} ({(t1 - t0) / 50:.1f} s), fell at step {int(fell[i])}"
    print(msg, flush=True)
    out.append(msg)
  return out
