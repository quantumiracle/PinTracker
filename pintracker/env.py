"""Vectorized MuJoCo Warp tracking environment (mjlab) with SuperMimic's MDP.

mjlab supplies the robot, PD actuators, physics and per-episode domain randomization. Resets,
reference bookkeeping, reward and termination are done here so that the simulator and the
world model evaluate exactly the same functions (``pintracker.task``).
"""

from __future__ import annotations

import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.envs.mdp import observations as mdp_obs
from mjlab.terrains import TerrainEntityCfg

from pintracker.kin import TreeFK
from pintracker.motion import ANCHOR_BODY, G1_JOINTS, TRACKED_BODIES, MotionLibrary
from pintracker.rot import quat_from_rotvec, quat_mul
from pintracker.task import NJ, STATE_DIM, XI_DIM, Tracking

FOOT_GEOMS = r"^(left|right)_foot[1-7]_collision$"

# BeyondMimic reset perturbations (reference-state initialization noise).
POSE_RANGE = torch.tensor([[-0.05, 0.05], [-0.05, 0.05], [-0.01, 0.01], [-0.1, 0.1], [-0.1, 0.1], [-0.2, 0.2]])
VEL_RANGE = torch.tensor([[-0.5, 0.5], [-0.5, 0.5], [-0.2, 0.2], [-0.52, 0.52], [-0.52, 0.52], [-0.78, 0.78]])
JOINT_RANGE = 0.1


def make_env_cfg(num_envs: int, randomize: bool, robot_cfg=None) -> ManagerBasedRlEnvCfg:
  fr, bias, com = ((0.3, 1.2), (-0.01, 0.01), {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)})
  if not randomize:  # evaluation protocol: friction 1.0, no randomization
    fr, bias, com = (1.0, 1.0), (0.0, 0.0), {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}
  events = {
    "foot_friction": EventTermCfg(
      mode="reset", func=dr.geom_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs",
              "ranges": fr, "shared_random": True},
    ),
    "encoder_bias": EventTermCfg(
      mode="reset", func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": bias},
    ),
    "base_com": EventTermCfg(
      mode="reset", func=dr.body_com_offset,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add", "ranges": com},
    ),
  }
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=num_envs,
                   entities={"robot": robot_cfg or get_g1_robot_cfg()}),
    observations={"policy": ObservationGroupCfg(
      terms={"joint_pos": ObservationTermCfg(func=mdp_obs.joint_pos_rel)}, concatenate_terms=True)},
    actions={"joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True)},
    commands={},
    events=events,
    rewards={},
    terminations={},
    sim=SimulationCfg(nconmax=35, njmax=250, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
    decimation=4,
    episode_length_s=1e9,
    auto_reset=False,
  )


class TrackingEnv:
  def __init__(self, motion_files, num_envs: int, device: str, randomize: bool = True,
               episode_steps: int = 500, reset_noise: bool = True, lib: MotionLibrary | None = None, robot_cfg=None):
    """lib: share another env's motion library (built from the same files) instead of loading a copy.
    robot_cfg: a different mjlab G1 entity config (e.g. other actuator gains) for evaluating outside policies."""
    self.env = ManagerBasedRlEnv(make_env_cfg(num_envs, randomize, robot_cfg), device=device)
    self.n = num_envs
    self.device = device
    # Silence per-step line-search overflow prints (they flood logs; physics is unchanged).
    for name in ("wp_model", "model"):
      try:
        getattr(self.env.sim, name).opt.warn_overflow = 0
      except Exception:
        pass
    self.robot = self.env.scene["robot"]
    r = self.robot
    mj = self.env.sim.mj_model
    self.jid = torch.as_tensor(r.find_joints(G1_JOINTS, preserve_order=True)[0], device=device)
    # FK over all robot bodies (entity order), root first.
    body_names = list(r.body_names)
    prefix = mj.body(int(r.indexing.body_ids[0])).name[: -len(body_names[0])]
    self.fk = TreeFK(mj, [prefix + b for b in body_names], [prefix + j for j in G1_JOINTS], device)
    self.tracked_ids = [body_names.index(b) for b in TRACKED_BODIES]
    # Short FK over the pelvis -> torso chain only, for the anchor pose (goal, WM termination features).
    chain, b = [], int(r.indexing.body_ids[body_names.index(ANCHOR_BODY)])
    while b != int(r.indexing.body_ids[0]):
      chain.append(b)
      b = int(mj.body_parentid[b])
    chain = [b] + chain[::-1]
    self.fk_anchor = TreeFK(mj, [mj.body(i).name for i in chain], [prefix + j for j in G1_JOINTS], device)
    self.lib = lib or MotionLibrary(motion_files, device, self.fk, self.tracked_ids)
    self.bad_steps = 0
    dq = r.data.default_joint_pos[0, self.jid]
    lim = r.data.soft_joint_pos_limits[0, self.jid]
    self.task = Tracking(self.fk, self.tracked_ids, TRACKED_BODIES.index(ANCHOR_BODY), self.lib, dq,
                         lim[:, 0], lim[:, 1], fk_anchor=self.fk_anchor)
    self.default_q = dq
    # Policy actions are in G1_JOINTS order; the action term orders joints by actuator.
    term = self.env.action_manager.get_term("joint_pos")
    tids = term._target_ids.tolist() if torch.is_tensor(term._target_ids) else list(term._target_ids)
    jlist = self.jid.tolist()
    self.act_perm = torch.as_tensor([jlist.index(t) for t in tids], device=device)
    self.foot_geoms = r.indexing.geom_ids[torch.as_tensor(r.find_geoms(FOOT_GEOMS)[0], device=device)]
    self.torso_body = int(r.indexing.body_ids[body_names.index("torso_link")])
    self.torso_ipos_default = torch.as_tensor(mj.body_ipos[self.torso_body], device=device, dtype=torch.float32)
    self.episode_steps = episode_steps
    self.reset_noise = reset_noise
    z = lambda *s, dt=torch.float32: torch.zeros(*s, device=device, dtype=dt)  # noqa: E731
    self.clip, self.frame, self.ep_step = z(num_envs, dt=torch.long), z(num_envs, dt=torch.long), z(num_envs, dt=torch.long)
    self.xi = z(num_envs, XI_DIM)
    self.prev_action = z(num_envs, NJ)
    self.sampler = None  # set by the trainer: callable(n) -> (clip, frame)

  def action_scale(self) -> torch.Tensor:
    """Per-joint action scale in G1_JOINTS order (the action term holds it in actuator order)."""
    term = self.env.action_manager.get_term("joint_pos")
    sc = torch.as_tensor(term._scale, device=self.device, dtype=torch.float32)
    sc = sc[0] if sc.ndim == 2 else sc.expand(len(self.act_perm))
    out = torch.empty(NJ, device=self.device)
    out[self.act_perm] = sc
    return out

  # ------------------------------------------------------------------ state io
  def state(self) -> torch.Tensor:
    d = self.robot.data
    return torch.cat([
      d.root_link_pos_w - self.env.scene.env_origins, d.root_link_quat_w, d.root_link_lin_vel_w,
      d.root_link_ang_vel_w, d.joint_pos[:, self.jid], d.joint_vel[:, self.jid]], dim=-1)

  def tracked_bodies(self):
    d, t = self.robot.data, self.tracked_ids
    return d.body_link_pos_w[:, t] - self.env.scene.env_origins[:, None], d.body_link_quat_w[:, t], \
      d.body_link_lin_vel_w[:, t], d.body_link_ang_vel_w[:, t]

  def _read_xi(self, ids):
    m = self.env.sim.model
    fr = m.geom_friction[ids][:, self.foot_geoms[0], 0]
    bias = self.robot.data.encoder_bias[ids][:, self.jid]
    com = m.body_ipos[ids][:, self.torso_body] - self.torso_ipos_default
    self.xi[ids] = torch.cat([fr[:, None], bias, com], dim=-1)

  # ------------------------------------------------------------------ reset / step
  def reset(self, ids: torch.Tensor, clip: torch.Tensor, frame: torch.Tensor, noise: bool | None = None):
    if ids.numel() == 0:
      return
    noise = self.reset_noise if noise is None else noise
    self.env.reset(env_ids=ids)  # fires the per-episode DR events
    L = self.lib
    g = L.gidx(clip, frame)
    pos, quat, v, w = L.root_pos[g].clone(), L.root_quat[g].clone(), L.root_v[g].clone(), L.root_w[g].clone()
    q, qd = L.joint_pos[g].clone(), L.joint_vel[g].clone()
    if noise:
      u = lambda r, n: r[:, 0] + (r[:, 1] - r[:, 0]) * torch.rand(n, r.shape[0], device=self.device)  # noqa: E731
      pr = u(POSE_RANGE.to(self.device), len(ids))
      pos = pos + pr[:, :3]
      quat = quat_mul(_quat_from_euler(pr[:, 3:]), quat)
      vr = u(VEL_RANGE.to(self.device), len(ids))
      v, w = v + vr[:, :3], w + vr[:, 3:]
      q = q + (torch.rand_like(q) * 2 - 1) * JOINT_RANGE
    lim = self.robot.data.soft_joint_pos_limits[ids][:, self.jid]
    q = torch.clamp(q, lim[..., 0], lim[..., 1])
    pos = pos + self.env.scene.env_origins[ids]
    jp = self.robot.data.default_joint_pos[ids].clone()
    jv = torch.zeros_like(jp)
    jp[:, self.jid], jv[:, self.jid] = q, qd
    self.robot.write_joint_state_to_sim(jp, jv, env_ids=ids)
    self.robot.write_root_state_to_sim(torch.cat([pos, quat, v, w], dim=-1), env_ids=ids)
    self.robot.reset(env_ids=ids)
    self.env.sim.forward()
    self.clip[ids], self.frame[ids], self.ep_step[ids] = clip, frame, 0
    self.prev_action[ids] = 0.0
    self._read_xi(ids)

  def step(self, action: torch.Tensor):
    """Apply action (normalized; joint setpoint = scale * a + default). Returns the transition. A non-finite action
    becomes zero, and a non-finite simulator state or reward ends the episode (counted in ``bad_steps``) with a
    finite placeholder, so NaNs reach neither the replay buffer nor the curriculum."""
    action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    self.env.step(action[:, self.act_perm])
    s_next = self.state()
    self.frame += 1
    self.ep_step += 1
    ref = self.task.ref(self.lib.gidx(self.clip, self.frame))
    bodies = self.tracked_bodies()
    r, _ = self.task.reward(s_next, ref, action, self.prev_action, bodies=bodies)
    an = self.task.anchor
    term = self.task.terminated(bodies[0][:, an], bodies[1][:, an], ref)
    bad = ~torch.isfinite(s_next).all(-1) | ~torch.isfinite(r)
    if bool(bad.any()):
      self.bad_steps += int(bad.sum())
      s_next, r, term = torch.nan_to_num(s_next, nan=0.0, posinf=0.0, neginf=0.0), torch.where(bad, 0.0, r), term | bad
    trunc = (self.frame >= self.lib.clip_len[self.clip] - 1) | (self.ep_step >= self.episode_steps)
    self.prev_action = action.clone()
    return s_next, r, term, trunc & ~term


def _quat_from_euler(rpy: torch.Tensor) -> torch.Tensor:
  ex = torch.zeros_like(rpy)
  qx = quat_from_rotvec(torch.stack([rpy[:, 0], ex[:, 0], ex[:, 0]], -1))
  qy = quat_from_rotvec(torch.stack([ex[:, 0], rpy[:, 1], ex[:, 0]], -1))
  qz = quat_from_rotvec(torch.stack([ex[:, 0], ex[:, 0], rpy[:, 2]], -1))
  return quat_mul(qz, quat_mul(qy, qx))
