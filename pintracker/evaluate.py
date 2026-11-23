"""Checkpoint benchmark (paper App. G): every clip plays once from frame 0; foot friction 1.0, no DR, no noise.

Normalized return = undiscounted reward sum over the whole clip / its largest possible value; a motion fails
if the training termination triggers. Several checkpoints can share one batched simulator (``group``): world i plays
clip i % num_clips with checkpoint i // num_clips.
"""

from __future__ import annotations

import torch

from pintracker.env import TrackingEnv
from pintracker.nets import Actor
from pintracker.task import HIST, R_MAX, obs_clean


@torch.no_grad()
def evaluate_actors(env: TrackingEnv, actors: list[Actor]):
  n, dev, lib = env.n, env.device, env.lib
  ids = torch.arange(n, device=dev)
  clip = ids % lib.num_clips
  env.reset(ids, clip, torch.zeros_like(clip), noise=False)
  o = obs_clean(env.state(), env.xi, env.prev_action, env.default_q)
  hist = o.unsqueeze(1).repeat(1, HIST, 1)
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  failed = torch.zeros(n, dtype=torch.bool, device=dev)
  ret = torch.zeros(n, device=dev)
  steps = int(lib.clip_len.max()) - 1
  C = lib.num_clips
  for _ in range(steps):
    s = env.state()
    ap, aq = env.task.anchor_pose(s)
    g = env.task.goal(env.clip, env.frame, ap, aq)
    a = torch.cat([actor(hist[i * C:(i + 1) * C], g[i * C:(i + 1) * C]) for i, actor in enumerate(actors)])
    _, r, term, _ = env.step(a)
    finished = env.frame >= lib.clip_len[env.clip] - 1
    ret += torch.where(alive, r, torch.zeros_like(r))
    failed |= alive & term
    alive &= ~term & ~finished
    if not alive.any():
      break
    o = obs_clean(env.state(), env.xi, env.prev_action, env.default_q)
    hist = torch.cat([hist[:, 1:], o.unsqueeze(1)], 1)
  norm = ret / (R_MAX * (lib.clip_len[clip] - 1).float())
  return norm.view(len(actors), C), failed.view(len(actors), C)


def evaluate_actor(env: TrackingEnv, actor: Actor):
  norm, failed = evaluate_actors(env, [actor])
  return norm[0], failed[0]
