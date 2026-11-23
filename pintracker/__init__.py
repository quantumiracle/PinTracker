"""PinTracker: a whole-body motion tracker for the Unitree G1 (29 DoF), trained with SuperMimic's model-based
recipe on about 1,600 h of motion extracted from Internet videos. See README.md for the interface."""

from __future__ import annotations

import torch

from pintracker.nets import Actor

__version__ = "0.1.0"


def load_policy(path: str, device: str = "cuda:0") -> Actor:
  """The deployed (EMA-averaged) policy from a checkpoint file; layer widths are read from the weights."""
  ck = torch.load(path, map_location=device, weights_only=True)
  policy = Actor.from_state_dict(ck["actor"]).to(device).eval().requires_grad_(False)
  return policy
