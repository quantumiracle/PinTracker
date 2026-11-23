# PinTracker

A whole-body motion tracker for the **Unitree G1 (29 DoF)**. Given a reference motion (joint angles and root pose
over time), it outputs joint position targets at 50 Hz that make the simulated robot follow it. Trained with the
SuperMimic model-based recipe on about **1,600 hours** of human motion extracted from Internet videos and retargeted
to the G1. Two sizes:

| | policy parameters | checkpoint | AMASS tracked to the end |
|---|---|---|---|
| **PinTracker-S** (recommended) | 8.9 M | [`weights/pintracker_g1_small_full.pt`](weights/pintracker_g1_small_full.pt), in this repo, 36 MB | **96.2%** |
| PinTracker-L | 90 M | [v0.1.0 release](https://github.com/quantumiracle/PinTracker/releases/tag/v0.1.0), 360 MB | 95.8% |

The small one is as good and ten times lighter: with this much data, model size is not the bottleneck.

**Blog post with results and sample videos:** <https://quantumiracle.github.io/PinTracker/>

## Results

Simulation results in MuJoCo (mjlab G1). Each clip plays once from its first frame, without domain randomization
or observation noise. Neither PinTracker nor SONIC was trained on this set; SONIC is NVIDIA's released GEAR-SONIC,
run through the same simulator.

**AMASS** — 1,202 clips of a random 1,602-clip draw from TWIST's retargeted AMASS (the clips SONIC finished on):

| | PinTracker-S | PinTracker-L | SONIC |
|---|---|---|---|
| tracked to the end | **96.2%** | 95.8% | 92.2% |
| success, SONIC paper criterion (root height off by > 0.25 m or orientation by > 1 rad = fail) | 94.0% | **94.2%** | 92.2% |
| normalized tracking return | **0.860** | 0.859 | 0.810 |
| root-relative MPJPE | 36 mm | 35 mm | **29 mm** |
| velocity / acceleration error (mm/frame, mm/frame²) | **5.2** / 1.5 | **5.2** / 1.5 | 5.9 / 1.5 |

PinTracker falls less often (PinTracker-S completes 57 AMASS clips where SONIC falls, SONIC 9 the other way round);
SONIC follows the body pose more closely on the clips both complete.

## Install

Linux with an NVIDIA GPU (mjlab simulates on MuJoCo Warp).

```bash
git clone https://github.com/quantumiracle/PinTracker.git && cd PinTracker
pip install -e .
```

PinTracker-S comes with the clone (`weights/pintracker_g1_small_full.pt`, sha256
`034f723de19dc5108f7d5b40090d6ce3bad6b138e686a0222567d21940c88189`). For PinTracker-L, run
`scripts/download_checkpoint.sh` (-> `checkpoints/pintracker_g1_big_full.pt`, 360 MB, from the
[v0.1.0 release](https://github.com/quantumiracle/PinTracker/releases/tag/v0.1.0); sha256
`811ea2f3dd5c7f3f66d2e2d7a08ddbd4510ee8e4d11fbd68aea1d5d12b6cf55c`).

## Quick start

```bash
# your motions -> 50 Hz clips (GMR / TWIST style pickles; 23- or 29-DoF G1)
python scripts/convert_motion.py motions/*.pkl --out clips/
# track them in simulation: per-clip metrics and videos (robot solid, reference blue ghost)
python scripts/track.py --ckpt weights/pintracker_g1_small_full.pt --clips clips/ --videos out/videos
```

From Python:

```python
from pintracker import load_policy
from pintracker.track import track, summarize, render

policy = load_policy("weights/pintracker_g1_small_full.pt")   # or checkpoints/pintracker_g1_big_full.pt
env, rows, rollout = track(policy, ["clips/walk.npz", "clips/dance.npz"])
print(summarize(rows))            # success rate, return, MPJPE, ...
render(env, rollout, "out/videos")
```

## Motion format

One `.npz` per clip, float32, 50 Hz (`scripts/convert_motion.py` / `pintracker.convert.to_clip` resample from any
frame rate and compute the velocities):

| key | shape | meaning |
|---|---|---|
| `root_pos` | (T, 3) | pelvis position, m, world frame, z up |
| `root_quat` | (T, 4) | pelvis orientation, **wxyz** |
| `root_v`, `root_w` | (T, 3) | pelvis linear / angular velocity, world frame |
| `joint_pos`, `joint_vel` | (T, 29) | joint angles (rad) and velocities, in `pintracker.motion.G1_JOINTS` order |
| `fps` | () | 50 |

Joint order is mjlab's G1 order: left leg (hip pitch, roll, yaw, knee, ankle pitch, roll), right leg, waist (yaw,
roll, pitch), left arm (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw), right arm. 23-DoF motions get
zero wrists.

## Policy interface

`policy(hist, goal) -> action`, called at 50 Hz; `pintracker/task.py` defines every quantity exactly.

- **`hist`** `(B, 10, 93)`: the last 10 proprioceptive observations, newest last:
  `[q − q_default (29), q̇ (29), pelvis angular velocity in the pelvis frame (3), gravity in the pelvis frame (3),
  previous action (29)]`.
- **`goal`** `(B, 670)`: the next 10 reference frames (t+1 … t+10), 67 numbers each:
  `[q_ref (29), q̇_ref / 5 (29), reference torso position in the robot's torso frame (3),
  reference torso orientation relative to the robot's torso as a 6-D rotation (6)]`.
- **`action`** `(B, 29)`: joint position target = `q_default + action_scale · action`; the checkpoint carries
  `q_default` and `action_scale` (`policy.ref_q0`, `policy.ref_scale`), and the PD gains are mjlab's G1 actuators.

Because the goal holds the reference torso pose relative to the robot's torso, the tracker corrects drift in the
world frame; on hardware this needs a global estimate of the torso pose.

## Model and training

The policy is a goal encoder and a proprioception decoder whose output is a residual on the next reference pose; the
checkpoints hold the exponential moving average of its weights. Hidden widths are 2048, 1024, 512, 256 for
PinTracker-S (8.9 M parameters) and 7680, 3840, 1920, 960 for PinTracker-L (90 M). Both were trained with SuperMimic's
model-based reinforcement learning (a learned world model and imagined rollouts), with randomized foot friction, joint
offsets and torso center of mass, plus observation and action noise: first on a 100 h subset of the motion data, then
on all of it. PinTracker-S took about 12 GPU-hours in total, PinTracker-L about 16.

## Limitations

- Simulation only; not yet tested on a real robot.
- Does not track falls and get-ups or lying on the floor.
- Root-relative MPJPE is 6–7 mm higher than SONIC's on AMASS.
- PinTracker-L is large (90 M parameters, 360 MB); prefer PinTracker-S.

## Acknowledgements and licenses

- Evaluation data, not included: [AMASS](https://amass.is.tue.mpg.de/) (retargeted by
  [TWIST](https://github.com/YanjieZe/TWIST)).
  The sample videos are simulator renders of AMASS motions, shown for illustration.
- [SONIC](https://nvlabs.github.io/GEAR-SONIC/) (NVIDIA Open Model License) is used for comparison only and is not
  included.
- The training recipe follows SuperMimic; the code in `pintracker/` comes from our reproduction of it.
