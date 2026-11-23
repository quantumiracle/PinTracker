# PinTracker

A whole-body motion tracker for the **Unitree G1 (29 DoF)**: given a reference motion (joint angles and root pose
over time), it outputs joint position targets at 50 Hz that make the simulated robot follow the motion. It was
trained with the SuperMimic model-based recipe on **1,592 hours** of Movens-5M reference motion.

> **Private research release.** The weights were trained on Movens-5M, whose license grants no redistribution;
> do not publish them or share this repository outside the group. See [License and data](#license-and-data).

An illustrated walkthrough with sample videos is in [`docs/index.html`](docs/index.html) (open it locally).

## Results

All numbers are simulation results in MuJoCo (mjlab G1), evaluation protocol: each clip plays once from its first
frame, no domain randomization, no observation noise. Neither PinTracker nor SONIC was trained on these sets.

**AMASS** — 1,602 clips drawn at random from TWIST's retargeted AMASS (5.7 h). The SONIC column is NVIDIA's
released GEAR-SONIC run through the same simulator; the table uses the 1,202 clips SONIC finished on.

| | PinTracker | SONIC |
|---|---|---|
| tracked to the end (our termination) | **95.8%** | 92.2% |
| success, SONIC paper criterion (root height > 0.25 m or orientation > 1 rad = fail) | **94.2%** | 92.2% |
| normalized return | **0.859** | 0.810 |
| root-relative MPJPE | 35 mm | **29 mm** |
| velocity / acceleration error (mm/frame, mm/frame²) | **5.2** / 1.5 | 5.9 / 1.5 |

On all 1,602 clips PinTracker tracks 1,536 (95.9%) to the end.

**LAFAN1** — all 40 clips (147 min; long, varied, including sprints, jumps and get-ups).

| | tracked to the end | share of time kept up | return |
|---|---|---|---|
| PinTracker | 24 / 40 | 82% | 0.54 |
| SONIC | 25 / 40 | 80% | 0.58 |
| reference: a tracker trained on these 40 clips | 40 / 40 | 100% | 0.90 |

SONIC does not observe the reference root position (its G1 mode reads joints and root orientation only), so it drifts
in the world frame; that lowers its return but not its success or MPJPE. Its lower MPJPE means it follows the body
pose more closely on the clips it completes.

## Install

Linux with an NVIDIA GPU (mjlab runs the simulator on MuJoCo Warp).

```bash
git clone git@github.com:quantumiracle/PinTracker.git && cd PinTracker
pip install -e .
scripts/download_checkpoint.sh          # needs `gh auth login` with access to this private repo
```

or download `pintracker_g1_big_full.pt` from the v0.1.0 release page into `checkpoints/`
(sha256 `811ea2f3dd5c7f3f66d2e2d7a08ddbd4510ee8e4d11fbd68aea1d5d12b6cf55c`, 360 MB).

## Quick start

```bash
# your motions -> 50 Hz clips (GMR / TWIST / Movens style pickles; 23- or 29-DoF G1)
python scripts/convert_motion.py motions/*.pkl --out clips/
# track them in simulation, write per-clip metrics and videos (robot solid, reference blue ghost)
python scripts/track.py --ckpt checkpoints/pintracker_g1_big_full.pt --clips clips/ --videos out/videos
```

From Python:

```python
from pintracker import load_policy
from pintracker.track import track, summarize, render

policy = load_policy("checkpoints/pintracker_g1_big_full.pt")
env, rows, rollout = track(policy, ["clips/walk.npz", "clips/dance.npz"])
print(summarize(rows))            # success rate, return, MPJPE, ...
render(env, rollout, "out/videos")
```

## Motion format

One `.npz` per clip, float32, 50 Hz (`scripts/convert_motion.py` / `pintracker.convert.to_clip` produce it from any
frame rate):

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

`policy(hist, goal) -> action`, called at 50 Hz (`pintracker/task.py` defines every quantity exactly).

- **`hist`** `(B, 10, 93)`: the last 10 proprioceptive observations, newest last. Each is
  `[q − q_default (29), q̇ (29), pelvis angular velocity in the pelvis frame (3), gravity in the pelvis frame (3),
  previous action (29)]`.
- **`goal`** `(B, 670)`: the next 10 reference frames (t+1 … t+10), 67 numbers each:
  `[q_ref (29), q̇_ref / 5 (29), reference torso position in the robot's torso frame (3),
  reference torso orientation relative to the robot's torso as a 6-D rotation (6)]` (`Tracking.goal`).
- **`action`** `(B, 29)`: normalized; joint position target = `q_default + action_scale · action`. The checkpoint
  carries `q_default` and `action_scale` (`policy.ref_q0`, `policy.ref_scale`, mjlab's G1 values); PD gains are
  mjlab's G1 actuator configuration.

The goal contains the reference torso pose **relative to the robot's own torso**, so the tracker corrects drift in the
world frame; on hardware that needs a global pose estimate of the torso (motion capture or state estimation).

## Model and training

- Policy: goal encoder (670 → 7680 → 3840 → 1920 → 960 → 32, tanh) and decoder (32 + 10 × 93 → 7680 → 3840 → 1920
  → 960 → 29), ELU, 90 M parameters; the action is a residual on the next reference pose. The checkpoint holds the
  exponential moving average of the weights (decay 0.995), which is the policy that was evaluated.
- Training: SuperMimic's model-based recipe (world model 3 × 3584, imagined 32-step rollouts, Muon), with domain
  randomization of foot friction, joint encoder bias and torso center of mass, and observation and action noise.
  1. 9,200 iterations on a 100 h Movens subset (8 × H100, data parallel).
  2. 2,000 iterations on all 1,592 h (8 × H200, 32,768 environments, learning rate 4e-4).
  3. 1,000 more iterations on all 1,592 h with the policy frozen for 300 iterations, then a 500-iteration warm-up
     to a policy learning rate of 1e-4.
  About 16 GPU-hours of training in total.

## Limitations

- Simulation only; not tested on a real robot.
- Large: 90 M parameters (360 MB in float32).
- Falls on all six LAFAN1 fall-and-get-up clips (lying on the floor is not tracked); its other LAFAN1 failures come
  late in multi-minute clips (walking, fighting, dancing, after 1–4 minutes).
- Its root-relative MPJPE on AMASS is 6 mm higher than SONIC's (35 vs 29 mm).

## License and data

- **Weights**: trained on Movens-5M (license "other", no redistribution granted). Do not redistribute; internal
  research use only until the Movens terms are confirmed.
- **Evaluation data** (not included): AMASS via TWIST (non-commercial research, no redistribution), LAFAN1
  (Ubisoft, CC BY-NC-ND 4.0). The sample videos in `docs/media` are renders of these motions; keep them private.
- **SONIC** (comparison only, not included): NVIDIA GEAR-SONIC, NVIDIA Open Model License.
- **Code**: the files in `pintracker/` come from the SuperMimic reproduction this model was trained with.
