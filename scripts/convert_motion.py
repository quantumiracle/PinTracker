"""Reference motions -> PinTracker clips (50 Hz npz).

  python scripts/convert_motion.py motions/*.pkl --out clips/            # GMR / TWIST pickles (xyzw quats)
  python scripts/convert_motion.py walk.npz --out clips/ --quat-order wxyz  # npz with fps, root_pos, root_rot, dof_pos
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pintracker.convert import convert_pickle, to_clip


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("inputs", nargs="+")
  ap.add_argument("--out", required=True)
  ap.add_argument("--quat-order", default="xyzw", choices=["xyzw", "wxyz"])
  a = ap.parse_args()
  Path(a.out).mkdir(parents=True, exist_ok=True)
  for p in a.inputs:
    out = Path(a.out) / (Path(p).stem + ".npz")
    if p.endswith(".pkl"):
      clip = convert_pickle(p, str(out), a.quat_order)
    else:
      d = np.load(p)
      clip = to_clip(d["root_pos"], d["root_rot"], d["dof_pos"], float(d["fps"]), a.quat_order)
      np.savez(out, **clip)
    print(f"{out}: {len(clip['joint_pos']) / 50:.1f} s")


if __name__ == "__main__":
  main()
