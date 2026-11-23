"""Track reference clips in simulation and report metrics (and optionally write videos).

  python scripts/track.py --ckpt checkpoints/pintracker_g1_big_full.pt --clips path/to/clips_dir --videos out/videos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pintracker import load_policy
from pintracker.track import render, summarize, track


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--clips", required=True, nargs="+", help="clip .npz files or folders of them")
  ap.add_argument("--out", default="results.json")
  ap.add_argument("--videos", default="", help="folder for one mp4 per clip (robot solid, reference ghost)")
  ap.add_argument("--max-seconds", type=float, default=40.0)
  ap.add_argument("--device", default="cuda:0")
  a = ap.parse_args()
  files = sorted(f for c in a.clips for f in (Path(c).glob("*.npz") if Path(c).is_dir() else [Path(c)]))
  policy = load_policy(a.ckpt, a.device)
  env, rows, roll = track(policy, files, a.device)
  summary = summarize(rows)
  json.dump(dict(summary=summary, rows=rows), open(a.out, "w"), indent=1)
  print(json.dumps(summary, indent=1))
  if a.videos:
    for p in render(env, roll, a.videos, max_seconds=a.max_seconds):
      print("wrote", p)


if __name__ == "__main__":
  main()
