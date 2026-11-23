#!/usr/bin/env bash
# Downloads the release checkpoint (private repo: needs `gh auth login` with access to it).
set -euo pipefail
mkdir -p checkpoints
gh release download v0.1.0 --repo quantumiracle/PinTracker --pattern "pintracker_g1_big_full.pt" --dir checkpoints
