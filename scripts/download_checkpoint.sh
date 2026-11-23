#!/usr/bin/env bash
# Downloads the v0.1.0 release checkpoint into checkpoints/.
set -euo pipefail
mkdir -p checkpoints
curl -L -o checkpoints/pintracker_g1_big_full.pt \
  https://github.com/quantumiracle/PinTracker/releases/download/v0.1.0/pintracker_g1_big_full.pt
