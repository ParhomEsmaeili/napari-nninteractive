#!/bin/bash
# Launch the nnInteractive fork against THIS worktree
# (feature/timing-instrumentation branch — the timing/gating/preset work).
#
# Only one worktree exists for this fork (unlike napari-clopa's dev/timing split),
# so this is mostly just activate-and-launch — the editable re-point is a cheap,
# harmless safety net in case the env's install ever drifts.
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /home/parhomesmaeili/Virtual_Environments/anaconda3/etc/profile.d/conda.sh
conda activate napari-nninteractive-fork-dev

echo "=== Pointing napari-nninteractive-fork-dev's editable install at $HERE ==="
pip install --no-deps -e "$HERE" --quiet

echo "=== Launching napari-nninteractive (timing) ==="
napari -w napari-nninteractive
