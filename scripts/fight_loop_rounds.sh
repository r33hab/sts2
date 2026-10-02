#!/usr/bin/env bash
# Run rl.prior_loop one round per process, until killed. A single long-lived loop process
# kept every round's training memory: a 52 GB footprint on a 36 GB Mac after 27 hours,
# swapping, with training 3.5x slower than in its first rounds (2026-10-02). A fresh
# process per round gives it all back. Resumes wherever the loop's folder says it is.
#
#   bash scripts/fight_loop_rounds.sh ROOT [more `rl.prior_loop run` options]
#   e.g. bash scripts/fight_loop_rounds.sh runs/prior_loop_big600 --workers 14 --device mps --no-export
set -uo pipefail

ROOT="$1"
shift
cd "$(dirname "$0")/.."
while true; do
  # A round is complete once its manifest is written (the loop's own `completed()`).
  done_rounds=$(find "$ROOT" -path "$ROOT/iter_*/manifest.json" 2>/dev/null | wc -l | tr -d ' ')
  PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m rl.prior_loop run \
    --root "$ROOT" --iterations $((done_rounds + 1)) "$@" || sleep 60
done
