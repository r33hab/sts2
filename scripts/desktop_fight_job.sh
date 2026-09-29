#!/usr/bin/env bash
# Run fight learning (rl.prior_loop) around the clock on the Linux desktop set up by
# desktop_setup.sh. Safe to re-run: it tops up the recorded fights and restarts nothing
# that is already running.
#
#   bash ~/sts2/scripts/desktop_fight_job.sh
#
# 1. Copies the recorded fight starts the loop trains on -- only the shard files of the
#    Mac's full-run and stage-2 jobs -- to the path the loop reads them from
#    (rl/prior_loop/real_fights.py JOBS: ~/Documents/git/sts2-wt/training/runs).
# 2. Starts the loop in tmux session "fight" (log: ~/fight_loop.log), exporting nothing.
# 3. Starts tmux session "sync": every 30 minutes the loop's small files (status, the
#    champion's networks, evaluations -- not the generated training data) go back to the
#    Mac under ~/Documents/git/sts2-wt/desktop/prior_loop.
#
# Overrides: MAC=user@host, WORKERS (default: all threads but 2).
set -euo pipefail

MAC="${MAC:-user@your-mac.local}"
WORKERS="${WORKERS:-$(($(nproc) > 3 ? $(nproc) - 2 : 1))}"
REPO="$HOME/sts2"
JOBS="$HOME/Documents/git/sts2-wt/training/runs"
ROOT="$REPO/runs/prior_loop"
export PATH="$HOME/.local/bin:$HOME/.dotnet:$PATH" DOTNET_ROOT="$HOME/.dotnet"
export PYTHONUTF8=1 LANG=C.UTF-8 UV_PYTHON=3.12

step() { printf '\n== %s\n' "$*"; }

step "recorded fights from $MAC"
for job in full stage2; do
  mkdir -p "$JOBS/$job"
  rsync -a --prune-empty-dirs \
    --include='gen_*/' --include='gen_*/shards/' --include='gen_*/shards/r*.jsonl.gz' \
    --exclude='*' "$MAC:Documents/git/sts2-wt/training/runs/$job/" "$JOBS/$job/"
  echo "$job: $(find "$JOBS/$job" -name 'r*.jsonl.gz' | wc -l) shards"
done

step "fight learning: tmux session 'fight', $WORKERS workers"
if tmux has-session -t fight 2>/dev/null; then
  echo "already running"
else
  tmux new-session -d -s fight \
    "cd '$REPO' && PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python -m rl.prior_loop run --root '$ROOT' --workers $WORKERS --no-export 2>&1 | tee -a '$HOME/fight_loop.log'"
  echo "started"
fi

step "sync back to the Mac: tmux session 'sync', every 30 minutes"
if tmux has-session -t sync 2>/dev/null; then
  echo "already running"
else
  tmux new-session -d -s sync "while true; do
      mkdir -p '$ROOT'
      rsync -a --exclude='generated_parts/' --exclude='*.npz' '$ROOT/' '$MAC:Documents/git/sts2-wt/desktop/prior_loop/' &&
        date '+%F %T synced' || date '+%F %T sync failed'
      sleep 1800
    done 2>&1 | tee -a '$HOME/fight_sync.log'"
  echo "started"
fi

step "done"
cat <<EOF
Watch it:      tmux attach -t fight     (leave with Ctrl+B, then D)
Its status:    cd ~/sts2 && PYTHONPATH=src uv run python -m rl.prior_loop status --root '$ROOT'
Stop it:       tmux kill-session -t fight; tmux kill-session -t sync
Keep this Ubuntu window open (minimised is fine): closing every window can stop WSL.
EOF
