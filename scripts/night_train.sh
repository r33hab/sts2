#!/usr/bin/env bash
# Full-run strategy training at real A10 on a machine that is only free at night (the
# work Mac), in the repository desktop_setup.sh made (~/sts2).
#
#   bash ~/sts2/scripts/night_train.sh start    # evening: begin or resume; stops itself at STOP_AT
#   bash ~/sts2/scripts/night_train.sh stop     # stop now: the loop ends its workers and flushes
#   bash ~/sts2/scripts/night_train.sh status
#
# The job lives in ~/sts2/runs/a10 and resumes where it stopped. While it runs, the Mac is
# kept awake on power (keep the lid open), and its small files -- status, log, every
# generation's evaluations, the exported champion; not the shards -- go back to the Mac
# under ~/Documents/git/sts2-wt/night/a10 every 5 minutes.
#
# Overrides: MAC=user@host, STOP_AT (default 07:30), WORKERS (default: cores but 2).
set -euo pipefail

export MAC="${MAC:-user@your-mac.local}"
REPO="$HOME/sts2"
export ROOT="$REPO/runs/a10"
STOP_AT="${STOP_AT:-07:30}"
CPUS=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
WORKERS="${WORKERS:-$((CPUS > 3 ? CPUS - 2 : 1))}"
export PATH="$HOME/.local/bin:$HOME/.dotnet:$PATH" DOTNET_ROOT="$HOME/.dotnet"
export PYTHONUTF8=1 LANG=C.UTF-8 UV_PYTHON=3.12
cd "$REPO"

train() { PYTHONPATH=src uv run python -m rl.run_train "$@"; }

sync_once() {
  rsync -a --prune-empty-dirs \
    --include='*/' --include='*.json' --include='*.log' --include='export/***' --exclude='*' \
    "$ROOT/" "$MAC:Documents/git/sts2-wt/night/a10/" && date '+%F %T synced' || date '+%F %T sync failed'
}

kill_pidfile() {
  local file="$ROOT/$1"
  if [ -f "$file" ]; then
    local pid
    pid=$(cat "$file")
    [ "$pid" != "$PPID" ] && kill "$pid" 2>/dev/null || true
    rm -f "$file"
  fi
}

case "${1:-status}" in
start)
  if [ -f "$ROOT/loop.pid" ] && kill -0 "$(cat "$ROOT/loop.pid")" 2>/dev/null; then
    echo "the loop is already running: refreshing the keep-awake, sync and auto-stop only"
  elif [ -f "$ROOT/config.json" ]; then
    train start --root "$ROOT"
  else
    train start --root "$ROOT" --workers "$WORKERS" --ascension 10 --export-dest "$ROOT/export"
  fi
  now=$(date +%s)
  target=$(date -j -f "%H:%M" "$STOP_AT" +%s 2>/dev/null || date -d "$STOP_AT" +%s)
  [ "$target" -le "$now" ] && target=$((target + 86400))
  seconds=$((target - now))
  kill_pidfile .caffeinate.pid
  nohup caffeinate -s -t "$seconds" >/dev/null 2>&1 &
  echo $! >"$ROOT/.caffeinate.pid"
  kill_pidfile .sync.pid
  nohup bash -c "$(declare -f sync_once); while true; do sync_once; sleep 300; done" \
    >>"$HOME/night_sync.log" 2>&1 &
  echo $! >"$ROOT/.sync.pid"
  kill_pidfile .autostop.pid
  nohup bash -c "sleep $seconds; bash '$REPO/scripts/night_train.sh' stop" >>"$HOME/night_sync.log" 2>&1 &
  echo $! >"$ROOT/.autostop.pid"
  echo "training with $WORKERS workers until $STOP_AT ($((seconds / 3600)) h); progress syncs to the Mac"
  ;;
stop)
  train stop --root "$ROOT" || true
  kill_pidfile .sync.pid
  kill_pidfile .caffeinate.pid
  kill_pidfile .autostop.pid
  sync_once
  ;;
status)
  train status --root "$ROOT"
  ;;
*)
  echo "usage: $0 start|stop|status" >&2
  exit 2
  ;;
esac
