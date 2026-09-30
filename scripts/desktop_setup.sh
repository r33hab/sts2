#!/usr/bin/env bash
# Set up this repository's simulator and trainers on a Linux desktop (WSL2 Ubuntu on
# Windows) or another Mac, pulling the code from the Mac over SSH. Safe to re-run: it updates the code,
# rebuilds and re-tests. Get and run it with one line (the Mac needs Remote Login on):
#
#   ssh user@your-mac.local cat Documents/git/sts2/scripts/desktop_setup.sh > setup.sh && bash setup.sh
#
# Overrides: MAC=user@host (where the code comes from and results go back to),
# BRANCH, SRC (clone source), RID (build target), SKIP_SSH=1 (no key setup).
set -euo pipefail

MAC="${MAC:-user@your-mac.local}"
BRANCH="${BRANCH:-spire-coach}"
SRC="${SRC:-$MAC:Documents/git/sts2}"
MACOS=$([ "$(uname -s)" = Darwin ] && echo 1 || true)
if [ -n "$MACOS" ]; then
  RID="${RID:-osx-arm64}"
  cpus() { sysctl -n hw.ncpu; }
  RC="$HOME/.zshrc"
else
  RID="${RID:-linux-x64}"
  cpus() { nproc; }
  RC="$HOME/.bashrc"
fi
REPO="$HOME/sts2"
SUDO=$([ "$(id -u)" = 0 ] || echo sudo)

step() { printf '\n== %s\n' "$*"; }

if [ -n "$MACOS" ]; then
  step "Xcode command line tools (the native build's linker)"
  if ! xcode-select -p >/dev/null 2>&1; then
    echo "Run: xcode-select --install   (click Install), then run this script again."
    exit 1
  fi
else
step "packages (asks for your Linux password once)"
need=()
for pkg in clang zlib1g-dev libicu-dev git rsync curl tmux ca-certificates procps; do
  dpkg -s "$pkg" >/dev/null 2>&1 || need+=("$pkg")
done
if ((${#need[@]})); then
  $SUDO apt-get update -q && $SUDO apt-get install -yq "${need[@]}"
fi
fi

if [ "${SKIP_SSH:-0}" != 1 ]; then
  step "SSH key for the Mac (asks for the Mac's password once)"
  [ -f "$HOME/.ssh/id_ed25519" ] || ssh-keygen -q -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$MAC" true 2>/dev/null; then
    ssh-copy-id -i "$HOME/.ssh/id_ed25519.pub" "$MAC"
  fi
fi

step "code: $SRC ($BRANCH)"
if [ -d "$REPO/.git" ]; then
  git -C "$REPO" fetch -q origin "$BRANCH"
  git -C "$REPO" checkout -q "$BRANCH"
  git -C "$REPO" merge -q --ff-only "origin/$BRANCH"
else
  git clone -q -b "$BRANCH" "$SRC" "$REPO"
fi

step "uv and .NET 9 (in your home folder)"
export PATH="$HOME/.local/bin:$HOME/.dotnet:$PATH" DOTNET_ROOT="$HOME/.dotnet" DOTNET_CLI_TELEMETRY_OPTOUT=1
export PYTHONUTF8=1 LANG=C.UTF-8 UV_PYTHON=3.12  # the code needs 3.12 (PEP 701 f-strings)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x "$HOME/.dotnet/dotnet" ] ||
  curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 9.0 --install-dir "$HOME/.dotnet"
grep -q '# sts2 desktop' "$RC" 2>/dev/null || cat >>"$RC" <<'EOF'
# sts2 desktop
export PATH="$HOME/.local/bin:$HOME/.dotnet:$PATH" DOTNET_ROOT="$HOME/.dotnet" DOTNET_CLI_TELEMETRY_OPTOUT=1
export PYTHONUTF8=1 LANG=C.UTF-8 UV_PYTHON=3.12
EOF

step "build the simulator ($RID)"
cd "$REPO"
bash scripts/build.sh "$RID"

step "tests"
# Known on Linux (2026-09-29): test_every_offer_the_history_names_is_the_emulators
# (the replay maps one recorded Sapphire Seed choice differently; run-level only) and,
# on a slow machine, the timing test test_a_pooled_call_takes_its_budget.
uv run python -m unittest discover -s tests/python 2>&1 | grep -E "^(FAIL|ERROR): |^Ran |^OK|^FAILED" ||
  echo "SOME TESTS FAILED: paste the lines above to Claude"

step "smoke run: 20 fights of the budget test, $(cpus) workers"
start=$(date +%s)
PYTHONPATH=src uv run python scripts/fight_budget_test.py \
  --limit 20 --workers "$(cpus)" --out "$HOME/fight_budget_smoke.json"
echo "smoke run took $(($(date +%s) - start)) s"

step "done: tell Claude the desktop is set up (and paste the last lines)"
