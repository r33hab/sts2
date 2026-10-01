#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Prefer the .NET 9 SDK desktop_setup.sh installs in ~/.dotnet: a shell that has not read
# its rc file finds none (WSL) or an older system SDK first (/usr/local/share/dotnet 8).
if [ -x "$HOME/.dotnet/dotnet" ]; then
  export PATH="$HOME/.dotnet:$PATH" DOTNET_ROOT="$HOME/.dotnet"
fi
# Default runtime to the host platform; override: bash scripts/build.sh linux-x64
if [ -n "${1:-}" ]; then
  RUNTIME="$1"
else
  case "$(uname -sm)" in
    "Darwin arm64")  RUNTIME="osx-arm64" ;;
    "Darwin x86_64") RUNTIME="osx-x64" ;;
    "Linux x86_64")  RUNTIME="linux-x64" ;;
    *)               RUNTIME="win-x64" ;;
  esac
fi

echo "Building Sts2Emulator for $RUNTIME..."

dotnet publish "$REPO_ROOT/src/Sts2Emulator/Sts2Emulator.csproj" \
  -c Release \
  -r "$RUNTIME" \
  --self-contained \
  -o "$REPO_ROOT/out/"

echo "Output: $REPO_ROOT/out/"
ls "$REPO_ROOT/out/"
