#!/usr/bin/env bash
# Convenience wrapper around docker compose for Wan2.2-S2V.
#
#   ./scripts/run.sh                     preflight report (doctor)
#   ./scripts/run.sh build               build the image
#   ./scripts/run.sh download            fetch the 45.6 GB checkpoint
#   ./scripts/run.sh plan shot-01        plan one performance (no GPU)
#   ./scripts/run.sh perform shot-01     render one performance
#   ./scripts/run.sh batch scene-01      render every performance in a folder
#   ./scripts/run.sh serve               HTTP API on :8000
#   ./scripts/run.sh shell               interactive shell in the image
#   ./scripts/run.sh <cmd...>            passed to the container verbatim
#
# Performances live in ./performances/<name>/ (start.jpg, audio.wav, prompt.txt)
# and are mounted at /performances. Use paths relative to that mount.
#
# Git Bash rewrites leading-slash arguments, hence MSYS_NO_PATHCONV.
set -euo pipefail

cd "$(dirname "$0")/.."
export MSYS_NO_PATHCONV=1
export COMPOSE_CONVERT_WINDOWS_PATHS="${COMPOSE_CONVERT_WINDOWS_PATHS:-1}"

command -v docker >/dev/null 2>&1 || { echo "error: docker not found on PATH" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "error: 'docker compose' (v2) is required" >&2; exit 1; }

cmd="${1:-doctor}"
shift || true

case "$cmd" in
  build)
    exec docker compose build "$@"
    ;;
  download)
    exec docker compose --profile download run --rm downloader "$@"
    ;;
  serve)
    exec docker compose run --rm --service-ports wan22 serve "$@"
    ;;
  doctor|plan|perform|batch|version|shell)
    exec docker compose run --rm wan22 "$cmd" "$@"
    ;;
  *)
    exec docker compose run --rm wan22 "$cmd" "$@"
    ;;
esac
