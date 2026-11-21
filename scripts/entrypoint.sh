#!/usr/bin/env bash
# Entrypoint for the Wan2.2-S2V image.
#
#   <no args>            doctor (preflight report)
#   doctor               preflight report
#   download             fetch the 45.6 GB checkpoint from Hugging Face
#   plan <performance>   what a render will produce, no GPU needed
#   perform <folder>     render one performance (start image + audio + prompt)
#   batch <folder>       render every performance under a folder
#   serve                HTTP API on $PORT (default 8000)
#   shell                interactive bash
#   <anything else>      executed as-is (e.g. python -c "...", pytest tests/)
#
# Every command runs from /opt/wan-docker so `python -m app.cli` resolves.
set -euo pipefail

cd /opt/wan-docker

case "${1:-doctor}" in
  doctor|download|plan|perform|batch|serve|selftest|version)
    exec python -m app.cli "$@"
    ;;
  shell|bash)
    shift || true
    exec /bin/bash "$@"
    ;;
  -*)
    exec python -m app.cli "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
