#!/usr/bin/env bash
# Download the Wan2.2-S2V-14B checkpoint (~45.6 GB) into /models.
#
# The repository is public (Apache-2.0); HF_TOKEN is only needed if Hugging Face
# rate-limits anonymous downloads.
#
#   export HF_TOKEN=hf_...     # optional
#   ./scripts/download-models.sh
#
# Env: MODEL_ID (default /models/Wan2.2-S2V-14B), HF_TOKEN, HF_MAX_WORKERS
set -euo pipefail

MODELS="${MODEL_ID:-/models/Wan2.2-S2V-14B}"
WORKERS="${HF_MAX_WORKERS:-8}"

echo "Wan2.2-S2V-14B weights -> ${MODELS}"
echo "  ~45.6 GB: 32.6 GB DiT (4 shards), 11.4 GB UMT5-XXL, 0.5 GB VAE,"
echo "           wav2vec2 + UMT5 tokenizer (~2.6 GB)"
echo "  assets/ and the duplicate wav2vec2 formats are skipped"

mkdir -p "${MODELS}"

if [ -x /opt/wan/.venv/bin/python ] || command -v python >/dev/null 2>&1; then
  exec python -m app.cli download --model-dir "${MODELS}" --workers "${WORKERS}"
fi

echo "error: python not found" >&2
exit 1
