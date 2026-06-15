#!/usr/bin/env bash
# Copy one model's results out of a (locally-synced) Google Drive folder into results/<MODEL_KEY>/.
# Usage: scripts/sync_results_from_drive.sh <DRIVE_CAUSAL_MIMICRY_DIR> <MODEL_KEY> [--with-figures]
set -euo pipefail
SRC="${1:?path to MyDrive/causal_mimicry}"; KEY="${2:?MODEL_KEY}"; WITH_FIG="${3:-}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/results/$KEY"
mkdir -p "$DEST"
shopt -s nullglob
cp -v "$SRC/$KEY/"*.csv "$DEST/" 2>/dev/null || echo "no CSVs found for $KEY"
if [ "$WITH_FIG" = "--with-figures" ]; then cp -v "$SRC/$KEY/"*.png "$DEST/" 2>/dev/null || true; fi
echo "done -> $DEST"
