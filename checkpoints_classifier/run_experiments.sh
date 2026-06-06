#!/usr/bin/env bash
# Legacy wrapper. Prefer calling retrain_pipeline.py directly.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/retrain_pipeline.py" "$@"
