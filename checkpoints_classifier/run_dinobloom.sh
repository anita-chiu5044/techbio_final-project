#!/usr/bin/env bash
# run_dinobloom.sh
# Waits for DinoBloom-B checkpoint, runs 3 DinoBloom configs, then eval_seeds on all checkpoints.

set -euo pipefail

CKPT="/home/yucheng/Desktop/techbio/artifacts/checkpoints/dinobloom/DinoBloom-B.pth"
SUMMARY_CSV="/home/yucheng/Desktop/techbio_pipeline_output/medsam_output/inference_summary.csv"
RUNS_DIR="/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="techbio"
MIN_CKPT_SIZE_MB=480  # DinoBloom-B is ~504MB; wait until fully downloaded

# ---- wait for checkpoint ----
echo "[$(date '+%H:%M:%S')] Waiting for DinoBloom-B checkpoint..."
while true; do
    if [[ -f "$CKPT" ]]; then
        SIZE_MB=$(du -m "$CKPT" | cut -f1)
        if [[ "$SIZE_MB" -ge "$MIN_CKPT_SIZE_MB" ]]; then
            echo "[$(date '+%H:%M:%S')] Checkpoint ready (${SIZE_MB}MB)."
            break
        fi
    fi
    sleep 30
    echo "[$(date '+%H:%M:%S')] Still waiting... ($(du -m "$CKPT" 2>/dev/null | cut -f1 || echo 0)MB)"
done

mkdir -p "$RUNS_DIR"

# ---- run 3 DinoBloom configs ----
CONFIGS=(
    "dinobloom_ce_uniform"
    "dinobloom_focal_wrs"
    "dinobloom_focal_wrs_stage2"
)

for CFG in "${CONFIGS[@]}"; do
    OUT_DIR="$RUNS_DIR/$CFG"
    LOG_FILE="$RUNS_DIR/${CFG}.log"

    if [[ -f "$OUT_DIR/metrics.json" ]]; then
        echo "[$(date '+%H:%M:%S')] $CFG already done, skipping."
        continue
    fi

    echo ""
    echo "================================================================"
    echo "[$(date '+%H:%M:%S')] Starting: $CFG"
    echo "================================================================"

    conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/train.py" \
        --config      "$CFG" \
        --summary-csv "$SUMMARY_CSV" \
        --output-dir  "$OUT_DIR" \
        --dinobloom-ckpt "$CKPT" \
        2>&1 | tee "$LOG_FILE"

    echo "[$(date '+%H:%M:%S')] $CFG done."
done

# ---- multi-seed eval on all finished checkpoints ----
echo ""
echo "================================================================"
echo "[$(date '+%H:%M:%S')] Running 5-seed eval on all checkpoints..."
echo "================================================================"

CKPTS=()
for d in "$RUNS_DIR"/*/; do
    if [[ -f "$d/best.pth" ]]; then
        CKPTS+=("$d/best.pth")
    fi
done

conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/eval_seeds.py" \
    --ckpt "${CKPTS[@]}" \
    --summary-csv "$SUMMARY_CSV" \
    --num-seeds 5 \
    --output "$RUNS_DIR/eval_seeds_report.json" \
    2>&1 | tee "$RUNS_DIR/eval_seeds.log"

# ---- comparison table ----
conda run -n "$CONDA_ENV" python "$SCRIPT_DIR/compare_runs.py" \
    --runs-dir "$RUNS_DIR" \
    --output   "$RUNS_DIR/comparison_report.txt"

echo ""
echo "[$(date '+%H:%M:%S')] ALL DONE."
echo "  Per-run comparison:  $RUNS_DIR/comparison_report.txt"
echo "  5-seed eval report:  $RUNS_DIR/eval_seeds_report.json"
echo "  5-seed eval log:     $RUNS_DIR/eval_seeds.log"
