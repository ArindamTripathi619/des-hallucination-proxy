#!/usr/bin/env bash
# run_expanded_pipeline.sh — Execute full pipeline with 9-model expanded data
# Run AFTER 02c_add_models.py completes
#
# Usage:
#   source .venv/bin/activate && bash run_expanded_pipeline.sh
set -euo pipefail

echo "============================================================"
echo "  EXPANDED PIPELINE (9 Models)"
echo "============================================================"
echo ""

SRC="$(dirname "$0")/src"

echo "Step 1/4: Scoring (03_scoring.py --expanded)"
echo "------------------------------------------------------------"
python "$SRC/03_scoring.py" --expanded
echo ""

echo "Step 2/4: Calibration & Tables (04_calibration.py --expanded)"
echo "------------------------------------------------------------"
python "$SRC/04_calibration.py" --expanded
echo ""

echo "Step 3/4: Analysis & AUROC curves (05_analysis.py --expanded)"
echo "------------------------------------------------------------"
python "$SRC/05_analysis.py" --expanded
echo ""

echo "Step 4/4: Robustness analyses (06_robustness.py --expanded)"
echo "------------------------------------------------------------"
python "$SRC/06_robustness.py" --expanded
echo ""

echo "============================================================"
echo "  ✅ ALL PIPELINE STEPS COMPLETE"
echo "============================================================"
echo "  Outputs in: outputs/tables/"
echo "  Next: re-run notebooks/figures.ipynb"
