#!/usr/bin/env bash
# CrossHealthAlign-300: reproduce metrics (and optionally figures)
# Run from 05_Reproducibility/
set -e
PYTHON=$(command -v python3 || command -v python)
echo "[1/2] Installing dependencies..."
$PYTHON -m pip install -q -r requirements.txt
echo "[2/2] Recalculating metrics (3-class Macro-F1)..."
$PYTHON recalc_metrics.py
echo "Done. Outputs in ../04_Evaluation/"
echo "Optional: python build_figures_v3.py  # writes to ./figures/"
