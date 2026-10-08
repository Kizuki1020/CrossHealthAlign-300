# Reproducibility Environment

## Python

Python 3.9 or later (tested with 3.11+).

## Install

```bash
cd 05_Reproducibility/
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`.

## Recalculate metrics

```bash
python recalc_metrics.py
```

Reads:
- `../01_Benchmark/expert_labels.json`
- `../02_Model_Outputs/{gpt-5.5,claude-sonnet-4-6,deepseek-v3.2}.json`

Writes into `../04_Evaluation/`:
- `evaluation_metrics.json`
- `evaluation_summary.csv` (3-class Macro-F1)
- `Frozen_Results_v1.csv`
- `advanced_stats.json`

Macro-F1 is the unweighted mean over the three gold classes
(EQUIVALENT, RELATED_NOT_EQUIVALENT, NO_JUSTIFIED_EQUIVALENT).
CONTEXT_DEPENDENT is a model-output label only and is excluded from Macro-F1.
ECE uses 10 equal-width confidence bins on [0, 1].

## Optional figures

```bash
python build_figures_v3.py
```

Writes PDF/PNG under `05_Reproducibility/figures/`. Figure files are not
part of this data deposit by default.

## One-click

```bash
bash run_all.sh
```
