# Reproducing quantitative results

## Setup

```bash
cd 05_Reproducibility/
pip install -r requirements.txt
```

## Recalculate all metrics

```bash
python recalc_metrics.py
```

Expected runtime: under 30 seconds after dependencies are installed.

### Metrics

| Metric | Definition | Manuscript |
|--------|------------|------------|
| Accuracy | Correct predictions / 300 | Table 5 |
| Macro-F1 (3-class) | Unweighted mean of per-gold-class F1 | Table 5 |
| Class-specific F1 | F1 for EQUIVALENT / RELATED / NO-EQUIV | Table 5 |
| OA | Non-equivalent items predicted EQUIVALENT / non-equivalent items | Table 5, Fig 2 |
| ARR | Correct NO_JUSTIFIED_EQUIVALENT / gold NO_JUSTIFIED_EQUIVALENT | Table 5, Fig 2 |
| ECE | Equal-width 10-bin expected calibration error | Table 6 |

### Authoritative frozen table

`../04_Evaluation/Frozen_Results_v1.csv` is the frozen long-format table used
for the revised manuscript. Re-running `recalc_metrics.py` should regenerate
matching values.

### Packaged legacy summary

`../04_Evaluation/evaluation_summary_packaged_4class_macro.csv` is retained
from an earlier packaging step. Its `macro_f1` column averages four labels
(including CONTEXT_DEPENDENT) and does **not** match the manuscript Macro-F1.
Use `Frozen_Results_v1.csv` or the regenerated `evaluation_summary.csv`.

## Optional figures

```bash
python build_figures_v3.py
```

Outputs go to `./figures/`. This deposit does not ship the manuscript figure
binaries.
