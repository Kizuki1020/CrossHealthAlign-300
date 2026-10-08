# CrossHealthAlign-300 (DHJ-26-2760) — public data deposit

Benchmark, item-level model outputs, frozen evaluation tables, and scripts
to reproduce the quantitative results for the Digital Health manuscript:

**Cognitive Misalignment Among Patients, Clinicians, and Machines:
A Cognitive Alignment Framework for Digital Health Interfaces**

Authors: Wenbo Gao, Yangyang Gao, Shubin Zhou, Jie Dong  
Journal / manuscript ID: *Digital Health* (SAGE), DHJ-26-2760

This folder is the public data repository for the manuscript. A DOI has not been minted.

## Contents

| Path | Description |
|------|-------------|
| `01_Benchmark/CrossHealthAlign-300.json` | 300 items (query, context, 4 candidates, gold label) |
| `01_Benchmark/expert_labels.json` | Gold labels keyed by `item_id` |
| `02_Model_Outputs/` | Item-level outputs: GPT-5.5, Claude-Sonnet-4.6, DeepSeek-V3.2 |
| `04_Evaluation/Frozen_Results_v1.csv` | **Authoritative** long-format metrics for the revised manuscript |
| `04_Evaluation/evaluation_summary.csv` | Regenerated wide summary (3-class Macro-F1) |
| `04_Evaluation/evaluation_metrics.json` | Regenerated detailed metrics |
| `04_Evaluation/advanced_stats.json` | McNemar / two-proportion / OA-confidence extras |
| `04_Evaluation/confusion_matrix.csv` | Packaged confusion matrices |
| `04_Evaluation/evaluation_*_packaged*` | Legacy packaged files (see note below) |
| `05_Reproducibility/` | `recalc_metrics.py`, optional `build_figures_v3.py`, requirements |
| `LICENSE.txt` | CC BY 4.0 + terminology notice |
| `MANIFEST.txt` | File list with sizes |

## Terminology source (important)

Candidate concepts were taken from the **OAC Consumer Health Vocabulary 2022**
flat-file release (concepts file dated **2011-02-04**). Legacy CUIs were
**not** re-verified against a current UMLS Metathesaurus release. This deposit
does **not** include the CHV source files or a UMLS Metathesaurus release.
Users of CrossHealthAlign-300 are not required to hold a UMLS license.

## Labels and AI assistance during construction

- Gold-standard labels were assigned by the authors.
- During pilot construction, Claude drafted candidate 3, candidate 4, and
  clinical contexts for a 48-item pilot; those drafts were marked for human
  review and were not used as gold labels.
- The three evaluated systems (GPT-5.5, Claude-Sonnet-4.6, DeepSeek-V3.2) are
  study objects described in the manuscript Methods; their outputs are in
  `02_Model_Outputs/`.

## Benchmark composition

300 consumer-health query items, 100 per gold class:

- `EQUIVALENT`
- `RELATED_NOT_EQUIVALENT`
- `NO_JUSTIFIED_EQUIVALENT`

Each item has four candidates (`c1`–`c4`). `CONTEXT_DEPENDENT` may appear as a
**model output** label; it is not a gold class.

## Reproduce metrics

```bash
cd 05_Reproducibility/
pip install -r requirements.txt
python recalc_metrics.py
```

Or: `bash run_all.sh`

Macro-F1 in the manuscript is the **unweighted mean over the three gold
classes** (CONTEXT_DEPENDENT excluded). ECE uses 10 equal-width bins on [0, 1].

Expected key values (also in `Frozen_Results_v1.csv`):

| Model | Accuracy | Macro-F1 (3-class) | OA | ARR | ECE |
|-------|----------|--------------------|----|-----|-----|
| GPT-5.5 | 0.390 | 0.313 | 0.375 | 0.020 | 0.513 |
| Claude-Sonnet-4.6 | 0.383 | 0.338 | 0.350 | 0.160 | 0.445 |
| DeepSeek-V3.2 | 0.433 | 0.347 | 0.685 | 0.240 | 0.526 |

### Legacy packaged Macro-F1

`evaluation_summary_packaged_4class_macro.csv` and
`evaluation_metrics_packaged.json` come from an earlier packaging step whose
`macro_f1` column averaged four labels (including CONTEXT_DEPENDENT). Those
values (~0.235 / 0.254 / 0.260) are **not** the manuscript Macro-F1. Prefer
`Frozen_Results_v1.csv` or the regenerated `evaluation_summary.csv`.

## What is intentionally excluded

- Raw OAC CHV zip / flat files / concept TSVs
- UMLS Metathesaurus or UMLs-master materials
- BioMedICUS weights and other large binaries
- Manuscript Word/PDF files and response letters
- Any clinical spreadsheet or file containing protected health information
- Author contact spreadsheets, CVs, phone numbers

## License

CC BY 4.0 for the benchmark, labels, model outputs, tables, and code in this
deposit, unless a file header says otherwise. See `LICENSE.txt` for the
terminology notice.

## Citation

Please cite the published *Digital Health* article (DHJ-26-2760) when
available, and this deposit once a DOI is assigned. Do not invent a DOI.
