#!/usr/bin/env python3
"""
Fixed figure generation for CrossHealthAlign-300 (manuscript Figures 2-4).

Corrections vs the original Supplementary_Materials/05_Reproducibility/build_figures_v3.py:

  1. Figure numbering matches the manuscript:
       Figure 2 = Over-Alignment Rate + performance metrics (2 panels)
       Figure 3 = Prediction distribution (stacked bars, 4 output labels)
       Figure 4 = Confidence calibration (reliability + class-level gaps)
     (Figure 1 is a conceptual schematic drawn by the authors -- not generated.)
  2. All confidence intervals are computed from the data (2,000 bootstrap
     resamples) -- no hardcoded CI values.
  3. Reliability diagrams use 10 EQUAL-WIDTH confidence bins (equal-width
     ECE, not equal-frequency).
  4. Paths are resolved relative to this script's location: reads from
     Supplementary_Materials/, writes figures to Revision_Working/figures/.

Reuses metric functions from recalc_metrics.py (single source of truth).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recalc_metrics as rm  # noqa: E402

FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Color palette (retained from the original script)
UC = "#B8C4CE"
HC = "#7A93A8"
RO = "#3D5A73"
SC = "#8FA69C"
SIG = "#C97064"
GRD = "#D9DDE1"
TXT = "#2B2F33"
BKG = "#F7F8F9"
DIV = "#E4E7EA"

MODELS = rm.MODELS
MODEL_LABELS = rm.MODEL_LABELS
MODEL_ORDER = [MODEL_LABELS[m] for m in MODELS]
MCOLORS = {
    "GPT-5.5": HC,
    "Claude-Sonnet-4.6": UC,
    "DeepSeek-V3.2": RO,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10, "figure.dpi": 200,
    "savefig.dpi": 200, "savefig.bbox": "tight",
    "text.color": TXT, "axes.labelcolor": TXT, "axes.edgecolor": TXT,
    "xtick.color": TXT, "ytick.color": TXT,
})


def savefig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[OK] {name}")


def oa_bootstrap_ci(metrics, n_resample=2000, seed=0):
    """Bootstrap CI for the OA rate (resampling the 200 non-equivalent items)."""
    vec = np.array([metrics["oa_vector"][iid] for iid in sorted(metrics["oa_vector"])], dtype=float)
    rng = np.random.default_rng(seed)
    rates = np.array([rng.choice(vec, size=len(vec), replace=True).mean() for _ in range(n_resample)])
    return float(vec.mean()), float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))


def reliability_bins(metrics, n_bins=10):
    """Equal-width confidence bins -> (bin_conf, bin_acc, bin_n)."""
    confs, corrects = [], []
    for iid in sorted(metrics["golds"]):
        g = metrics["golds"][iid]
        if g in rm.GOLD_CLASSES:
            confs.append(metrics["confs"][iid])
            corrects.append(1.0 if metrics["preds"][iid] == g else 0.0)
    confs = np.array(confs, dtype=float)
    corrects = np.array(corrects, dtype=float)
    bin_conf, bin_acc, bin_n = [], [], []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        bin_n.append(int(mask.sum()))
        if mask.sum() > 0:
            bin_conf.append(float(confs[mask].mean()))
            bin_acc.append(float(corrects[mask].mean()))
        else:
            bin_conf.append(float("nan"))
            bin_acc.append(float("nan"))
    return bin_conf, bin_acc, bin_n


def class_gaps(metrics):
    """Per-gold-class calibration gap = mean confidence - observed accuracy."""
    gaps = {}
    for g in rm.GOLD_CLASSES:
        confs, accs = [], []
        for iid in sorted(metrics["golds"]):
            if metrics["golds"][iid] == g:
                confs.append(metrics["confs"][iid])
                accs.append(1.0 if metrics["preds"][iid] == g else 0.0)
        mean_conf = float(np.mean(confs)) if confs else 0.0
        acc = float(np.mean(accs)) if accs else 0.0
        gaps[g] = mean_conf - acc
    return gaps


# ============================================================
# Figure 2: Over-Alignment Rate (forest) + performance (Cleveland dot)
# ============================================================
def build_fig2(metrics_by_model, mcnemar):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

    # --- Panel A: OA forest plot with bootstrap CIs ---
    ax = axes[0]
    ax.set_facecolor(BKG)
    y_positions = [2.5, 1.5, 0.5]
    for i, m in enumerate(MODELS):
        lb = MODEL_LABELS[m]
        mm = metrics_by_model[m]
        rate, lo, hi = oa_bootstrap_ci(mm)
        y = y_positions[i]
        color = MCOLORS[lb]
        ax.plot([lo * 100, hi * 100], [y, y], "-", color=color, linewidth=2.0, alpha=0.8)
        ax.plot(rate * 100, y, "D", color=color, markersize=10, markeredgecolor="white", markeredgewidth=0.8)
        ax.text(rate * 100, y + 0.25, f"{rate*100:.1f}%", ha="center", fontsize=9, fontweight="bold", color=color)
        ax.text(20, y, lb, ha="right", fontsize=11, fontweight="bold", color=color, va="center")
    ax.axvline(x=33.3, color=SIG, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.text(33.3, 3.2, "Chance\n(33.3%)", ha="center", fontsize=7, color=SIG, alpha=0.8)
    ax.set_xlabel("Over-Alignment Rate (%, 95% CI, 2,000 bootstrap resamples)", fontweight="bold")
    ax.set_xlim(20, 80)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks([])
    ax.grid(axis="x", color=GRD, linewidth=0.5)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title("A. Over-Alignment Rate (non-equivalent items, n=200)", fontweight="bold", fontsize=12, loc="left")

    # --- Panel B: Cleveland dot plot of performance metrics ---
    ax = axes[1]
    ax.set_facecolor(BKG)
    metric_names = ["Accuracy", "Macro-F1", "F1-EQUIV", "F1-RELATED", "F1-NO-EQUIV", "ARR"]
    for i, m in enumerate(MODELS):
        lb = MODEL_LABELS[m]
        mm = metrics_by_model[m]
        vals = [
            mm["accuracy"],
            mm["macro_f1"],
            mm["per_class"]["EQUIVALENT"]["f1"],
            mm["per_class"]["RELATED_NOT_EQUIVALENT"]["f1"],
            mm["per_class"]["NO_JUSTIFIED_EQUIVALENT"]["f1"],
            mm["appropriate_rejection_rate"],
        ]
        color = MCOLORS[lb]
        ys = np.arange(len(metric_names))[::-1]
        off = (i - 1) * 0.12
        ax.plot(vals, ys + off, "o", color=color, markersize=9, label=lb, alpha=0.9)
    ax.set_yticks(np.arange(len(metric_names))[::-1])
    ax.set_yticklabels(metric_names, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Value (higher is better)", fontweight="bold")
    ax.grid(axis="x", color=GRD, linewidth=0.5)
    ax.grid(axis="y", color=GRD, linewidth=0.5, alpha=0.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85, edgecolor=DIV)
    ax.set_title("B. Performance Metrics", fontweight="bold", fontsize=12, loc="left")

    fig.suptitle("Figure 2. Over-Alignment and Performance Metrics", fontweight="bold", fontsize=13, y=1.01)
    plt.tight_layout()
    savefig(fig, "Figure_2_over_alignment_metrics")


# ============================================================
# Figure 3: Prediction distribution (stacked bars)
# ============================================================
def build_fig3(metrics_by_model):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="white")
    pred_labels = rm.ALL_PRED_LABELS
    pred_colors = [RO, SC, HC, UC]  # EQUIV, RELATED, NO-EQUIV, CONTEXT
    gold_rows = ["EQUIVALENT", "RELATED_NOT_EQUIVALENT", "NO_JUSTIFIED_EQUIVALENT"]
    gold_short = ["EQUIV", "RELATED", "NO-EQUIV"]

    for ax_idx, m in enumerate(MODELS):
        ax = axes[ax_idx]
        ax.set_facecolor(BKG)
        lb = MODEL_LABELS[m]
        mm = metrics_by_model[m]
        cm = mm["confusion_matrix"]
        color = MCOLORS[lb]

        y_offsets = [2.5, 1.5, 0.5]
        for i_row, g in enumerate(gold_rows):
            cum = 0.0
            for j_col, p in enumerate(pred_labels):
                count = cm[g][p]
                total = sum(cm[g].values())
                pct = count / total if total else 0.0
                if pct < 0.02:
                    continue
                rect = ax.barh(y_offsets[i_row], pct, height=0.45, left=cum,
                               color=pred_colors[j_col], alpha=0.85, edgecolor="white", linewidth=0.5)
                # red outline on EQUIVALENT predictions among non-equivalent gold
                if p == "EQUIVALENT" and g in rm.NON_EQUIV_CLASSES:
                    rect[0].set_edgecolor(SIG)
                    rect[0].set_linewidth(2.0)
                if pct > 0.08:
                    ax.text(cum + pct / 2, y_offsets[i_row], f"{count}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color="white" if pct > 0.3 else TXT)
                cum += pct

        ax.set_ylim(-0.5, 3.3)
        ax.set_yticks(y_offsets)
        ax.set_yticklabels(gold_short, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Proportion of Predictions", fontsize=9)
        ax.set_title(lb, fontweight="bold", color=color, fontsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=GRD, linewidth=0.5, alpha=0.7)

    legend_elements = [
        Patch(facecolor=RO, alpha=0.85, label="Predicted: EQUIVALENT"),
        Patch(facecolor=SC, alpha=0.85, label="Predicted: RELATED"),
        Patch(facecolor=HC, alpha=0.85, label="Predicted: NO-EQUIV"),
        Patch(facecolor=UC, alpha=0.85, label="Predicted: CONTEXT"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=4, fontsize=9,
               framealpha=0.85, edgecolor=DIV, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Figure 3. Prediction Distribution by Gold Label", fontweight="bold", fontsize=13, y=1.01)
    fig.subplots_adjust(top=0.85, wspace=0.25)
    savefig(fig, "Figure_3_prediction_distribution")


# ============================================================
# Figure 4: Confidence calibration (reliability + class-level gaps)
# ============================================================
def build_fig4(metrics_by_model):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor="white")

    # --- Panels A-C: reliability diagrams (10 equal-width bins) ---
    for i, m in enumerate(MODELS):
        ax = axes[i // 2, i % 2]
        ax.set_facecolor(BKG)
        lb = MODEL_LABELS[m]
        mm = metrics_by_model[m]
        bin_conf, bin_acc, bin_n = reliability_bins(mm, n_bins=10)
        color = MCOLORS[lb]
        ax.plot([0, 1], [0, 1], "--", color=SC, linewidth=1.5, alpha=0.8, label="Perfect calibration")
        for bc, ba, bn in zip(bin_conf, bin_acc, bin_n):
            if np.isnan(bc) or np.isnan(ba):
                continue
            ax.plot(bc, ba, "o", color=color, markersize=max(3, min(14, bn ** 0.5)), alpha=0.85)
        ax.plot(bin_conf, bin_acc, "-", color=color, linewidth=1.5, alpha=0.6)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.grid(color=GRD, linewidth=0.5, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{chr(65+i)}. {lb}  (ECE={mm['ece']:.3f})", fontweight="bold", fontsize=11, loc="left")

    # --- Panel D: class-level calibration gaps ---
    ax = axes[1, 1]
    ax.set_facecolor(BKG)
    classes = ["EQUIV", "RELATED", "NO-EQUIV"]
    x = np.arange(len(classes))
    width = 0.25
    for i, m in enumerate(MODELS):
        lb = MODEL_LABELS[m]
        mm = metrics_by_model[m]
        gaps = class_gaps(mm)
        vals = [gaps["EQUIVALENT"], gaps["RELATED_NOT_EQUIVALENT"], gaps["NO_JUSTIFIED_EQUIVALENT"]]
        color = MCOLORS[lb]
        ax.bar(x + (i - 1) * width, vals, width, color=color, alpha=0.85, label=lb, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color=TXT, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_ylabel("Calibration Gap (Confidence - Accuracy)")
    ax.set_title("D. Class-Level Calibration Gaps", fontweight="bold", fontsize=11, loc="left")
    ax.grid(axis="y", color=GRD, linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.85, edgecolor=DIV)

    fig.suptitle("Figure 4. Confidence Calibration", fontweight="bold", fontsize=13, y=1.0)
    plt.tight_layout()
    savefig(fig, "Figure_4_calibration")


def main():
    expert = rm.load_expert_labels()
    metrics_by_model = {m: rm.compute_model_metrics(m, expert) for m in MODELS}

    pairs = [("gpt-5.5", "claude-sonnet-4-6"), ("gpt-5.5", "deepseek-v3.2"),
             ("claude-sonnet-4-6", "deepseek-v3.2")]
    mcnemar = {}
    for a, b in pairs:
        res = rm.pairwise_mcnemar(metrics_by_model[a], metrics_by_model[b])
        res["pair"] = f"{MODEL_LABELS[a]} vs {MODEL_LABELS[b]}"
        res["bonferroni_p"] = min(res["exact_p_two_sided"] * 3, 1.0)
        mcnemar[res["pair"]] = res

    print("Building fixed figures (manuscript numbering)...")
    build_fig2(metrics_by_model, mcnemar)
    build_fig3(metrics_by_model)
    build_fig4(metrics_by_model)
    print(f"\nDone. Files in {FIG_DIR}:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(" ", f)


if __name__ == "__main__":
    main()
