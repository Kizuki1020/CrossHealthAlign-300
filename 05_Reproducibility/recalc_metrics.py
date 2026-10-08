#!/usr/bin/env python3
"""
Fixed metric recalculation for CrossHealthAlign-300.

Corrections vs the original Supplementary_Materials/05_Reproducibility/recalc_metrics.py:

  1. Macro-F1 is computed over the 3 GOLD classes only (unweighted mean):
     EQUIVALENT, RELATED_NOT_EQUIVALENT, NO_JUSTIFIED_EQUIVALENT.
     CONTEXT_DEPENDENT is a *model-output* label, not a gold class, so it is
     excluded from the average (this was the root cause of the wrong 4-class
     Macro-F1 in Table S4).
  2. Expected Calibration Error (ECE) uses 10 EQUAL-WIDTH confidence bins
     on [0,1], not equal-frequency bins.
  3. Pairwise Over-Alignment comparisons use the McNemar exact test (two-sided),
     Bonferroni correction (family size 3), Cohen's h, and the paired risk
     difference with 95% CI.
  4. RELATED vs NO-JUSTIFIED-EQUIVALENT OA is compared with a two-proportion
     test (independent groups, n=100 each) -- NOT McNemar.
  5. OA confidence analysis: median/IQR, Mann-Whitney U, Cliff's delta,
     bootstrap CI (not just a t-test).
  6. OA uncertainty: numerator / denominator / proportion / 95% Wilson CI,
     stratified by gold class.

All confidence intervals are computed from the data (no hardcoding).
Paths are resolved relative to this script's location.
Reads from 01_Benchmark/ and 02_Model_Outputs/ (READ-ONLY); writes to 04_Evaluation/.
"""

import json
import os
import math
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
BENCHMARK_DIR = os.path.join(ROOT, "01_Benchmark")
MODEL_OUTPUTS_DIR = os.path.join(ROOT, "02_Model_Outputs")
OUT_DIR = os.path.join(ROOT, "04_Evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

GOLD_CLASSES = ["EQUIVALENT", "RELATED_NOT_EQUIVALENT", "NO_JUSTIFIED_EQUIVALENT"]
NON_EQUIV_CLASSES = ["RELATED_NOT_EQUIVALENT", "NO_JUSTIFIED_EQUIVALENT"]
ALL_PRED_LABELS = GOLD_CLASSES + ["CONTEXT_DEPENDENT"]
MODELS = ["gpt-5.5", "claude-sonnet-4-6", "deepseek-v3.2"]

MODEL_LABELS = {
    "gpt-5.5": "GPT-5.5",
    "claude-sonnet-4-6": "Claude-Sonnet-4.6",
    "deepseek-v3.2": "DeepSeek-V3.2",
}


def load_expert_labels():
    with open(os.path.join(BENCHMARK_DIR, "expert_labels.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_outputs(model_key):
    with open(os.path.join(MODEL_OUTPUTS_DIR, f"{model_key}.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((center - margin) / denom, (center + margin) / denom)


def cohens_h(p1, p2):
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def exact_mcnemar_p(b, c):
    """Two-sided exact McNemar p-value (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    m = min(b, c)
    p = 0.0
    for k in range(0, m + 1):
        p += math.comb(n, k) * (0.5 ** n)
    p *= 2.0
    return min(p, 1.0)


def paired_risk_diff_ci(b, c, n):
    """Paired risk difference d=(b-c)/n with 95% CI; var=(b+c)/n^2-(b-c)^2/n^3."""
    if n == 0:
        return (float("nan"), float("nan"))
    d = (b - c) / n
    var = (b + c) / (n ** 2) - ((b - c) ** 2) / (n ** 3)
    var = max(var, 0.0)
    se = math.sqrt(var)
    return (d - 1.96 * se, d + 1.96 * se)


def equal_width_ece(confidences, corrects, n_bins=10):
    """ECE with equal-width bins spanning [0,1]."""
    confs = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=float)
    n = len(confs)
    if n == 0:
        return 0.0
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confs[mask].mean()
        bin_acc = corr[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def two_proportion_test(k1, n1, k2, n2):
    """Two-proportion z-test (independent groups). Returns dict with z, p, risk diff CI."""
    p1 = k1 / n1 if n1 else 0.0
    p2 = k2 / n2 if n2 else 0.0
    pooled = (k1 + k2) / (n1 + n2) if (n1 + n2) else 0.0
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)) if (n1 and n2) else float("nan")
    z = (p1 - p2) / se if se else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se_rd = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if (n1 and n2) else float("nan")
    rd = p1 - p2
    rd_lo = rd - 1.96 * se_rd
    rd_hi = rd + 1.96 * se_rd
    return {
        "p1": p1, "p2": p2, "risk_difference": rd,
        "risk_diff_ci_low": rd_lo, "risk_diff_ci_high": rd_hi,
        "z": z, "p_value": p,
    }


def fisher_p(k1, n1, k2, n2):
    """Fisher exact test (two-sided) for 2x2 independent-proportion table."""
    a, b = k1, n1 - k1
    c, d = k2, n2 - k2
    _, p = stats.fisher_exact([[a, b], [c, d]], alternative="two-sided")
    return float(p)


def cliff_delta(x, y):
    """Cliff's delta effect size for two independent samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    greater = sum(1 for a in x for b in y if a > b)
    less = sum(1 for a in x for b in y if a < b)
    return (greater - less) / (nx * ny)


def bootstrap_ci(x, y, stat_fn, n_resample=2000, seed=0, alpha=0.05):
    """Bootstrap percentile CI for a two-sample statistic (pooled resampling)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    est = stat_fn(x, y)
    stats_boot = []
    for _ in range(n_resample):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        stats_boot.append(stat_fn(xb, yb))
    lo = np.percentile(stats_boot, 100 * alpha / 2)
    hi = np.percentile(stats_boot, 100 * (1 - alpha / 2))
    return est, lo, hi


def compute_model_metrics(model_key, expert):
    data = load_model_outputs(model_key)
    results = data["results"]

    golds, preds, confs = {}, {}, {}
    for r in results:
        iid = r["item_id"]
        golds[iid] = expert.get(iid, {}).get("gold_label", "")
        preds[iid] = r.get("pred_label", "")
        confs[iid] = r.get("confidence", 0.0) or 0.0

    n = len(results)

    # Confusion matrix: gold (3) x pred (4)
    cm = {g: {p: 0 for p in ALL_PRED_LABELS} for g in GOLD_CLASSES}
    for iid in golds:
        g, p = golds[iid], preds[iid]
        if g in cm and p in cm[g]:
            cm[g][p] += 1

    # Accuracy over 3 gold classes
    correct = total = 0
    for iid in golds:
        if golds[iid] in GOLD_CLASSES:
            total += 1
            if preds[iid] == golds[iid]:
                correct += 1
    accuracy = correct / total if total else 0.0

    # Per-class precision/recall/F1 for the 3 gold classes
    per_class = {}
    for g in GOLD_CLASSES:
        tp = cm[g][g]
        fp = sum(cm[g2][g] for g2 in GOLD_CLASSES) - tp
        fn = sum(cm[g][p] for p in ALL_PRED_LABELS) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(cm[g].values())
        per_class[g] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }

    # Macro-F1: unweighted mean over the 3 gold classes ONLY
    macro_f1 = float(np.mean([per_class[g]["f1"] for g in GOLD_CLASSES]))

    # Over-alignment: non-equivalent items predicted EQUIVALENT
    oa_by_label = {}
    oa_sum = oa_total = 0
    for g in NON_EQUIV_CLASSES:
        c = cm[g]["EQUIVALENT"]
        t = sum(cm[g].values())
        oa_by_label[g] = {"count": c, "total": t}
        oa_sum += c
        oa_total += t
    oa_rate = oa_sum / oa_total if oa_total else 0.0

    # Appropriate rejection rate = recall of NO_JUSTIFIED_EQUIVALENT
    arr = per_class["NO_JUSTIFIED_EQUIVALENT"]["recall"]
    arr_count = per_class["NO_JUSTIFIED_EQUIVALENT"]["tp"]
    arr_total = per_class["NO_JUSTIFIED_EQUIVALENT"]["support"]

    # ECE (equal-width) and calibration gap
    conf_arr, corr_arr = [], []
    for iid in golds:
        if golds[iid] in GOLD_CLASSES:
            conf_arr.append(confs[iid])
            corr_arr.append(1.0 if preds[iid] == golds[iid] else 0.0)
    ece = equal_width_ece(conf_arr, corr_arr)
    mean_conf = float(np.mean(conf_arr)) if conf_arr else 0.0
    conf_acc_gap = mean_conf - accuracy

    high_conf_errors = sum(
        1 for iid in golds
        if golds[iid] in GOLD_CLASSES and preds[iid] != golds[iid] and (confs[iid] or 0) > 0.8
    )

    # OA vector (for paired McNemar): non-equivalent item -> 1 if over-aligned
    oa_vector = {}
    for r in results:
        iid = r["item_id"]
        if golds[iid] in NON_EQUIV_CLASSES:
            oa_vector[iid] = 1 if preds[iid] == "EQUIVALENT" else 0

    return {
        "model": model_key,
        "model_label": MODEL_LABELS.get(model_key, model_key),
        "n": n,
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "over_alignment_rate": oa_rate,
        "over_alignment_count": oa_sum,
        "non_equiv_total": oa_total,
        "over_alignment_by_label": oa_by_label,
        "appropriate_rejection_rate": arr,
        "appropriate_rejection_count": arr_count,
        "noequiv_total": arr_total,
        "ece": ece,
        "mean_confidence": mean_conf,
        "conf_acc_gap": conf_acc_gap,
        "high_conf_errors": high_conf_errors,
        "oa_vector": oa_vector,
        "preds": preds,
        "golds": golds,
        "confs": confs,
        "confusion_matrix": cm,
    }


def pairwise_mcnemar(metrics_a, metrics_b):
    """McNemar comparison of two models' OA on the same non-equivalent items."""
    ids = sorted(set(metrics_a["oa_vector"]) & set(metrics_b["oa_vector"]))
    b = c = 0
    for iid in ids:
        a = metrics_a["oa_vector"][iid]
        bb = metrics_b["oa_vector"][iid]
        if a == 1 and bb == 0:
            b += 1
        elif a == 0 and bb == 1:
            c += 1
    n = len(ids)
    p = exact_mcnemar_p(b, c)
    h = cohens_h(metrics_a["over_alignment_rate"], metrics_b["over_alignment_rate"])
    rd_lo, rd_hi = paired_risk_diff_ci(b, c, n)
    return {
        "n_items": n,
        "discordant_b": b,
        "discordant_c": c,
        "exact_p_two_sided": p,
        "cohens_h": h,
        "paired_risk_difference": (b - c) / n if n else float("nan"),
        "paired_risk_diff_ci_low": rd_lo,
        "paired_risk_diff_ci_high": rd_hi,
    }


def related_vs_noequiv(metrics):
    """Two-proportion test: OA in RELATED (n=100) vs NO-EQUIV (n=100), independent."""
    rel = metrics["over_alignment_by_label"]["RELATED_NOT_EQUIVALENT"]
    noeq = metrics["over_alignment_by_label"]["NO_JUSTIFIED_EQUIVALENT"]
    k1, n1 = rel["count"], rel["total"]
    k2, n2 = noeq["count"], noeq["total"]
    tp = two_proportion_test(k1, n1, k2, n2)
    fp = fisher_p(k1, n1, k2, n2)
    h = cohens_h(k1 / n1 if n1 else 0.0, k2 / n2 if n2 else 0.0)
    return {
        "related_oa": {"count": k1, "total": n1, "rate": k1 / n1 if n1 else float("nan")},
        "noequiv_oa": {"count": k2, "total": n2, "rate": k2 / n2 if n2 else float("nan")},
        "risk_difference": tp["risk_difference"],
        "risk_diff_ci_low": tp["risk_diff_ci_low"],
        "risk_diff_ci_high": tp["risk_diff_ci_high"],
        "z": tp["z"],
        "z_test_p": tp["p_value"],
        "fisher_exact_p": fp,
        "cohens_h": h,
    }


def oa_confidence_analysis(metrics):
    """Compare confidence between over-aligned vs correctly-rejected non-equivalent items."""
    conf_oa, conf_not = [], []
    for iid, oa in metrics["oa_vector"].items():
        conf = metrics["confs"][iid]
        if oa == 1:
            conf_oa.append(conf)
        else:
            conf_not.append(conf)
    conf_oa = np.asarray(conf_oa, dtype=float)
    conf_not = np.asarray(conf_not, dtype=float)

    def med(x):
        return float(np.median(x)) if len(x) else float("nan")

    def iqr(x):
        return (float(np.percentile(x, 25)), float(np.percentile(x, 75))) if len(x) else (float("nan"), float("nan"))

    if len(conf_oa) > 0 and len(conf_not) > 0:
        u, mw_p = stats.mannwhitneyu(conf_oa, conf_not, alternative="two-sided")
    else:
        u, mw_p = float("nan"), float("nan")

    cd = cliff_delta(conf_oa, conf_not)
    mean_diff, md_lo, md_hi = bootstrap_ci(conf_oa, conf_not, lambda a, b: float(np.mean(a) - np.mean(b)))
    cd_est, cd_lo, cd_hi = bootstrap_ci(conf_oa, conf_not, cliff_delta)

    return {
        "overaligned_n": int(len(conf_oa)),
        "not_overaligned_n": int(len(conf_not)),
        "overaligned_median": med(conf_oa),
        "overaligned_iqr": iqr(conf_oa),
        "not_overaligned_median": med(conf_not),
        "not_overaligned_iqr": iqr(conf_not),
        "mannwhitney_U": float(u),
        "mannwhitney_p": float(mw_p),
        "cliffs_delta": float(cd),
        "cliffs_delta_ci_low": float(cd_lo),
        "cliffs_delta_ci_high": float(cd_hi),
        "mean_conf_diff": float(mean_diff),
        "mean_conf_diff_ci_low": float(md_lo),
        "mean_conf_diff_ci_high": float(md_hi),
    }


def oa_uncertainty(metrics):
    """OA numerator/denominator/proportion + 95% Wilson CI, overall and stratified."""
    out = {}
    k, n = metrics["over_alignment_count"], metrics["non_equiv_total"]
    lo, hi = wilson_ci(k, n)
    out["overall"] = {"numerator": k, "denominator": n, "proportion": k / n if n else float("nan"),
                      "ci_low": lo, "ci_high": hi}
    for g in NON_EQUIV_CLASSES:
        c = metrics["over_alignment_by_label"][g]["count"]
        t = metrics["over_alignment_by_label"][g]["total"]
        lo, hi = wilson_ci(c, t)
        out[g] = {"numerator": c, "denominator": t, "proportion": c / t if t else float("nan"),
                  "ci_low": lo, "ci_high": hi}
    return out


def main():
    expert = load_expert_labels()
    all_metrics = {}
    for m in MODELS:
        all_metrics[m] = compute_model_metrics(m, expert)

    # ---- Pairwise McNemar + Bonferroni (family size 3) ----
    pairs = [("gpt-5.5", "claude-sonnet-4-6"), ("gpt-5.5", "deepseek-v3.2"),
             ("claude-sonnet-4-6", "deepseek-v3.2")]
    mcnemar = {}
    for a, b in pairs:
        res = pairwise_mcnemar(all_metrics[a], all_metrics[b])
        res["pair"] = f"{MODEL_LABELS[a]} vs {MODEL_LABELS[b]}"
        mcnemar[res["pair"]] = res
    for key, res in mcnemar.items():
        res["bonferroni_p"] = min(res["exact_p_two_sided"] * 3, 1.0)

    # ---- RELATED vs NO-EQUIV (two-proportion) ----
    rel_noeq = {all_metrics[m]["model_label"]: related_vs_noequiv(all_metrics[m]) for m in MODELS}

    # ---- OA confidence ----
    oa_conf = {all_metrics[m]["model_label"]: oa_confidence_analysis(all_metrics[m]) for m in MODELS}

    # ---- OA uncertainty ----
    oa_unc = {all_metrics[m]["model_label"]: oa_uncertainty(all_metrics[m]) for m in MODELS}

    # ---- Serialize metrics (drop per-item dicts) ----
    serializable = {}
    for m, mm in all_metrics.items():
        s = dict(mm)
        s.pop("oa_vector", None)
        s.pop("preds", None)
        s.pop("golds", None)
        s.pop("confs", None)
        serializable[m] = s

    metrics_path = os.path.join(OUT_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    # ---- evaluation_summary.csv (correct 3-class Macro-F1) ----
    summary_path = os.path.join(OUT_DIR, "evaluation_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("model,accuracy,macro_f1,f1_EQUIVALENT,f1_RELATED_NOT_EQUIVALENT,"
                "f1_NO_JUSTIFIED_EQUIVALENT,over_alignment_rate,appropriate_rejection_rate,"
                "ece,mean_confidence,conf_acc_gap,high_conf_errors\n")
        for m in MODELS:
            mm = all_metrics[m]
            f.write(",".join([
                MODEL_LABELS[m],
                f"{mm['accuracy']:.4f}",
                f"{mm['macro_f1']:.4f}",
                f"{mm['per_class']['EQUIVALENT']['f1']:.4f}",
                f"{mm['per_class']['RELATED_NOT_EQUIVALENT']['f1']:.4f}",
                f"{mm['per_class']['NO_JUSTIFIED_EQUIVALENT']['f1']:.4f}",
                f"{mm['over_alignment_rate']:.4f}",
                f"{mm['appropriate_rejection_rate']:.4f}",
                f"{mm['ece']:.4f}",
                f"{mm['mean_confidence']:.4f}",
                f"{mm['conf_acc_gap']:+.4f}",
                str(mm['high_conf_errors']),
            ]) + "\n")

    # ---- advanced_stats.json ----
    adv_path = os.path.join(OUT_DIR, "advanced_stats.json")
    with open(adv_path, "w", encoding="utf-8") as f:
        json.dump({
            "mcnemar_pairwise_oa": mcnemar,
            "related_vs_noequiv_oa": rel_noeq,
            "oa_confidence": oa_conf,
            "oa_uncertainty": oa_unc,
        }, f, ensure_ascii=False, indent=2)

    # ---- Frozen_Results_v1.csv (long format) ----
    frozen_path = os.path.join(OUT_DIR, "Frozen_Results_v1.csv")
    rows = [("section", "metric", "group", "value", "ci_low", "ci_high", "note")]

    def add(section, metric, group, value, lo="", hi="", note=""):
        rows.append((section, metric, group, value, lo, hi, note))

    for m in MODELS:
        mm = all_metrics[m]
        lb = MODEL_LABELS[m]
        add("accuracy", "accuracy", lb, f"{mm['accuracy']:.4f}")
        add("f1", "macro_f1_3class", lb, f"{mm['macro_f1']:.4f}",
            note="unweighted mean over 3 gold classes; CONTEXT_DEPENDENT excluded")
        for g in GOLD_CLASSES:
            add("f1", f"f1_{g}", lb, f"{mm['per_class'][g]['f1']:.4f}")
        add("over_alignment", "oa_rate", lb, f"{mm['over_alignment_rate']:.4f}")
        add("over_alignment", "oa_related", lb, f"{mm['over_alignment_by_label']['RELATED_NOT_EQUIVALENT']['count']/100:.4f}")
        add("over_alignment", "oa_noequiv", lb, f"{mm['over_alignment_by_label']['NO_JUSTIFIED_EQUIVALENT']['count']/100:.4f}")
        add("rejection", "arr", lb, f"{mm['appropriate_rejection_rate']:.4f}")
        add("calibration", "ece_equal_width", lb, f"{mm['ece']:.4f}")
        add("calibration", "mean_confidence", lb, f"{mm['mean_confidence']:.4f}")
        add("calibration", "conf_acc_gap", lb, f"{mm['conf_acc_gap']:+.4f}")

    for key, res in mcnemar.items():
        add("mcnemar", "discordant_b", key, str(res["discordant_b"]))
        add("mcnemar", "discordant_c", key, str(res["discordant_c"]))
        add("mcnemar", "exact_p_two_sided", key, f"{res['exact_p_two_sided']:.6f}")
        add("mcnemar", "bonferroni_p", key, f"{res['bonferroni_p']:.6f}", note="family size 3")
        add("mcnemar", "cohens_h", key, f"{res['cohens_h']:.4f}")
        add("mcnemar", "paired_risk_difference", key, f"{res['paired_risk_difference']:.4f}",
            f"{res['paired_risk_diff_ci_low']:.4f}", f"{res['paired_risk_diff_ci_high']:.4f}")

    for lb, res in rel_noeq.items():
        add("related_vs_noequiv", "risk_difference", lb, f"{res['risk_difference']:.4f}",
            f"{res['risk_diff_ci_low']:.4f}", f"{res['risk_diff_ci_high']:.4f}")
        add("related_vs_noequiv", "z_test_p", lb, f"{res['z_test_p']:.6f}")
        add("related_vs_noequiv", "fisher_exact_p", lb, f"{res['fisher_exact_p']:.6f}")
        add("related_vs_noequiv", "cohens_h", lb, f"{res['cohens_h']:.4f}")

    for lb, res in oa_conf.items():
        add("oa_confidence", "overaligned_median", lb, f"{res['overaligned_median']:.4f}")
        add("oa_confidence", "not_overaligned_median", lb, f"{res['not_overaligned_median']:.4f}")
        add("oa_confidence", "mannwhitney_p", lb, f"{res['mannwhitney_p']:.6f}")
        add("oa_confidence", "cliffs_delta", lb, f"{res['cliffs_delta']:.4f}",
            f"{res['cliffs_delta_ci_low']:.4f}", f"{res['cliffs_delta_ci_high']:.4f}")

    for lb, res in oa_unc.items():
        for grp in ["overall", "RELATED_NOT_EQUIVALENT", "NO_JUSTIFIED_EQUIVALENT"]:
            g = res[grp]
            add("oa_uncertainty", "proportion", f"{lb}:{grp}", f"{g['proportion']:.4f}",
                f"{g['ci_low']:.4f}", f"{g['ci_high']:.4f}",
                note=f"{g['numerator']}/{g['denominator']}")

    with open(frozen_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    # ---- Console summary ----
    print("=== KEY RESULTS (3-class Macro-F1, equal-width ECE) ===")
    for m in MODELS:
        mm = all_metrics[m]
        print(f"\n{MODEL_LABELS[m]}:")
        print(f"  accuracy        = {mm['accuracy']:.4f}")
        print(f"  macro_f1 (3cls) = {mm['macro_f1']:.4f}")
        print(f"  F1 EQ/REL/NOEQ  = {mm['per_class']['EQUIVALENT']['f1']:.4f} / "
              f"{mm['per_class']['RELATED_NOT_EQUIVALENT']['f1']:.4f} / "
              f"{mm['per_class']['NO_JUSTIFIED_EQUIVALENT']['f1']:.4f}")
        print(f"  OA rate         = {mm['over_alignment_rate']:.4f} ({mm['over_alignment_count']}/{mm['non_equiv_total']})")
        print(f"  OA-REL / OA-NOEQ= {mm['over_alignment_by_label']['RELATED_NOT_EQUIVALENT']['count']/100:.4f} / "
              f"{mm['over_alignment_by_label']['NO_JUSTIFIED_EQUIVALENT']['count']/100:.4f}")
        print(f"  ARR             = {mm['appropriate_rejection_rate']:.4f}")
        print(f"  ECE (eq-width)  = {mm['ece']:.4f}")
        print(f"  mean conf       = {mm['mean_confidence']:.4f}")
        print(f"  conf-acc gap    = {mm['conf_acc_gap']:+.4f}")
        print(f"  high-conf errors= {mm['high_conf_errors']}")

    print("\n=== McNemar (pairwise OA) ===")
    for key, res in mcnemar.items():
        print(f"  {key}: b={res['discordant_b']} c={res['discordant_c']} "
              f"exact_p={res['exact_p_two_sided']:.4f} bonf_p={res['bonferroni_p']:.4f} "
              f"h={res['cohens_h']:.3f}")

    print("\n=== RELATED vs NO-EQUIV (two-proportion) ===")
    for lb, res in rel_noeq.items():
        print(f"  {lb}: RD={res['risk_difference']:.4f} z_p={res['z_test_p']:.4f} "
              f"fisher_p={res['fisher_exact_p']:.4f} h={res['cohens_h']:.3f}")

    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
