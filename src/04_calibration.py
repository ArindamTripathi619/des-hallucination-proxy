"""
04_calibration.py — Calibration curves, AUROC, architecture gap, domain analysis
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"

Reads:  data/results/scored_results.jsonl
Writes:
  outputs/tables/table1_model_accuracy.csv
  outputs/tables/table2_calibration_metrics.csv
  outputs/tables/table3_classification_performance.csv
  outputs/tables/table4_architecture_gap.csv
  outputs/tables/calibration_raw.csv   (for Figure 1 in notebook)

Usage:
  python 04_calibration.py                    # All 9 models
  python 04_calibration.py --expanded         # All 9 models
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from scipy.stats import pearsonr

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import (
    DATA_RESULTS, OUTPUTS_TABLES, MODELS, MODELS_ALL, FAMILY_MAP,
    WITHIN_FAMILY_PAIRS, CROSS_FAMILY_PAIRS, CROSS_FAMILY_PAIRS_EXPANDED,
    normalize_answer, get_embedder, strip_thinking_tags, extract_for_embedding,
)

OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

# NOTE (issue #19): ACTIVE_MODELS and ACTIVE_CROSS_PAIRS are mutable module-level
# globals. These are intentionally set exactly once in the __main__ block below
# based on the --expanded CLI flag. Do NOT mutate them from helper functions, as
# this would cause silent analysis inconsistencies if functions are called from
# external scripts or tests.
ACTIVE_MODELS: dict = MODELS
ACTIVE_CROSS_PAIRS: list = CROSS_FAMILY_PAIRS


# ─────────────────────────────────────────────────────────────────
# Load scored data
# ─────────────────────────────────────────────────────────────────
def load_scored_df(path: pathlib.Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Convert any_error to numeric
    df["any_error"] = pd.to_numeric(df["any_error"], errors="coerce")
    # Error rate per question: fraction of models that got it wrong
    def _error_rate(flags):
        vals = [v for v in flags.values() if v is not None]
        return 1 - np.mean(vals) if vals else np.nan

    df["error_rate"] = df["correctness_flags"].apply(_error_rate)
    return df


# ─────────────────────────────────────────────────────────────────
# Table 1 — Per-model accuracy per dataset
# ─────────────────────────────────────────────────────────────────
def build_table1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for alias in ACTIVE_MODELS:
        row = {"Model": alias, "Family": FAMILY_MAP[alias]}
        for source in ["triviaqa", "truthfulqa", "mmlu"]:
            sub = df[df["source"] == source]
            correct = sub["correctness_flags"].apply(
                lambda f: f.get(alias)
            )
            correct = correct.dropna()
            acc = correct.mean() * 100 if len(correct) > 0 else np.nan
            row[f"Acc_{source}"] = round(acc, 1)
        rows.append(row)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Table 2 — Calibration metrics per dataset (ECE + Pearson r)
# ─────────────────────────────────────────────────────────────────
def calibration_curve(df_sub: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Bin by DES, compute mean error rate per bin."""
    valid = df_sub.dropna(subset=["DES", "error_rate"])
    valid = valid.copy()
    valid["des_bin"] = pd.cut(valid["DES"], bins=n_bins, labels=False, include_lowest=True)
    return valid.groupby("des_bin").agg(
        mean_DES=("DES", "mean"),
        mean_error_rate=("error_rate", "mean"),
        count=("DES", "count"),
    ).reset_index()


def expected_calibration_error(cal_df: pd.DataFrame) -> float:
    total = cal_df["count"].sum()
    ece = (
        (cal_df["count"] / total)
        * np.abs(cal_df["mean_DES"] - cal_df["mean_error_rate"])
    ).sum()
    return float(ece)


def build_table2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in ["triviaqa", "truthfulqa", "mmlu", "all"]:
        sub = df if source == "all" else df[df["source"] == source]
        sub = sub.dropna(subset=["DES", "error_rate"])
        if len(sub) < 10:
            continue
        cal = calibration_curve(sub)
        ece = expected_calibration_error(cal)
        r, p = pearsonr(sub["DES"], sub["error_rate"])
        rows.append({
            "Dataset": source,
            "N": len(sub),
            "ECE": round(ece, 4),
            "Pearson_r": round(r, 3),
            "p_value": round(p, 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# AUROC helper (with Bootstrap CI)
# ─────────────────────────────────────────────────────────────────
def compute_auroc_f1(y_true, y_score, label="", bootstrap=True, n_bootstraps=1000):
    valid = [(yt, ys) for yt, ys in zip(y_true, y_score)
             if not np.isnan(yt) and not np.isnan(ys)]
    if len(valid) < 2 or len(set(yt for yt, _ in valid)) < 2:
        return {"label": label, "AUROC": np.nan, "AUROC_95CI": "", "Precision": np.nan,
                "Recall": np.nan, "F1": np.nan, "Threshold": np.nan}
    y_t = np.array([v[0] for v in valid])
    y_s = np.array([v[1] for v in valid])
    auroc = roc_auc_score(y_t, y_s)
    
    ci_str = ""
    if bootstrap:
        np.random.seed(42)
        bootstrapped_scores = []
        indices = np.arange(len(y_t))
        for _ in range(n_bootstraps):
            sample_idx = np.random.choice(indices, size=len(indices), replace=True)
            if len(np.unique(y_t[sample_idx])) < 2:
                continue
            score = roc_auc_score(y_t[sample_idx], y_s[sample_idx])
            bootstrapped_scores.append(score)
        
        if bootstrapped_scores:
            ci_lower = np.percentile(bootstrapped_scores, 2.5)
            ci_upper = np.percentile(bootstrapped_scores, 97.5)
            ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"

    prec, rec, thresholds = precision_recall_curve(y_t, y_s)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_i = np.argmax(f1s)
    return {
        "label": label,
        "AUROC": round(auroc, 4),
        "AUROC_95CI": ci_str,
        "Precision": round(prec[best_i], 3),
        "Recall": round(rec[best_i], 3),
        "F1": round(f1s[best_i], 3),
        "Threshold": round(float(thresholds[best_i]) if best_i < len(thresholds) else np.nan, 3),
    }

# ─────────────────────────────────────────────────────────────────
# Sensitivity Analysis (Appendix C)
# ─────────────────────────────────────────────────────────────────
def alpha_sensitivity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Computes AUROC as α sweeps from 0.0 to 1.0 (β = 1 - α)."""
    y_true = df["any_error"].values
    surf = df["surface_DES"].values
    sem = df["semantic_DES"].values
    
    records = []
    for alpha in np.arange(0.0, 1.1, 0.1):
        alpha = round(alpha, 1)
        beta = round(1.0 - alpha, 1)
        des_scores = (alpha * surf) + (beta * sem)
        
        valid = [(yt, ys) for yt, ys in zip(y_true, des_scores)
                 if not np.isnan(yt) and not np.isnan(ys)]
        if len(valid) < 2 or len(set(v[0] for v in valid)) < 2:
            auroc = np.nan
        else:
            auroc = roc_auc_score([v[0] for v in valid], [v[1] for v in valid])
            
        records.append({
            "Alpha": alpha,
            "Beta": beta,
            "AUROC": round(auroc, 4) if not np.isnan(auroc) else np.nan
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
# Table 3 — Classification performance (DES variants vs. SelfCheck baseline)
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
# Table 3 — Classification performance (DES variants vs. SelfCheck baseline)
# ─────────────────────────────────────────────────────────────────
def selfcheck_baseline_surface(df: pd.DataFrame) -> pd.Series:
    """Simulate SelfCheckGPT: use 3 Llama variants as 'stochastic samples' (surface)."""
    def _sc_score(row):
        llama_models = ["llama-large", "llama-small", "llama4-scout"]
        answers = [
            row["model_responses"].get(m, {}).get("response")
            for m in llama_models
        ]
        answers = [a for a in answers if a]
        if len(answers) < 2:
            return np.nan
        pairs = list(combinations(answers, 2))
        disagree = sum(1 for a, b in pairs if
                       normalize_answer(a, row["question_type"]) !=
                       normalize_answer(b, row["question_type"]))
        return disagree / len(pairs)
    return df.apply(_sc_score, axis=1)

def selfcheck_baseline_semantic(df: pd.DataFrame) -> pd.Series:
    """Simulate SelfCheckGPT: use 3 Llama variants as 'stochastic samples' (semantic)."""
    emb = get_embedder()
    def _sc_score(row):
        llama_models = ["llama-large", "llama-small", "llama4-scout"]
        answers = [
            row["model_responses"].get(m, {}).get("response")
            for m in llama_models
        ]
        # Use extract_for_embedding to handle think tags properly
        answers = [extract_for_embedding(a) for a in answers if a]
        answers = [a for a in answers if a]
        if len(answers) < 2:
            return np.nan
        embeddings = emb.encode(answers, convert_to_numpy=True)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        pairs = list(combinations(range(len(embeddings)), 2))
        sims = [np.dot(embeddings[i], embeddings[j]) for i, j in pairs]
        return float(1.0 - np.mean(sims))
    return df.apply(_sc_score, axis=1)

def build_table3(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["any_error"].values
    rows = [
        compute_auroc_f1(y_true, df["DES"].values,          "DES (combined)"),
        compute_auroc_f1(y_true, df["surface_DES"].values,  "DES (surface)"),
        compute_auroc_f1(y_true, df["semantic_DES"].values, "DES (semantic)"),
        compute_auroc_f1(y_true, selfcheck_baseline_surface(df).values, "SelfCheckGPT (surface)"),
        compute_auroc_f1(y_true, selfcheck_baseline_semantic(df).values, "SelfCheckGPT (semantic)"),
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Table 4 — Architecture Gap Analysis (RQ3)
# ─────────────────────────────────────────────────────────────────

def pair_semantic_DES(row: dict, pair: tuple[str, str]) -> float | None:
    """Compute semantic distance between a pair of model responses."""
    emb = get_embedder()
    r1 = row["model_responses"].get(pair[0], {}).get("response")
    r2 = row["model_responses"].get(pair[1], {}).get("response")
    if not r1 or not r2:
        return None
    # Use extract_for_embedding to handle think tags properly
    r1 = extract_for_embedding(r1)
    r2 = extract_for_embedding(r2)
    if not r1 or not r2:
        return None
    embeddings = emb.encode([r1, r2], convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1) + 1e-12
    cos_sim = np.dot(embeddings[0] / norms[0], embeddings[1] / norms[1])
    return float(1.0 - cos_sim)


def build_table4(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = df["any_error"].values

    def _run_pairs(pair_list, pair_type):
        for pair in pair_list:
            label = f"{pair[0]} × {pair[1]}"
            pair_scores = df.apply(lambda r: pair_semantic_DES(r, pair), axis=1)
            row = compute_auroc_f1(y_true, pair_scores.fillna(0.5).values, label)
            row["Pair_Type"] = pair_type
            rows.append(row)

    _run_pairs(WITHIN_FAMILY_PAIRS, "within-family")
    _run_pairs(ACTIVE_CROSS_PAIRS, "cross-family")

    # Full N-model DES
    full_row = compute_auroc_f1(y_true, df["DES"].fillna(0.5).values,
                                f"Full DES ({len(ACTIVE_MODELS)} models)")
    full_row["Pair_Type"] = "all-models"
    rows.append(full_row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Domain sensitivity
# ─────────────────────────────────────────────────────────────────
def domain_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain, sub in df.groupby("domain"):
        sub = sub.dropna(subset=["DES", "any_error"])
        if len(sub) < 10 or sub["any_error"].nunique() < 2:
            continue
        auroc = roc_auc_score(sub["any_error"], sub["DES"])
        rows.append({
            "Domain": domain,
            "N": len(sub),
            "Mean_DES": round(sub["DES"].mean(), 3),
            "Mean_Error_Rate": round(sub["error_rate"].mean(), 3),
            "AUROC": round(auroc, 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration and tables")
    parser.add_argument("--expanded", action="store_true",
                        help="Use all 9 models (after 02c_add_models.py)")
    args = parser.parse_args()

    if args.expanded:
        ACTIVE_MODELS = MODELS_ALL
        ACTIVE_CROSS_PAIRS = CROSS_FAMILY_PAIRS_EXPANDED
    else:
        ACTIVE_MODELS = MODELS
        ACTIVE_CROSS_PAIRS = CROSS_FAMILY_PAIRS

    print(f"Using {len(ACTIVE_MODELS)} models: {list(ACTIVE_MODELS.keys())}")

    scored_file = DATA_RESULTS / "scored_results.jsonl"
    if not scored_file.exists():
        print(f"[ERROR] {scored_file} not found. Run 03_scoring.py first.")
        sys.exit(1)

    print("Loading scored results...")
    df = load_scored_df(scored_file)
    print(f"  Loaded {len(df)} records.")

    # --- Table 1 ---
    print("Building Table 1: Model accuracy...")
    t1 = build_table1(df)
    t1.to_csv(OUTPUTS_TABLES / "table1_model_accuracy.csv", index=False)
    t1.to_latex(OUTPUTS_TABLES / "table1_model_accuracy.tex", index=False, float_format="%.1f")

    # --- Table 2 ---
    print("Building Table 2: Calibration metrics...")
    t2 = build_table2(df)
    t2.to_csv(OUTPUTS_TABLES / "table2_calibration_metrics.csv", index=False)
    t2.to_latex(OUTPUTS_TABLES / "table2_calibration_metrics.tex", index=False, float_format="%.4f")

    # Calibration raw data for Figure 1
    cal_frames = []
    for source in ["triviaqa", "truthfulqa", "mmlu"]:
        sub = df[df["source"] == source].dropna(subset=["DES", "error_rate"])
        if len(sub) < 10:
            continue
        cal = calibration_curve(sub)
        cal["source"] = source
        cal_frames.append(cal)
    if cal_frames:
        pd.concat(cal_frames).to_csv(OUTPUTS_TABLES / "calibration_raw.csv", index=False)

    # --- Table 3 ---
    print("Building Table 3: Classification performance...")
    t3 = build_table3(df)
    t3.to_csv(OUTPUTS_TABLES / "table3_classification_performance.csv", index=False)
    t3.to_latex(OUTPUTS_TABLES / "table3_classification_performance.tex", index=False, float_format="%.4f")

    # --- Table 4 ---
    print("Building Table 4: Architecture gap (this may take a few minutes)...")
    t4 = build_table4(df)
    t4.to_csv(OUTPUTS_TABLES / "table4_architecture_gap.csv", index=False)
    t4.to_latex(OUTPUTS_TABLES / "table4_architecture_gap.tex", index=False, float_format="%.4f")

    # --- Domain analysis ---
    print("Building domain sensitivity table...")
    td = domain_analysis(df)
    td.to_csv(OUTPUTS_TABLES / "domain_sensitivity.csv", index=False)

    # --- Alpha Sensitivity Analysis (Appendix C) ---
    print("Building alpha sensitivity analysis...")
    t_sens = alpha_sensitivity_analysis(df)
    t_sens.to_csv(OUTPUTS_TABLES / "alpha_sensitivity.csv", index=False)

    print(f"\n✅ All tables saved to {OUTPUTS_TABLES}")
    print("\n--- Table 2 Preview ---")
    print(t2.to_string(index=False))
    print("\n--- Table 3 Preview ---")
    print(t3.to_string(index=False))
