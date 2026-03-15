"""
05_analysis.py — RQ4 (Qwen ablation) + cost-efficiency curve + Table 5
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"

Reads:  data/results/scored_results.jsonl
Writes:
  outputs/tables/table5_qwen_ablation.csv
  outputs/tables/auroc_vs_n_models.csv   (for Figure 5 in notebook)

Usage:
  python 05_analysis.py                       # All 9 models
  python 05_analysis.py --expanded            # All 9 models
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import (
    DATA_RESULTS, OUTPUTS_TABLES, MODELS, MODELS_ALL, FAMILY_MAP,
    normalize_answer, DES_ALPHA, DES_BETA,
    get_embedder, strip_thinking_tags, extract_for_embedding,
)

OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

# Will be set in main() based on --expanded flag
ACTIVE_MODELS: dict = MODELS


# ─────────────────────────────────────────────────────────────────
# Table 5 — Qwen Thinking vs. No-Think Ablation (RQ4)
# ─────────────────────────────────────────────────────────────────
def build_table5(df: pd.DataFrame) -> pd.DataFrame:
    """Compare DES calibration with Qwen in thinking vs. no-think mode.

    Only records that have the 'qwen-nothink' key are included (first ~200 Qs).

    Returns:
        DataFrame with per-condition AUROC and accuracy.
    """
    df_sub = df[df["model_responses"].apply(lambda r: "qwen-nothink" in r)].copy()
    if df_sub.empty:
        print("  [WARN] No qwen-nothink records found — skipping Table 5")
        return pd.DataFrame()

    def _qwen_accuracy(row, alias):
        resp = row["model_responses"].get(alias, {}).get("response")
        if resp is None:
            return np.nan
        norm = normalize_answer(resp, row["question_type"])
        if norm is None:
            return np.nan
        correct_answers = row["correct_answers"]
        qtype = row["question_type"]
        # Standardized with canonical is_correct() (issue #L246 suggestion)
        return float(is_correct(norm, correct_answers, qtype))

    df_sub["qwen_think_correct"]   = df_sub.apply(lambda r: _qwen_accuracy(r, "qwen"), axis=1)
    df_sub["qwen_nothink_correct"] = df_sub.apply(lambda r: _qwen_accuracy(r, "qwen-nothink"), axis=1)

    # Recompute DES variants replacing qwen with qwen-nothink
    def _des_with_nothink(row):
        """Compute DES replacing qwen response with qwen-nothink response."""
        responses = {k: v for k, v in row["model_responses"].items() if k != "qwen-nothink"}
        if "qwen-nothink" in row["model_responses"]:
            responses["qwen"] = row["model_responses"]["qwen-nothink"]   # swap
        # Use extract_for_embedding to handle think tags properly
        raw_answers = []
        for r in responses.values():
            raw = r.get("response")
            if raw:
                cleaned = extract_for_embedding(raw)
                if cleaned:
                    raw_answers.append(cleaned)
        if len(raw_answers) < 2:
            return np.nan
        embedder = get_embedder()
        emb = embedder.encode(raw_answers, convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        pairs = list(combinations(range(len(emb)), 2))
        sims = [np.dot(emb[i], emb[j]) for i, j in pairs]
        return float(1.0 - np.mean(sims))

    df_sub["des_nothink"] = df_sub.apply(_des_with_nothink, axis=1)
    y_true = df_sub["any_error"].values

    def _auroc(scores):
        valid = [(yt, ys) for yt, ys in zip(y_true, scores)
                 if not np.isnan(float(yt)) and not np.isnan(float(ys))]
        if len(valid) < 2 or len(set(v[0] for v in valid)) < 2:
            return np.nan
        return roc_auc_score([v[0] for v in valid], [v[1] for v in valid])

    rows = [
        {
            "Condition": "Qwen thinking ON (default)",
            "N": len(df_sub),
            "Qwen_Accuracy_%": round(df_sub["qwen_think_correct"].mean() * 100, 1),
            "AUROC_with_Qwen_think": round(_auroc(df_sub["DES"].values), 4),
            "AUROC_with_Qwen_nothink": round(_auroc(df_sub["des_nothink"].values), 4),
        },
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# AUROC vs. Number of Models — Figure 5 data
# ─────────────────────────────────────────────────────────────────
def auroc_vs_n_models(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AUROC for all subsets of size k from 2..N models.

    Picks the best cross-family subset for each k (greedy diversity).
    Returns mean ± std AUROC over all C(N,k) subsets for each k.
    """
    all_model_keys = list(ACTIVE_MODELS.keys())
    y_true = df["any_error"].values
    records = []
    
    # Pre-compute all pairwise distances for every record
    print("  Pre-computing full distance matrices per record...")
    dist_matrices = []
    embedder = get_embedder()
    
    for _, row in df.iterrows():
        raws = []
        # Keep track of which model aligns to which index
        idx_map = {}
        idx = 0
        for m in all_model_keys:
            r = row.get("model_responses", {}).get(m, {}).get("response")
            if r:
                # Use extract_for_embedding to handle think tags properly
                cleaned = extract_for_embedding(r)
                if cleaned:
                    raws.append(cleaned)
                    idx_map[m] = idx
                    idx += 1
                
        if not raws:
            dist_matrices.append((idx_map, np.array([])))
            continue
            
        # Encode all available responses for this row
        emb = embedder.encode(raws, convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
        
        # Calculate full N x N distance matrix for this row
        dist_mat = 1.0 - np.dot(emb, emb.T) 
        dist_matrices.append((idx_map, dist_mat))

    print("  Computing subset AUROCs using pre-computed distances...")
    for k in range(2, len(all_model_keys) + 1):
        aurocs = []
        for subset in combinations(all_model_keys, k):
            subset = list(subset)
            scores = []
            
            for (idx_map, dist_mat) in dist_matrices:
                # Find indices of the subset models in the distance matrix
                valid_idxs = [idx_map[m] for m in subset if m in idx_map]
                if len(valid_idxs) < 2:
                    scores.append(np.nan)
                    continue
                
                # Extract the submatrix for the valid indices and compute mean distance of upper triangle
                sub_dist = dist_mat[np.ix_(valid_idxs, valid_idxs)]
                iu1 = np.triu_indices(len(valid_idxs), 1)
                sims = sub_dist[iu1]
                scores.append(float(np.mean(sims)))
                
            valid = [(yt, ys) for yt, ys in zip(y_true, scores)
                     if not np.isnan(float(yt)) and not np.isnan(float(ys))]
            
            if len(valid) < 2 or len(set(v[0] for v in valid)) < 2:
                continue
                
            auroc = roc_auc_score([v[0] for v in valid], [v[1] for v in valid])
            aurocs.append(auroc)

        if aurocs:
            records.append({
                "N_models": k,
                "Mean_AUROC": round(np.mean(aurocs), 4),
                "Std_AUROC": round(np.std(aurocs), 4),
                "Max_AUROC": round(np.max(aurocs), 4),
                "N_subsets": len(aurocs),
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis and AUROC curves")
    parser.add_argument("--expanded", action="store_true",
                        help="Use all 9 models (after 02c_add_models.py)")
    ap_args = parser.parse_args()

    if ap_args.expanded:
        ACTIVE_MODELS = MODELS_ALL
    else:
        ACTIVE_MODELS = MODELS

    print(f"Using {len(ACTIVE_MODELS)} models: {list(ACTIVE_MODELS.keys())}")

    scored_file = DATA_RESULTS / "scored_results.jsonl"
    if not scored_file.exists():
        print(f"[ERROR] {scored_file} not found. Run 03_scoring.py first.")
        sys.exit(1)

    print("Loading scored results...")
    records = []
    with open(scored_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["any_error"] = pd.to_numeric(df["any_error"], errors="coerce")
    print(f"  Loaded {len(df)} records.")

    # --- Table 5: Qwen ablation ---
    print("Building Table 5: Qwen thinking ablation...")
    t5 = build_table5(df)
    if not t5.empty:
        t5.to_csv(OUTPUTS_TABLES / "table5_qwen_ablation.csv", index=False)
        t5.to_latex(OUTPUTS_TABLES / "table5_qwen_ablation.tex", index=False)
        print(t5.to_string(index=False))

    # --- Figure 5 data: AUROC vs. N models ---
    print(f"\nComputing AUROC vs. number of models (for Figure 5)...")
    print(f"  NOTE: This runs C({len(ACTIVE_MODELS)},k) subsets for k=2..{len(ACTIVE_MODELS)}. May take a few minutes.")
    auroc_df = auroc_vs_n_models(df)
    auroc_df.to_csv(OUTPUTS_TABLES / "auroc_vs_n_models.csv", index=False)
    print(auroc_df.to_string(index=False))
    print(f"\n✅ Analysis tables saved to {OUTPUTS_TABLES}")
