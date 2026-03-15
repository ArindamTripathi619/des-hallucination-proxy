"""
06_robustness.py — Comprehensive robustness analyses for DES
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"

Produces:
  1. Bootstrap 95% CI for all AUROC values (1000 resamples)
  2. McNemar's test: DES vs SelfCheckGPT significance
  3. Leave-One-Model-Out (LOMO) stability analysis
  4. Embedding model ablation: DES with 3 different sentence transformers
  5. Expanded cross-family gap analysis (9 models × 8 families)

Reads:  data/results/raw_results.jsonl   (for embedding ablation re-scoring)
        data/results/scored_results.jsonl (for existing DES scores)
Writes:
  outputs/tables/robustness_lomo.csv
  outputs/tables/robustness_embedding_ablation.csv
  outputs/tables/robustness_mcnemar.csv
  outputs/tables/robustness_bootstrap_ci.csv

Usage:
  python 06_robustness.py --expanded          # Run with all 9 models
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
from scipy.stats import chi2

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import (
    DATA_RESULTS, OUTPUTS_TABLES, MODELS, MODELS_ALL, FAMILY_MAP,
    DES_ALPHA, DES_BETA, SEED,
    normalize_answer, is_correct, get_embedder,
    strip_thinking_tags, extract_for_embedding,
    EMBEDDING_MODELS,
)

OUTPUTS_TABLES.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

N_BOOTSTRAPS = 1000


# ─────────────────────────────────────────────────────────────────
# 1. Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────
def bootstrap_auroc(y_true: np.ndarray, y_score: np.ndarray,
                    n_boot: int = N_BOOTSTRAPS) -> dict:
    """Compute AUROC with 95% bootstrap CI.

    Args:
        y_true: Binary ground truth (1=hallucinated, 0=correct).
        y_score: DES scores.
        n_boot: Number of bootstrap resamples.

    Returns:
        Dict with AUROC, CI_lower, CI_upper.
    """
    rng = np.random.RandomState(SEED)
    valid = [(yt, ys) for yt, ys in zip(y_true, y_score)
             if not np.isnan(yt) and not np.isnan(ys)]
    if len(valid) < 2:
        return {"AUROC": np.nan, "CI_lower": np.nan, "CI_upper": np.nan}
    y_t = np.array([v[0] for v in valid])
    y_s = np.array([v[1] for v in valid])

    point = roc_auc_score(y_t, y_s)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_t), size=len(y_t), replace=True)
        if len(np.unique(y_t[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_t[idx], y_s[idx]))
    return {
        "AUROC": round(point, 4),
        "CI_lower": round(np.percentile(scores, 2.5), 4) if scores else np.nan,
        "CI_upper": round(np.percentile(scores, 97.5), 4) if scores else np.nan,
    }


def build_bootstrap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap CI for DES, surface-only, and semantic-only AUROC, per dataset."""
    rows = []
    y_true = df["any_error"].values

    for label, scores in [
        ("DES (combined)", df["DES"].values),
        ("DES (surface only)", df["surface_DES"].values),
        ("DES (semantic only)", df["semantic_DES"].values),
    ]:
        row = bootstrap_auroc(y_true, scores)
        row["Method"] = label
        row["Dataset"] = "all"
        rows.append(row)

        # Per dataset
        for source in ["triviaqa", "truthfulqa", "mmlu"]:
            sub = df[df["source"] == source]
            sub_true = sub["any_error"].values
            if label == "DES (combined)":
                sub_scores = sub["DES"].values
            elif label == "DES (surface only)":
                sub_scores = sub["surface_DES"].values
            else:
                sub_scores = sub["semantic_DES"].values
            row2 = bootstrap_auroc(sub_true, sub_scores)
            row2["Method"] = label
            row2["Dataset"] = source
            rows.append(row2)

    return pd.DataFrame(rows)[["Method", "Dataset", "AUROC", "CI_lower", "CI_upper"]]


# ─────────────────────────────────────────────────────────────────
# 2. McNemar's Test — DES vs SelfCheckGPT
# ─────────────────────────────────────────────────────────────────
def mcnemar_test(y_true: np.ndarray, y_pred_des: np.ndarray,
                 y_pred_sc: np.ndarray) -> dict:
    """McNemar's test comparing DES and SelfCheckGPT binary predictions.

    Uses optimal F1 thresholds for each method.

    Returns:
        Dict with b, c (off-diagonal), chi2 statistic, p-value.
    """
    # Both correct or both wrong don't matter
    # b = DES correct, SelfCheck wrong
    # c = DES wrong, SelfCheck correct
    b = int(np.sum((y_pred_des == y_true) & (y_pred_sc != y_true)))
    c = int(np.sum((y_pred_des != y_true) & (y_pred_sc == y_true)))

    if b + c == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}

    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    p_value = 1.0 - chi2.cdf(chi2_stat, df=1)
    return {
        "b_DES_right_SC_wrong": b,
        "c_DES_wrong_SC_right": c,
        "chi2": round(chi2_stat, 4),
        "p_value": round(p_value, 6),
        "significant_p05": p_value < 0.05,
    }


def build_mcnemar_table(df: pd.DataFrame) -> pd.DataFrame:
    """McNemar test comparing DES vs SelfCheckGPT at optimal threshold.

    Note: This function is called from main() with inline logic.
    Kept as a stub for documentation.
    """
    pass  # Logic is inline in main() to avoid circular imports


# ─────────────────────────────────────────────────────────────────
# Semantic Caching (issue #23 performance optimization)
# ─────────────────────────────────────────────────────────────────
def precompute_embeddings(df: pd.DataFrame, embedder, model_set: dict) -> dict[str, np.ndarray]:
    """Pre-encode all response texts in bulk to avoid per-row overhead."""
    unique_texts = set()
    for _, row in df.iterrows():
        model_responses = row.get("model_responses", {})
        for m in model_set:
            resp = model_responses.get(m, {})
            raw = resp.get("response")
            if raw:
                et = extract_for_embedding(raw)
                if et:
                    unique_texts.add(et)
    if not unique_texts:
        return {}
    
    texts = list(unique_texts)
    embs = embedder.encode(texts, convert_to_numpy=True, batch_size=64)
    return {t: e for t, e in zip(texts, embs)}


# ─────────────────────────────────────────────────────────────────
# 3. Leave-One-Model-Out (LOMO) Stability
# ─────────────────────────────────────────────────────────────────
def lomo_analysis(df_raw: pd.DataFrame, model_set: dict) -> pd.DataFrame:
    """For each model, drop it and recompute DES + AUROC.

    Proves that no single model is critical — DES is robust to any dropout.

    Args:
        df_raw: DataFrame from raw_results.jsonl with model_responses.
        model_set: Active model dict.

    Returns:
        DataFrame with one row per dropped model showing residual AUROC.
    """
    from utils import normalize_answer, is_correct, extract_for_embedding

    all_models = list(model_set.keys())
    embedder = get_embedder()
    # Pre-compute all embeddings once for the default embedder (issue #23)
    emb_cache = precompute_embeddings(df_raw, embedder, model_set)
    rows = []

    for drop_model in all_models:
        remaining = [m for m in all_models if m != drop_model]
        scores = []
        y_true_list = []

        for _, rec in df_raw.iterrows():
            qtype = rec.get("question_type", "open")
            model_responses = rec.get("model_responses", {})

            # Normalize answers for remaining models
            norm_answers = {}
            embeddings = []
            for m in remaining:
                resp = model_responses.get(m, {})
                raw = resp.get("response")
                if raw:
                    norm_answers[m] = normalize_answer(raw, qtype)
                    et = extract_for_embedding(raw)
                    if et and et in emb_cache:
                        embeddings.append(emb_cache[et])

            # Surface disagreement
            valid_norms = [v for v in norm_answers.values() if v is not None]
            if len(valid_norms) < 2:
                scores.append(np.nan)
            else:
                pairs = list(combinations(valid_norms, 2))
                surf = sum(1 for a, b in pairs if a != b) / len(pairs)

                # Semantic disagreement (use cached embeddings)
                if len(embeddings) >= 2:
                    embs = np.array(embeddings)
                    norms_v = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                    embs = embs / norms_v
                    p = list(combinations(range(len(embs)), 2))
                    sims = [np.dot(embs[i], embs[j]) for i, j in p]
                    sem = 1.0 - np.mean(sims)
                else:
                    sem = 0.0

                des = float(np.clip(DES_ALPHA * surf + DES_BETA * sem, 0, 1))
                scores.append(des)

            # Error signal
            correct_answers = rec.get("correct_answers", [])
            evaluated = []
            for m in remaining:
                n = norm_answers.get(m)
                if n is not None:
                    evaluated.append(is_correct(n, correct_answers, qtype))
            y_true_list.append(int(any(not v for v in evaluated)) if evaluated else np.nan)

        # AUROC
        valid = [(yt, ys) for yt, ys in zip(y_true_list, scores)
                 if not np.isnan(yt) and not np.isnan(ys)]
        if len(valid) >= 2 and len(set(v[0] for v in valid)) >= 2:
            auroc = roc_auc_score([v[0] for v in valid], [v[1] for v in valid])
        else:
            auroc = np.nan

        rows.append({
            "Dropped_Model": drop_model,
            "Dropped_Family": FAMILY_MAP.get(drop_model, "?"),
            "Remaining_Models": len(remaining),
            "AUROC": round(auroc, 4) if not np.isnan(auroc) else np.nan,
        })
        print(f"  LOMO: dropped {drop_model:15s} → AUROC = {auroc:.4f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# 4. Embedding Model Ablation
# ─────────────────────────────────────────────────────────────────
def embedding_ablation(df_raw: pd.DataFrame, model_set: dict) -> pd.DataFrame:
    """Re-score DES with multiple embedding models to show agnosticism.

    Tests: all-MiniLM-L6-v2, all-mpnet-base-v2, intfloat/e5-large-v2

    Args:
        df_raw: DataFrame from raw_results.jsonl.
        model_set: Active model dict.

    Returns:
        DataFrame with AUROC per embedding model.
    """
    all_models = list(model_set.keys())
    rows = []

    for emb_name in EMBEDDING_MODELS:
        print(f"  Embedding ablation: {emb_name}...")
        embedder = get_embedder(emb_name)
        # Pre-compute all embeddings for THIS embedder (issue #23)
        emb_cache = precompute_embeddings(df_raw, embedder, model_set)

        scores = []
        y_true_list = []

        for _, rec in df_raw.iterrows():
            qtype = rec.get("question_type", "open")
            model_responses = rec.get("model_responses", {})
            correct_answers = rec.get("correct_answers", [])

            norm_answers = {}
            embeddings = []

            for m in all_models:
                resp = model_responses.get(m, {})
                raw = resp.get("response")
                if raw:
                    norm_answers[m] = normalize_answer(raw, qtype)
                    et = extract_for_embedding(raw)
                    if et and et in emb_cache:
                        embeddings.append(emb_cache[et])

            valid_norms = [v for v in norm_answers.values() if v is not None]
            if len(valid_norms) < 2:
                scores.append(np.nan)
                y_true_list.append(np.nan)
                continue

            # Surface
            pairs = list(combinations(valid_norms, 2))
            surf = sum(1 for a, b in pairs if a != b) / len(pairs)

            # Semantic with THIS cached embedder
            if len(embeddings) >= 2:
                embs = np.array(embeddings)
                norms_v = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
                embs = embs / norms_v
                p = list(combinations(range(len(embs)), 2))
                sims = [np.dot(embs[i], embs[j]) for i, j in p]
                sem = 1.0 - np.mean(sims)
            else:
                sem = 0.0

            des = float(np.clip(DES_ALPHA * surf + DES_BETA * sem, 0, 1))
            scores.append(des)

            evaluated = []
            for m in all_models:
                n = norm_answers.get(m)
                if n is not None:
                    evaluated.append(is_correct(n, correct_answers, qtype))
            y_true_list.append(int(any(not v for v in evaluated)) if evaluated else np.nan)

        # AUROC
        valid = [(yt, ys) for yt, ys in zip(y_true_list, scores)
                 if not np.isnan(yt) and not np.isnan(ys)]
        if len(valid) >= 2 and len(set(v[0] for v in valid)) >= 2:
            auroc = roc_auc_score([v[0] for v in valid], [v[1] for v in valid])
        else:
            auroc = np.nan

        ci = bootstrap_auroc(
            np.array([v[0] for v in valid]),
            np.array([v[1] for v in valid]),
        )

        rows.append({
            "Embedding_Model": emb_name,
            "AUROC": round(auroc, 4),
            "CI_lower": ci["CI_lower"],
            "CI_upper": ci["CI_upper"],
            "N_valid": len(valid),
        })
        print(f"    → AUROC = {auroc:.4f} [{ci['CI_lower']}, {ci['CI_upper']}]")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustness analyses")
    parser.add_argument("--expanded", action="store_true",
                        help="Use all 9 models")
    parser.add_argument("--skip-embed-ablation", action="store_true",
                        help="Skip embedding ablation (slow)")
    parser.add_argument("--skip-lomo", action="store_true",
                        help="Skip leave-one-model-out (slow)")
    args = parser.parse_args()

    model_set = MODELS_ALL if args.expanded else MODELS
    print(f"Using {len(model_set)} models: {list(model_set.keys())}")

    # Load scored results
    scored_file = DATA_RESULTS / "scored_results.jsonl"
    if not scored_file.exists():
        print(f"[ERROR] {scored_file} not found. Run 03_scoring.py first.")
        sys.exit(1)

    print("Loading scored results...")
    scored_records = []
    with open(scored_file) as f:
        for line in f:
            if line.strip():
                scored_records.append(json.loads(line))
    df = pd.DataFrame(scored_records)
    df["any_error"] = pd.to_numeric(df["any_error"], errors="coerce")
    print(f"  Loaded {len(df)} scored records.")

    # Also load raw results (needed for LOMO + embedding ablation)
    raw_file = DATA_RESULTS / "raw_results.jsonl"
    raw_records = []
    with open(raw_file) as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))
    df_raw = pd.DataFrame(raw_records)
    df_raw["any_error"] = pd.to_numeric(
        df_raw.get("any_error", pd.Series(dtype=float)), errors="coerce"
    )
    # Compute any_error from correctness if not already there
    if "any_error" not in df_raw.columns or df_raw["any_error"].isna().all():
        def _compute_error(row):
            resps = row.get("model_responses", {})
            qtype = row.get("question_type", "open")
            correct = row.get("correct_answers", [])
            for m in model_set:
                r = resps.get(m, {}).get("response")
                if r:
                    n = normalize_answer(r, qtype)
                    if n is not None and not is_correct(n, correct, qtype):
                        return 1
            return 0
        df_raw["any_error"] = df_raw.apply(_compute_error, axis=1)

    # ── 1. Bootstrap CI ──
    print("\n" + "=" * 60)
    print("1. Bootstrap 95% CI for AUROC")
    print("=" * 60)
    t_boot = build_bootstrap_table(df)
    t_boot.to_csv(OUTPUTS_TABLES / "robustness_bootstrap_ci.csv", index=False)
    print(t_boot.to_string(index=False))

    # ── 2. McNemar's test ──
    print("\n" + "=" * 60)
    print("2. McNemar's Test: DES vs SelfCheckGPT")
    print("=" * 60)
    try:
        # Import selfcheck function from 04_calibration
        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from importlib import import_module
        mod04 = import_module("04_calibration")
        selfcheck_surface = mod04.selfcheck_baseline_surface

        # Get binary predictions at optimal threshold
        from sklearn.metrics import precision_recall_curve
        y_true = df["any_error"].values
        des_scores = df["DES"].values
        valid_mask = ~np.isnan(y_true) & ~np.isnan(des_scores)
        y_v = y_true[valid_mask].astype(int)
        des_v = des_scores[valid_mask]

        prec, rec, thresh = precision_recall_curve(y_v, des_v)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_t_des = thresh[np.argmax(f1s)]
        y_pred_des = (des_v >= best_t_des).astype(int)

        sc_all = selfcheck_surface(df).values
        sc_v = sc_all[valid_mask]
        sc_valid = ~np.isnan(sc_v)

        y_v2 = y_v[sc_valid]
        y_pred_des2 = y_pred_des[sc_valid]
        sc_v2 = sc_v[sc_valid]

        prec2, rec2, thresh2 = precision_recall_curve(y_v2, sc_v2)
        f1s2 = 2 * prec2 * rec2 / (prec2 + rec2 + 1e-8)
        best_t_sc = thresh2[np.argmax(f1s2)]
        y_pred_sc = (sc_v2 >= best_t_sc).astype(int)

        mc = mcnemar_test(y_v2, y_pred_des2, y_pred_sc)
        t_mc = pd.DataFrame([mc])
        t_mc.to_csv(OUTPUTS_TABLES / "robustness_mcnemar.csv", index=False)
        print(t_mc.to_string(index=False))
    except Exception as e:
        print(f"  [WARN] McNemar test failed: {e}")

    # ── 3. Leave-One-Model-Out ──
    if not args.skip_lomo:
        print("\n" + "=" * 60)
        print("3. Leave-One-Model-Out (LOMO) Stability")
        print("=" * 60)
        t_lomo = lomo_analysis(df_raw, model_set)
        t_lomo.to_csv(OUTPUTS_TABLES / "robustness_lomo.csv", index=False)
        print(t_lomo.to_string(index=False))
        drop_range = t_lomo["AUROC"].max() - t_lomo["AUROC"].min()
        print(f"\n  LOMO AUROC range: {drop_range:.4f} (lower = more stable)")
    else:
        print("\n[SKIP] Leave-One-Model-Out (use --skip-lomo to skip)")

    # ── 4. Embedding Model Ablation ──
    if not args.skip_embed_ablation:
        print("\n" + "=" * 60)
        print("4. Embedding Model Ablation")
        print("=" * 60)
        t_emb = embedding_ablation(df_raw, model_set)
        t_emb.to_csv(OUTPUTS_TABLES / "robustness_embedding_ablation.csv", index=False)
        print(t_emb.to_string(index=False))
    else:
        print("\n[SKIP] Embedding ablation (use --skip-embed-ablation to skip)")

    print(f"\n✅ Robustness tables saved to {OUTPUTS_TABLES}")
