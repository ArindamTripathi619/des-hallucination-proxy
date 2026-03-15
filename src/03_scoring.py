"""
03_scoring.py — Compute DES (Disagreement Entropy Score) on raw results
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"

Reads:  data/results/raw_results.jsonl
Writes: data/results/scored_results.jsonl

Each output record adds:
  - correctness_flags: {model_alias: True|False}
  - any_error: 1 if any model was wrong, 0 otherwise
  - surface_DES: D_S  (surface-level pairwise disagreement fraction)
  - semantic_DES: D_Sem (1 - mean cosine similarity of embeddings)
  - DES: α·D_S + β·D_Sem  (combined score)
  - null_models: list of model aliases that returned None

Usage:
  python 03_scoring.py                        # Score with all 9 models
  python 03_scoring.py --expanded             # Score with all 9 models
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from itertools import combinations

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import (
    DATA_RESULTS, DES_ALPHA, DES_BETA,
    normalize_answer, is_correct, MODELS, MODELS_ALL, get_embedder,
    strip_thinking_tags, extract_for_embedding, get_embedding_text_from_response,
)

# ─────────────────────────────────────────────────────────────────
# Surface Disagreement  D_S
# ─────────────────────────────────────────────────────────────────
def surface_disagreement(normalized_answers: list[str | None]) -> float | None:
    """Fraction of model-pair combinations with different normalized answers.

    D_S = (# disagreeing pairs) / C(k, 2)

    Args:
        normalized_answers: List of normalized answer strings (or None for nulls).

    Returns:
        Float in [0, 1] or None if fewer than 2 valid answers.
    """
    valid = [a for a in normalized_answers if a is not None]
    if len(valid) < 2:
        return None
    pairs = list(combinations(valid, 2))
    n_disagree = sum(1 for a, b in pairs if a != b)
    return n_disagree / len(pairs)


# ─────────────────────────────────────────────────────────────────
# Semantic Disagreement  D_Sem
# ─────────────────────────────────────────────────────────────────
def _cosine_dist(emb1, emb2):
    """Cosine distance between 1D numpy arrays."""
    if emb1 is None or emb2 is None:
        return 0.0
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9)
    return 1.0 - sim

def semantic_disagreement(pairs: list[tuple[str, str]], sem_cache: dict) -> float:
    """Mean pairwise semantic distance (1 - cosine_sim) using precomputed embeddings."""
    if not pairs:
        return 0.0
    
    distances = []
    for a, b in pairs:
        if not a or not b:
            distances.append(1.0) # Max disagree if missing
            continue
            
        emb_a = sem_cache.get(a)
        emb_b = sem_cache.get(b)
        
        distances.append(_cosine_dist(emb_a, emb_b))
        
    return float(np.mean(distances))


# ─────────────────────────────────────────────────────────────────
# Combined DES
# ─────────────────────────────────────────────────────────────────
def compute_DES(surface: float | None, semantic: float | None, alpha: float = DES_ALPHA, beta: float = DES_BETA) -> float | None:
    """Weighted combination: DES = α·D_S + β·D_Sem, clamped to [0, 1].

    Accepts per-question-type alpha/beta so experiments can use
    type-adaptive weighting (e.g., alpha_mc vs alpha_open).
    """
    if surface is None or semantic is None:
        return None
    return float(np.clip(alpha * surface + beta * semantic, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────
# Per-question scoring
# ─────────────────────────────────────────────────────────────────
def score_record(record: dict, sem_cache: dict, model_set: dict | None = None,
                 mc_embed_choices: bool = False, alpha_mc: float = 0.8, alpha_open: float = 0.2) -> dict:
    """Enrich one raw result record with DES scores and correctness flags.

    Args:
        record: Raw question record from JSONL.
        sem_cache: Pre-computed embeddings dict (text → numpy array).
        model_set: Dict of {alias: litellm_name} to score. Defaults to MODELS.
    """
    active_models = model_set or MODELS
    qtype = record.get("question_type", "open")
    model_responses = record.get("model_responses", {})

    embed_answers = {}   # For semantic DES — keeps truncated think content
    norm_answers = {}    # For surface DES + correctness — strips think fully
    null_models = []

    for alias in active_models:
        resp = model_responses.get(alias, {})
        raw = resp.get("response")
        if raw:
            # For embedding: prefer mapping MC letters to full choice text when requested
            embed_text = get_embedding_text_from_response(raw, question=record, mc_embed_choices=mc_embed_choices, model_alias=alias)
            embed_answers[alias] = embed_text if embed_text else None
            # For correctness/surface: strip think tags fully, then normalize
            norm_answers[alias] = normalize_answer(raw, qtype)
        else:
            null_models.append(alias)
            embed_answers[alias] = None
            norm_answers[alias] = None

    # Correctness flags
    correctness_flags = {}
    for alias in active_models:
        norm = norm_answers[alias]
        choices = record.get("choices")
        if norm is not None:
            correctness_flags[alias] = is_correct(norm, record.get("correct_answers", []), qtype, choices=choices)
        else:
            correctness_flags[alias] = None   # Null response — cannot evaluate

    # Error signal: any non-null model got it wrong
    evaluated = {k: v for k, v in correctness_flags.items() if v is not None}
    any_error = int(any(not v for v in evaluated.values())) if evaluated else None

    # DES computation
    surf = surface_disagreement(list(norm_answers.values()))

    # Semantic disagreement pairs — use embed_answers (may contain choice text for MC)
    valid_embeds = [a for a in embed_answers.values() if a]
    embed_pairs = list(combinations(valid_embeds, 2)) if len(valid_embeds) > 1 else []
    sem = semantic_disagreement(embed_pairs, sem_cache)

    # Choose per-question alpha/beta based on question type
    if qtype == "mc":
        alpha = float(alpha_mc)
    else:
        alpha = float(alpha_open)
    beta = 1.0 - alpha

    des = compute_DES(surf, sem, alpha=alpha, beta=beta)

    return {
        **record,
        "correctness_flags": correctness_flags,
        "any_error": any_error,
        "surface_DES": surf,
        "semantic_DES": sem,
        "DES": des,
        "null_models": null_models,
    }

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score raw results with DES")
    parser.add_argument("--expanded", action="store_true",
                        help="Score with all 9 models (after 02c_add_models.py)")
    parser.add_argument("--mc-embed-choices", action="store_true",
                        help="For MC questions, embed the full choice text instead of a single letter")
    parser.add_argument("--alpha-mc", type=float, default=0.8,
                        help="DES alpha (surface weight) for MC questions; beta = 1-alpha")
    parser.add_argument("--alpha-open", type=float, default=0.2,
                        help="DES alpha (surface weight) for open-ended questions; beta = 1-alpha")
    parser.add_argument("--embed-model", type=str, default=None,
                        help="Override embedding model (for ablation)")
    parser.add_argument("--exclude-null-models", action="store_true",
                        help="If set, exclude models whose null/empty response rate exceeds the threshold")
    parser.add_argument("--exclude-null-models-threshold", type=float, default=0.3,
                        help="Null-rate threshold (fraction) above which models are excluded when --exclude-null-models is passed. Default 0.3")
    parser.add_argument("--data-results", type=str, default=None,
                        help="Override DATA_RESULTS directory (for testing)")
    args = parser.parse_args()

    model_set = MODELS_ALL if args.expanded else MODELS
    print(f"Candidate models: {len(model_set)} -> {list(model_set.keys())}")

    # Allow overriding DATA_RESULTS for tests or alternate runs
    data_results = pathlib.Path(args.data_results) if args.data_results else DATA_RESULTS
    input_file  = data_results / "raw_results.jsonl"
    output_file = data_results / "scored_results.jsonl"

    if not input_file.exists():
        print(f"[ERROR] {input_file} not found. Run 02_query_engine.py first.")
        sys.exit(1)

    print(f"Scoring: {input_file}")
    
    records = []
    with open(input_file) as fin:
        for line in fin:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # --- Compute per-model null rates (treat missing/empty response as null) ---
    null_counts = {alias: 0 for alias in model_set}
    total = len(records)
    for rec in records:
        responses = rec.get("model_responses", {})
        for alias in model_set:
            raw = responses.get(alias, {}).get("response")
            if not raw:
                null_counts[alias] += 1

    null_rates = {alias: null_counts[alias] / total if total > 0 else 1.0 for alias in model_set}

    # Save null rates to DATA_RESULTS for auditing
    null_rates_file = data_results / "null_model_null_rates.json"
    try:
        with open(null_rates_file, "w") as jf:
            json.dump({"total_records": total, "null_counts": null_counts, "null_rates": null_rates}, jf, indent=2)
    except Exception:
        print(f"[WARN] Could not write null rates to {null_rates_file}")

    print("Per-model null rates saved to:", null_rates_file)
    for a, r in sorted(null_rates.items(), key=lambda x: x[1], reverse=True):
        print(f"  {a}: {r:.3f} ({null_counts[a]} / {total})")

    # Optionally exclude high-null-rate models (opt-in only)
    if args.exclude_null_models:
        threshold = float(args.exclude_null_models_threshold)
        excluded = [a for a, r in null_rates.items() if r > threshold]
        if excluded:
            print(f"Excluding {len(excluded)} models with null-rate > {threshold}: {excluded}")
            filtered = {k: v for k, v in model_set.items() if k not in excluded}
            if len(filtered) < 2:
                print(f"[ERROR] Excluding models would leave fewer than 2 models ({len(filtered)}). Aborting.")
                sys.exit(1)
            model_set = filtered
        else:
            print("No models exceed null-rate threshold; no exclusions applied.")

    print(f"Using {len(model_set)} models for scoring: {list(model_set.keys())}")

    # Pre-compute all embeddings for embedding-ready answers
    # Uses extract_for_embedding: keeps truncated think content for semantic DES
    print("Pre-computing embeddings for all unique answers...")
    embedder = get_embedder(args.embed_model)
    unique_answers = set()
    for rec in records:
        for alias, data in rec.get("model_responses", {}).items():
            raw = data.get("response")
            if raw:
                embed_text = get_embedding_text_from_response(raw, question=rec, mc_embed_choices=args.mc_embed_choices)
                if embed_text:
                    unique_answers.add(embed_text)
    
    unique_answers = list(unique_answers)
    sem_cache = {}
    if unique_answers:
        embeddings = embedder.encode(unique_answers, show_progress_bar=True)
        for ans, emb in zip(unique_answers, embeddings):
            sem_cache[ans] = emb

    n_ok = n_err = n_null = 0

    with open(output_file, "w") as fout:
        for record in tqdm(records, desc="Scoring"):
            scored = score_record(record, sem_cache, model_set=model_set)
            fout.write(json.dumps(scored) + "\n")

            if scored.get("DES") is not None:
                n_ok += 1
            else:
                n_null += 1
            if scored.get("any_error") == 1:
                n_err += 1

    print(f"\\n{'='*50}")
    print("Scoring Summary")
    print(f"{'='*50}")
    print(f"  Records scored:     {n_ok + n_null}")
    print(f"  Valid DES scores:   {n_ok}")
    print(f"  Null DES (skipped): {n_null}")
    print(f"  Hallucinated (any error): {n_err}")
    print(f"  Output: {output_file}")
