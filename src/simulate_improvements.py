"""
simulate_improvements.py — Simulate AUROC with improved normalize/extract
"""
import json, sys, numpy as np, re, pathlib
from sklearn.metrics import roc_auc_score
from itertools import combinations

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import is_correct, get_embedder, DES_ALPHA, DES_BETA

with open(pathlib.Path(__file__).parent.parent / "data/results/scored_results.jsonl") as f:
    scored = [json.loads(l) for l in f if l.strip()]

def extract_final_answer(raw):
    """Extract final answer, stripping CoT reasoning."""
    if raw is None:
        return None
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    elif "<think>" in raw:
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
    return raw

def improved_normalize(raw, qtype):
    """Improved normalization: extract final answer first, then normalize."""
    if raw is None:
        return None
    answer = extract_final_answer(raw)
    if not answer:
        return None
    answer = answer.strip()
    if qtype == "mc":
        match = re.search(r"\b([ABCD])\b", answer.upper())
        return match.group(1) if match else None
    else:
        answer = re.sub(
            r"^(the answer is|answer:|it is)\s*",
            "", answer.lower(), flags=re.IGNORECASE,
        )
        return answer.lower().strip().rstrip(".,!?;:")

def improved_extract_for_embedding(raw):
    """Always embed final answer only, not CoT."""
    if not raw:
        return ""
    answer = extract_final_answer(raw)
    return answer.strip() if answer else ""

# ────────────────────────────────────────────────────
MODELS = [
    "llama-large", "llama-small", "llama4-scout",
    "gpt-oss-large", "qwen", "kimi",
    "gemma", "mistral", "deepseek-r1",
]
embedder = get_embedder()

improved_des = []
improved_any_error = []
original_des = []

for rec in scored:
    qtype = rec["question_type"]
    norms = {}
    embed_texts = []
    for m in MODELS:
        resp = rec["model_responses"].get(m, {}).get("response")
        if resp:
            norms[m] = improved_normalize(resp, qtype)
            et = improved_extract_for_embedding(resp)
            if et:
                embed_texts.append(et)
        else:
            norms[m] = None

    valid_norms = [v for v in norms.values() if v is not None]
    if len(valid_norms) < 2:
        improved_des.append(np.nan)
        improved_any_error.append(np.nan)
        original_des.append(rec["DES"])
        continue

    pairs = list(combinations(valid_norms, 2))
    surf = sum(1 for a, b in pairs if a != b) / len(pairs)

    if len(embed_texts) >= 2:
        embs = embedder.encode(embed_texts, convert_to_numpy=True)
        norm_v = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / norm_v
        p = list(combinations(range(len(embs)), 2))
        sims = [np.dot(embs[i], embs[j]) for i, j in p]
        sem = 1.0 - np.mean(sims)
    else:
        sem = 0.0

    des = float(np.clip(DES_ALPHA * surf + DES_BETA * sem, 0, 1))
    improved_des.append(des)
    original_des.append(rec["DES"])

    correct_answers = rec["correct_answers"]
    any_err = 0
    for m in MODELS:
        n = norms.get(m)
        if n is not None and not is_correct(n, correct_answers, qtype):
            any_err = 1
            break
    improved_any_error.append(any_err)

# ── Results ──
orig_y = np.array([r["any_error"] for r in scored], dtype=float)
orig_des_arr = np.array(original_des, dtype=float)
imp_des_arr = np.array(improved_des, dtype=float)
imp_y = np.array(improved_any_error, dtype=float)

v_o = ~np.isnan(orig_y) & ~np.isnan(orig_des_arr)
auroc_orig = roc_auc_score(orig_y[v_o], orig_des_arr[v_o])

v_i = ~np.isnan(imp_y) & ~np.isnan(imp_des_arr)
auroc_improved = roc_auc_score(imp_y[v_i], imp_des_arr[v_i])

v_m = ~np.isnan(orig_y) & ~np.isnan(imp_des_arr)
auroc_mixed = roc_auc_score(orig_y[v_m], imp_des_arr[v_m])

print("AUROC Comparison:")
print(f"  Original DES + Original labels:  {auroc_orig:.4f}")
print(f"  Improved DES + Original labels:  {auroc_mixed:.4f}  (DES fix only)")
print(f"  Improved DES + Improved labels:  {auroc_improved:.4f}  (DES + label fix)")
print(f"  Delta (DES fix):   {auroc_mixed - auroc_orig:+.4f}")
print(f"  Delta (both fix):  {auroc_improved - auroc_orig:+.4f}")

orig_hall = np.nanmean(orig_y)
imp_hall = np.nanmean(imp_y)
print(f"\nHallucination rate: {orig_hall:.3f} -> {imp_hall:.3f} ({imp_hall-orig_hall:+.3f})")

# ── DS-R1 specific: embedding distance before/after ──
ds_dists_before = []
ds_dists_after = []
for rec in scored[:200]:
    ds_raw = rec["model_responses"].get("deepseek-r1", {}).get("response")
    ll_raw = rec["model_responses"].get("llama-large", {}).get("response")
    if not ds_raw or not ll_raw:
        continue
    # Before: embed full text
    from utils import extract_for_embedding as orig_extract
    e_before = embedder.encode([orig_extract(ds_raw), orig_extract(ll_raw)])
    n1 = np.linalg.norm(e_before, axis=1, keepdims=True) + 1e-12
    e_before = e_before / n1
    ds_dists_before.append(1.0 - np.dot(e_before[0], e_before[1]))
    # After: embed final answer only
    e_after = embedder.encode([improved_extract_for_embedding(ds_raw), improved_extract_for_embedding(ll_raw)])
    n2 = np.linalg.norm(e_after, axis=1, keepdims=True) + 1e-12
    e_after = e_after / n2
    ds_dists_after.append(1.0 - np.dot(e_after[0], e_after[1]))

print(f"\nDS-R1 vs llama-large semantic distance (200 Qs):")
print(f"  Before (full text):     mean={np.mean(ds_dists_before):.3f}")
print(f"  After (final answer):   mean={np.mean(ds_dists_after):.3f}")
print(f"  Reduction:              {np.mean(ds_dists_before)-np.mean(ds_dists_after):.3f}")

# ── Per-source AUROC ──
print("\nPer-source AUROC (improved DES + improved labels):")
for src in ["triviaqa", "truthfulqa", "mmlu"]:
    mask = np.array([r["source"] == src for r in scored])
    v = mask & v_i
    if v.sum() > 10 and len(np.unique(imp_y[v])) >= 2:
        a = roc_auc_score(imp_y[v], imp_des_arr[v])
        # Original
        v2 = mask & v_o
        a_orig = roc_auc_score(orig_y[v2], orig_des_arr[v2])
        print(f"  {src:15s}: {a_orig:.4f} -> {a:.4f} ({a-a_orig:+.4f})")
