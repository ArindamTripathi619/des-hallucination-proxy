"""
analyze_alpha.py — Alpha sweep per question type + type-adaptive alpha

NOTE: Pure semantic DES (alpha=0.0) often shows weak AUROC (~0.52) in this sweep.
This is likely due to semantic entropy on short answers (1-3 words) where the 
All-MiniLM-L6-v2 model fails to capture fine-grained semantic distinctions,
leading to compressed distance distributions. The fused DES (alpha=0.4)
outperforms components by capturing both lexical and distributed signal.
"""
import json, sys, numpy as np, pathlib
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(pathlib.Path(__file__).parent))

with open(pathlib.Path(__file__).parent.parent / "data/results/scored_results.jsonl") as f:
    scored = [json.loads(l) for l in f if l.strip()]

mc = [r for r in scored if r["question_type"] == "mc"]
op = [r for r in scored if r["question_type"] == "open"]

print("Alpha sweep for MC questions:")
y_mc = np.array([r["any_error"] for r in mc], dtype=float)
surf_mc = np.array([r["surface_DES"] for r in mc], dtype=float)
sem_mc = np.array([r["semantic_DES"] for r in mc], dtype=float)
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    des = alpha * surf_mc + (1 - alpha) * sem_mc
    v = ~np.isnan(y_mc) & ~np.isnan(des)
    auroc = roc_auc_score(y_mc[v], des[v])
    print(f"  alpha={alpha:.1f}: AUROC={auroc:.4f}")

print()
print("Alpha sweep for open-ended questions:")
y_op = np.array([r["any_error"] for r in op], dtype=float)
surf_op = np.array([r["surface_DES"] for r in op], dtype=float)
sem_op = np.array([r["semantic_DES"] for r in op], dtype=float)
for alpha in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    des = alpha * surf_op + (1 - alpha) * sem_op
    v = ~np.isnan(y_op) & ~np.isnan(des)
    auroc = roc_auc_score(y_op[v], des[v])
    print(f"  alpha={alpha:.1f}: AUROC={auroc:.4f}")

# Type-adaptive alpha
print()
print("Type-adaptive alpha experiments:")
for a_mc, a_op in [(1.0, 0.2), (1.0, 0.3), (0.8, 0.2), (0.8, 0.3), (0.4, 0.4)]:
    all_des = []
    all_y = []
    for r in scored:
        if r["question_type"] == "mc":
            des = a_mc * r["surface_DES"] + (1 - a_mc) * r["semantic_DES"]
        else:
            des = a_op * r["surface_DES"] + (1 - a_op) * r["semantic_DES"]
        all_des.append(des)
        all_y.append(r["any_error"])
    all_des = np.array(all_des)
    all_y = np.array(all_y, dtype=float)
    v = ~np.isnan(all_y) & ~np.isnan(all_des)
    auroc = roc_auc_score(all_y[v], all_des[v])
    print(f"  alpha_mc={a_mc:.1f} alpha_open={a_op:.1f}: AUROC={auroc:.4f}")

# Current uniform alpha=0.4 result (computed dynamically to avoid stale values):
all_des_uniform = 0.4 * np.array([r["surface_DES"] for r in scored], dtype=float) + \
                  0.6 * np.array([r["semantic_DES"] for r in scored], dtype=float)
all_y_uniform = np.array([r["any_error"] for r in scored], dtype=float)
v_u = ~np.isnan(all_y_uniform) & ~np.isnan(all_des_uniform)
auroc_uniform = roc_auc_score(all_y_uniform[v_u], all_des_uniform[v_u])
print(f"\n  Current (uniform alpha=0.4): AUROC={auroc_uniform:.4f}")

# GPT-OSS null handling: what if we impute DES for null GPT-OSS pairs
# instead of dropping them?
print()
print("GPT-OSS null impact:")
has_gpt = [r for r in scored if r["model_responses"].get("gpt-oss-large", {}).get("response") is not None]
no_gpt = [r for r in scored if r["model_responses"].get("gpt-oss-large", {}).get("response") is None]
print(f"  Records with GPT-OSS: {len(has_gpt)}")
print(f"  Records without GPT-OSS: {len(no_gpt)}")

y_has = np.array([r["any_error"] for r in has_gpt], dtype=float)
des_has = np.array([r["DES"] for r in has_gpt], dtype=float)
v = ~np.isnan(y_has) & ~np.isnan(des_has)
print(f"  AUROC (with GPT-OSS): {roc_auc_score(y_has[v], des_has[v]):.4f}")

y_no = np.array([r["any_error"] for r in no_gpt], dtype=float)
des_no = np.array([r["DES"] for r in no_gpt], dtype=float)
v2 = ~np.isnan(y_no) & ~np.isnan(des_no)
print(f"  AUROC (without GPT-OSS): {roc_auc_score(y_no[v2], des_no[v2]):.4f}")
