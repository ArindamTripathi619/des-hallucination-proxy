"""
analyze_extraction.py — Accuracy impact of final-answer extraction per model
"""
import json, sys, numpy as np, re, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import normalize_answer, is_correct

with open(pathlib.Path(__file__).parent.parent / "data/results/scored_results.jsonl") as f:
    scored = [json.loads(l) for l in f if l.strip()]

def extract_final(raw):
    if raw is None:
        return None
    if "</think>" in raw:
        return raw.split("</think>")[-1].strip()
    elif "<think>" in raw:
        return re.sub(r"<think>.*", "", raw, flags=re.DOTALL).strip()
    return raw

models = [
    "llama-large", "llama-small", "llama4-scout", "gpt-oss-large",
    "qwen", "kimi", "gemma", "mistral", "deepseek-r1",
]

print("MC accuracy comparison by model (old vs new extraction):")
for model in models:
    mc_recs = [r for r in scored if r["question_type"] == "mc"]
    old_c = new_c = total = 0
    for rec in mc_recs:
        resp = rec["model_responses"].get(model, {}).get("response")
        if not resp:
            continue
        total += 1
        gt = rec["correct_answers"]
        old_norm = normalize_answer(resp, "mc")
        old_ok = is_correct(old_norm, gt, "mc") if old_norm else False
        if old_ok:
            old_c += 1
        final = extract_final(resp)
        if final:
            match = re.search(r"\b([ABCD])\b", final.upper())
            new_norm = match.group(1) if match else None
        else:
            new_norm = None
        new_ok = is_correct(new_norm, gt, "mc") if new_norm else False
        if new_ok:
            new_c += 1
    delta = new_c - old_c
    flag = " <<<" if abs(delta) > 5 else ""
    print(f"  {model:20s}: {old_c}/{total} ({100*old_c/total:.1f}%) -> {new_c}/{total} ({100*new_c/total:.1f}%)  delta={delta:+d}{flag}")

print()
print("Open-ended DS-R1 accuracy (old vs new extraction):")
open_recs = [r for r in scored if r["question_type"] == "open"]
ds_old = ds_new = ds_total = 0
for rec in open_recs:
    resp = rec["model_responses"].get("deepseek-r1", {}).get("response")
    if not resp:
        continue
    ds_total += 1
    gt = rec["correct_answers"]
    old_norm = normalize_answer(resp, "open")
    old_ok = any(old_norm in a.lower() or a.lower() in old_norm for a in gt) if old_norm else False
    if old_ok:
        ds_old += 1
    final = extract_final(resp)
    if final:
        new_norm = final.lower().strip().rstrip(".,;:")
    else:
        new_norm = None
    new_ok = any(new_norm in a.lower() or a.lower() in new_norm for a in gt) if new_norm else False
    if new_ok:
        ds_new += 1
print(f"  deepseek-r1 open: {ds_old}/{ds_total} ({100*ds_old/ds_total:.1f}%) -> {ds_new}/{ds_total} ({100*ds_new/ds_total:.1f}%)  delta={ds_new-ds_old:+d}")

# Surface DES sensitivity: how does improved normalization for open-ended reduce noise?
print()
print("Surface DES noise analysis (TriviaQA, first 200 questions, excluding DS-R1):")
from itertools import combinations
terse_models = ["llama-large", "llama-small", "llama4-scout", "kimi", "gemma", "mistral"]
triviaqa = [r for r in scored if r["source"] == "triviaqa"][:200]
both_correct_disagree = 0
genuine_disagree = 0
agree = 0
total_pairs = 0
for rec in triviaqa:
    norms = {}
    for m in terse_models:
        resp = rec["model_responses"].get(m, {}).get("response")
        if resp:
            norms[m] = normalize_answer(resp, "open")
    for (m1, n1), (m2, n2) in combinations(norms.items(), 2):
        if n1 is None or n2 is None:
            continue
        total_pairs += 1
        if n1 == n2:
            agree += 1
        else:
            gt = rec["correct_answers"]
            c1 = any(n1 in a.lower() or a.lower() in n1 for a in gt)
            c2 = any(n2 in a.lower() or a.lower() in n2 for a in gt)
            if c1 and c2:
                both_correct_disagree += 1
            else:
                genuine_disagree += 1

print(f"  Total pairs: {total_pairs}")
print(f"  Agree (same text): {agree} ({100*agree/total_pairs:.1f}%)")
print(f"  Both correct, different text: {both_correct_disagree} ({100*both_correct_disagree/total_pairs:.1f}%)")
print(f"  Genuine disagree (>=1 wrong): {genuine_disagree} ({100*genuine_disagree/total_pairs:.1f}%)")
