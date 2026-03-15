"""
01_data_prep.py — Dataset loading, sampling, and JSONL export
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"

Outputs (in data/processed/):
  triviaqa_800.jsonl
  truthfulqa_817.jsonl
  mmlu_400.jsonl

Each record schema:
  {
    "id": str,
    "source": "triviaqa" | "truthfulqa" | "mmlu",
    "domain": str,
    "question": str,
    "correct_answers": list[str],   # for MC: ["A"] or ["B"] etc.
    "choices": list[str] | null,    # for MC: [choice_a, choice_b, ...]
    "question_type": "open" | "mc"
  }
"""

import json
import pathlib
import sys

# Allow importing from src/
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import DATA_PROCESSED, SEED

from datasets import load_dataset
from tqdm import tqdm

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# 1. TriviaQA — 800 open-ended questions
# ─────────────────────────────────────────────────────────────────
def prepare_triviaqa(n=800, output_file=None):
    print("Loading TriviaQA...")
    output_file = output_file or DATA_PROCESSED / "triviaqa_800.jsonl"
    tqa = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    tqa = tqa.filter(lambda x: len(x["answer"]["aliases"]) > 0)
    tqa = tqa.shuffle(seed=SEED).select(range(min(n, len(tqa))))

    records = []
    for item in tqdm(tqa, desc="TriviaQA"):
        # Use ONLY answer["value"] (the single canonical answer string).
        # Both "aliases" and "normalized_aliases" contain Wikipedia entity
        # redirects / disambiguation noise (e.g., "Yukon Optics" for "Pulsar",
        # "Administrative divisions of Nicaragua" for "Nicaragua").
        # The correctness checker (normalize_answer + is_correct) already handles
        # minor casing/article/punctuation variations via normalization.
        primary = item["answer"]["value"].strip()

        record = {
            "id": f"tqa_{len(records):04d}",
            "source": "triviaqa",
            "domain": "general",
            "question": item["question"],
            "correct_answers": [primary],
            "choices": None,
            "question_type": "open",
        }
        records.append(record)

    with open(output_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"  → Saved {len(records)} TriviaQA questions to {output_file}")
    return len(records)


# ─────────────────────────────────────────────────────────────────
# 2. TruthfulQA — 817 multiple-choice questions
# ─────────────────────────────────────────────────────────────────
def prepare_truthfulqa(output_file=None):
    print("Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

    output_file = output_file or DATA_PROCESSED / "truthfulqa_817.jsonl"
    with open(output_file, "w") as f:
        for i, item in enumerate(tqdm(ds, desc="TruthfulQA")):
            mc_labels = item["mc1_targets"]["labels"]
            mc_choices = item["mc1_targets"]["choices"]

            # Find the correct answer index
            correct_idx = mc_labels.index(1)
            letter_map = ["A", "B", "C", "D"]

            # Use first 4 choices only
            choices_4 = mc_choices[:4]
            correct_letter = letter_map[correct_idx] if correct_idx < 4 else None

            if correct_letter is None:
                continue  # Skip if correct answer outside first 4

            record = {
                "id": f"tfqa_{i:04d}",
                "source": "truthfulqa",
                "domain": "mixed",
                "question": item["question"],
                "correct_answers": [correct_letter],
                "choices": choices_4,
                "question_type": "mc",
            }
            f.write(json.dumps(record) + "\n")

    print(f"  → Saved TruthfulQA questions to {output_file}")


# ─────────────────────────────────────────────────────────────────
# 3. MMLU — 4 domains × 100 questions
# ─────────────────────────────────────────────────────────────────
MMLU_DOMAINS = {
    "high_school_us_history":      "history",
    "high_school_biology":    "biology",
    "professional_law":       "law",
    "high_school_mathematics":"math",
}

def prepare_mmlu(n_per_domain=100, output_file=None):
    print("Loading MMLU...")
    output_file = output_file or DATA_PROCESSED / "mmlu_400.jsonl"
    letter_map = ["A", "B", "C", "D"]
    total = 0

    with open(output_file, "w") as f:
        for hf_name, domain_label in MMLU_DOMAINS.items():
            ds = load_dataset("cais/mmlu", hf_name, split="test")
            ds = ds.shuffle(seed=SEED).select(range(min(n_per_domain, len(ds))))

            for i, item in enumerate(tqdm(ds, desc=f"MMLU-{domain_label}")):
                correct_letter = letter_map[item["answer"]]
                record = {
                    "id": f"mmlu_{domain_label}_{i:03d}",
                    "source": "mmlu",
                    "domain": domain_label,
                    "question": item["question"],
                    "correct_answers": [correct_letter],
                    "choices": item["choices"],
                    "question_type": "mc",
                }
                f.write(json.dumps(record) + "\n")
                total += 1

    print(f"  → Saved {total} MMLU questions to {output_file}")
    return total


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    n_tqa  = prepare_triviaqa()
    prepare_truthfulqa()
    n_mmlu = prepare_mmlu()

    print("\n" + "=" * 50)
    print("Dataset Summary")
    print("=" * 50)
    print(f"  TriviaQA (open):   {n_tqa}")
    print(f"  TruthfulQA (MC):   ~817")
    print(f"  MMLU (MC, 4 dom):  {n_mmlu}")
    print(f"  TOTAL:             ~{n_tqa + 817 + n_mmlu}")
    print("\nAll JSONL files saved to:", DATA_PROCESSED)
