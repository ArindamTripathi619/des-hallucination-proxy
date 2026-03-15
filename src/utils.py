"""
utils.py — Shared constants, model definitions, and helper utilities
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"
"""

from __future__ import annotations

import pathlib
import re
import os

# ─────────────────────────────────────────────────────────────────
# Paths & Directories
# ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_RESULTS = DATA_DIR / "results"
OUTPUTS_TABLES = PROJECT_ROOT / "outputs" / "tables"

# ─────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────
SEED = 42

# LiteLLM Rotator Endpoint (local proxy)
# ─────────────────────────────────────────────────────────────────
LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:8000/v1")
LITELLM_API_KEY  = os.environ.get("LITELLM_API_KEY", "sk-local")

# ─────────────────────────────────────────────────────────────────
# Original 6-Model Lineup (baseline experiment)
# ─────────────────────────────────────────────────────────────────
MODELS_ORIGINAL = {
    "llama-large":   "groq-llama",        # Meta Llama 3.3 70B
    "llama-small":   "groq-llama-small",  # Meta Llama 3.1 8B
    "llama4-scout":  "groq-scout",        # Meta Llama 4 Scout 17B
    "gpt-oss-large": "groq-gpt-oss",      # GPT-OSS 120B
    "qwen":          "groq-qwen",         # Qwen3 32B
    "kimi":          "groq-kimi",         # Kimi K2
}

# ─────────────────────────────────────────────────────────────────
# Expanded 9-Model Lineup (robustness expansion: +3 new families)
# ─────────────────────────────────────────────────────────────────
MODELS_EXPANDED = {
    "gemma":       "scw-gemma",          # Google Gemma 3 27B  → family: google
    "mistral":     "scw-mistral-small",  # Mistral Small 3.2 24B → family: mistral
    "deepseek-r1": "scw-deepseek-r1",    # DeepSeek R1 671B → family: deepseek
}

# MODELS = {**MODELS_ORIGINAL} is the default (6 models, original paper baseline).
# Scripts that need all 9 models for the expanded analysis must use MODELS_ALL.
# Most scripts accept --expanded flag which switches to MODELS_ALL at runtime.
MODELS = {**MODELS_ORIGINAL}             # ← default: original 6

MODELS_ALL = {**MODELS_ORIGINAL, **MODELS_EXPANDED}  # 9 models total
# NOTE: gpt-oss-mini (GPT-OSS 20B) dropped — 80% empty-response rate on Groq,
# no working fallback provider. gpt-oss-large (120B) still represents Microsoft-OSS.

# ─────────────────────────────────────────────────────────────────
# Provider Fallback Map  (model_alias → [fallback LiteLLM names])
# Used when the primary Groq endpoint returns empty/errors.
# Only models with verified alternative providers are listed.
# ─────────────────────────────────────────────────────────────────
FALLBACK_MODELS = {
    "gpt-oss-large": ["scw-gpt-oss"],     # Scaleway GPT-OSS 120B — tested OK
}

# ─────────────────────────────────────────────────────────────────
# Architectural Family Mapping  (all 9 models)
# ─────────────────────────────────────────────────────────────────
FAMILY_MAP = {
    # ── Original 6 ──
    "llama-large":   "meta",
    "llama-small":   "meta",
    "llama4-scout":  "meta-next-gen",    # Different architecture generation
    "gpt-oss-large": "microsoft-oss",
    "qwen":          "alibaba",
    "kimi":          "moonshot",
    # ── Expansion 3 ──
    "gemma":         "google",
    "mistral":        "mistral",
    # NOTE (issue #11): DeepSeek-R1 shows anomalously low MMLU accuracy (22.9%).
    # This is most likely a response-format extraction failure, not a true capability
    # collapse. The full pipeline owner should inspect raw DeepSeek MMLU responses
    # and verify the normalize_answer() extraction path for this model.
    "deepseek-r1":   "deepseek",
}

# ─────────────────────────────────────────────────────────────────
# Pair Definitions for Architecture-Gap Analysis (RQ3)
# ─────────────────────────────────────────────────────────────────
WITHIN_FAMILY_PAIRS = [
    ("llama-large",   "llama-small"),    # Meta: large vs small
    ("llama-large",   "llama4-scout"),   # Meta: gen2 vs gen3
]

CROSS_FAMILY_PAIRS = [
    ("llama-large",   "gpt-oss-large"),  # Meta vs Microsoft
    ("llama-large",   "qwen"),           # Meta vs Alibaba
    ("llama-large",   "kimi"),           # Meta vs Moonshot
    ("gpt-oss-large", "qwen"),           # Microsoft vs Alibaba
    ("gpt-oss-large", "kimi"),           # Microsoft vs Moonshot
    ("qwen",          "kimi"),           # Alibaba vs Moonshot
]

# Expanded pairs — adds the 3 new families to the cross-family grid
CROSS_FAMILY_PAIRS_EXPANDED = CROSS_FAMILY_PAIRS + [
    ("llama-large",   "gemma"),          # Meta vs Google
    ("llama-large",   "mistral"),        # Meta vs Mistral
    ("llama-large",   "deepseek-r1"),    # Meta vs DeepSeek
    ("gpt-oss-large", "gemma"),          # Microsoft vs Google
    ("gpt-oss-large", "mistral"),        # Microsoft vs Mistral
    ("gpt-oss-large", "deepseek-r1"),    # Microsoft vs DeepSeek
    ("qwen",          "gemma"),          # Alibaba vs Google
    ("qwen",          "mistral"),        # Alibaba vs Mistral
    ("qwen",          "deepseek-r1"),    # Alibaba vs DeepSeek
    ("kimi",          "gemma"),          # Moonshot vs Google
    ("kimi",          "mistral"),        # Moonshot vs Mistral
    ("kimi",          "deepseek-r1"),    # Moonshot vs DeepSeek
    ("gemma",         "mistral"),        # Google vs Mistral
    ("gemma",         "deepseek-r1"),    # Google vs DeepSeek
    ("mistral",       "deepseek-r1"),    # Mistral vs DeepSeek
]

# ─────────────────────────────────────────────────────────────────
#  DES Weighting
# ─────────────────────────────────────────────────────────────────
DES_ALPHA = 0.4   # Weight for surface disagreement (D_S)
DES_BETA  = 0.6   # Weight for semantic disagreement (D_Sem)

# ─────────────────────────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────────────────────────
OPEN_ENDED_PROMPT = (
    "Answer the following question in 1-3 words only.\n"
    "Do not explain. Just give the answer.\n\n"
    "Question: {question}\n"
    "Answer:"
)

MC_PROMPT = (
    "Choose the correct answer. Reply with only the letter (A, B, C, or D).\n\n"
    "Question: {question}\n"
    "A) {choice_a}\n"
    "B) {choice_b}\n"
    "C) {choice_c}\n"
    "D) {choice_d}\n\n"
    "Answer:"
)

QWEN_NO_THINK_SYSTEM = "/no_think"

# ─────────────────────────────────────────────────────────────────
# Answer Normalization
# ─────────────────────────────────────────────────────────────────
def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model responses (e.g., Qwen3).

    Handles both closed (<think>...</think>) and unclosed (<think>... EOF)
    blocks — the latter occurs when the model's reasoning exceeds max_tokens
    and the response is truncated before the closing tag.
    """
    if not text:
        return text
    # 1. Closed tags: <think>...</think> (non-greedy across newlines)
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    # 2. Unclosed tags: <think> with no matching </think> — strip to end
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    return text.strip()





def extract_for_embedding(text: str, prefer_final: bool = True) -> str:
    """Extract text suitable for semantic embedding from a model response.

    By default (prefer_final=True) this will return the final answer portion of
    a response (text after a closing </think> or the last non-empty line). If
    that extraction yields nothing, it falls back to returning the cleaned
    response per legacy behavior. This avoids inflating semantic distances when
    one model emits a long CoT while others emit terse answers.
    """
    if not text:
        return ""

    # If closed think block exists, prefer the post-think answer
    if "<think>" in text and "</think>" in text:
        # Prefer the text after the last closing </think> as the final answer
        after = text.split("</think>")[-1].strip()
        if after:
            return after
        # If nothing after the tag, fall back to removing the think block
        cleaned = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        if cleaned:
            return cleaned
        return text.strip()

    # If prefer_final and there's evidence of a think block (possibly truncated),
    # try to extract the last non-empty line as the final answer
    if prefer_final and "<think>" in text:
        # Remove opening tag then take last non-empty line
        without_tag = re.sub(r'<think>\s*', '', text, count=1).strip()
        lines = [l.strip() for l in without_tag.splitlines() if l.strip()]
        if lines:
            # Heuristic: last short line is likely the final answer
            last = lines[-1]
            if len(last) < 500:  # avoid returning entire CoT
                return last
        # if unsuccessful fall back to returning the reasoning content
        return without_tag

    # If unclosed think block and prefer_final==False, or no think tags at all,
    # preserve the legacy behavior: strip tags and return remainder
    if "<think>" in text:
        return re.sub(r'<think>\s*', '', text, count=1).strip()

    # No think tags — return as-is
    return text.strip()


# ─────────────────────────────────────────────────────────────────
# Levenshtein Distance (issue #15: hoisted to module level)
# Previously defined inside is_correct()'s inner loop, which caused
# Python to recreate the function object on every call. Now defined
# once here as a private helper.
# ─────────────────────────────────────────────────────────────────
def _levenshtein(s1: str, s2: str) -> int:
    """Compute the edit distance between two strings (Wagner-Fischer)."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)
    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0
    return v0[len(s2)]

def get_embedding_text_from_response(raw: str, question: dict | None = None, mc_embed_choices: bool = False, model_alias: str | None = None) -> str:
    """Return the text to be embedded for a given model response.

    For multiple-choice questions, when `mc_embed_choices` is True, prefer
    embedding the full choice text corresponding to the model's selected
    letter (A/B/C/D). This avoids embedding single-letter tokens which
    are semantically vacuous. Falls back to extracting the final answer
    portion when choice text cannot be resolved.

    Args:
        raw: Raw model response string.
        question: The question dict from dataset (may contain 'choices').
        mc_embed_choices: Whether to map MC letter to full choice text.

    Returns:
        String suitable for semantic embedding (may be empty).
    """
    if not raw:
        return ""
    qtype = None
    choices = None
    if question:
        qtype = question.get("question_type")
        choices = question.get("choices")

    # If MC and user requested choice-text embedding, map letter -> text
    if mc_embed_choices and qtype == "mc":
        # Normalize answer to a single letter first
        letter = normalize_answer(raw, "mc")
        if letter and choices:
            idx_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            idx = idx_map.get(letter.upper())
            if idx is not None and idx < len(choices):
                choice_text = choices[idx].strip()
                if choice_text:
                    return choice_text
    # fallback to final extraction if mapping failed
    return extract_for_embedding(raw, prefer_final=True)

    # NOTE (issue #10): The DeepSeek-R1 specific block that was here (L260-L283)
    # was dead code — the early `return` above made it unreachable.
    # The logic was also redundant with `extract_for_embedding()`. Removed.

def normalize_answer(answer: str, question_type: str) -> str | None:
    """Normalize model response based on question type.

    Args:
        answer: Raw string response from LLM.
        question_type: "mc" for multiple-choice, "open" for open-ended.

    Returns:
        Normalized string or None if unparseable.
    """
    if answer is None:
        return None
        
    # Prefer the model's final answer (handles CoT outputs like DeepSeek R1)
    final = extract_for_embedding(answer, prefer_final=True) if answer else None
    if final:
        answer = final
    else:
        # Fallback: remove any <think> blocks entirely
        answer = strip_thinking_tags(answer)
    answer = answer.strip()

    if question_type == "mc":
        match = re.search(r'\b([ABCD])\b', answer.upper())
        return match.group(1) if match else None
    else:
        # Open-ended: lowercase, strip punctuation
        return answer.lower().strip().rstrip('.,!?;:')


def is_correct(predicted: str, correct_answers: list, question_type: str, choices: list | None = None) -> bool:
    """Check if a normalized prediction matches any correct alias.

    Args:
        predicted: Normalized prediction string.
        correct_answers: List of acceptable answers (aliases or single letter).
        question_type: "mc" or "open".

    Returns:
        True if prediction is correct.
    """
    if predicted is None:
        return False
    if question_type == "mc":
        # Allow multiple correct-answer formats: letters (A/B/C/D) or full choice text.
        pred = predicted.strip().upper()
        # If predicted is single letter, check against provided correct letters
        if len(pred) == 1 and pred in "ABCD":
            for a in correct_answers:
                if isinstance(a, str) and len(a.strip()) == 1 and a.strip().upper() == pred:
                    return True
            # If ground truth provided as full choice text, map predicted letter -> choice
            if choices:
                idx_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                idx = idx_map.get(pred)
                if idx is not None and idx < len(choices):
                    gt = choices[idx].strip().lower()
                    for a in correct_answers:
                        if a and a.strip().lower() == gt:
                            return True
            return False
        # If predicted is longer text (not a single letter), compare to choices and aliases
        pred_text = predicted.strip().lower()
        for a in correct_answers:
            if a and a.strip().lower() == pred_text:
                return True
        if choices:
            for c in choices:
                if c and c.strip().lower() == pred_text:
                    # If the model accidentally returned full choice text, accept it
                    return True
        return False
    else:
        # Stricter matching for open-ended answers to avoid CoT-inflated matches.
        pred = predicted.lower().strip()
        def normalize_text(s: str) -> str:
            s = s.lower().strip()
            # remove leading articles
            s = re.sub(r'^the\s+|^a\s+|^an\s+', '', s)
            # strip punctuation
            s = re.sub(r'[.,!?;:]$', '', s)
            return s

        pred_norm = normalize_text(pred)

        for a in correct_answers:
            if not a:
                continue
            gt = a.lower().strip()
            gt_norm = normalize_text(gt)
            # Exact match
            if pred_norm == gt_norm:
                return True
            # Token-overlap heuristic: require high overlap with the ground truth
            pred_tokens = pred_norm.split()
            gt_tokens = gt_norm.split()
            if gt_tokens:
                common = len(set(pred_tokens) & set(gt_tokens))
                if (common / len(gt_tokens)) >= 0.8:
                    return True
            # Normalized edit (Levenshtein) distance threshold
            max_len = max(len(pred_norm), len(gt_norm))
            if max_len > 0:
                ed = _levenshtein(pred_norm, gt_norm)
                if (ed / max_len) <= 0.2:
                    return True
        return False


def build_prompt(question: dict) -> tuple[str, str | None]:
    """Build the prompt string and optional system message for a question.

    Args:
        question: Dict with keys: question, question_type, choices (for MC).

    Returns:
        (prompt_text, system_prompt_or_None)
    """
    qtype = question.get("question_type", "open")
    if qtype == "mc":
        choices = question.get("choices", [])
        # Pad to 4 choices if necessary
        while len(choices) < 4:
            choices.append("N/A")
        prompt = MC_PROMPT.format(
            question=question["question"],
            choice_a=choices[0],
            choice_b=choices[1],
            choice_c=choices[2],
            choice_d=choices[3],
        )
    else:
        prompt = OPEN_ENDED_PROMPT.format(question=question["question"])
    return prompt, None

# ─────────────────────────────────────────────────────────────────
# --- Singleton Embedder ---
_EMBEDDER = None
_EMBEDDER_NAME = "all-MiniLM-L6-v2"

# Embedding model names for ablation study
EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",       # Default (dim 384, fast)
    "all-mpnet-base-v2",      # Higher quality (dim 768)
    "intfloat/e5-large-v2",   # Instruction-tuned (dim 1024)
]

def get_embedder(model_name: str | None = None):
    """Lazy load the sentence transformer model as a singleton.

    Args:
        model_name: Override model name. If None, uses default (all-MiniLM-L6-v2).
                    Passing a different name resets the singleton.
    """
    global _EMBEDDER, _EMBEDDER_NAME
    target = model_name or "all-MiniLM-L6-v2"
    if _EMBEDDER is None or _EMBEDDER_NAME != target:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(target)
        _EMBEDDER_NAME = target
    return _EMBEDDER
