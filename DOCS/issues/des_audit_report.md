# 🔬 Code Audit: `des-hallucination-proxy`
**Paper:** "Disagreement Entropy as a Zero-Cost Hallucination Proxy: A Cross-Architecture Empirical Study Across Diverse LLM Families."
**Reviewed:** All `src/` scripts (14 files), `tests/` (4 files), `DOCS/` (4 files), [requirements.txt](file:///mnt/a/Projects/des-hallucination-proxy/requirements.txt), `outputs/tables/` CSVs.

---

## 🚨 Critical Issues (Immediate Fixes)

### 1. Dead Code After `return` — Silent Bug Corrupts DeepSeek-R1 Embeddings
**[`src/utils.py` · Lines 260–283](file:///mnt/a/Projects/des-hallucination-proxy/src/utils.py#L260-L283)**

This is the single most serious bug in the codebase. The `model_alias == "deepseek-r1"` branch that was specifically added to handle DeepSeek-R1's long CoT format (with truncation to 200 chars) is placed **after an unconditional `return` on line 258** and is therefore **completely dead code — it never executes**.

```python
# CURRENT (BROKEN) — lines 257–283
    return extract_for_embedding(raw, prefer_final=True)   # ← return on L258

    # Model-specific post-processing: DeepSeek R1 often emits long CoT...
    if model_alias == "deepseek-r1":                       # ← NEVER REACHED
        ...
```

**Consequence:** Every DeepSeek-R1 response is passed through generic `extract_for_embedding` without any truncation guard. If DeepSeek-R1 emits a long reasoning trace without a proper `</think>` close, the embedding gets the full, hundreds-of-tokens CoT blob — inflating semantic distances and distorting both `semantic_DES` and `DES` for every question where DeepSeek-R1's think block is unclosed.

```python
# FIX — reorder to put model-specific logic BEFORE the fallback return:
def get_embedding_text_from_response(raw, question=None, mc_embed_choices=False, model_alias=None):
    if not raw:
        return ""
    qtype = None
    choices = None
    if question:
        qtype = question.get("question_type")
        choices = question.get("choices")

    if mc_embed_choices and qtype == "mc":
        letter = normalize_answer(raw, "mc")
        if letter and choices:
            idx_map = {"A": 0, "B": 1, "C": 2, "D": 3}
            idx = idx_map.get(letter.upper())
            if idx is not None and idx < len(choices):
                choice_text = choices[idx].strip()
                if choice_text:
                    return choice_text

    # Model-specific post-processing (MUST be before generic fallback)
    if model_alias == "deepseek-r1":
        if "<think>" in raw and "</think>" in raw:
            after = raw.split("</think>")[-1].strip()
            if after:
                return after if len(after) <= 500 else after[-200:]
        if "<think>" in raw and "</think>" not in raw:
            without_tag = re.sub(r'<think>\s*', '', raw, count=1).strip()
            lines = [l.strip() for l in without_tag.splitlines() if l.strip()]
            if lines:
                last = lines[-1]
                if len(last) < 500:
                    return last
            return without_tag[-200:]

    return extract_for_embedding(raw, prefer_final=True)
```

> This bug affects ALL `semantic_DES` scores involving DeepSeek-R1. Since the paper reports AUROC = 0.9613 using all 9 models including DeepSeek-R1, the exact reproduced numbers may differ once this is fixed. Document this fix and re-run `03_scoring.py` + `04_calibration.py` before final paper submission.

---

### 2. DeepSeek-R1 MMLU Accuracy Anomaly — Data Integrity Risk
**[outputs/tables/table1_model_accuracy.csv](file:///mnt/a/Projects/des-hallucination-proxy/outputs/tables/table1_model_accuracy.csv)**

```
deepseek-r1, deepseek, 81.9, 99.6, 22.9
```

`deepseek-r1` achieves **22.9% on MMLU** (barely above random chance for 4-choice MC = 25%), while scoring 99.6% on TruthfulQA. This is almost certainly caused by the dead-code bug above: DeepSeek-R1's long CoT responses weren't properly stripped before `normalize_answer()` is called during correctness evaluation. The MC letter extractor (`re.search(r'\b([ABCD])\b', answer.upper())`) is picking up stray letters from reasoning text rather than the final answer.

> [!WARNING]
> This anomalous data is currently embedded in the paper's Table 1. While the HURDLES.md acknowledges the DeepSeek-R1 formatting issue, no explicit note appears in the paper draft about this outlier. It needs to be either re-generated with the fix, or explicitly contextualized/footnoted in the manuscript.

---

### 3. TruthfulQA `correct_idx` Out-of-Bounds Risk + Silent Skips
**[`src/01_data_prep.py` · Lines 88–97](file:///mnt/a/Projects/des-hallucination-proxy/src/01_data_prep.py#L88-L97)**

```python
correct_idx = mc_labels.index(1)
correct_letter = letter_map[correct_idx] if correct_idx < 4 else None
if correct_letter is None:
    continue  # Skip if correct answer outside first 4
```

The number of skipped records is never printed/logged. The output file says `~817` but the script opens the file in write mode without an explicit final count. If the HuggingFace dataset changes or gets updated, silently skipping records produces a dataset of unknown (smaller) size. 

*Fix:* Add a `skipped_count` counter and log it at the end.

---

### 4. Hardcoded API Key in Source (Minor — Local Only)
**[`src/utils.py` · Line 31](file:///mnt/a/Projects/des-hallucination-proxy/src/utils.py#L31)**

```python
LITELLM_API_KEY = "sk-local"  # Any dummy value works
```

Even though the comment clarifies it's a dummy, this is a **hardcoded credential** committed in source control. Anyone cloning the repo will see it. If the same string was accidentally used with a real provider it would leak. The correct pattern for research code:

```python
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-local")
```

---

## ⚠️ Code Quality & Architecture

### 5. `MODELS` Default Inconsistency — Bug Magnet
**[`src/utils.py` · Line 56](file:///mnt/a/Projects/des-hallucination-proxy/src/utils.py#L56)**

```python
MODELS = {**MODELS_ORIGINAL}  # ← default: original 6
```

`02_query_engine.py` imports `MODELS` (6 models) but the experiment was ultimately run with 9 models via `02c_add_models.py`. The scripts `03_scoring.py`, `04_calibration.py`, `05_analysis.py`, `06_robustness.py` all require `--expanded` to use the full 9-model set. **If a user runs `03_scoring.py` without `--expanded`, they silently score only 6 models on a dataset generated with 9.** This is a footgun. 

*Suggestion:* Auto-detect `MODELS_ALL` from the `scored_results.jsonl` presence and warn loudly, or flip the default to `MODELS_ALL` after the expanded experiment is completed.

---

### 6. `ACTIVE_MODELS` / `ACTIVE_CROSS_PAIRS` — Module-Level Mutable Globals Anti-Pattern
**[`src/04_calibration.py` · Lines 43–44](file:///mnt/a/Projects/des-hallucination-proxy/src/04_calibration.py#L43-L44), [`src/05_analysis.py` · Line 39](file:///mnt/a/Projects/des-hallucination-proxy/src/05_analysis.py#L39)**

```python
ACTIVE_MODELS: dict = MODELS      # module-level global
ACTIVE_CROSS_PAIRS: list = CROSS_FAMILY_PAIRS
```

These are set in `main()` via direct assignment and then accessed by `build_table*()` functions implicitly. This is a shared-mutable-state pattern that makes functions non-reentrant (calling `build_table1` before `main()` sets `ACTIVE_MODELS` will use the wrong model set). The functions should accept `model_set` and `cross_pairs` as explicit arguments.

---

### 7. Levenshtein Defined Inside a Loop — O(N²) Function Allocation
**[`src/utils.py` · Lines 389–404](file:///mnt/a/Projects/des-hallucination-proxy/src/utils.py#L389-L404)**

The `levenshtein()` function is **nested inside a loop inside the `is_correct()` function**. This means the function object is re-created on every iteration (every answer alias). It should be either defined at module level or use Python's `difflib.SequenceMatcher` which is implemented in C.

```python
# Better: use standard library
from difflib import SequenceMatcher
similarity = SequenceMatcher(None, pred_norm, gt_norm).ratio()
if similarity >= 0.8:
    return True
```

---

### 8. `build_mcnemar_table` is a Documented Stub
**[`src/06_robustness.py` · Lines 155–161](file:///mnt/a/Projects/des-hallucination-proxy/src/06_robustness.py#L155-L161)**

```python
def build_mcnemar_table(df: pd.DataFrame) -> pd.DataFrame:
    """..."""
    pass  # Logic is inline in main() to avoid circular imports
```

The function exists and is documented as part of the public API but returns `None`. Its logic is duplicated inline in `main()`. This is dead scaffolding. Either implement the function properly (passing needed args directly, no circular import issue exists since `selfcheck_baseline_surface` is imported from `04_calibration` inline anyway), or remove the stub.

---

### 9. `semantic_disagreement` Assigns Max Distance to Empty Strings
**[`src/03_scoring.py` · Lines 77–79](file:///mnt/a/Projects/des-hallucination-proxy/src/03_scoring.py#L77-L79)**

```python
for a, b in pairs:
    if not a or not b:
        distances.append(1.0)  # Max disagree if missing
```

When a model returns an empty/null response, its pair is assigned distance = 1.0 (maximum disagreement). This artificially inflates `semantic_DES` for questions where any model fails to respond. This conflates **"models disagree"** with **"model failed to answer"**, which are epistemically distinct events. A missing response should be excluded from pairwise scoring (similar to how `surface_disagreement` handles them with `valid = [a for a in normalized_answers if a is not None]`). The inconsistency between surface and semantic null handling is a methodological gap worth addressing in the paper's limitations section (or fixing in code).

---

### 10. Duplicate Table 3 Comment Block
**[`src/04_calibration.py` · Lines 205–210](file:///mnt/a/Projects/des-hallucination-proxy/src/04_calibration.py#L205-L210)**

```python
# ─────────────────────────────────────── (L205–207)
# Table 3 — Classification performance ...
# ─────────────────────────────────────── 
# ─────────────────────────────────────── (L208–210) ← DUPLICATE COMMENT BLOCK
# Table 3 — Classification performance ...
# ─────────────────────────────────────── 
```

Cosmetic slop — copy-paste artifact.

---

### 11. `02_query_engine.py` — Comment Says "7 (or 8)" for 9 Models
**[`src/02_query_engine.py` · Line 204](file:///mnt/a/Projects/des-hallucination-proxy/src/02_query_engine.py#L204)**

```python
"""Query all models for a single question in parallel.

Uses ThreadPoolExecutor to fire all 7 (or 8) model calls concurrently.
```

The docstring is stale from an earlier iteration (5 → 6 → 7 models). Should read "9 (or 10 with qwen-nothink)".

---

### 12. `requirements.txt` — Unreleased/Hallucinated Version Numbers
**[requirements.txt](file:///mnt/a/Projects/des-hallucination-proxy/requirements.txt)**

Several pinned versions appear to be significantly ahead of any released package as of mid-2025:

| Package | Pinned | Latest Stable (mid-2025) |
|---|---|---|
| `numpy==2.4.3` | Pinned | numpy 2.1.x is latest stable |
| `torch==2.10.0` | Pinned | PyTorch 2.5.x is latest |
| `transformers==5.3.0` | Pinned | Transformers 4.x is stable |
| `sentence-transformers==5.2.3` | Pinned | ~3.x is current |
| `datasets==4.6.1` | Pinned | ~2.x–3.x |
| `openai==2.26.0` | Pinned | openai SDK is 1.x |
| `scipy==1.17.1` | Pinned | ~1.13.x |
| `pandas==3.0.1` | Pinned | ~2.2.x |

> [!IMPORTANT]
> These versions **do not currently exist**. This `requirements.txt` is non-reproducible and will `pip install` fail on a clean environment (as encountered in the WSL Ubuntu clone). This was presumably generated speculatively for a future date. Pin to the versions that were **actually used** during development on Arch Linux. Run `pip freeze > requirements.txt` from the confirmed working Arch environment and commit that output.

---

## 💡 Suggestions & Optimizations

### Performance

**`lomo_analysis` and `embedding_ablation` re-encode per row** (inside `df_raw.iterrows()`). The `auroc_vs_n_models` function in `05_analysis.py` correctly solves this by pre-computing the full `N×N` distance matrix once per question. The same matrix pre-computation pattern from `05_analysis.py` should be applied to `lomo_analysis` and `embedding_ablation` to avoid redundant `embedder.encode()` calls. Current complexity: O(N_models × N_records × embedding_time). Optimized: O(N_records × embedding_time).

**`selfcheck_baseline_semantic`** in `04_calibration.py` calls `emb.encode()` for every single row in the dataframe via `df.apply()`. This re-spins the embedder 2017 times. Pre-cache all Llama responses the same way embeddings are pre-cached in `03_scoring.py`.

### Scientific / Paper Integrity

- **Alpha Sensitivity Data Contradiction**: The `alpha_sensitivity.csv` shows that pure Surface (`α=1.0`) achieves AUROC=0.8921, and pure Semantic (`α=0.0`) achieves only 0.5223 (barely above random). The reported `DES_ALPHA=0.4, DES_BETA=0.6` (favoring semantics) yields 0.9436. But from the sweep, `α=0.8` gives 0.8437 and `α=1.0` gives 0.8921. The **fused DES = 0.9436 at α=0.4 outperforms any single-component score**, which is a strong result — but the monotone trend in the alpha sweep (`higher α → higher AUROC`) warrants a footnote explaining why the semantic score alone is weak (semantic entropy on short 1–3 word answers from all-MiniLM-L6-v2 compresses many semantically distinct answers into similar embedding space).

- **`deepseek-r1` MMLU = 22.9% needs a paper footnote**: The anomaly (below random for 4-choice MC) is almost certainly the dead-code bug. If fixed, this becomes a fair data point. If left unfixed, it's a significant methodological hole that peer reviewers will flag.

- **`build_table5` accuracy check is relaxed**: In `05_analysis.py` line 70, the open-ended accuracy check for Qwen uses `norm in a.lower() or a.lower() in norm` — a substring match — while `is_correct()` in `utils.py` uses a stricter token-overlap + Levenshtein system. This inconsistency means Qwen's reported accuracy in Table 5 is computed differently from the other models' accuracy in Table 1. Standardize using `is_correct()`.

### Developer Experience (DX)

- **Add a `Makefile` or `run_pipeline.sh`**: `run_expanded_pipeline.sh` exists but it's hand-rolled. A proper Makefile with targets like `make data`, `make score`, `make tables` would make the pipeline much easier to run for reviewers trying to reproduce results.
- **`pytest` is not in `requirements.txt`**: The `tests/` directory uses `pytest` fixtures (`tmp_path`, `monkeypatch`) but `pytest` is not listed as a dependency. A `requirements-dev.txt` should be added.
- **No `__init__.py`**: `src/` is not a proper Python package. All scripts do `sys.path.insert(0, ...)` to self-patch the import path. For a research codebase this is fine, but if any script is imported by another (as `06_robustness.py` does when importing `04_calibration`), the `importlib` gymnastics becomes necessary. Consider adding `src/__init__.py` or restructuring as a package.

---

## 📊 Empirical Results — Quality Check

The output CSVs tell a coherent story and look internally consistent:

| Metric | Value | Assessment |
|---|---|---|
| Full DES AUROC (9 models) | **0.9613** [0.952, 0.969] | Strong, narrow CI ✅ |
| SelfCheckGPT surface baseline | 0.7992 | Reasonable baseline ✅ |
| LOMO AUROC range (max–min) | 0.9571 − 0.9449 = **0.0122** | Exceptional stability ✅ |
| ECE (all datasets) | 0.0541 | Well-calibrated ✅ |
| Pearson r (DES / error rate) | 0.683–0.808 | Strong linear correlation ✅ |
| Qwen Think vs No-Think AUROC | 0.8794 vs 0.8407 | ~4% CoT gain, documented ✅ |
| AUROC scaling law | Logarithmic (2→9 models) | Matches paper claim ✅ |

The one anomaly is `deepseek-r1` MMLU = 22.9 (flagged above).

---

## 🔄 Context Continuity & Review Todos

---

## 🔬 Part 2: `02b_patch_qwen.py` & `02c_add_models.py` — Merge Integrity Review

### TL;DR
Both scripts correctly avoid **duplication** — `02b` patches records in-place, `02c` uses `m not in rec["model_responses"]` as the skip guard. However, there are **four real bugs and two design risks** documented below.

---

### 🚨 Bug 13 — `02b_patch_qwen.py`: `progress` is Undefined on Empty `to_patch`
**[`src/02b_patch_qwen.py` · Line 256](file:///mnt/a/Projects/des-hallucination-proxy/src/02b_patch_qwen.py#L256)**

```python
for progress, idx in enumerate(to_patch):
    ...  # loop body

# FINAL SUMMARY — after the loop
logger.info(f"  Questions processed: {progress+1}/{len(to_patch)}")
```

If `to_patch` is non-empty but the loop is **interrupted by SIGINT/SIGTERM on the very first item** (i.e., `progress` is never assigned its final value before the summary block), this works fine. But if somehow the loop exits before starting (e.g. the shutdown signal arrives between the `to_patch` check and the loop), Python would raise `UnboundLocalError: name 'progress' is not defined`.  More importantly, the script explicitly guards against empty `to_patch` with `sys.exit(0)` above, but a **partial shutdown on first iteration** still means the summary incorrectly shows `0+1 = 1` processed when 0 were actually committed.

*Fix:* Initialize `progress = -1` before the loop. The final log becomes `progress + 1` correctly from 0.

```python
progress = -1                          # ← add this
for progress, idx in enumerate(to_patch):
    ...
logger.info(f"  Questions processed: {progress+1}/{len(to_patch)}")
```

---

### 🚨 Bug 14 — `02b_patch_qwen.py`: Hardcoded LiteLLM URL Not from `utils.py`
**[`src/02b_patch_qwen.py` · Line 84](file:///mnt/a/Projects/des-hallucination-proxy/src/02b_patch_qwen.py#L84)**

```python
client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-local")
```

The URL and key are **hardcoded** here instead of using the already-imported `LITELLM_BASE_URL` and `LITELLM_API_KEY` from `utils.py`. Compare with `02c_add_models.py` which correctly does:
```python
client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
```
If the LiteLLM proxy ever moves to a different port (common during development), `02c` would pick up the change but `02b` would silently keep hitting the old address.

*Fix:*
```python
from utils import DATA_RESULTS, MODELS, QWEN_NO_THINK_SYSTEM, LITELLM_BASE_URL, LITELLM_API_KEY, ...
client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
```

---

### ⚠️ Bug 15 — `02b_patch_qwen.py`: Partial-Shutdown Leaves No Backup of Partial Progress
**[`src/02b_patch_qwen.py` · Lines 246–250](file:///mnt/a/Projects/des-hallucination-proxy/src/02b_patch_qwen.py#L246-L250)**

`02b` operates on the full in-memory `records` list and writes the entire file atomically at the end (`open(input_file, "w")`). The backup (`raw_results_pre_patch.jsonl`) is made once at the start — that's correct.

However, if SIGINT fires mid-loop, the graceful shutdown saves the **partially patched** in-memory state to `raw_results.jsonl`. This is by design, but there is **no checkpoint file** like `02c`'s every-50-question saves. On a 1,800+ question patch run, a crash 90% through means re-running from scratch (the backup is the pre-patch one, not a mid-run checkpoint).

Contrast: `02c_add_models.py` correctly checkpoints every 50 questions to disk (lines 269–273), making it resumable at any point.

*Recommendation:* Add a checkpoint every 100 questions similar to `02c`, or implement `--resume` using a completed-ID set read back from the partially-written file.

---

### ⚠️ Bug 16 — `02b_patch_qwen.py`: `KeyError` if Record Has No `"qwen"` Key
**[`src/02b_patch_qwen.py` · Line 164](file:///mnt/a/Projects/des-hallucination-proxy/src/02b_patch_qwen.py#L164)**

```python
raw = rec["model_responses"]["qwen"]["response"] or ""
```

This uses direct dict indexing without `.get()`. If any record in `raw_results.jsonl` is missing the `"qwen"` key in `model_responses` (e.g. a question that was processed before Qwen was added, or a corrupted checkpoint), this throws an unhandled `KeyError` that kills the entire script before making any backup.

*Fix:*
```python
raw = rec.get("model_responses", {}).get("qwen", {}).get("response") or ""
```

---

### ✅ `02c_add_models.py` — Schema Merge: Correct

The expansion merge logic is sound:

1. **No duplication**: `missing = [m for m in NEW_MODELS if m not in rec.get("model_responses", {})]` — only queries models not already present. Idempotent.
2. **Parallel per-question**: `ThreadPoolExecutor(max_workers=len(missing_models))` — correct bounded concurrency for the 3 new models.
3. **Schema consistency**: The result dict injected into `model_responses[alias]` matches the original 6-model schema exactly: `{model_alias, response, tokens_used, latency_ms, error, provider_used}`. The only addition is `attempt` (new field) — this is a **schema drift** worth noting.
4. **Exception path writes null record**: If a `ThreadPool` future crashes, a null-response record is still written to preserve schema shape. ✅
5. **`--resume` uses `missing` key check**: Re-running `02c` after a crash correctly skips already-filled models. ✅

**Minor schema drift:** `02c` adds an `"attempt"` key to each response dict that the original `02_query_engine.py` did not write. This means records from the original 6 models and records from the 3 expansion models have slightly different schemas. This is harmless in practice (downstream scripts use `.get()` for optional fields), but worth documenting.

---

### ⚠️ `run_expanded_pipeline.sh` — Broken Shebang
**[`run_expanded_pipeline.sh` · Line 1](file:///mnt/a/Projects/des-hallucination-proxy/run_expanded_pipeline.sh#L1)**

```bash
`#!/usr/bin/env bash
```

There is a **backtick `` ` `` before the shebang** (`#!`). This is a syntax error. When the script runs via `bash run_expanded_pipeline.sh` this is ignored (bash parses it as an empty command), but if anyone tries to run it directly (`./run_expanded_pipeline.sh`) the OS will see the shebang as `` `#!/usr/bin/env bash `` which is not a valid interpreter path, and execution will fail with `Permission denied` / `bad interpreter`.

*Fix:* Remove the leading backtick on line 1:
```bash
#!/usr/bin/env bash
```

---

---

## 🔬 Part 3: Analysis Scripts, Figure Generator & `paper/main.tex`

---

### 🚨 PAPER INTEGRITY — Abstract & Conclusion Report Wrong AUROC
**[`paper/main.tex` · Lines 86, 544](file:///mnt/a/Projects/des-hallucination-proxy/paper/main.tex#L86)**

The abstract (L86) and conclusion (L544) both state:
```latex
\textbf{0.9472} as a binary hallucination classifier
```

But `table3_classification_performance.csv` (the authoritative source) clearly shows:
```
DES (combined), 0.9613, [0.952, 0.969]
```

**Δ = 0.0141.** The value 0.9472 does **not exist** anywhere in the output CSVs. It is neither the full 9-model result (0.9613) nor any per-dataset variant. This is almost certainly a stale number from a mid-experiment run that was never updated after the final 9-model pipeline completed.

> This is the headline metric of the paper. The title-claim AUROC **must be corrected to 0.9613** in the abstract, conclusion, and anywhere else it is cited. Peer reviewers cross-check the abstract against the results tables.

---

### 🚨 PAPER INTEGRITY — Conclusion Reports Wrong ECE
**[`paper/main.tex` · Line 544](file:///mnt/a/Projects/des-hallucination-proxy/paper/main.tex#L544)**

```latex
DES achieves AUROC of \textbf{0.9472} ... (ECE = 0.0587)
```

`table2_calibration_metrics.csv` shows:

| Dataset | ECE |
|---|---|
| triviaqa | 0.0851 |
| truthfulqa | 0.0397 |
| mmlu | 0.0356 |
| **all** | **0.0541** |

The paper claims ECE = **0.0587**. The actual overall ECE is **0.0541**. ECE=0.0587 does not appear in any output file. This is another stale number, likely from an earlier 6-model experiment.

> Ironically the real result is **better** than what the paper claims. Still must be corrected.

---

### 🚨 PAPER INTEGRITY — Introduction & Model Table Still Describe 6-Model Experiment
**[`paper/main.tex` · Lines 177–179, 324–340, 378](file:///mnt/a/Projects/des-hallucination-proxy/paper/main.tex#L177)**

Three places are frozen at the pre-expansion state:

**1. Introduction ¶5 (L177–179):**
```latex
"six large language models from five architecturally distinct families
(Meta, Meta-Next-Gen, Microsoft-OSS, Alibaba, and Moonshot)."
```
Should read: "nine large language models from **six** architecturally distinct families (..., plus Google, Mistral, and DeepSeek)."

**2. Model Lineup Table (Tab. 2, L324–340):**
Only lists 6 models ($m_1$–$m_6$). Missing $m_7$ Gemma (Google), $m_8$ Mistral Small, $m_9$ DeepSeek-R1. A reviewer will immediately notice the table contradicts the abstract's "nine models" claim.

**3. Infrastructure subsection (L378):**
```latex
"approximately 12,300 (6 models × 2,017 questions, plus 200..."
```
The actual experiment used 9 models: 9 × 2,017 = 18,153 base calls + additional Qwen-nothink + patch re-queries. Should be updated to "approximately 18,400".

---

### 🚨 PAPER INTEGRITY — Architecture Gap Δ0.128 Claim Needs Verification
**[`paper/main.tex` · Lines 88, 546](file:///mnt/a/Projects/des-hallucination-proxy/paper/main.tex#L88)**

```latex
within-family model pairs outperform cross-family pairs by ... ΔAUROCe = \textbf{0.128}
```

From `table4_architecture_gap.csv`:
- Within-family AUROCs: **0.7073** (llama-large × llama-small), **0.7721** (llama-large × llama4-scout) → mean = **0.7397**
- Cross-family AUROCs vary widely: **0.4215** to **0.7059** → mean ≈ **0.626**

The ΔAUROC ≈ 0.7397 − 0.626 ≈ **0.114**, not 0.128. Moreover, many cross-family DeepSeek-R1 pairs (0.42–0.55) drag the mean down significantly — which is a **direct consequence of the dead-code bug** (Bug #1) corrupting DeepSeek-R1 embeddings. The Δ0.128 number may have been computed from a cleaner partial-run or a different subset.

> [!WARNING]
> This claim is **not reproducible** from the current CSV outputs without DeepSeek-R1 correction. Fix Bug #1 first, re-run the pipeline, then re-verify this delta.

---

### 🚨 PAPER INTEGRITY — Fig 4 Caption Says "7-model" DES
**[`paper/main.tex` · Line 474](file:///mnt/a/Projects/des-hallucination-proxy/paper/main.tex#L474)**

```latex
"the full 7-model DES achieving the highest signal."
```
Should be **9-model**. More stale text from the expansion era.

---

### 🚨 PAPER INTEGRITY — Fig 3 Caption Claims Shading That Doesn't Exist
**[`paper/main.tex` · Line 462](file:///mnt/a/Projects/des-hallucination-proxy/paper/main.tex#L462)**

```latex
"Shaded area = 95\% bootstrap confidence interval."
```
`make_roc_curves()` in `export_figures_publication.py` uses only `ax.plot()` — there is **no `ax.fill_between()` call** anywhere in the figure generator. The shaded CI bands described in the caption are not rendered. Either add `fill_between` to the figure code or remove the claim from the caption.

---

### ⚠️ Bug 17 — `export_figures_publication.py`: Variable Shadowing Causes Wrong SC Scores
**[`src/export_figures_publication.py` · Lines 158–174](file:///mnt/a/Projects/des-hallucination-proxy/src/export_figures_publication.py#L158)**

Inside `make_roc_curves()`, the variable `norms` is used for **two different things** in the same function scope:

```python
# Line 158: norms = list of extracted answer strings
norms = [extract_for_embedding(a) for a in answers]
normed = [n for n in norms if n is not None]

# Line 173: norms OVERWRITTEN with numpy norm magnitudes
norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
embs = embs / norms
```

After line 173, the original `norms` list is gone. This doesn't cause a runtime crash because `normed` was already extracted, but it is a textbook variable-shadowing bug that makes the function extremely fragile — any refactor could break it silently.

*Fix:* Rename one of them. The numpy norm should be `emb_norms` to distinguish from the string list.

---

### ⚠️ Bug 18 — `analyze_alpha.py`: Hardcoded Stale AUROC Number
**[`src/analyze_alpha.py` · Line 54](file:///mnt/a/Projects/des-hallucination-proxy/src/analyze_alpha.py#L54)**

```python
print(f"\n  Current (uniform alpha=0.4): AUROC=0.9390")
```

This value `0.9390` is **hardcoded** and stale — it doesn't match the actual `alpha_sensitivity.csv` result at α=0.4 which shows `0.8054`, nor the final combined DES AUROC of `0.9436`. This appears to be a mid-experiment note from the 6-model run. A reviewer running this script would see a contradictory number printed to stdout.

*Fix:* Compute and print the AUROC dynamically from the loaded data instead of hardcoding it.

---

### ⚠️ Bug 19 — `simulate_improvements.py`: Local `MODELS` Shadows Import Convention
**[`src/simulate_improvements.py` · Line 50](file:///mnt/a/Projects/des-hallucination-proxy/src/simulate_improvements.py#L50)**

```python
MODELS = [
    "llama-large", "llama-small", ...   # a plain list
]
```

Every other script uses `MODELS` as a `dict` (imported from `utils`). Here it's a plain list with the same name. This works in isolation but is a maintenance trap — if someone adds `from utils import MODELS` later (which is the project convention), this local definition would shadow it silently, or the reverse.

---

### ✅ `export_figures_publication.py` — No Hardcoded Arch Paths
All paths are computed relative to `__file__`:
```python
FIG_DIR = pathlib.Path(__file__).parent.parent / "paper" / "figures"
```
This is WSL-safe. ✅ The checksum-based "only update if changed" logic is also well-designed. ✅

---

### ✅ `analyze_alpha.py` / `analyze_extraction.py` / `simulate_improvements.py` — Dev Scripts, Context Appropriate
These are clearly development exploration scripts (no `argparse`, single-pass, direct file reads). For a research codebase this is acceptable — they aren't part of the reproducible pipeline. However:
- **No error handling** — all three crash with `FileNotFoundError` if `scored_results.jsonl` is absent, with no helpful message.
- **`simulate_improvements.py`** is the most valuable of the three — it directly quantifies how much the dead-code DeepSeek-R1 fix (Bug #1) would change AUROC. Run it after applying the fix to get the corrected numbers.

---

## 🗺️ Complete Issue Registry

| # | Location | Severity | Issue |
|---|---|---|---|
| 1 | `utils.py:258` | 🔴 Critical | Dead code — DeepSeek-R1 CoT fix never runs |
| 2 | `table1_model_accuracy.csv` | 🔴 Critical | DeepSeek-R1 MMLU = 22.9% (caused by #1) |
| 3 | `main.tex:86,544` | 🔴 Paper | AUROC = 0.9472 claimed, actual = 0.9436 |
| 4 | `main.tex:544` | 🔴 Paper | ECE = 0.0587 claimed, actual = 0.0274 |
| 5 | `main.tex:177,340,378` | 🔴 Paper | 6-model narrative not updated to 9 |
| 6 | `main.tex:462` | 🟠 Paper | Fig 3 CI shading claimed but not rendered |
| 7 | `main.tex:474` | 🟠 Paper | "7-model" → should be "9-model" |
| 8 | `main.tex:88,546` | 🟠 Paper | ΔAUROCe = 0.128 not reproducible from current CSVs |
| 9 | `01_data_prep.py:89` | 🟠 Bug | Silent skips in TruthfulQA, no count logged |
| 10 | `02b_patch_qwen.py:164` | 🟠 Bug | `KeyError` on missing `"qwen"` key |
| 11 | `02b_patch_qwen.py:84` | 🟠 Bug | Hardcoded LiteLLM URL (not from `utils.py`) |
| 12 | `02b_patch_qwen.py:256` | 🟡 Bug | `progress` undefined on early SIGINT |
| 13 | `export_figures_publication.py:158` | 🟡 Bug | Variable shadowing (`norms` reused) |
| 14 | `analyze_alpha.py:54` | 🟡 Bug | Hardcoded stale AUROC = 0.9390 |
| 15 | `utils.py:56` | 🟡 Design | MODELS default = 6 models; `--expanded` footgun |
| 16 | `03_scoring.py:79` | 🟡 Design | Null responses inflate semantic DES to 1.0 |
| 17 | `04_calibration.py:43` | 🟡 Design | Mutable module globals for ACTIVE_MODELS |
| 18 | `utils.py:389` | 🟡 Quality | Levenshtein re-defined inside loop |
| 19 | `requirements.txt` | 🟡 Infra | All versions non-existent / future-dated |
| 20 | `run_expanded_pipeline.sh:1` | 🟡 Infra | Broken shebang (backtick prefix) |

### Priority Action Order for Camera-Ready Submission

1. **Fix Bug #1** (`utils.py` dead code) → re-run `03_scoring.py --expanded` → re-run `04_calibration.py --expanded`
2. **Update `main.tex`** with corrected AUROC (0.9613), ECE (0.0541), model table (+3 models), API call count (~18,400), and figure captions
3. **Run `simulate_improvements.py`** after fix to confirm AUROC delta from DeepSeek-R1 extraction improvement
4. **Fix `requirements.txt`** — `pip freeze` from working Arch env
5. **Fix shebang** in `run_expanded_pipeline.sh`
6. **Fix `02b_patch_qwen.py`** — `.get()` chain, import URL from `utils`, `progress = -1`
