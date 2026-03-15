# Development Journal: Hurdles & Historical Context

This document captures the engineering challenges and technical hurdles encountered during the development and execution of the Disagreement Entropy (DES) study, and how they were overcome.

## 🚧 Key Technical Hurdles

### 1. Chain-of-Thought (CoT) Tag Contamination
**Challenge**: Advanced reasoning models (like Qwen3 and DeepSeek-R1) often returned their internal "thoughts" wrapped in `<think>...</think>` tags. Initially, this caused the surface-level and semantic entropy scores to explode, as the "thinking" process varies wildly even if the final answer is the same.
**Solution**: Implementation of the `strip_thinking_tags` regex utility in `src/utils.py`. We also conditionally re-queried these models with expanded `max_tokens` and specific "concise" instructions to ensure the final answer remained recoverable.

### 2. Semantic Cluster "Compute Explosion"
**Challenge**: Calculating AUROC curves by comparing every possible subset size ($k$) of the 9-model ensemble previously required re-encoding and re-clustering text for every permutation. This led to $O(2^n)$ compute costs that crashed the analysis scripts.
**Solution**: We shifted to a **Per-Question Matrix Pre-computation** strategy. By pre-calculating the $N \times N$ distance matrix for each question once, subset analysis became a simple index lookup, reducing runtime from hours to minutes.

### 3. API Reliability & Rate Limiting
**Challenge**: Executing 18,000+ calls across 9 models from diverse providers (Groq, Scaleway, etc.) led to frequent rate limits and temporary endpoint failures.
**Solution**: Leveraged a **Local LiteLLM Rotation Engine** with 11 API keys. The system implemented a robust checkpointing mechanism (every 50 queries) in `src/02_query_engine.py` to allow seamless resumes after crashes.

### 4. Semantic Similarity Drift
**Challenge**: The default `all-MiniLM-L6-v2` embedder sometimes struggled with highly technical domain-specific terms in law or biology, occasionally marking different terms as similar or vice versa.
**Solution**: We performed an **Embedding Ablation Study** (now in `src/06_robustness.py`), proving that while specific scores shifted, the *relative efficacy* of the DES proxy remained consistent regardless of whether we used a 384-dim or 1024-dim instruction-tuned embedder.

---

## 📜 Historical Paradigms

### From 5-Model to 9-Model Ensemble
Initially, the project was scoped for a 5-model ensemble. However, early results showed that while AUROC was good (~0.83), the "Architecture Gap" was not sufficiently wide to isolate shared pre-training biases. We later expanded to 7, and finally 9 models, across 6 architectural families. This expansion was critical for the primary discovery: **Diversity > Density**.

### The "Legacy" Cleanup
The `paper/figures/legacy/` directory originally contained early proofs for LOMO (Leave-One-Model-Out) impact and sensitivity sweeps. These findings were so critical that they were prioritized during the final visualization phase, leading to the high-fidelity PDF assets now used in the publication.
