# Empirical Results & Analysis Summary

This document synthesizes the core scientific findings and proofs from the DES study.

## 📊 Performance Proofs

### Result 1: Classification Superiority
The fused ensemble (Surface + Semantic) achieved an **AUROC of 0.9613**, outperforming the standard SelfCheckGPT baseline.
- **Standard SelfCheck**: ~0.86 AUROC
- **Fused DES**: **~0.96 AUROC**

### Result 2: The Architecture Gap
We analyzed "Within-Family" vs "Cross-Family" disagreement.
- **Within-Family**: Models from the same family (e.g., Llama 3.1 & Llama 3.3) tend to share hallucinations, leading to low entropy and high False Negatives.
- **Cross-Family**: Pitting models with different training data (e.g., Llama vs Qwen) breaks this shared bias, resulting in much sharper hallucination detection.

### Result 3: Reasoning (CoT) Impact
Ablating the "thinking" process of Qwen3 revealed that internal reasoning acts as a truth serum:
- **No-Think Mode**: 0.8407 AUROC
- **Think Mode**: **0.8794 AUROC**

---

## 📈 Scaling Laws
The study found that the benefit of adding models to the ensemble is **logarithmic**.
- **1-3 Models**: Rapid surge in detection accuracy.
- **5-9 Models**: Diminishing returns, suggesting 3-5 diverse models is the "sweet spot" for industrial deployment.

---

## 📝 SWOT Analysis

| Strengths (S) | Weaknesses (W) |
|---|---|
| **Zero-Cost**: No logprobs needed. | **Shared Bias**: Catastrophic failure if all models share the same hallucination. |
| **Architectural Rigor**: Actively separates family-biases. | **Embedding Latency**: CPU-clustering can be slow for massive datasets. |

| Opportunities (O) | Threats (T) |
|---|---|
| **Online Guardrails**: Real-time hallucination scoring. | **Ensemble Homogenization**: Synthetic data loops might converge model behaviors. |
| **RAG Hybridization**: Using DES to trigger RAG lookups. | **API Volatility**: Upstream changes to model output formats. |
