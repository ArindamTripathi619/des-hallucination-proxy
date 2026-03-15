# Model Ensemble Details

To maximize the architectural "Architecture Gap," we curated a diverse ensemble of **9 models across 6 architectural families**. 

## 🧱 The Ensemble

| Alias | Family | Base Architecture | Key Role |
|---|---|---|---|
| `llama-3.3-70b` | Meta | Llama-3 (Standard) | Anchor model, high factual recall. |
| `llama-4-scout` | Meta | Llama-Next (Internal) | Next-gen architectural diversity. |
| `gpt-oss-120b` | Microsoft | OSS (MoE) | Massive-scale sparse representation. |
| `qwen-3-32b` | Alibaba | Qwen (Reasoning) | Primary Reasoning / CoT focus. |
| `kimi-k2` | Moonshot | Moonshot | Geographically distinct training data. |
| `gemma-3-27b` | Google | Gemma (Hybrid) | Distinct RLHF and alignment layer. |
| `deepseek-r1` | Mistral | DeepSeek (MoE/Think) | SOTA reasoning and structural divergence. |
| `mistral-small` | Mistral | Mistral | Efficiency-focused control model. |
| `llama-3.1-8b` | Meta | Llama-3 | Small-scale architectural baseline. |

## 📐 Selection Logic
1. **Diversity Over Density**: We prioritized models from different companies (Microsoft, Google, Meta, Alibaba) to ensure they do not share the same internet pre-training snapshots.
2. **Structural Variance**: The ensemble mixes Dense models and Mixture-of-Expert (MoE) models to test if disagreement persists across compute paradigms.
3. **Reasoning mix**: We specifically included Reasoning models (DeepSeek-R1, Qwen3) to evaluate if internal scratchpads enhance the consensus signal.
