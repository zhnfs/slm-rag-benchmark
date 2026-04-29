# SLM RAG Benchmark: Fine-Tuned SLM vs. Frontier Cloud Model

## Objective
Prove that a fine-tuned Small Language Model (Qwen2.5-0.5B-Instruct) using RAG
achieves equivalent accuracy to a frontier cloud model (Gemini) with **10x lower
latency** and **zero data exfiltration risk**.

## Domain
Oil & Gas Petroleum Engineering — 21 reference documents covering drilling,
completion, production, and reservoir topics.

## Methodology
1. **Synthetic QA Generation** — Use Gemini to generate 300 context-grounded QA
   pairs from the domain corpus.
2. **Fine-Tuning** — Behaviorally tune Qwen2.5-0.5B-Instruct via LoRA using
   LLaMA-Factory to strictly extract from provided context.
3. **Head-to-Head Benchmark** — Run 50 test queries through both the local SLM
   and Gemini API, measuring latency, accuracy, and cost.

## Quick Start
```bash
# 1. Install dependencies
uv sync

# 2. Generate training data
uv run python scripts/generate_rag_dataset.py

# 3. Fine-tune (uses LLaMA-Factory from finetune_slm_llamaFactory venv)
# See configs/rag_finetune.yaml

# 4. Run benchmark
uv run python scripts/benchmark_slm_vs_cloud.py
```
