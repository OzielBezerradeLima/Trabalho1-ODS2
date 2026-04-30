#!/bin/bash
# Experimento 01: qwen2.5:0.5b com top-k 5 (baseline rapido)
.venv/bin/python avaliar.py \
  --pdf backend/pdf/artigoteste.pdf \
  --dataset backend/evaluation/dataset_exemplo.json \
  --output results/exp01_qwen05b_topk5.csv \
  --top-k 5 \
  --use-ragas \
  --ragas-llm-model qwen2.5:0.5b \
  --ragas-embedding-model nomic-embed-text:latest \
  --ragas-timeout 300 \
  --ragas-max-workers 1 \
  --ragas-max-retries 2
