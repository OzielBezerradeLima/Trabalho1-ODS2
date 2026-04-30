#!/bin/bash
# Experimento 05: qwen2.5:0.5b com top-k 10 (mais contexto)
.venv/bin/python avaliar.py \
  --pdf backend/pdf/artigoteste.pdf \
  --dataset backend/evaluation/dataset_exemplo.json \
  --output results/exp05_qwen05b_topk10.csv \
  --top-k 10 \
  --use-ragas \
  --ragas-llm-model qwen2.5:0.5b \
  --ragas-embedding-model nomic-embed-text:latest \
  --ragas-timeout 300 \
  --ragas-max-workers 1 \
  --ragas-max-retries 2
