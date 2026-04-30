#!/bin/bash
# Experimento 02: qwen2.5:1.5b com top-k 5 (juiz mais robusto)
.venv/bin/python avaliar.py \
  --pdf backend/pdf/artigoteste.pdf \
  --dataset backend/evaluation/dataset_exemplo.json \
  --output results/exp02_qwen15b_topk5.csv \
  --top-k 5 \
  --use-ragas \
  --ragas-llm-model qwen2.5:1.5b \
  --ragas-embedding-model nomic-embed-text:latest \
  --ragas-timeout 600 \
  --ragas-max-workers 1 \
  --ragas-max-retries 2
