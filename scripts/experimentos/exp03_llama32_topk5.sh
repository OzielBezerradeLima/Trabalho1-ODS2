#!/bin/bash
# Experimento 03: llama3.2:latest (3.2B) com top-k 5
.venv/bin/python avaliar.py \
  --pdf backend/pdf/artigoteste.pdf \
  --dataset backend/evaluation/dataset_rapido.json \
  --output results/exp03_llama32_topk5.csv \
  --top-k 5 \
  --use-ragas \
  --ragas-llm-model llama3.2:latest \
  --ragas-embedding-model nomic-embed-text:latest \
  --ragas-timeout 900 \
  --ragas-max-workers 1 \
  --ragas-max-retries 2
