#!/bin/bash
# Experimento 06: baseline sem RAGAS (apenas metricas classicas, rapido)
.venv/bin/python avaliar.py \
  --pdf backend/pdf/artigoteste.pdf \
  --dataset backend/evaluation/dataset_exemplo.json \
  --output results/exp06_sem_ragas.csv \
  --top-k 5
