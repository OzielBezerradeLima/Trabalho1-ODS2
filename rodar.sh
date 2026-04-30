#!/bin/bash
.venv/bin/python avaliar.py --pdf backend/pdf/artigoteste.pdf --dataset backend/evaluation/dataset_exemplo.json --output results.csv --top-k 5 --use-ragas --ragas-llm-model qwen2.5:0.5b --ragas-embedding-model nomic-embed-text:latest --ragas-timeout 300
