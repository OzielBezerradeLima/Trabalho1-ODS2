#!/bin/bash
# Roda todos os experimentos sequencialmente e registra o tempo de cada um.
# Logs sao salvos em results/<exp>.log

set -u
mkdir -p results

experimentos=(
  "scripts/experimentos/exp01_qwen05b_topk5.sh"
  "scripts/experimentos/exp02_qwen15b_topk5.sh"
  "scripts/experimentos/exp03_llama32_topk5.sh"
  "scripts/experimentos/exp04_qwen05b_topk3.sh"
  "scripts/experimentos/exp05_qwen05b_topk10.sh"
  "scripts/experimentos/exp06_qwen05b_sem_ragas.sh"
)

for exp in "${experimentos[@]}"; do
  nome=$(basename "$exp" .sh)
  echo "=========================================="
  echo "Rodando $nome ..."
  echo "=========================================="
  inicio=$(date +%s)
  bash "$exp" 2>&1 | tee "results/${nome}.log"
  fim=$(date +%s)
  echo "$nome levou $((fim - inicio)) segundos" | tee -a "results/${nome}.log"
done

echo "Todos os experimentos finalizados. CSVs e logs em results/"
