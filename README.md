# Trabalho1-ODS2

RAG local para perguntas sobre PDFs. Frontend React, backend FastAPI, embeddings via Ollama, base vetorial Chroma e LLM aberto local — tudo executável offline depois do setup inicial.

## Como funciona

1. O usuário envia um PDF pela interface ou pela API.
2. O backend extrai o texto, faz chunking e gera embeddings via Ollama (`nomic-embed-text`).
3. Os chunks são indexados no Chroma (banco vetorial local).
4. Na pergunta, o sistema recupera os trechos semanticamente mais próximos.
5. O contexto recuperado é enviado para um LLM local (via Hugging Face) que gera a resposta.

## Estrutura

| Componente | Local | Tecnologia |
|---|---|---|
| Frontend | `frontend/` | React |
| API | `backend/main.py` | FastAPI |
| Pipeline RAG | `backend/rag/`, `backend/llm/`, `backend/pdf/` | Chroma + Ollama + HF |
| Avaliação | `backend/evaluation/` | métricas próprias + RAGAS |

## Limitações conhecidas

- A qualidade depende fortemente do PDF e da recuperação dos chunks.
- Documentos longos ou heterogêneos podem trazer trechos pouco relevantes.
- Desempenho depende do hardware local (geração do LLM é o gargalo).
- Avaliação automática deve ser complementada com análise manual de erros.

## Requisitos

- Python 3.11+
- Node.js (para o frontend)
- Ollama instalado e rodando
- **Windows**: usar PowerShell para Python/Node, e Git Bash ou WSL para rodar os scripts `.sh` dos experimentos

## Setup

### 1. Ambiente virtual e dependências

**Linux / macOS:**

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 2. Modelos do Ollama

Mínimo para rodar a aplicação (mesmo comando nos dois sistemas):

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

Para os experimentos (opcional), adicione:

```bash
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:1.5b
```

Conferir o que está instalado: `ollama list`

## Rodar

### Backend

**Linux / macOS:**

```bash
.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

**Linux / macOS:**

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

**Windows (PowerShell):**

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

### Endpoints

- `GET /health`
- `POST /upload-pdf`
- `POST /chat`

## Avaliação

O script `avaliar.py` indexa um PDF, roda perguntas de um dataset JSON e gera um CSV com métricas.

Métricas básicas (rápido, sem RAGAS):

**Linux / macOS:**

```bash
.venv/bin/python avaliar.py --pdf backend/pdf/artigoteste.pdf --dataset backend/evaluation/dataset_exemplo.json --output results.csv --top-k 5
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe avaliar.py --pdf backend\pdf\artigoteste.pdf --dataset backend\evaluation\dataset_exemplo.json --output results.csv --top-k 5
```

Com RAGAS local (juiz via Ollama):

**Linux / macOS:**

```bash
.venv/bin/python avaliar.py --pdf backend/pdf/artigoteste.pdf --dataset backend/evaluation/dataset_exemplo.json --output results.csv --top-k 5 --use-ragas --ragas-llm-model llama3.2:latest --ragas-timeout 600
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe avaliar.py --pdf backend\pdf\artigoteste.pdf --dataset backend\evaluation\dataset_exemplo.json --output results.csv --top-k 5 --use-ragas --ragas-llm-model llama3.2:latest --ragas-timeout 600
```

Se o juiz local estiver lento ou indisponível, as métricas RAGAS aparecem como `indisponivel` — o CSV básico ainda é gerado.

### Flags úteis

- `--top-k` — chunks recuperados por pergunta; afeta precision/recall do retrieval.
- `--ragas-llm-model` — modelo Ollama usado como juiz; precisa existir em `ollama list`.
- `--ragas-timeout` — segundos por operação; modelos maiores precisam de mais.
- `--ragas-max-workers` (default `1`) — manter `1` em CPU/Ollama local; aumentar só com GPU.
- `--ragas-max-retries` (default `2`) — tentativas antes de marcar como `indisponivel`.

## Experimentos

Para comparar modelos juiz e configurações de retrieval, há scripts pré-prontos em `scripts/experimentos/`. Cada experimento usa o mesmo `avaliar.py` com flags diferentes e gera um CSV em `results/`.

**Linux / macOS** — rodar um experimento:

```bash
bash scripts/experimentos/exp01_qwen05b_topk5.sh
```

**Windows** — usar Git Bash ou WSL (os scripts são `.sh`):

```bash
# No Git Bash:
bash scripts/experimentos/exp01_qwen05b_topk5.sh
```

Rodar todos sequencialmente (com cronometragem):

```bash
bash scripts/experimentos/rodar_todos.sh
```

Cada execução gera `results/<exp>.csv` (métricas) e `results/<exp>.log` (saída + tempo).

> **Windows sem Git Bash/WSL?** Você pode chamar o comando do experimento diretamente em PowerShell. Por exemplo, o conteúdo de `exp01_qwen05b_topk5.sh` é só uma chamada para `avaliar.py` — basta traduzir o caminho da venv (`.\.venv\Scripts\python.exe`).

### Matriz de experimentos

| Script | Modelo Juiz | top-k | Timeout | Objetivo |
|---|---|---|---|---|
| `exp01_qwen05b_topk5.sh` | qwen2.5:0.5b | 5 | 300s | Baseline rápido |
| `exp02_qwen15b_topk5.sh` | qwen2.5:1.5b | 5 | 600s | Juiz mais robusto |
| `exp03_llama32_topk5.sh` | llama3.2:latest | 5 | 900s | Baseline 3.2B |
| `exp04_qwen05b_topk3.sh` | qwen2.5:0.5b | 3 | 300s | Menos contexto |
| `exp05_qwen05b_topk10.sh` | qwen2.5:0.5b | 10 | 300s | Mais contexto |
| `exp06_qwen05b_sem_ragas.sh` | — | 5 | — | Apenas métricas clássicas |

### Adicionar novo experimento

1. Copie um script existente em `scripts/experimentos/`.
2. Renomeie seguindo o padrão `expNN_<modelo>_<config>.sh`.
3. Ajuste `--ragas-llm-model`, `--top-k`, `--ragas-timeout`, etc.
4. Mude `--output` para `results/<nome>.csv`.
5. Adicione a entrada na matriz acima e em `rodar_todos.sh`.
