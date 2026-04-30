# Trabalho1-ODS2 - Frontend React + Backend FastAPI

## Arquitetura

- Frontend: React (pasta `frontend/`)
- Backend: FastAPI (`backend/main.py`)
- RAG: pipeline Python existente em `rag/`, `llm/`, `pdf/`

## Visao geral do fluxo

1. O usuario envia um PDF pela interface web ou pela API.
2. O backend extrai o texto do documento e faz o chunking em blocos menores.
3. Os chunks sao transformados em embeddings com Ollama e armazenados no banco vetorial Chroma.
4. Na pergunta do usuario, o sistema recupera os trechos semanticamente mais proximos.
5. O contexto recuperado e enviado para um LLM local, que gera a resposta final.

## Decisoes tecnicas

- Uso de Chroma como banco vetorial local para simplificar persistencia e reproducibilidade.
- Uso de Ollama para embeddings, evitando dependencia obrigatoria de APIs externas.
- Uso de um LLM local via Hugging Face para manter o projeto executavel offline depois da configuracao inicial.
- Exposicao de uma API FastAPI simples para facilitar integracao com o frontend e com testes.

## Limitacoes conhecidas

- A qualidade da resposta depende fortemente da qualidade do PDF e da recuperação dos chunks.
- Em documentos longos ou muito heterogêneos, o chunking pode recuperar trechos pouco relevantes.
- O desempenho depende do hardware local, principalmente na etapa de geracao do LLM.
- A avaliacao automatica e util para comparação, mas ainda deve ser complementada com análise manual de casos de erro.

## 1) Criar ambiente virtual e instalar dependencias

No Windows (PowerShell), execute os comandos abaixo na raiz do projeto:

```powershell
cd C:\Users\User\Documents\Projetos\Trabalho1-ODS2
```

Criar o ambiente virtual (`.venv`):

```powershell
py -3.11 -m venv .venv
```

Ativar o ambiente virtual:

```powershell
.\.venv\Scripts\Activate.ps1
```

Se aparecer erro de politica de execucao no PowerShell, rode uma vez e tente ativar novamente:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Atualizar o `pip` e instalar os pacotes do projeto:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Opcional: validar se o ambiente foi ativado e os pacotes foram instalados:

```powershell
python --version
pip list
```

## 2) Iniciar backend da API

```powershell
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

No Linux ou macOS, usando a venv do projeto:

```bash
.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

## 3) Garantir Ollama e embeddings

```powershell
ollama pull nomic-embed-text
```

## 4) Rodar frontend React

```powershell
cd frontend
copy .env.example .env
npm install
npm run dev
```

RAG local para perguntas sobre PDFs, com frontend React, backend FastAPI, embeddings via Ollama, base vetorial Chroma e resposta final gerada por LLM aberto local.

## O que faz

1. Recebe um PDF pela interface ou pela API.
2. Extrai o texto, faz chunking e gera embeddings com Ollama.
3. Indexa os trechos no Chroma.
4. Recupera os trechos mais próximos para a pergunta.
5. Gera a resposta com um LLM local via Hugging Face.

## Stack

- Frontend: React em `frontend/`
- API: FastAPI em `backend/main.py`
- RAG: `backend/rag/`, `backend/llm/`, `backend/pdf/`
- Avaliação: métricas próprias e RAGAS local com Ollama

## Requisitos

- Python 3.11+
- Ollama instalado
- Modelos locais:
	- `nomic-embed-text`
	- `llama3.2`

Instalação dos modelos:

```bash
ollama pull nomic-embed-text
# Trabalho1-ODS2

RAG local para perguntas sobre PDFs, com frontend React, backend FastAPI, embeddings via Ollama, base vetorial Chroma e LLM aberto local.

## O que faz

1. Recebe um PDF pela interface ou pela API.
2. Extrai o texto, faz chunking e gera embeddings com Ollama.
3. Indexa os trechos no Chroma.
4. Recupera os trechos mais próximos para a pergunta.
5. Gera a resposta com um LLM local via Hugging Face.

## Stack

- Frontend: React em `frontend/`
- API: FastAPI em `backend/main.py`
- RAG: `backend/rag/`, `backend/llm/`, `backend/pdf/`
- Avaliação: métricas próprias e RAGAS local com Ollama

## Requisitos

- Python 3.11+
- Ollama instalado
- Modelos locais:
	- `nomic-embed-text`
	- `llama3.2`

Baixe os modelos:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

## Rodar

Backend:

```bash
.venv/bin/uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

## Avaliação

Métricas básicas:

```bash
.venv/bin/python avaliar.py --pdf backend/pdf/artigoteste.pdf --dataset backend/evaluation/dataset_exemplo.json --output results.csv --top-k 5
```

Com RAGAS local:

```bash
.venv/bin/python avaliar.py --pdf backend/pdf/artigoteste.pdf --dataset backend/evaluation/dataset_exemplo.json --output results.csv --top-k 5 --use-ragas --ragas-llm-model llama3.2:latest --ragas-timeout 600
```

Se você usar outro nome de saída, ajuste `--output`. Se passar apenas `results.csv`, o arquivo será criado na raiz do projeto.

O RAGAS usa lista de contextos por pergunta. Se o juiz local estiver lento ou indisponível, as métricas RAGAS podem aparecer como indisponíveis, mas o CSV básico ainda é gerado.

## Experimentos

Para comparar diferentes modelos juiz e configurações de retrieval, há scripts pré-prontos em `scripts/experimentos/`. Cada experimento gera um CSV separado em `results/`.

### Preparação

1. Instalar os modelos no Ollama (uma vez só):

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
ollama pull qwen2.5:1.5b
ollama pull llama3.2
```

2. Verificar modelos disponíveis:

```bash
ollama list
```

3. Garantir que o Ollama está rodando:

```bash
curl -s http://localhost:11434/api/tags
```

### Rodar um experimento isolado

```bash
bash scripts/experimentos/exp01_qwen05b_topk5.sh
```

### Rodar todos sequencialmente

```bash
bash scripts/experimentos/rodar_todos.sh
```

Cada experimento gera:
- `results/<exp>.csv` — métricas por pergunta
- `results/<exp>.log` — saída completa + tempo de execução

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

1. Copie um script existente em `scripts/experimentos/`
2. Renomeie seguindo o padrão `expNN_<modelo>_<config>.sh`
3. Ajuste `--ragas-llm-model`, `--top-k`, `--ragas-timeout`, etc.
4. Mude `--output` para `results/<nome>.csv`
5. Adicione a entrada na matriz acima e em `rodar_todos.sh`

### Flags úteis para tuning

- `--ragas-max-workers` (default `1`) — manter `1` em CPU/Ollama local; aumentar só com GPU
- `--ragas-max-retries` (default `2`) — quantas tentativas antes de marcar como `indisponivel`
- `--ragas-timeout` — segundos por operação; modelos maiores precisam de mais
- `--top-k` — chunks recuperados por pergunta; afeta precision/recall do retrieval

## API

- `GET /health`
- `POST /upload-pdf`
- `POST /chat`

## Observações

- O sistema usa LLM aberto/self-hosted para cumprir o requisito do projeto.
- O timeout padrão do RAGAS foi aumentado para 600 segundos por operação.
- O modelo juiz deve existir no Ollama; o nome em uso pode ser conferido com `ollama list`.
