# Trabalho1-ODS2

RAG local para perguntas sobre PDFs. Frontend React, backend FastAPI, embeddings via Ollama, base vetorial Chroma e LLM aberto local — tudo executável offline depois do setup inicial.

## Problema e contexto

Durante o período de TCC, a pesquisa bibliográfica consome muito tempo: o aluno acumula uma pilha de artigos para ler e, várias vezes, só percebe que um artigo não tem aderência ao seu tema depois de ler boa parte dele. Tempo é o recurso mais escasso nessa fase.

Esta solução ataca essa dor: o aluno carrega o PDF do artigo, faz perguntas em linguagem natural ("este artigo trata de X?", "qual a metodologia usada?", "ele cita autor Y?") e recebe respostas baseadas no próprio conteúdo do PDF. Assim é possível **filtrar rapidamente artigos irrelevantes** e **localizar trechos específicos** sem precisar ler o documento inteiro.

**Público-alvo:** estudantes em fase de TCC, IC ou revisão bibliográfica.

**Por que local:** os PDFs podem ser materiais ainda não publicados, drafts de orientadores ou artigos com restrição de uso. Rodar 100% offline (Ollama + Chroma + LLM local) evita expor esses documentos a APIs externas.

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

## Decisões técnicas

| Decisão | Por quê |
|---|---|
| **Chroma como banco vetorial** | Persistência local em arquivo, sem servidor externo. Reprodutibilidade garantida (cada execução recria a coleção a partir do PDF). |
| **Ollama para embeddings** | Roda 100% offline. Evita dependência de APIs externas (custo + privacidade). O modelo `nomic-embed-text` é leve (137M params) e tem ótimo desempenho em PT-BR. |
| **Qwen2.5-1.5B-Instruct (HuggingFace) para geração** | LLM aberto/self-hosted (requisito do projeto). Pequeno o suficiente para CPU (1-3min/pergunta) e tem boa aderência ao contexto (Faithfulness 1.0 nos testes). |
| **RAGAS para avaliação** | Padrão de fato para avaliação automática de RAG. Usa LLM-as-judge. Configurado para usar Ollama local (sem custo de API). |
| **FastAPI no backend** | Tipagem com Pydantic, OpenAPI/Swagger automático em `/docs`, fácil integração com o frontend React. |
| **React no frontend** | Stack mais comum no contexto acadêmico, fácil de demonstrar. |
| **`RecursiveCharacterTextSplitter`** | Quebra hierárquica (`\n\n` → `\n` → `.` → ` `): preserva parágrafos e sentenças quando possível, evitando cortar no meio de frases. |

## Base de conhecimento

Etapa de ingestão (`backend/rag/ingestion.py` + `database.py`):

| Parâmetro | Valor | Justificativa |
|---|---|---|
| **Chunk size** | 1000 caracteres | Equilíbrio: grande o suficiente para conter um parágrafo completo (~150-200 palavras), pequeno o suficiente para caber no contexto do LLM gerador junto com a pergunta. |
| **Chunk overlap** | 100 caracteres (10%) | Evita perda de contexto em fronteiras de chunk. Frase quebrada no fim do chunk N reaparece no início do chunk N+1. |
| **Separadores (em ordem)** | `\n\n`, `\n`, `.`, ` `, `""` | Tenta quebrar primeiro em parágrafos, depois linhas, depois sentenças. Só corta no meio de palavra se nada melhor existir. |
| **Modelo de embeddings** | `nomic-embed-text` (Ollama) | 137M params, 768-d vectors, contexto de 8192 tokens. Open-weight, multilíngue, suporte robusto a PT-BR. |
| **Banco vetorial** | ChromaDB (persistente em `db/chroma_db/`) | Cada execução de avaliação cria uma coleção separada com timestamp para isolamento. |
| **Top-k default** | 5 chunks | Validado empiricamente nos experimentos: top-3 perde info, top-10 traz ruído (ver matriz de experimentos). |

**Estatística do PDF de teste** (`artigoteste.pdf`, TCC sobre saúde mental, 34 páginas):
- 58.702 caracteres extraídos
- **66 chunks** gerados após o splitter
- Indexação completa em ~5s (embeddings via Ollama)

## Limitações conhecidas

- A qualidade depende fortemente do PDF e da recuperação dos chunks.
- Documentos longos ou heterogêneos podem trazer trechos pouco relevantes.
- Desempenho depende do hardware local (geração do LLM é o gargalo).
- Avaliação automática deve ser complementada com análise manual de erros.

## Desempenho e tempo de execução

O projeto roda **100% local em CPU** por padrão. Isso afeta os tempos de resposta de duas formas distintas:

### Uso normal (chat / API)

Cada pergunta executa: 1 busca semântica no Chroma + 1 chamada ao LLM gerador.

| Hardware | Tempo por pergunta |
|---|---|
| CPU comum (Intel Core Ultra 5 / Ryzen 5) | ~1-3 minutos |
| CPU + GPU NVIDIA com CUDA configurado no HF | ~5-15 segundos |
| Servidor com GPU dedicada | <5 segundos |

### Avaliação automática (`avaliar.py` com `--use-ragas`)

A avaliação é **muito mais lenta** que o uso normal porque executa **dois LLMs sequencialmente para cada pergunta do dataset**:

1. **LLM gerador** (`Qwen2.5-1.5B-Instruct` via HuggingFace) — gera a resposta a partir do contexto recuperado.
2. **LLM juiz** (Ollama, ex.: `qwen2.5:0.5b`) — o RAGAS chama 4 métricas (`faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`), e cada métrica faz múltiplas chamadas ao juiz.

Para um dataset de 8 perguntas em CPU, isso resulta em **~80-120 chamadas a LLMs** no total, o que leva **25-40 minutos** em um Core Ultra 5.

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

A API expõe 3 endpoints. Documentação interativa Swagger disponível em `http://localhost:8000/docs` quando o backend está rodando.

#### `GET /health`

Verifica se a API está no ar.

```bash
curl http://localhost:8000/health
```

Resposta:
```json
{ "status": "ok" }
```

#### `POST /upload-pdf`

Faz upload de um PDF, indexa no Chroma e retorna um `session_id` para usar nas perguntas.

```bash
curl -X POST http://localhost:8000/upload-pdf \
  -F "file=@backend/pdf/artigoteste.pdf"
```

Resposta:
```json
{
  "session_id": "abc123XYZ_token_seguro",
  "doc_name": "artigoteste.pdf",
  "chunks_count": 66,
  "collection_name": "session-abc123XYZ"
}
```

Erros possíveis:
- `400` — arquivo não é PDF, está vazio ou nome inválido
- `500` — falha ao indexar (verificar se Ollama está rodando)

#### `POST /chat`

Faz uma pergunta sobre o PDF previamente enviado.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc123XYZ_token_seguro",
    "question": "Qual o tema principal do trabalho?",
    "top_k": 5
  }'
```

**Body schema:**
- `session_id` (string, obrigatório, mín. 8 chars) — token retornado por `/upload-pdf`
- `question` (string, obrigatório, mín. 2 chars) — pergunta em linguagem natural
- `top_k` (int, opcional, default `5`, faixa `1-15`) — quantos chunks recuperar

Resposta:
```json
{
  "answer": "O tema principal é a saúde mental como desafio para a saúde pública, com foco no Programa Saúde da Família.",
  "context": "...trechos do PDF que embasaram a resposta..."
}
```

Erros possíveis:
- `404` — `session_id` não encontrado (PDF precisa ser reenviado)
- `500` — falha na geração da resposta

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

### Análise de casos de falha

Após rodar o experimento com 4 perguntas (`results/exp01_qwen05b_topk5.csv`), inspecionamos manualmente cada resposta para entender onde o pipeline diverge do gabarito. Os padrões observados:

**Caso 1 — Resposta correta porém mais detalhada que o gabarito (Q1, Q3):**

| Campo | Conteúdo |
|---|---|
| Pergunta | Qual o tema principal do trabalho? |
| Gabarito | Saúde mental como desafio para a saúde pública, com foco no Programa Saúde da Família. |
| Predito | "O tema principal do trabalho é o cenário da saúde mental no Programa Saúde da Família (PSF) - Areias I - situado no bairro Areias no município de Ribeirão das Neves - MG." |
| F1 | 0.22 ⚠️ |
| Faithfulness | 1.00 ✅ |

**Diagnóstico:** o LLM gerador acerta o conteúdo mas adiciona contexto extra (município, bairro, ESF). O F1 baseado em token overlap penaliza isso. **Não é uma falha real** — o gabarito é que está conciso demais para ser ideal.

**Caso 2 — Resposta correta mas excessivamente sucinta (Q2):**

| Campo | Conteúdo |
|---|---|
| Pergunta | Em qual município e bairro foi realizado o estudo? |
| Gabarito | No bairro Areias, no município de Ribeirão das Neves, em Minas Gerais. |
| Predito | "Ribeirão das Neves - MG" |
| F1 | 0.27 ⚠️ |

**Diagnóstico:** o LLM gerador resumiu demais e omitiu "bairro Areias". Falha real do prompt de geração — pode ser corrigida instruindo o modelo a responder de forma completa.

**Caso 3 — Juiz pequeno (qwen 0.5b) penaliza injustamente (Q1, Q2, Q3):**

Em 3 de 4 perguntas, `RAGAS Answer Relevancy = 0.00` mesmo quando a resposta está correta e Faithfulness = 1.00. Comparando com o experimento exp02 (mesmo dataset, juiz qwen 1.5b), Answer Relevancy sobe para 0.58 — confirmando que o juiz pequeno é severo demais e gera **falsos negativos**.

**Recomendação:** usar `qwen2.5:1.5b` ou maior como juiz para Answer Relevancy. Métricas baseadas em embeddings (Context Precision/Recall) são robustas mesmo com juiz pequeno.

**Conclusões da análise:**

1. **Pipeline RAG está sólido** — Faithfulness 1.0 + Context Precision/Recall ~1.0 mostra que o retrieval acha o contexto certo e o LLM não inventa.
2. **Métricas de overlap (F1) subestimam a qualidade** — penalizam respostas mais detalhadas mesmo corretas.
3. **Juiz menor que 1B não é confiável** para Answer Relevancy — produz falsos negativos.
4. **Prompt de geração pode ser refinado** para evitar respostas excessivamente curtas (Caso 2).

## Experimentos

Para comparar modelos juiz e configurações de retrieval, há scripts pré-prontos em `scripts/experimentos/`. Cada experimento usa o mesmo `avaliar.py` com flags diferentes e gera um CSV em `results/`.

> Os experimentos usam um dataset reduzido (`dataset_rapido.json`, 4 perguntas) para que cada execução termine em tempo razoável em CPU. Para avaliação final, use `dataset_exemplo.json` (8 perguntas).

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

### Resultados — comparação de modelos juiz (dataset 4 perguntas)

Todos com top-k 5, mesmo dataset (`dataset_rapido.json`). Métricas clássicas idênticas entre experimentos (geração e retrieval são determinísticos e independem do juiz).

| Métrica | exp01 (qwen 0.5b) | exp02 (qwen 1.5b) | exp03 (llama3.2 3.2B) |
|---|---|---|---|
| **Duração** | 17min | 23min | **38min** |
| Answer F1 | 0.39 | 0.39 | 0.39 |
| Retrieval F1 | 0.47 | 0.47 | 0.47 |
| Retrieval MRR | 0.63 | 0.63 | 0.63 |
| RAGAS Faithfulness | 1.00 ✅ | 1.00 ✅ | **0.81** ⚠️ |
| **RAGAS Answer Relevancy** | 0.15 ⚠️ | **0.58** ↑ | **0.60** ↑ |
| RAGAS Context Precision | 1.00 ✅ | 0.95 | 0.99 |
| RAGAS Context Recall | 1.00 ✅ | 1.00 ✅ | 1.00 ✅ |

**Insights da comparação:**

- **Métricas clássicas são iguais entre experimentos** — não dependem do juiz, apenas do gerador (Qwen2.5-1.5B HF) que é o mesmo em todos.
- **Answer Relevancy precisa de juiz ≥ 1.5B** — qwen 0.5b dá falsos negativos (0.15). Tanto qwen 1.5b (0.58) quanto llama3.2 (0.60) convergem para um valor similar e mais confiável.
- **Faithfulness inesperadamente cai com llama3.2** (0.81 vs 1.00 nos qwen) — investigando os logs, vimos `OUTPUT_PARSING_FAILURE` ocasional no llama3.2: o modelo às vezes retorna JSON malformado para o RAGAS, e métricas com parse falho viram 0. **Conclusão**: modelos maiores nem sempre são melhores juízes — qwen 2.5 segue melhor o schema esperado pelo RAGAS.
- **Context Precision/Recall são robustas com qualquer juiz** ≥ 0.5B.
- **Trade-off velocidade vs qualidade**:
  - qwen 0.5b: 17min, mas Answer Relevancy não é confiável
  - qwen 1.5b: 23min, **melhor custo-benefício** — métricas confiáveis, sem parsing failures
  - llama3.2 3.2B: 38min, mais lento e ainda com parsing issues — não compensa neste setup

**Recomendação prática**: usar **qwen2.5:1.5b** como juiz padrão para avaliações. Reservar qwen2.5:0.5b apenas para tuning rápido onde só importa Faithfulness/Context.

### Adicionar novo experimento

1. Copie um script existente em `scripts/experimentos/`.
2. Renomeie seguindo o padrão `expNN_<modelo>_<config>.sh`.
3. Ajuste `--ragas-llm-model`, `--top-k`, `--ragas-timeout`, etc.
4. Mude `--output` para `results/<nome>.csv`.
5. Adicione a entrada na matriz acima e em `rodar_todos.sh`.
