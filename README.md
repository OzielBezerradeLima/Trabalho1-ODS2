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

Frontend em `http://127.0.0.1:5173`.

## 5) Build do frontend

```powershell
cd frontend
npm run build
```

## Estrutura recomendada

- `frontend/`: interface React principal
- `backend/main.py`: backend FastAPI
- `rag/`: pipeline de ingestao e recuperacao

## Endpoints principais

- `GET /health`
- `POST /upload-pdf`
- `POST /chat`

## Testes

```powershell
pytest -q evaluation/test_metrics.py
```

No Linux ou macOS, usando a venv do projeto:

```bash
.venv/bin/pytest -q backend/evaluation/test_metrics.py
```

## Avaliacao com metricas

Para executar a avaliacao do RAG com calculo de metricas de resposta e recuperacao, use o atalho da raiz do projeto:

```powershell
python avaliar.py --pdf caminho/do/documento.pdf --dataset backend/evaluation/dataset_exemplo.json --output backend/evaluation/results.csv --top-k 5
```

No Linux ou macOS, usando a venv do projeto:

```bash
.venv/bin/python avaliar.py --pdf caminho/do/documento.pdf --dataset backend/evaluation/dataset_exemplo.json --output backend/evaluation/results.csv --top-k 5
```

O processo gera um CSV com as metricas por pergunta e imprime no terminal os resumos medios de precision, recall, F1 e MRR para resposta e recuperacao.

Para incluir as metricas do RAGAS, execute o mesmo comando com a flag `--use-ragas` e garanta que o Ollama esteja rodando com os modelos locais disponiveis. Nesse modo, a avaliacao passa a medir:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall

Exemplo:

```powershell
python avaliar.py --pdf caminho/do/documento.pdf --dataset backend/evaluation/dataset_exemplo.json --output backend/evaluation/results.csv --top-k 5 --use-ragas
```

Antes de rodar, voce pode preparar os modelos locais assim:

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b-instruct
```

Se quiser usar outro modelo de juiz, informe `--ragas-llm-model` e `--ragas-embedding-model` no comando, ou configure `RAGAS_LLM_MODEL` e `RAGAS_EMBEDDING_MODEL` no ambiente.

### Exemplo de execucao validada

Na execucao realizada em 27/04/2026 com backend/pdf/artigoteste.pdf e backend/evaluation/dataset_exemplo.json, o pipeline gerou o arquivo backend/evaluation/results.csv com os resultados por pergunta. Os valores medios obtidos foram:

- Answer Precision: 0.0429
- Answer Recall: 0.2727
- Answer F1: 0.0741
- Retrieval Precision: 0.2000
- Retrieval Recall: 0.5000
- Retrieval F1: 0.2857
- Retrieval MRR: 0.2500

Isso documenta que a avaliacao foi executada de ponta a ponta e tambem evidencia os pontos de fragilidade do sistema, principalmente na qualidade da resposta final e na relevancia dos trechos recuperados.
