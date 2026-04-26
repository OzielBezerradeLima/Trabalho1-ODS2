# Trabalho1-ODS2 - Frontend React + Backend FastAPI

## Arquitetura

- Frontend: React (pasta `frontend/`)
- Backend: FastAPI (`backend/main.py`)
- RAG: pipeline Python existente em `rag/`, `llm/`, `pdf/`

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
