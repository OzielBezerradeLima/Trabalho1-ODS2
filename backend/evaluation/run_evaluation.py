import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.llm.chat import generate_rag_answer
from backend.llm.model import load_llm
from backend.rag.database import get_vector_db, retrieve_context
from backend.rag.ingestion import processar_documento

from backend.evaluation.metrics import mrr_at_k, retrieval_precision_recall_f1, token_f1


def parse_args():
    parser = argparse.ArgumentParser(description="Avaliacao de Qualidade do RAG")
    parser.add_argument("--pdf", required=True, help="Caminho do PDF a ser indexado")
    parser.add_argument("--dataset", required=True, help="Arquivo JSON com perguntas de avaliacao")
    parser.add_argument("--output", default="backend/evaluation/results.csv", help="Arquivo CSV de saida")
    parser.add_argument("--top-k", type=int, default=5, help="Quantidade de chunks recuperados")
    parser.add_argument("--use-ragas", action="store_true", help="Ativa avaliacao adicional com RAGAS")
    parser.add_argument(
        "--ragas-llm-model",
        default=os.getenv("RAGAS_LLM_MODEL", "llama3.2:3b-instruct"),
        help="Modelo Ollama usado como juiz do RAGAS",
    )
    parser.add_argument(
        "--ragas-embedding-model",
        default=os.getenv("RAGAS_EMBEDDING_MODEL", "nomic-embed-text"),
        help="Modelo de embeddings Ollama usado pelo RAGAS",
    )
    return parser.parse_args()


def load_dataset(dataset_path):
    with open(dataset_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_ragas_scores(result_frame):
    metric_columns = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    rows_scores = []
    for _, row in result_frame.iterrows():
        row_scores = {}
        for column in metric_columns:
            row_scores[f"ragas_{column}"] = float(row[column])
        rows_scores.append(row_scores)

    averages = {}
    for column in metric_columns:
        averages[f"ragas_{column}_mean"] = float(result_frame[column].mean())

    return {
        "rows": rows_scores,
        "averages": averages,
    }


def optional_ragas_scores(rows, llm_model, embedding_model):
    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except Exception as error:
        print(f"RAGAS indisponivel neste ambiente: {error}")
        return {}

    ragas_dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row["predicted_answer"] for row in rows],
            "contexts": [row["retrieved_contexts"] for row in rows],
            "ground_truth": [row["expected_answer"] for row in rows],
        }
    )

    llm = ChatOllama(model=llm_model, temperature=0)
    embeddings = OllamaEmbeddings(model=embedding_model)

    try:
        result = evaluate(
            ragas_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
    except Exception as error:
        print(f"Falha ao executar RAGAS com Ollama local: {error}")
        return {}

    return extract_ragas_scores(result.to_pandas())


def run_evaluation(args):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    collection_name = f"evaluation-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    persist_directory = os.path.join("./db/chroma_db", collection_name)

    processar_documento(
        args.pdf,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    vector_db = get_vector_db(
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    if not vector_db:
        raise RuntimeError("Falha ao carregar banco vetorial apos indexacao")

    model, tokenizer, device = load_llm()
    if not model or not tokenizer:
        raise RuntimeError("Falha ao carregar modelo LLM para avaliacao")

    dataset = load_dataset(args.dataset)
    rows = []

    for item in dataset:
        question = item["question"]
        expected_answer = item["expected_answer"]
        expected_terms = item.get("expected_context_terms", [])

        docs, context = retrieve_context(vector_db, question, k=args.top_k)
        retrieved_contexts = [doc.page_content for doc in docs]

        predicted_answer = generate_rag_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            context=context,
            device=device,
        )

        answer_scores = token_f1(predicted_answer, expected_answer)
        retrieval_scores = retrieval_precision_recall_f1(retrieved_contexts, expected_terms)

        rows.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "predicted_answer": predicted_answer,
                "answer_precision": answer_scores["precision"],
                "answer_recall": answer_scores["recall"],
                "answer_f1": answer_scores["f1"],
                "retrieval_precision": retrieval_scores["precision"],
                "retrieval_recall": retrieval_scores["recall"],
                "retrieval_f1": retrieval_scores["f1"],
                "retrieval_mrr": mrr_at_k(retrieved_contexts, expected_terms),
                "retrieved_contexts": " ||| ".join(retrieved_contexts),
            }
        )

    ragas_scores = optional_ragas_scores(rows, args.ragas_llm_model, args.ragas_embedding_model) if args.use_ragas else {}

    with open(args.output, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "question",
            "expected_answer",
            "predicted_answer",
            "answer_precision",
            "answer_recall",
            "answer_f1",
            "retrieval_precision",
            "retrieval_recall",
            "retrieval_f1",
            "retrieval_mrr",
            "retrieved_contexts",
        ]

        if ragas_scores:
            fieldnames.extend(
                [
                    "ragas_faithfulness",
                    "ragas_answer_relevancy",
                    "ragas_context_precision",
                    "ragas_context_recall",
                ]
            )

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, row in enumerate(rows):
            row_data = dict(row)
            if ragas_scores:
                row_data.update(ragas_scores["rows"][index])
            writer.writerow(row_data)

    averages = {
        "answer_precision": sum(row["answer_precision"] for row in rows) / len(rows),
        "answer_recall": sum(row["answer_recall"] for row in rows) / len(rows),
        "answer_f1": sum(row["answer_f1"] for row in rows) / len(rows),
        "retrieval_precision": sum(row["retrieval_precision"] for row in rows) / len(rows),
        "retrieval_recall": sum(row["retrieval_recall"] for row in rows) / len(rows),
        "retrieval_f1": sum(row["retrieval_f1"] for row in rows) / len(rows),
        "retrieval_mrr": sum(row["retrieval_mrr"] for row in rows) / len(rows),
    }

    print("=== RESUMO DA AVALIACAO ===")
    print(f"Perguntas avaliadas: {len(rows)}")
    print(f"Answer Precision medio: {averages['answer_precision']:.4f}")
    print(f"Answer Recall medio: {averages['answer_recall']:.4f}")
    print(f"Answer F1 medio: {averages['answer_f1']:.4f}")
    print(f"Retrieval Precision medio: {averages['retrieval_precision']:.4f}")
    print(f"Retrieval Recall medio: {averages['retrieval_recall']:.4f}")
    print(f"Retrieval F1 medio: {averages['retrieval_f1']:.4f}")
    print(f"Retrieval MRR medio: {averages['retrieval_mrr']:.4f}")
    if ragas_scores:
        print("RAGAS habilitado com metricas adicionais.")
        print(f"RAGAS Faithfulness medio: {ragas_scores['averages']['ragas_faithfulness_mean']:.4f}")
        print(f"RAGAS Answer Relevancy medio: {ragas_scores['averages']['ragas_answer_relevancy_mean']:.4f}")
        print(f"RAGAS Context Precision medio: {ragas_scores['averages']['ragas_context_precision_mean']:.4f}")
        print(f"RAGAS Context Recall medio: {ragas_scores['averages']['ragas_context_recall_mean']:.4f}")
    print(f"CSV salvo em: {args.output}")


def main():
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
