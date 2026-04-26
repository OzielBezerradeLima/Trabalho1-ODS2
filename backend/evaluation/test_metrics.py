from backend.evaluation.metrics import mrr_at_k, retrieval_precision_recall_f1, token_f1


def test_token_f1_has_overlap():
    scores = token_f1(
        "nivel de carga leve e media",
        "niveis de carga leve media e pesada",
    )
    assert scores["f1"] > 0


def test_retrieval_scores_detect_relevant_context():
    contexts = [
        "O documento descreve niveis de carga e sua classificacao.",
        "Trecho sem relacao.",
    ]
    scores = retrieval_precision_recall_f1(contexts, ["niveis de carga", "classificacao"])

    assert scores["precision"] == 0.5
    assert scores["recall"] == 1.0
    assert scores["f1"] > 0


def test_mrr_at_k_returns_inverse_rank():
    contexts = [
        "irrelevante",
        "a classificacao considera demanda e faixa",
        "outro trecho",
    ]
    score = mrr_at_k(contexts, ["demanda"])
    assert score == 0.5
