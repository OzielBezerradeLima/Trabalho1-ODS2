import re
from typing import Iterable, List, Sequence, Set


TOKEN_REGEX = re.compile(r"[a-zA-Z0-9_À-ÿ]+")


def normalize_tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_REGEX.findall(text)]


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def precision_recall_f1(predicted_tokens: Sequence[str], reference_tokens: Sequence[str]):
    pred_set = set(predicted_tokens)
    ref_set = set(reference_tokens)

    true_positive = len(pred_set.intersection(ref_set))
    precision = safe_div(true_positive, len(pred_set))
    recall = safe_div(true_positive, len(ref_set))
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def token_f1(predicted_text: str, reference_text: str):
    return precision_recall_f1(
        normalize_tokens(predicted_text),
        normalize_tokens(reference_text),
    )


def _is_relevant(document_text: str, expected_terms: Iterable[str]) -> bool:
    lowered = document_text.lower()
    terms = [term.strip().lower() for term in expected_terms if term.strip()]
    if not terms:
        return False
    return any(term in lowered for term in terms)


def retrieval_precision_recall_f1(retrieved_contexts: Sequence[str], expected_terms: Sequence[str]):
    if not retrieved_contexts:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    relevant_retrieved = [
        context
        for context in retrieved_contexts
        if _is_relevant(context, expected_terms)
    ]

    precision = safe_div(len(relevant_retrieved), len(retrieved_contexts))
    recall = 1.0 if relevant_retrieved else 0.0
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def mrr_at_k(retrieved_contexts: Sequence[str], expected_terms: Sequence[str]):
    for index, context in enumerate(retrieved_contexts, start=1):
        if _is_relevant(context, expected_terms):
            return 1.0 / index
    return 0.0
