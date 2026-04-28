import pandas as pd

from backend.evaluation.run_evaluation import extract_ragas_scores


def test_extract_ragas_scores_builds_row_scores_and_averages():
    result_frame = pd.DataFrame(
        [
            {
                "faithfulness": 0.25,
                "answer_relevancy": 0.50,
                "context_precision": 0.75,
                "context_recall": 1.0,
            },
            {
                "faithfulness": 0.75,
                "answer_relevancy": 0.25,
                "context_precision": 0.50,
                "context_recall": 0.25,
            },
        ]
    )

    scores = extract_ragas_scores(result_frame)

    assert scores["rows"][0]["ragas_faithfulness"] == 0.25
    assert scores["rows"][0]["ragas_context_recall"] == 1.0
    assert scores["rows"][1]["ragas_answer_relevancy"] == 0.25
    assert scores["averages"]["ragas_faithfulness_mean"] == 0.5
    assert scores["averages"]["ragas_context_precision_mean"] == 0.625