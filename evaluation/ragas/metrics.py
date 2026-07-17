from typing import Any, List

from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_correctness,
    answer_relevancy,
)


def get_ragas_metrics() -> List[Any]:
    """
    Returns the list of classic Ragas metrics instances used for evaluation.

    Returns:
        List of Ragas metrics.
    """
    return [
        context_precision,
        context_recall,
        faithfulness,
        answer_correctness,
        answer_relevancy,
    ]
