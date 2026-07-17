"""
Low-level scikit-learn metrics wrapper for classification evaluation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from evaluation.models import ClassificationMetrics

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
) -> ClassificationMetrics:
    """
    Computes classification performance scores using scikit-learn.

    Args:
        y_true: List of ground truth labels.
        y_pred: List of predicted labels.

    Returns:
        ClassificationMetrics object.
    """
    if not y_true or not y_pred:
        logger.warning("Empty labels lists received. Returning zeroed metrics.")
        return ClassificationMetrics(
            accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            precision_weighted=0.0,
            recall_weighted=0.0,
            f1_weighted=0.0,
            confusion_matrix=[[]],
            per_label_metrics={},
        )

    # Compute overall accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Compute macro-averaged scores
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Compute weighted-averaged scores
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Unique labels in y_true and y_pred (sorted to maintain consistency)
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # Compute confusion matrix
    cm_array = confusion_matrix(y_true, y_pred, labels=labels)
    cm_list = cm_array.tolist()

    # Calculate per-label metrics
    per_label: Dict[str, Dict[str, float]] = {}
    p_label, r_label, f1_label, sup_label = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    for i, label in enumerate(labels):
        per_label[label] = {
            "precision": float(p_label[i]),
            "recall": float(r_label[i]),
            "f1_score": float(f1_label[i]),
            "support": float(sup_label[i]),
        }

    return ClassificationMetrics(
        accuracy=acc,
        precision_macro=float(prec_macro),
        recall_macro=float(rec_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(prec_weighted),
        recall_weighted=float(rec_weighted),
        f1_weighted=float(f1_weighted),
        confusion_matrix=cm_list,
        per_label_metrics=per_label,
    )
