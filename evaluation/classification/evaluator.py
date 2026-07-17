"""
Evaluator for classification metrics (status, product_area, request_type).
"""

from __future__ import annotations

import logging
from typing import Dict, List

from evaluation.classification.metrics import calculate_classification_metrics
from evaluation.models import ClassificationMetrics
from evaluation.models import ClassificationPrediction
from evaluation.models import EvaluationSample

logger = logging.getLogger(__name__)


class ClassificationEvaluator:
    """
    Coordinates metrics extraction for the multi-domain classification outputs.
    """

    def evaluate(
        self,
        predictions: List[ClassificationPrediction],
        samples: List[EvaluationSample],
    ) -> Dict[str, ClassificationMetrics]:
        """
        Executes classification evaluations across targets status, request_type, and product_area.

        Args:
            predictions: List of classification prediction outputs from the SupportAgent.
            samples: List of matching ground truth samples.

        Returns:
            Dictionary containing metrics keyed by column name ('request_type', 'product_area', 'status').
        """
        if len(predictions) != len(samples):
            error_msg = f"Size mismatch between predictions ({len(predictions)}) and samples ({len(samples)})."
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Running classification evaluation on {len(samples)} tickets.")

        # Extract true and predicted labels for each field
        req_type_true = [s.expected_request_type for s in samples]
        req_type_pred = [p.predicted_request_type for p in predictions]

        prod_area_true = [s.expected_product_area for s in samples]
        prod_area_pred = [p.predicted_product_area for p in predictions]

        status_true = [s.expected_status for s in samples]
        status_pred = [p.predicted_status for p in predictions]

        # Calculate metrics for each category
        logger.info("Computing metrics for request_type classification.")
        req_type_metrics = calculate_classification_metrics(req_type_true, req_type_pred)

        logger.info("Computing metrics for product_area classification.")
        prod_area_metrics = calculate_classification_metrics(prod_area_true, prod_area_pred)

        logger.info("Computing metrics for status classification.")
        status_metrics = calculate_classification_metrics(status_true, status_pred)

        logger.info("Classification evaluation completed successfully.")
        return {
            "request_type": req_type_metrics,
            "product_area": prod_area_metrics,
            "status": status_metrics,
        }
